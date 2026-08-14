"""
Trusty Chapter 12 — Generic retry/backoff decorator.

Wraps any callable that might fail transiently (network hiccup, rate
limit, temporary unavailability) with exponential backoff up to a
configurable cap.

Designed as a shared primitive per blueprint §9.7:
- Option 1: wraps ask_llm() for Groq 429s and transient errors
- Option 2 (future): wraps sandbox execution calls
- Option 3 (future): wraps peer-agent A2A requests

The decorator itself never changes between options — only the function
it wraps and the retry parameters change.
"""

import logging
import time
from functools import wraps
from typing import Callable, Tuple, Type

logger = logging.getLogger(__name__)


def with_retry(
    max_attempts: int = 3,
    initial_delay_s: float = 1.0,
    backoff_factor: float = 2.0,
    max_delay_s: float = 16.0,
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,),
    retryable_status_codes: Tuple[int, ...] = (429, 500, 502, 503, 504),
):
    """Decorator factory: wrap a function with exponential backoff retry.

    Args:
        max_attempts:          Total attempts including the first (not just retries).
                               max_attempts=3 means: try, fail, wait, try, fail, wait, try.
        initial_delay_s:       Seconds to wait before the second attempt.
        backoff_factor:        Multiply delay by this after each failure.
                               initial_delay_s=1, backoff_factor=2 → waits: 1s, 2s, 4s...
        max_delay_s:           Cap on how long a single wait can be.
        retryable_exceptions:  Exception types that trigger a retry.
                               Non-listed exceptions propagate immediately — no retry.
        retryable_status_codes: HTTP status codes that trigger a retry, checked on
                               exceptions that carry a `.status_code` attribute
                               (e.g. openai.APIStatusError).
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay_s
            last_exception = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)

                except retryable_exceptions as exc:
                    last_exception = exc

                    # If the exception carries a status code, only retry
                    # on the codes we've declared retryable — let others
                    # propagate immediately even if the exception type matches.
                    status_code = getattr(exc, "status_code", None)
                    if status_code is not None and status_code not in retryable_status_codes:
                        logger.warning(
                            f"{func.__name__} failed with non-retryable "
                            f"status {status_code}: {exc}"
                        )
                        raise

                    if attempt == max_attempts:
                        logger.error(
                            f"{func.__name__} failed after {max_attempts} "
                            f"attempts. Last error: {exc}"
                        )
                        raise

                    logger.warning(
                        f"{func.__name__} attempt {attempt}/{max_attempts} "
                        f"failed ({exc}). Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                    delay = min(delay * backoff_factor, max_delay_s)

            # Unreachable — the loop always either returns or raises,
            # but satisfies type checkers that want an explicit exit.
            raise last_exception  # pragma: no cover

        return wrapper
    return decorator