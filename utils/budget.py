"""
Trusty Chapter 12 — Parametrizable Budget primitive.

Tracks consumption of a finite resource against a ceiling and raises
a clear, typed error when the ceiling is exceeded.

Designed as a shared primitive per blueprint §9.7:
- Option 1 (now):    dimension="llm_calls", tracks per-session LLM call count
- Option 2 (future): dimension="compute_seconds", tracks sandbox wall-clock time
- Option 3 (future): dimension="agent_calls", tracks outbound A2A request count

The Budget class itself never changes between options — only the
dimension label and limit value change at the call site.
"""

import logging

logger = logging.getLogger(__name__)


class BudgetExceededError(Exception):
    """Raised when a Budget's ceiling has been reached.

    Carries the dimension name and limit so callers can surface a
    meaningful error message rather than a generic exception string.
    """
    def __init__(self, dimension: str, limit: float, used: float):
        self.dimension = dimension
        self.limit = limit
        self.used = used
        super().__init__(
            f"Budget exceeded: {used:.1f}/{limit:.1f} {dimension} used."
        )


class Budget:
    """Track consumption of a finite resource against a hard ceiling.

    Args:
        dimension:  Human-readable label for what's being tracked
                    (e.g. "llm_calls", "compute_seconds", "agent_calls").
                    Used in log messages and error text only — no
                    behaviour depends on the label's value.
        limit:      Hard ceiling. check() raises BudgetExceededError
                    once `used` reaches this value.
        warn_at:    Optional fraction of `limit` at which to log a
                    warning (e.g. 0.8 warns at 80% usage). Pass None
                    to disable warnings entirely.
    """

    def __init__(
        self,
        dimension: str,
        limit: float,
        warn_at: float | None = 0.8,
    ):
        if limit <= 0:
            raise ValueError(f"Budget limit must be positive, got {limit}.")
        if warn_at is not None and not (0 < warn_at < 1):
            raise ValueError(f"warn_at must be between 0 and 1 exclusive, got {warn_at}.")

        self.dimension = dimension
        self.limit = limit
        self.warn_at = warn_at
        self._used: float = 0.0
        self._warned: bool = False   # emit the warning at most once

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def consume(self, amount: float = 1.0) -> None:
        """Record consumption of `amount` units.

        Does not enforce the ceiling — call check() before or after
        consuming to enforce. This separation lets callers decide
        whether to check before (pre-flight) or after (post-flight)
        consuming a unit. Option 1 uses pre-flight: check() before
        the LLM call so the call never happens if already over budget.
        """
        if amount <= 0:
            raise ValueError(f"Consumed amount must be positive, got {amount}.")
        self._used += amount
        self._maybe_warn()

    def check(self) -> None:
        """Raise BudgetExceededError if the ceiling has been reached.

        Call this before an operation (pre-flight) to refuse it when
        over budget, or after (post-flight) to enforce a hard stop
        after the fact. Raises immediately — never silently continues.
        """
        if self._used >= self.limit:
            raise BudgetExceededError(self.dimension, self.limit, self._used)

    def consume_and_check(self, amount: float = 1.0) -> None:
        """Convenience: consume then immediately check.

        Use when you want a strict pre-flight: consume the unit and
        raise if now over budget in a single call. Equivalent to
        calling consume() then check().
        """
        self.consume(amount)
        self.check()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def used(self) -> float:
        return self._used

    @property
    def remaining(self) -> float:
        return max(0.0, self.limit - self._used)

    @property
    def exhausted(self) -> bool:
        return self._used >= self.limit

    def summary(self) -> dict:
        """Return a dict suitable for inclusion in a trace or API response."""
        return {
            "dimension": self.dimension,
            "used": round(self._used, 3),
            "limit": self.limit,
            "remaining": round(self.remaining, 3),
            "exhausted": self.exhausted,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _maybe_warn(self) -> None:
        if self.warn_at is None or self._warned:
            return
        if self._used >= self.limit * self.warn_at:
            logger.warning(
                f"Budget warning: {self._used:.1f}/{self.limit:.1f} "
                f"{self.dimension} used ({self.warn_at * 100:.0f}% threshold reached)."
            )
            self._warned = True