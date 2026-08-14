"""
Trusty Chapter 14 -- Response cache.

Caches /ask pipeline results keyed by (normalised question, content hash
of the session's current document set). A cache hit returns the stored
result immediately, skipping the full relevance->research->verification
pipeline and consuming no LLM budget.

Design decisions:
- In-memory only, consistent with session_docs/retrievers (Ch7's deliberate
  choice -- no persistence across restarts).
- Content-hash keying: the cache key includes a SHA-256 hash of the
  session's full accumulated document text, not just a session ID. This
  means a new upload (which changes the document set) automatically
  invalidates all prior cache entries for that session -- no explicit
  invalidation logic needed.
- Question normalisation: strip whitespace and lowercase before hashing,
  so "What is X?" and "  what is x?  " hit the same cache entry.
- Size cap: a simple LRU-style eviction (drop oldest entry) keeps memory
  bounded. Default cap of 500 entries is conservative for a local dev app.
- Cache hits are flagged in the response with "cached": True so the caller
  (and the UI) can distinguish a live pipeline result from a stored one.
"""

import hashlib
from collections import OrderedDict
from typing import Optional


class ResponseCache:
    """In-memory response cache keyed by (question, document-content hash).

    Args:
        max_size: Maximum number of entries before oldest are evicted.
                  Each entry is one cached /ask response dict.
    """

    def __init__(self, max_size: int = 500):
        if max_size <= 0:
            raise ValueError(f"max_size must be positive, got {max_size}.")
        self._max_size = max_size
        self._store: OrderedDict[str, dict] = OrderedDict()

    # ------------------------------------------------------------------
    # Key construction
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_question(question: str) -> str:
        """Strip whitespace and lowercase for consistent key matching."""
        return question.strip().lower()

    @staticmethod
    def _docs_hash(docs: list) -> str:
        """SHA-256 of the full concatenated text of all session documents.

        Using content rather than session ID means:
        - A new upload changes the hash -> old entries don't match -> miss.
        - Two sessions that happen to upload the same document share cache
          entries, which is both correct and efficient.
        """
        combined = "".join(doc.page_content for doc in docs)
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    def _make_key(self, question: str, docs: list) -> str:
        q = self._normalise_question(question)
        h = self._docs_hash(docs)
        return f"{h}::{q}"

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def get(self, question: str, docs: list) -> Optional[dict]:
        """Return the cached result for this question+docs pair, or None.

        Moves the entry to the end of the OrderedDict (most-recently-used)
        so it is the last to be evicted when the cache is full.
        """
        key = self._make_key(question, docs)
        if key not in self._store:
            return None
        # Move to end -- marks as recently used
        self._store.move_to_end(key)
        return self._store[key]

    def set(self, question: str, docs: list, result: dict) -> None:
        """Store a result for this question+docs pair.

        Evicts the oldest entry if the cache is at capacity.
        Never caches a result that the pipeline flagged as a failure
        (verified=NO, or the fallback "best_effort" answer) -- those
        should be retried, not served from cache indefinitely.
        """
        # Don't cache best-effort or unsupported results
        parsed = result.get("parsed_report", {})
        supported = parsed.get("Supported", "").strip().upper()
        if supported == "NO":
            return

        key = self._make_key(question, docs)
        if key in self._store:
            self._store.move_to_end(key)
        else:
            if len(self._store) >= self._max_size:
                self._store.popitem(last=False)   # evict oldest
            self._store[key] = result

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        return len(self._store)

    def summary(self) -> dict:
        """Return a dict suitable for logging or a health-check endpoint."""
        return {
            "entries": self.size,
            "max_size": self._max_size,
        }