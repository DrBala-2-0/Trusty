"""
Trusty Chapter 11 — Per-request decision tracer.

A Tracer is created fresh for every /ask call, accumulates structured
events as the pipeline runs, and is serialised into the response's
`trace` field at the end. It is stateless between requests — nothing
here persists to disk or survives the request lifecycle.

Design note: the tracer is passed explicitly into the pipeline rather
than stored in a global or thread-local. This keeps it visible in the
call graph (you can always see where the trace comes from) and makes
it trivially safe under concurrent requests — each request owns its
own Tracer instance with no shared state.
"""

import time
from typing import Any


class Tracer:
    def __init__(self):
        self._start = time.perf_counter()
        self._steps: list[dict] = []

    # ------------------------------------------------------------------
    # Recording methods — one per pipeline step
    # ------------------------------------------------------------------

    def record_retrieval(self, session_id: str, chunk_count: int, chunks: list) -> None:
        """Record what the retriever returned for this question."""
        self._steps.append({
            "step": "retrieval",
            "session_id": session_id,
            "chunks_retrieved": chunk_count,
            "chunk_sources": list({
                c.metadata.get("source", "unknown") for c in chunks
            }),
        })

    def record_relevance(self, label: str, is_relevant: bool) -> None:
        """Record the relevance checker's classification and decision."""
        self._steps.append({
            "step": "relevance_check",
            "label": label,
            "is_relevant": is_relevant,
            "elapsed_s": round(time.perf_counter() - self._start, 3),
        })

    def record_research(self, attempt: int, draft_answer: str) -> None:
        """Record each research attempt — called once per attempt including retries."""
        self._steps.append({
            "step": "research",
            "attempt": attempt,
            "draft_answer_preview": draft_answer[:200],
            "elapsed_s": round(time.perf_counter() - self._start, 3),
        })

    def record_verification(
        self,
        attempt: int,
        supported: str,
        relevant: str,
        unsupported_claims: str,
        contradictions: str,
    ) -> None:
        """Record the verification agent's parsed report for each attempt."""
        self._steps.append({
            "step": "verification",
            "attempt": attempt,
            "supported": supported,
            "relevant": relevant,
            "unsupported_claims": unsupported_claims,
            "contradictions": contradictions,
            "elapsed_s": round(time.perf_counter() - self._start, 3),
        })

    def record_retry(self, attempt: int, reason: str) -> None:
        """Record that a retry was triggered and why."""
        self._steps.append({
            "step": "retry_triggered",
            "attempt": attempt,
            "reason": reason,
            "elapsed_s": round(time.perf_counter() - self._start, 3),
        })

    def record_outcome(self, outcome: str) -> None:
        """Record the final pipeline outcome label."""
        self._steps.append({
            "step": "outcome",
            "outcome": outcome,
            "total_elapsed_s": round(time.perf_counter() - self._start, 3),
        })

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return the complete trace as a plain dict, ready for JSON serialisation."""
        return {
            "total_elapsed_s": round(time.perf_counter() - self._start, 3),
            "steps": self._steps,
        }