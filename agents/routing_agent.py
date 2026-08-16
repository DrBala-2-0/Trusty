"""
Trusty routing agent — Chapter 22.
Classifies an incoming question into one of four routing paths:

    rag_only                       — answer from uploaded documents alone (Option 1)
    rag_with_analysis              — answer requires Python code execution (Option 2)
    rag_with_external              — answer requires data from a peer Trusty instance (Option 3)
    rag_with_analysis_and_external — answer requires both code execution AND peer data
                                     (Option 2 + Option 3)

The routing decision is made before the relevance check so the correct
pipeline path is chosen before any retrieval work begins.

Design principle: when in doubt, route to rag_only. The more capable
paths are only chosen when the question clearly signals a need for them.
False negatives (routing a computation question to rag_only) are
recoverable — the answer will be weaker but the system won't crash.
False positives (routing a simple question to analysis) waste sandbox
time and budget.
"""
from __future__ import annotations

from utils.llm_client import ask_llm
from utils.logging import logger


ROUTING_SYSTEM = """You are a question router for a document question-answering system.
Classify the question into exactly one of these four categories:

rag_only                       — The question can be answered from uploaded documents
                                 using text retrieval alone. No computation or external
                                 data needed.
                                 Examples: "What does the report say about X?",
                                 "Summarise section 3", "Who is mentioned in the document?"

rag_with_analysis              — The question requires computation, statistics, charts,
                                 or data analysis on structured data (CSV, Excel, numbers).
                                 No external data needed.
                                 Examples: "What is the total revenue?",
                                 "Show a chart of sales by region",
                                 "Which product has the highest margin?"

rag_with_external              — The question requires data from an external source or
                                 peer instance. No computation needed.
                                 Examples: "Compare our results with industry average",
                                 "What do other sources say about this?",
                                 "Get the latest benchmark data for X"

rag_with_analysis_and_external — The question requires BOTH computation on local data
                                 AND data from an external source or peer instance.
                                 Examples: "Compare our Q3 revenue trend with industry average",
                                 "Show a chart of our sales vs the market benchmark",
                                 "How does our margin compare to sector data from peers?"

Rules:
- Respond with ONLY the category name, exactly as written above
- No explanation, no punctuation, nothing else
- Default to rag_only when uncertain
"""

ROUTING_USER = "Question: {question}"

VALID_ROUTES = {
    "rag_only",
    "rag_with_analysis",
    "rag_with_external",
    "rag_with_analysis_and_external",
}


def route(question: str) -> str:
    """
    Classify a question into a routing path.

    Always returns one of:
        "rag_only"
        "rag_with_analysis"
        "rag_with_external"
        "rag_with_analysis_and_external"

    Falls back to "rag_only" on any error or unexpected model response.
    """
    prompt = f"{ROUTING_SYSTEM}\n\n{ROUTING_USER.format(question=question)}"
    try:
        response = ask_llm(prompt).strip().lower()
        # Extract just the first token in case the model adds punctuation
        first_word = response.split()[0].rstrip(".,;:") if response else ""
        if first_word in VALID_ROUTES:
            logger.info(f"[router] question={question[:60]!r} -> {first_word}")
            return first_word
        # Model returned something unexpected — default to rag_only
        logger.warning(
            f"[router] Unexpected response {response!r} — defaulting to rag_only"
        )
        return "rag_only"
    except Exception as e:
        logger.error(f"[router] Routing failed: {e} — defaulting to rag_only")
        return "rag_only"