from typing import TypedDict, List, Dict, Optional

from langgraph.graph import StateGraph, END

from utils.logging import logger
from utils.tracer import Tracer
from .research_agent import ResearchAgent
from .verification_agent import VerificationAgent
from .relevance_checker import RelevanceChecker

MAX_RESEARCH_ATTEMPTS = 2


class AgentState(TypedDict):
    question: str
    documents: List[dict]
    draft_answer: str
    verification_report: str
    parsed_report: dict
    is_relevant: bool
    research_attempts: int
    tracer: Optional[object]   # Tracer instance, injected per request


class AgentWorkflow:
    def __init__(self):
        self.researcher = ResearchAgent()
        self.verifier = VerificationAgent()
        self.relevance_checker = RelevanceChecker()
        self.compiled_workflow = self.build_workflow()

    def build_workflow(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("check_relevance", self._check_relevance_step)
        workflow.add_node("research", self._research_step)
        workflow.add_node("verify", self._verification_step)
        workflow.set_entry_point("check_relevance")
        workflow.add_conditional_edges(
            "check_relevance",
            self._decide_after_relevance_check,
            {"relevant": "research", "irrelevant": END},
        )
        workflow.add_edge("research", "verify")
        workflow.add_conditional_edges(
            "verify",
            self._decide_next_step,
            {"re_research": "research", "end": END},
        )
        return workflow.compile()

    def _check_relevance_step(self, state: AgentState) -> Dict:
        tracer: Optional[Tracer] = state.get("tracer")
        classification = self.relevance_checker.check(
            state["question"], state["documents"]
        )
        is_relevant = classification in ("CAN_ANSWER", "PARTIAL")

        if tracer:
            tracer.record_relevance(classification, is_relevant)

        if is_relevant:
            return {"is_relevant": True}
        return {
            "is_relevant": False,
            "draft_answer": "This question isn't related to (or isn't covered by) the provided documents.",
        }

    def _decide_after_relevance_check(self, state: AgentState) -> str:
        return "relevant" if state["is_relevant"] else "irrelevant"

    def _research_step(self, state: AgentState) -> Dict:
        tracer: Optional[Tracer] = state.get("tracer")
        result = self.researcher.generate(
            state["question"], state["documents"]
        )
        attempt = state.get("research_attempts", 0) + 1

        if tracer:
            tracer.record_research(attempt, result["draft_answer"])

        return {
            "draft_answer": result["draft_answer"],
            "research_attempts": attempt,
        }

    def _verification_step(self, state: AgentState) -> Dict:
        tracer: Optional[Tracer] = state.get("tracer")
        result = self.verifier.check(
            state["draft_answer"], state["documents"]
        )
        parsed = result["parsed_report"]

        if tracer:
            tracer.record_verification(
                attempt=state.get("research_attempts", 1),
                supported=parsed.get("Supported", ""),
                relevant=parsed.get("Relevant", ""),
                unsupported_claims=parsed.get("Unsupported Claims", ""),
                contradictions=parsed.get("Contradictions", ""),
            )

        return {
            "verification_report": result["raw_report"],
            "parsed_report": parsed,
        }

    def _decide_next_step(self, state: AgentState) -> str:
        tracer: Optional[Tracer] = state.get("tracer")
        parsed = state.get("parsed_report", {})
        attempts = state.get("research_attempts", 0)
        supported = parsed.get("Supported", "").strip().upper()
        relevant = parsed.get("Relevant", "").strip().upper()
        needs_retry = supported != "YES" or relevant != "YES"

        if needs_retry and attempts < MAX_RESEARCH_ATTEMPTS:
            logger.info(
                f"Verification failed (attempt {attempts}). Re-researching."
            )
            if tracer:
                tracer.record_retry(
                    attempt=attempts,
                    reason=f"Supported={supported}, Relevant={relevant}",
                )
            return "re_research"

        if needs_retry:
            logger.warning(
                f"Max research attempts ({MAX_RESEARCH_ATTEMPTS}) reached. "
                f"Returning best-effort answer."
            )
            if tracer:
                tracer.record_outcome("best_effort_max_retries_reached")
        else:
            if tracer:
                tracer.record_outcome("verified_pass")

        return "end"

    def full_pipeline(
        self, question: str, documents: list, tracer: Optional[Tracer] = None
    ) -> Dict:
        initial_state = AgentState(
            question=question,
            documents=documents,
            draft_answer="",
            verification_report="",
            parsed_report={},
            is_relevant=False,
            research_attempts=0,
            tracer=tracer,
        )
        final_state = self.compiled_workflow.invoke(initial_state)

        # Record the irrelevant-question outcome here since
        # _decide_after_relevance_check routes to END directly,
        # bypassing _decide_next_step where other outcomes are recorded.
        if tracer and not final_state.get("is_relevant"):
            tracer.record_outcome("irrelevant_no_answer")

        return {
            "draft_answer": final_state["draft_answer"],
            "verification_report": final_state.get("verification_report", ""),
            "parsed_report": final_state.get("parsed_report", {}),
        }