from typing import TypedDict, List, Dict

from langgraph.graph import StateGraph, END

from utils.logging import logger
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
        classification = self.relevance_checker.check(state["question"], state["documents"])
        if classification in ("CAN_ANSWER", "PARTIAL"):
            return {"is_relevant": True}
        return {
            "is_relevant": False,
            "draft_answer": "This question isn't related to (or isn't covered by) the provided documents.",
        }

    def _decide_after_relevance_check(self, state: AgentState) -> str:
        return "relevant" if state["is_relevant"] else "irrelevant"

    def _research_step(self, state: AgentState) -> Dict:
        result = self.researcher.generate(state["question"], state["documents"])
        return {
            "draft_answer": result["draft_answer"],
            "research_attempts": state.get("research_attempts", 0) + 1,
        }

    def _verification_step(self, state: AgentState) -> Dict:
        result = self.verifier.check(state["draft_answer"], state["documents"])
        return {"verification_report": result["raw_report"], "parsed_report": result["parsed_report"]}

    def _decide_next_step(self, state: AgentState) -> str:
        parsed = state.get("parsed_report", {})
        attempts = state.get("research_attempts", 0)
        supported = parsed.get("Supported", "").strip().upper()
        relevant = parsed.get("Relevant", "").strip().upper()
        needs_retry = supported != "YES" or relevant != "YES"

        if needs_retry and attempts < MAX_RESEARCH_ATTEMPTS:
            logger.info(f"Verification failed (attempt {attempts}). Re-researching.")
            return "re_research"
        if needs_retry:
            logger.warning(f"Max research attempts ({MAX_RESEARCH_ATTEMPTS}) reached. Returning best-effort answer.")
        return "end"

    def full_pipeline(self, question: str, documents: list) -> Dict:
        initial_state = AgentState(
            question=question, documents=documents, draft_answer="",
            verification_report="", parsed_report={}, is_relevant=False, research_attempts=0,
        )
        final_state = self.compiled_workflow.invoke(initial_state)
        return {
            "draft_answer": final_state["draft_answer"],
            "verification_report": final_state.get("verification_report", ""),
            "parsed_report": final_state.get("parsed_report", {}),
        }