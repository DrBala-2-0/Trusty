from utils.logging import logger
from utils.llm_client import ask_llm

VALID_LABELS = {"CAN_ANSWER", "PARTIAL", "NO_MATCH"}

RELEVANCE_PROMPT_TEMPLATE = """You are an AI relevance checker between a user's question and provided document content.

Instructions:
- Classify how well the document content addresses the user's question.
- Respond with only one of the following labels: CAN_ANSWER, PARTIAL, NO_MATCH.
- Do not include any additional text or explanation.

Labels:
1) "CAN_ANSWER": The passages contain enough explicit information to fully answer the question.
2) "PARTIAL": The passages mention or discuss the question's topic but do not provide all the details needed for a complete answer.
3) "NO_MATCH": The passages do not discuss or mention the question's topic at all.

Important: If the passages mention or reference the topic or timeframe of the question in any way, even if incomplete, respond with "PARTIAL" instead of "NO_MATCH".

Question: {question}
Passages: {document_content}

Respond ONLY with one of the following labels: CAN_ANSWER, PARTIAL, NO_MATCH
"""


class RelevanceChecker:
    def check(self, question: str, documents: list) -> str:
        if not documents:
            logger.debug("No documents provided. Classifying as NO_MATCH.")
            return "NO_MATCH"

        document_content = "\n\n".join(doc.page_content for doc in documents[:3]) #doc["content"] changed to doc.page_content
        prompt = RELEVANCE_PROMPT_TEMPLATE.format(question=question, document_content=document_content)

        try:
            response = ask_llm(prompt).strip().upper()
        except Exception as e:
            logger.error(f"Relevance check failed: {e}")
            return "NO_MATCH"

        if response not in VALID_LABELS:
            logger.debug(f"Invalid label returned ('{response}'). Forcing NO_MATCH.")
            return "NO_MATCH"
        return response