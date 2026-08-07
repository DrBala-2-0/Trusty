import logging
from utils.llm_client import ask_llm

logger = logging.getLogger(__name__)

VERIFICATION_PROMPT_TEMPLATE = """You are an AI assistant designed to verify the accuracy and relevance of answers based on the provided context.

Instructions:
- Verify the following answer against the provided context.
- Check for:
1. Direct/indirect factual support (YES/NO)
2. Relevance to the question (YES/NO)
- Respond in the exact format specified below without adding any unrelated information.

Format:
Supported: YES/NO
Relevant: YES/NO

Answer: {answer}
Context:
{context}

Respond ONLY with the above format.
"""


class VerificationAgent:
    def check(self, answer: str, documents: list) -> dict:
        context = "\n\n".join(doc.page_content for doc in documents)
        prompt = VERIFICATION_PROMPT_TEMPLATE.format(answer=answer, context=context)

        try:
            response = ask_llm(prompt).strip()
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            response = "Supported: NO\nRelevant: NO"

        return {"verification_report": response, "context_used": context}