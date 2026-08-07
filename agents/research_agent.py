import logging
from utils.llm_client import ask_llm

logger = logging.getLogger(__name__)

RESEARCH_PROMPT_TEMPLATE = """You are an AI assistant designed to provide precise and factual answers based on the given context.

Instructions:
- Answer the following question using only the provided context.
- Be clear, concise, and factual.
- Return as much information as you can get from the context.

Question: {question}
Context:
{context}

Provide your answer below:
"""

FALLBACK_ANSWER = "I cannot answer this question based on the provided documents."


class ResearchAgent:
    def generate(self, question: str, documents: list) -> dict:
        context = "\n\n".join(doc.page_content for doc in documents)
        prompt = RESEARCH_PROMPT_TEMPLATE.format(question=question, context=context)

        try:
            draft_answer = ask_llm(prompt).strip()
        except Exception as e:
            logger.error(f"Research generation failed: {e}")
            draft_answer = FALLBACK_ANSWER

        return {"draft_answer": draft_answer or FALLBACK_ANSWER, "context_used": context}