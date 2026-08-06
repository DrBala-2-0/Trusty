# Trusty

A trustworthy document Q&A app: minimal infrastructure, zero/near-zero cost,
built entirely on free and open-source/open-weight models, APIs, and UI —
while targeting enterprise-grade reliability in *how it fails*, not just how it answers.

Philosophy, closely follows:
- Don't trust a single LLM call to both answer and audit itself.
- Fail closed: an API hiccup should degrade to "I don't know," never a
  confident wrong answer.
- Every non-obvious architectural choice should trace back to that objective.

Stack: Groq (free tier) as primary inference, OpenRouter as fallback,
FastAPI as the orchestration layer, LangGraph for the agent pipeline.