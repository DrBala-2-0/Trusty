from fastapi import FastAPI
from pydantic import BaseModel
from utils.llm_client import ask_llm

app = FastAPI(title="Trusty (Chapter 2 — naive version)")

class Question(BaseModel):
    text: str
    context: str  # hand-pasted for now; real retrieval comes in a later chapter

@app.post("/ask")
def ask(q: Question):
    prompt = f"Context:\n{q.context}\n\nQuestion: {q.text}\n\nAnswer using only the context above."
    return {"answer": ask_llm(prompt)}