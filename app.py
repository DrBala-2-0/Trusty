from fastapi import FastAPI
from pydantic import BaseModel
from agents.workflow import AgentWorkflow

app = FastAPI(title="Trusty (Chapter 3 — agent pipeline)")
workflow = AgentWorkflow()  # built once at startup, reused across requests

class Question(BaseModel):
    text: str
    documents: list[str]  # hand-supplied doc chunks for now; real retrieval comes later

@app.post("/ask")
def ask(q: Question):
    docs = [{"content": d} for d in q.documents]
    result = workflow.full_pipeline(q.text, docs)
    return result