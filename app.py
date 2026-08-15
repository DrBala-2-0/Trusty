import os
import shutil
import subprocess
import sys
from typing import Optional
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from agents.workflow import AgentWorkflow
from document_processor.file_handler import chunk_text, load_and_chunk
from document_processor.url_loader import fetch_url_text
from retriever.builder import RetrieverBuilder
from utils.logging import logger
from utils.session import resolve_session_id
from utils.tracer import Tracer
from utils.budget import Budget, BudgetExceededError
from utils.cache import ResponseCache
from utils.formatter import FormatterError, apply_format, validate_format
from config.settings import settings

app = FastAPI(title="Trusty (Chapter 7 — multi-user retrieval)")
workflow = AgentWorkflow()
retriever_builder = RetrieverBuilder()

# Session-scoped state. Both are plain in-memory dicts — wiped on process
# restart (including uvicorn --reload). That's a deliberate choice, not an
# oversight: see docs/chapters/chapter-7.md for why on-disk session
# persistence isn't warranted by anything the blueprint actually requires.
session_docs: dict[str, list] = {}   # session_id -> every chunk uploaded so far, across all /upload calls
retrievers: dict[str, object] = {}   # session_id -> that session's current hybrid retriever
session_budgets: dict[str, Budget] = {}   # session_id -> that session's LLM call budget

# Maximum LLM calls allowed per session before /ask returns 429.
# One /ask call consumes one unit regardless of how many internal
# agent steps run — the budget tracks requests, not tokens.
SESSION_LLM_CALL_LIMIT = settings.SESSION_LLM_CALL_LIMIT

# Shared across all sessions — keyed by (content hash, normalised question)
# so cross-session cache hits are correct (same docs + same question = same answer).
response_cache = ResponseCache(max_size=settings.RESPONSE_CACHE_MAX_SIZE)


UPLOAD_DIR = ".cache/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.post("/upload")
def upload(
    file: Optional[UploadFile] = File(None),
    url: Optional[str] = Form(None),
    session_id: str = Depends(resolve_session_id),
):
    file_given = file is not None and bool(file.filename)
    if file_given and url:
        raise HTTPException(status_code=400, detail="Provide either a file or a url, not both.")
    if not file_given and not url:
        raise HTTPException(status_code=400, detail="No file or url provided.")

    if url:
        try:
            raw_text = fetch_url_text(url)
        except ValueError as e:
            raise HTTPException(status_code=415, detail=str(e))
        except RuntimeError as e:
            raise HTTPException(status_code=502, detail=str(e))
        new_docs = chunk_text(raw_text, source=url)
        skipped = []
        label = url
    else:
        dest_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(dest_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        try:
            new_docs, skipped = load_and_chunk(dest_path)
        except ValueError as e:
            raise HTTPException(status_code=415, detail=str(e))
        except RuntimeError as e:
            raise HTTPException(status_code=502, detail=str(e))
        label = file.filename

    if not new_docs:
        if skipped:
            raise HTTPException(
                status_code=422,
                detail=f"No extractable content found. All files were skipped: {skipped}",
            )
        raise HTTPException(status_code=422, detail="No extractable text found.")

    session_docs.setdefault(session_id, []).extend(new_docs)
    all_docs = session_docs[session_id]

    retrievers[session_id] = retriever_builder.build_hybrid_retriever(all_docs, session_id)
    logger.info(
        f"[{session_id}] Indexed {label}: +{len(new_docs)} chunks "
        f"({len(all_docs)} total for this session)"
    )
    return {
        "status": "indexed",
        "chunks_added": len(new_docs),
        "chunks_total": len(all_docs),
        "session_id": session_id,
        "skipped": skipped,
    }



class Question(BaseModel):
    text: str
    response_format: str = "text"
    response_template: str | None = None


@app.post("/ask")
def ask(q: Question, session_id: str = Depends(resolve_session_id)):
    if not q.text.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if session_id not in retrievers:
        raise HTTPException(
            status_code=400,
            detail="No document indexed yet for this session. Call /upload first.",
        )

    # Validate format before cache check or any LLM work -- bad requests
    # are rejected immediately, before consuming budget or retrieval time.
    try:
        validate_format(q.response_format, q.response_template)
    except FormatterError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Check cache first -- a hit returns immediately, consuming no budget.
    docs = session_docs.get(session_id, [])
    cached = response_cache.get(q.text, docs)
    if cached is not None:
        if session_id not in session_budgets:
            session_budgets[session_id] = Budget(
                dimension="llm_calls",
                limit=SESSION_LLM_CALL_LIMIT,
                warn_at=0.8,
            )
        budget = session_budgets[session_id]
        logger.info(
            f"[{session_id}] Cache hit for question: {q.text[:60]!r}"
        )
        # Re-apply format on cache hit -- the cached result has draft_answer
        # and parsed_report; chunk_sources was stored at cache time.
        formatted = apply_format(
            q.response_format,
            q.response_template,
            cached["draft_answer"],
            cached.get("parsed_report", {}),
            cached.get("chunk_sources", []),
        )
        return {**cached, "cached": True, "budget": budget.summary(), **formatted}

    # Cache miss -- enforce budget before doing any LLM work.
    if session_id not in session_budgets:
        session_budgets[session_id] = Budget(
            dimension="llm_calls",
            limit=SESSION_LLM_CALL_LIMIT,
            warn_at=0.8,
        )
    budget = session_budgets[session_id]

    try:
        budget.check()
        budget.consume()
    except BudgetExceededError as e:
        logger.warning(f"[{session_id}] Budget exhausted: {e}")
        raise HTTPException(
            status_code=429,
            detail=(
                f"Session budget exhausted: {int(e.used)}/{int(e.limit)} "
                f"LLM calls used. Start a new session to continue."
            ),
        )

    tracer = Tracer()
    try:
        documents = retrievers[session_id].invoke(q.text)
        tracer.record_retrieval(session_id, len(documents), documents)
        result = workflow.full_pipeline(
            q.text,
            documents,
            tracer=tracer,
            response_format=q.response_format,
        )

        # Store chunk_sources in result so cache hits can apply formatting
        # without re-invoking the retriever.
        result["chunk_sources"] = [
            {"source": doc.metadata.get("source", "unknown")} for doc in documents
        ]

        formatted = apply_format(
            q.response_format,
            q.response_template,
            result["draft_answer"],
            result.get("parsed_report", {}),
            documents,
        )
        response_cache.set(q.text, docs, result)
        logger.info(
            f"[{session_id}] Answered question: {q.text[:60]!r} "
            f"(budget: {budget.used:.0f}/{budget.limit:.0f})"
        )
        return {
            **result,
            "cached": False,
            "trace": tracer.to_dict(),
            "budget": budget.summary(),
            **formatted,
        }
    except Exception as e:
        logger.error(f"[{session_id}] Pipeline failed for question {q.text!r}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Something went wrong processing this question: {e}",
        )

    
@app.post("/evaluate")
def evaluate(base_url: str = "http://localhost:8000"):
    """Run the golden test set against a live server and return structured results.

    Calls tests/evaluator.py as a subprocess rather than importing it directly,
    so the evaluator runs in a clean process with no shared state from the
    current server — the same isolation a real CI invocation would have.

    Args:
        base_url: the server URL the evaluator should point at.
                  Defaults to http://localhost:8000 (i.e. itself).
    """
    evaluator_path = os.path.join(os.path.dirname(__file__), "tests", "evaluator.py")

    if not os.path.exists(evaluator_path):
        raise HTTPException(
            status_code=500,
            detail="Evaluator not found. Expected at tests/evaluator.py."
        )

    try:
        proc = subprocess.run(
            [sys.executable, "-X", "utf8", evaluator_path, "--base-url", base_url],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=300,
        )
        passed = proc.returncode == 0
        return {
            "passed": passed,
            "exit_code": proc.returncode,
            "output": proc.stdout,
            "errors": proc.stderr or None,
        }
    except subprocess.TimeoutExpired:
        raise HTTPException(
            status_code=504,
            detail="Evaluation timed out after 5 minutes."
        )
    except Exception as e:
        logger.error(f"Evaluation subprocess failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Evaluation failed to run: {e}"
        )