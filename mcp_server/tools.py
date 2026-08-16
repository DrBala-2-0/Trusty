"""
Trusty MCP server tool definitions — Chapter 21 (§9.16).
Each function here is a named MCP tool that wraps an existing
FastAPI endpoint. No logic lives here — only translation between
MCP tool call arguments and HTTP requests to localhost FastAPI.

Tools exposed:
    trusty_upload       — ingest a document into a session
    trusty_ask          — ask a question, get a verified answer
    trusty_run_analysis — run sandboxed data analysis
"""
from __future__ import annotations

import requests

FASTAPI_BASE = "http://localhost:8000"


def trusty_ask(
    question: str,
    session_id: str,
    response_format: str = "text",
) -> dict:
    """
    Ask Trusty a question against previously uploaded documents.

    Args:
        question        : the question to ask
        session_id      : session with documents already uploaded
        response_format : "text" (default), "json", "markdown_table"

    Returns a dict with: answer, sources, verification, confidence,
    supported, relevant, cached, verification_mode.
    """
    response = requests.post(
        f"{FASTAPI_BASE}/ask",
        json={"text": question, "response_format": response_format},
        headers={"X-Session-ID": session_id},
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()
    parsed = data.get("parsed_report", {})
    return {
        "answer": data.get("draft_answer", ""),
        "formatted_answer": data.get("formatted_answer", ""),
        "sources": [
            s.get("source", "unknown")
            for s in data.get("chunk_sources", [])
        ],
        "supported": parsed.get("Supported", "NO"),
        "relevant": parsed.get("Relevant", "NO"),
        "confidence": data.get("confidence", "LOW"),
        "cached": data.get("cached", False),
        "verification_mode": data.get("verification_mode", "document"),
    }


def trusty_run_analysis(
    question: str,
    session_id: str,
    data_description: str,
    data_csv: str,
    sandbox_backend: str = "docker",
) -> dict:
    """
    Ask Trusty a data analysis question, executing Python in the sandbox.

    Args:
        question         : the analysis question
        session_id       : session with documents already uploaded
        data_description : description of the DataFrame
        data_csv         : raw CSV content to analyse
        sandbox_backend  : "docker" (default) or "colab"

    Returns a dict with: answer, code, stdout, chart_b64,
    supported, verification_mode, backend, error.
    """
    response = requests.post(
        f"{FASTAPI_BASE}/ask",
        json={
            "text": question,
            "enable_analysis": True,
            "data_description": data_description,
            "data_csv": data_csv,
            "sandbox_backend": sandbox_backend,
        },
        headers={"X-Session-ID": session_id},
        timeout=180,
    )
    response.raise_for_status()
    data = response.json()
    code_result = data.get("code_result") or {}
    return {
        "answer": data.get("draft_answer", ""),
        "code": code_result.get("code", ""),
        "stdout": code_result.get("stdout", ""),
        "chart_b64": code_result.get("chart_b64"),
        "supported": data.get("parsed_report", {}).get("Supported", "NO"),
        "verification_mode": data.get("verification_mode", "document"),
        "backend": code_result.get("backend", "unknown"),
        "error": code_result.get("error"),
    }


def trusty_upload(
    file_path: str,
    session_id: str,
) -> dict:
    """
    Upload a file to a Trusty session.

    Args:
        file_path  : local path to the file to upload
        session_id : session to upload into

    Returns a dict with: status, chunks_added, chunks_total.
    """
    with open(file_path, "rb") as f:
        filename = file_path.replace("\\", "/").split("/")[-1]
        response = requests.post(
            f"{FASTAPI_BASE}/upload",
            files={"file": (filename, f)},
            headers={"X-Session-ID": session_id},
            timeout=120,
        )
    response.raise_for_status()
    data = response.json()
    return {
        "status": data.get("status", "unknown"),
        "chunks_added": data.get("chunks_added", 0),
        "chunks_total": data.get("chunks_total", 0),
    }


# Tool registry — maps tool name to function and description
# Used by mcp_server/server.py to register tools with the MCP framework
TOOL_REGISTRY = {
    "trusty_ask": {
        "fn": trusty_ask,
        "description": (
            "Ask Trusty a question against uploaded documents. "
            "Returns a verified, sourced answer with confidence level."
        ),
    },
    "trusty_run_analysis": {
        "fn": trusty_run_analysis,
        "description": (
            "Run a data analysis question using sandboxed Python execution. "
            "Returns the answer, generated code, stdout, and optional chart."
        ),
    },
    "trusty_upload": {
        "fn": trusty_upload,
        "description": (
            "Upload a local file to a Trusty session for subsequent querying."
        ),
    },
}