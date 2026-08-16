"""
Trusty code agent — Chapter 18.
Given a user question and a structured data context (column names + sample rows),
writes Python code, executes it in the sandbox, and returns the result.
The agent never knows which sandbox backend ran — that is sandbox_client's concern.
"""
from __future__ import annotations

from typing import Optional

from utils.sandbox_client import run as sandbox_run, SandboxResult
from utils.logging import logger
from utils.tracer import Tracer
from utils.llm_client import ask_llm


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_CODE_SYSTEM = """You are a Python data-analysis expert.
You will be given a question and a description of a pandas DataFrame that is
already loaded as the variable `df`.
Your job is to write a single self-contained block of Python code that answers
the question using `df`.

Rules:
- `df` is already defined — do NOT read any file or fetch any data.
- Print the final answer clearly with print().
- If a chart would help, create it with matplotlib. Do NOT call plt.show().
- Do NOT import anything outside: pandas, matplotlib, scipy, numpy, openpyxl.
- Output ONLY the Python code, no explanation, no markdown fences.
"""

_CODE_USER = """Question: {question}

DataFrame description:
{data_description}

Write the Python code now."""


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def run_code_agent(
    question: str,
    data_description: str,
    data_csv: Optional[str] = None,
    sandbox_backend: str = "docker",
    colab_url: Optional[str] = None,
    tracer: Optional[Tracer] = None,
) -> dict:
    """
    Write and execute Python code to answer `question` about the described data.

    Returns a dict with keys:
        code        : str   — the generated Python code
        stdout      : str   — execution output
        stderr      : str   — execution errors (empty on success)
        chart_b64   : str | None — base64 PNG if a chart was produced
        backend     : str   — which sandbox backend ran
        error       : str | None — sandbox-level error (not a code error)
        supported   : bool  — False if execution failed or produced no output
    """
    # --- Step 1: generate code ---
    if tracer:
        tracer.record_research(0, "[code_agent] generating code")

    prompt = _CODE_USER.format(
        question=question,
        data_description=data_description,
    )
    try:
        code = ask_llm(
            f"{_CODE_SYSTEM}\n\n{prompt}"
        ).strip()
        # Strip markdown fences if the model adds them despite instructions
        if code.startswith("```"):
            lines = code.splitlines()
            code = "\n".join(
                l for l in lines
                if not l.strip().startswith("```")
            ).strip()
    except Exception as e:
        logger.error(f"[code_agent] LLM call failed: {e}")
        return _failure(code="", error=f"Code generation failed: {e}")

    logger.info(f"[code_agent] Generated code ({len(code)} chars)")

    # --- Step 2: execute in sandbox ---
    result: SandboxResult = sandbox_run(
        code=code,
        backend=sandbox_backend,
        colab_url=colab_url,
        data_csv=data_csv,
    )

    if tracer:
        tracer.record_outcome(
            f"code_agent:{'error' if result.error else 'ok'}:backend={result.backend_used}"
        )

    # --- Step 3: assess success ---
    if result.error:
        logger.warning(f"[code_agent] Sandbox error: {result.error}")
        return _failure(code=code, error=result.error)

    if result.stderr and not result.stdout and not result.chart_b64:
        # Code ran but produced only errors and no usable output
        logger.warning(f"[code_agent] Code produced only stderr: {result.stderr[:200]}")
        return _failure(code=code, error=f"Code error: {result.stderr[:500]}")

    return {
        "code": code,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "chart_b64": result.chart_b64,
        "backend": result.backend_used,
        "error": None,
        "supported": True,
    }


def _failure(code: str, error: str) -> dict:
    return {
        "code": code,
        "stdout": "",
        "stderr": "",
        "chart_b64": None,
        "backend": "unknown",
        "error": error,
        "supported": False,
    }