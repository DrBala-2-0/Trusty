"""
Trusty Chapter 15 -- Flexible output formatter.

Validates, fills, and shapes pipeline output into the format the caller
requested. Four formats are supported at Option 1 stage:

    text           -- plain prose (default, current behaviour unchanged)
    json           -- structured JSON with fixed keys
    markdown_table -- answer formatted as a markdown table where possible
    template       -- caller-supplied string with named placeholders

Option 1 placeholders (filled from current pipeline output):
    {answer}       -- the draft answer text
    {sources}      -- comma-separated source file names from retrieved chunks
    {verification} -- Supported/Relevant verdict as a short string
    {confidence}   -- "HIGH" / "MEDIUM" / "LOW" derived from parsed_report

Placeholders reserved for Options 2/3 (rejected at Option 1 stage):
    {chart}  {image}  {visualization}  {code_output}
    {external_data}  {live_table}

Per blueprint §9.10: future-option placeholders are rejected with a clear
error rather than silently left unfilled, so callers know exactly what
capability gap they've hit.
"""

import json
import re
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_FORMATS = {"text", "json", "markdown_table", "template"}

# Placeholders valid at Option 1 stage
OPTION1_PLACEHOLDERS = {"answer", "sources", "verification", "confidence"}

# Placeholders that require Option 2 or 3 -- rejected now, reserved for later
FUTURE_PLACEHOLDERS = {
    "chart", "image", "visualization", "code_output",   # Option 2
    "external_data", "live_table",                       # Option 3
}

_PLACEHOLDER_RE = re.compile(r"\{(\w+)\}")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class FormatterError(ValueError):
    """Raised for invalid format requests -- maps to HTTP 400 at the call site."""
    pass


def validate_format(response_format: str, response_template: Optional[str]) -> None:
    """Raise FormatterError if the format request is invalid.

    Called before the pipeline runs so bad requests are rejected early,
    before any LLM calls are made.
    """
    if response_format not in VALID_FORMATS:
        raise FormatterError(
            f"Unknown response_format '{response_format}'. "
            f"Valid values: {sorted(VALID_FORMATS)}."
        )

    if response_format == "template":
        if not response_template or not response_template.strip():
            raise FormatterError(
                "response_template is required when response_format='template'."
            )
        # Check for future-option placeholders
        placeholders = set(_PLACEHOLDER_RE.findall(response_template))
        future_used = placeholders & FUTURE_PLACEHOLDERS
        if future_used:
            raise FormatterError(
                f"Template contains placeholder(s) not available at Option 1 stage: "
                f"{sorted(future_used)}. These require Option 2 (code execution) or "
                f"Option 3 (A2A data) to be enabled. Available placeholders: "
                f"{sorted(OPTION1_PLACEHOLDERS)}."
            )
        unknown = placeholders - OPTION1_PLACEHOLDERS - FUTURE_PLACEHOLDERS
        if unknown:
            raise FormatterError(
                f"Unknown placeholder(s) in template: {sorted(unknown)}. "
                f"Available at Option 1 stage: {sorted(OPTION1_PLACEHOLDERS)}."
            )

    if response_format != "template" and response_template:
        raise FormatterError(
            "response_template is only used when response_format='template'. "
            f"Got response_format='{response_format}'."
        )


# ---------------------------------------------------------------------------
# Placeholder extraction helpers
# ---------------------------------------------------------------------------

def _extract_sources(documents: list) -> str:
    """Comma-separated unique source filenames from retrieved chunks.

    Handles both LangChain Document objects (live pipeline path) and
    plain dicts with a 'source' key (cache hit path, where chunk_sources
    was stored as [{'source': '...'}] rather than Document objects).
    """
    seen = []
    for doc in documents:
        if hasattr(doc, "metadata"):
            src = doc.metadata.get("source", "unknown")
        elif isinstance(doc, dict):
            src = doc.get("source", "unknown")
        else:
            src = "unknown"
        name = src.replace("\\", "/").split("/")[-1]
        if name not in seen:
            seen.append(name)
    return ", ".join(seen) if seen else "unknown"


def _extract_confidence(parsed_report: dict) -> str:
    """Derive a confidence label from the verification report."""
    supported = parsed_report.get("Supported", "").strip().upper()
    relevant = parsed_report.get("Relevant", "").strip().upper()
    unsupported = parsed_report.get("Unsupported Claims", "None")

    if supported == "YES" and relevant == "YES":
        if unsupported.strip().lower() in ("none", ""):
            return "HIGH"
        return "MEDIUM"
    return "LOW"


def _extract_verification(parsed_report: dict) -> str:
    """Short human-readable verification verdict."""
    supported = parsed_report.get("Supported", "--")
    relevant = parsed_report.get("Relevant", "--")
    return f"Supported: {supported} | Relevant: {relevant}"


# ---------------------------------------------------------------------------
# Format application
# ---------------------------------------------------------------------------

def apply_format(
    response_format: str,
    response_template: Optional[str],
    draft_answer: str,
    parsed_report: dict,
    documents: list,
) -> dict:
    """Shape the pipeline result into the requested format.

    Returns a dict with:
        formatted_answer  -- the shaped output (str for text/template,
                             dict for json, str for markdown_table)
        response_format   -- echoed back so callers know what shape arrived
    """
    sources = _extract_sources(documents)
    confidence = _extract_confidence(parsed_report)
    verification = _extract_verification(parsed_report)

    if response_format == "text":
        return {
            "formatted_answer": draft_answer,
            "response_format": "text",
        }

    elif response_format == "json":
        # If the research agent returned valid JSON ({"answer": "..."}),
        # extract the answer string from it. If not (refusal path or LLM
        # didn't comply with the instruction), use draft_answer as-is.
        # Either way, the formatter assembles the full structured envelope.
        answer_text = draft_answer
        try:
            parsed_draft = json.loads(draft_answer)
            if isinstance(parsed_draft, dict) and "answer" in parsed_draft:
                answer_text = parsed_draft["answer"]
        except (json.JSONDecodeError, ValueError):
            # LLM returned prose instead of JSON -- use it directly.
            # This happens on refusal paths ("I cannot answer...") and is
            # correct behaviour: wrap the prose in the structured envelope
            # rather than failing the whole request.
            pass

        structured = {
            "answer": answer_text,
            "sources": sources,
            "verification": {
                "supported": parsed_report.get("Supported", "--"),
                "relevant": parsed_report.get("Relevant", "--"),
                "unsupported_claims": parsed_report.get("Unsupported Claims", "None"),
                "contradictions": parsed_report.get("Contradictions", "None"),
            },
            "confidence": confidence,
        }
        return {
            "formatted_answer": json.dumps(structured, indent=2),
            "response_format": "json",
        }

    elif response_format == "markdown_table":
        table = (
            "| Field | Value |\n"
            "|---|---|\n"
            f"| Answer | {draft_answer.replace(chr(10), ' ')} |\n"
            f"| Sources | {sources} |\n"
            f"| Supported | {parsed_report.get('Supported', '--')} |\n"
            f"| Relevant | {parsed_report.get('Relevant', '--')} |\n"
            f"| Confidence | {confidence} |\n"
        )
        return {
            "formatted_answer": table,
            "response_format": "markdown_table",
        }

    elif response_format == "template":
        filled = response_template.format(
            answer=draft_answer,
            sources=sources,
            verification=verification,
            confidence=confidence,
        )
        return {
            "formatted_answer": filled,
            "response_format": "template",
        }

    # Unreachable -- validate_format() guards against unknown formats
    raise FormatterError(f"Unhandled response_format: {response_format}")