"""
Trusty Chapter 20 -- Thin-client Gradio UI (updated for Option 2 analysis).

Changes from Ch15:
- Analysis toggle (enable_analysis) in Ask section
- Colab URL input (sandbox_backend selector)
- Data description + CSV inputs for structured data questions
- Chart display panel (base64 PNG rendered inline)
- Code output accordion showing generated code + stdout
- Execution-grounded verification shown alongside document verification

Runs as a separate process from the FastAPI server:
    Terminal 1: uvicorn app:app --reload
    Terminal 2: python ui/app_ui.py
"""

import base64
import uuid

import gradio as gr
import pandas as pd
import requests

API_BASE = "http://localhost:8000"
SUPPORTED_EXTENSIONS = [
    ".pdf", ".txt", ".csv", ".xlsx",
    ".png", ".jpg", ".jpeg", ".webp",
    ".mp3", ".wav", ".m4a", ".zip",
]

FORMAT_CHOICES = ["Text", "JSON", "Markdown Table", "Template"]
FORMAT_MAP = {
    "Text": "text",
    "JSON": "json",
    "Markdown Table": "markdown_table",
    "Template": "template",
}

TEMPLATE_HINT = (
    "Option 1 placeholders: {answer}, {sources}, {verification}, {confidence}\n"
    "Option 2 placeholders: {chart}, {code_output}\n"
    "Example: Answer: {answer}\nSources: {sources}\nConfidence: {confidence}"
)


# ---------------------------------------------------------------------------
# API calls
# ---------------------------------------------------------------------------

def upload_file(file_path: str, session_id: str) -> dict:
    with open(file_path, "rb") as f:
        filename = file_path.replace("\\", "/").split("/")[-1]
        response = requests.post(
            f"{API_BASE}/upload",
            files={"file": (filename, f)},
            headers={"X-Session-ID": session_id},
            timeout=120,
        )
    response.raise_for_status()
    return response.json()


def upload_url(url: str, session_id: str) -> dict:
    response = requests.post(
        f"{API_BASE}/upload",
        data={"url": url},
        headers={"X-Session-ID": session_id},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def ask_question(
    question: str,
    session_id: str,
    response_format: str,
    response_template: str | None = None,
    enable_analysis: bool = False,
    sandbox_backend: str = "docker",
    colab_url: str = "",
    data_description: str = "",
    data_csv: str = "",
) -> dict:
    payload = {"text": question, "response_format": response_format}
    if response_template:
        payload["response_template"] = response_template
    if enable_analysis:
        payload["enable_analysis"] = True
        payload["sandbox_backend"] = sandbox_backend or "docker"
        colab_url = colab_url or ""
        if colab_url.strip():
            payload["colab_url"] = colab_url.strip()
        payload["data_description"] = (data_description or "").strip()
        payload["data_csv"] = (data_csv or "").strip()
    response = requests.post(
        f"{API_BASE}/ask",
        json=payload,
        headers={"X-Session-ID": session_id},
        timeout=180,   # analysis cases need extra time
    )
    response.raise_for_status()
    return response.json()


# ---------------------------------------------------------------------------
# Response formatting helpers
# ---------------------------------------------------------------------------

def format_verification(parsed_report: dict, code_parsed_report: dict | None,
                        verification_mode: str) -> str:
    if not parsed_report:
        return ""
    supported = parsed_report.get("Supported", "--")
    relevant = parsed_report.get("Relevant", "--")
    unsupported = parsed_report.get("Unsupported Claims", "None")
    contradictions = parsed_report.get("Contradictions", "None")
    supported_label = "[YES]" if supported == "YES" else "[NO]"
    relevant_label = "[YES]" if relevant == "YES" else "[NO]"
    lines = [
        "### Verification Report",
        f"**Mode:** `{verification_mode}`",
        f"**Supported:** {supported_label}",
        f"**Relevant:** {relevant_label}",
    ]
    if unsupported and unsupported.lower() not in ("none", ""):
        lines.append(f"**Unsupported Claims:** {unsupported}")
    if contradictions and contradictions.lower() not in ("none", ""):
        lines.append(f"**Contradictions:** {contradictions}")

    if code_parsed_report:
        code_supported = code_parsed_report.get("Supported", "--")
        code_label = "[YES]" if code_supported == "YES" else "[NO]"
        lines.append(f"\n**Execution check:** {code_label}")
        code_unsupported = code_parsed_report.get("Unsupported Claims", "None")
        if code_unsupported and code_unsupported.lower() not in ("none", ""):
            lines.append(f"**Execution unsupported:** {code_unsupported}")

    return "\n\n".join(lines)


def format_budget(budget: dict) -> str:
    if not budget:
        return ""
    used = int(budget.get("used", 0))
    limit = int(budget.get("limit", 0))
    remaining = int(budget.get("remaining", 0))
    exhausted = budget.get("exhausted", False)
    bar_filled = int((used / limit) * 20) if limit else 0
    bar_empty = 20 - bar_filled
    bar = "#" * bar_filled + "-" * bar_empty
    status = " [EXHAUSTED]" if exhausted else ""
    return (
        f"### Budget\n"
        f"`[{bar}]` {used}/{limit} calls used "
        f"-- {remaining} remaining{status}"
    )


def format_trace(trace: dict) -> str:
    if not trace:
        return ""
    steps = trace.get("steps", [])
    total = trace.get("total_elapsed_s", 0)
    lines = [f"**Total time:** {total}s\n"]
    for step in steps:
        name = step.get("step", "")
        elapsed = step.get("elapsed_s", "")
        time_str = f" (+{elapsed}s)" if elapsed else ""
        if name == "retrieval":
            sources = ", ".join(step.get("chunk_sources", []))
            count = step.get("chunks_retrieved", 0)
            lines.append(f"**[Retrieval]** -- {count} chunks from `{sources}`")
        elif name == "relevance_check":
            label = step.get("label", "")
            is_relevant = step.get("is_relevant", False)
            verdict = "[RELEVANT]" if is_relevant else "[NOT RELEVANT]"
            lines.append(f"**[Relevance]** -- `{label}` {verdict}{time_str}")
        elif name == "research":
            attempt = step.get("attempt", 1)
            preview = step.get("draft_answer_preview", "")
            lines.append(f"**[Research]** attempt {attempt}{time_str}")
            lines.append(f"> {preview[:120]}...")
        elif name == "verification":
            supported = step.get("supported", "")
            verdict = "[SUPPORTED]" if supported == "YES" else "[NOT SUPPORTED]"
            lines.append(f"**[Verification]** -- {verdict}{time_str}")
        elif name == "retry_triggered":
            reason = step.get("reason", "")
            lines.append(f"**[Retry]** -- {reason}{time_str}")
        elif name == "outcome":
            outcome = step.get("outcome", "")
            total_s = step.get("total_elapsed_s", "")
            lines.append(f"**[Outcome]** -- `{outcome}` (total: {total_s}s)")
    return "\n\n".join(lines)


def format_code_result(code_result: dict | None) -> tuple[str, str]:
    """Returns (code_text, stdout_text) for the code output accordion."""
    if not code_result:
        return "", ""
    code = code_result.get("code", "")
    stdout = code_result.get("stdout", "").strip()
    stderr = code_result.get("stderr", "").strip()
    error = code_result.get("error", "")
    backend = code_result.get("backend", "")

    stdout_parts = []
    if stdout:
        stdout_parts.append(stdout)
    if stderr:
        stdout_parts.append(f"[stderr]\n{stderr}")
    if error:
        stdout_parts.append(f"[error]\n{error}")
    if backend:
        stdout_parts.append(f"\n[backend: {backend}]")

    return code, "\n\n".join(stdout_parts)


def b64_to_image_path(chart_b64: str | None) -> str | None:
    """Decode a base64 PNG and save to a temp file for gr.Image."""
    if not chart_b64:
        return None
    try:
        import tempfile, os
        img_bytes = base64.b64decode(chart_b64)
        tmp = tempfile.NamedTemporaryFile(
            delete=False, suffix=".png", prefix="trusty_chart_"
        )
        tmp.write(img_bytes)
        tmp.close()
        return tmp.name
    except Exception:
        return None


def parse_markdown_table(md_table: str) -> pd.DataFrame | None:
    try:
        lines = [
            line.strip() for line in md_table.strip().splitlines()
            if line.strip() and not line.strip().startswith("|---")
        ]
        if len(lines) < 2:
            return None
        headers = [h.strip() for h in lines[0].strip("|").split("|")]
        rows = []
        for line in lines[1:]:
            cells = [c.strip() for c in line.strip("|").split("|")]
            rows.append(cells)
        return pd.DataFrame(rows, columns=headers)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Event handlers
# ---------------------------------------------------------------------------

def handle_new_session():
    new_id = uuid.uuid4().hex
    state = {"session_id": new_id, "uploads": 0}
    label = f"**Active session:** `{new_id[:12]}...`"
    return state, label


def handle_format_change(format_choice: str):
    return gr.update(visible=(format_choice == "Template"))


def handle_analysis_toggle(enabled: bool):
    return gr.update(visible=enabled)


def handle_backend_change(backend: str):
    return gr.update(visible=(backend == "colab"))


def handle_upload(file, url_input, session_state):
    session_id = session_state.get("session_id")
    if not session_id:
        return "[WARNING] No session -- click 'New Session' first.", session_state, gr.update()
    if file is None and not url_input.strip():
        return "[WARNING] Please select a file or enter a URL.", session_state, gr.update()
    try:
        if file is not None:
            result = upload_file(file.name, session_id)
            filename = file.name.replace("\\", "/").split("/")[-1]
            added = result.get("chunks_added", "?")
            total = result.get("chunks_total", "?")
            skipped = result.get("skipped", [])
            msg = (
                f"[OK] **{filename}** indexed -- "
                f"{added} chunks added ({total} total this session)"
            )
            if skipped:
                msg += f"\n[WARNING] Skipped: {', '.join(skipped)}"
        else:
            result = upload_url(url_input.strip(), session_id)
            added = result.get("chunks_added", "?")
            total = result.get("chunks_total", "?")
            msg = f"[OK] URL indexed -- {added} chunks added ({total} total this session)"
        session_state = dict(session_state)
        session_state["uploads"] = session_state.get("uploads", 0) + 1
        return msg, session_state, gr.update(value=None)
    except requests.HTTPError as e:
        try:
            detail = e.response.json().get("detail", str(e))
        except Exception:
            detail = str(e)
        return f"[ERROR] Upload failed: {detail}", session_state, gr.update()
    except Exception as e:
        return f"[ERROR] Upload error: {e}", session_state, gr.update()


def handle_ask(
    question, format_choice, template_input,
    enable_analysis, backend_choice, colab_url_input,
    data_description_input, data_csv_input,
    session_state,
):
    """
    Returns 11 values:
    answer_md, answer_json, answer_df,
    chart_img, code_text, stdout_text,
    cached_label, verification, budget, trace, error
    """
    session_id = session_state.get("session_id")
    empty = (
        gr.update(value="", visible=False),
        gr.update(value="", visible=False),
        gr.update(value=None, visible=False),
        gr.update(value=None, visible=False),
        gr.update(value="", visible=False),
        gr.update(value="", visible=False),
        "", "", "", "",
    )

    if not session_id:
        return *empty, "[WARNING] No session -- click 'New Session' first."
    if not question.strip():
        return *empty, "[WARNING] Please enter a question."
    if session_state.get("uploads", 0) == 0:
        return *empty, "[WARNING] No documents uploaded yet."

    fmt_key = FORMAT_MAP.get(format_choice, "text")
    template = template_input.strip() if fmt_key == "template" else None
    backend = "colab" if (backend_choice or "") == "Colab GPU" else "docker"

    try:
        result = ask_question(
            question.strip(), session_id, fmt_key, template,
            enable_analysis=enable_analysis,
            sandbox_backend=backend,
            colab_url=colab_url_input,
            data_description=data_description_input,
            data_csv=data_csv_input,
        )

        formatted_answer = result.get("formatted_answer", result.get("draft_answer", ""))
        fmt = result.get("response_format", "text")
        cached = result.get("cached", False)
        cached_label = "**[Cached result -- no pipeline trace]**" if cached else ""

        verification_mode = result.get("verification_mode", "document")
        verification = format_verification(
            result.get("parsed_report", {}),
            result.get("code_parsed_report"),
            verification_mode,
        )
        budget_str = format_budget(result.get("budget", {}))
        trace_str = format_trace(result.get("trace", {}))

        # Chart
        code_result = result.get("code_result")
        chart_b64 = (code_result or {}).get("chart_b64")
        chart_path = b64_to_image_path(chart_b64)
        chart_update = gr.update(value=chart_path, visible=bool(chart_path))

        # Code output
        code_text, stdout_text = format_code_result(code_result)
        code_update = gr.update(value=code_text, visible=bool(code_text))
        stdout_update = gr.update(value=stdout_text, visible=bool(stdout_text))

        # Answer format routing
        if fmt == "json":
            ans_md = gr.update(value="", visible=False)
            ans_json = gr.update(value=formatted_answer, visible=True)
            ans_df = gr.update(value=None, visible=False)
        elif fmt == "markdown_table":
            df = parse_markdown_table(formatted_answer)
            ans_md = gr.update(value="", visible=False)
            ans_json = gr.update(value="", visible=False)
            ans_df = gr.update(value=df, visible=True)
        else:
            ans_md = gr.update(value=formatted_answer, visible=True)
            ans_json = gr.update(value="", visible=False)
            ans_df = gr.update(value=None, visible=False)

        return (
            ans_md, ans_json, ans_df,
            chart_update, code_update, stdout_update,
            cached_label, verification, budget_str, trace_str, "",
        )

    except requests.HTTPError as e:
        try:
            detail = e.response.json().get("detail", str(e))
        except Exception:
            detail = str(e)
        if e.response.status_code == 429:
            return *empty, f"[WARNING] {detail}"
        if e.response.status_code == 400:
            return *empty, f"[ERROR] Bad request: {detail}"
        return *empty, f"[ERROR] Ask failed: {detail}"
    except Exception as e:
        return *empty, f"[ERROR] {e}"


# ---------------------------------------------------------------------------
# UI layout
# ---------------------------------------------------------------------------

def build_ui():
    with gr.Blocks(title="Trusty", theme=gr.themes.Soft()) as demo:

        gr.Markdown(
            "# Trusty\n"
            "*Trustworthy document question-answering -- "
            "grounded, verified, and observable.*"
        )

        session_state = gr.State({})

        # ── Session row ──────────────────────────────────────────────────
        with gr.Row():
            with gr.Column(scale=1):
                new_session_btn = gr.Button("New Session", variant="secondary")
            with gr.Column(scale=4):
                session_label = gr.Markdown(
                    "*No active session -- click New Session to start*"
                )

        gr.Markdown("---")

        # ── Upload section ───────────────────────────────────────────────
        gr.Markdown("## Upload Documents")
        with gr.Row():
            file_input = gr.File(
                label="Select a file",
                file_types=SUPPORTED_EXTENSIONS,
                scale=3,
            )
            url_input = gr.Textbox(
                label="Or paste a URL",
                placeholder="https://example.com/report.pdf",
                scale=3,
            )
        upload_btn = gr.Button("Upload", variant="primary")
        upload_status = gr.Markdown("")

        gr.Markdown("---")

        # ── Ask section ──────────────────────────────────────────────────
        gr.Markdown("## Ask a Question")
        question_input = gr.Textbox(
            label="Question",
            placeholder="What does this document say about...?",
            lines=2,
        )

        format_selector = gr.Radio(
            choices=FORMAT_CHOICES,
            value="Text",
            label="Response format",
        )

        template_input = gr.Textbox(
            label="Response template",
            placeholder=TEMPLATE_HINT,
            lines=3,
            visible=False,
        )

        # ── Analysis panel (Option 2) ────────────────────────────────────
        with gr.Accordion("Data Analysis (Option 2)", open=False):
            gr.Markdown(
                "*Enable this when your question requires computation on "
                "structured data (CSV / Excel). The code agent will write "
                "and execute Python code to answer the question.*"
            )
            enable_analysis_toggle = gr.Checkbox(
                label="Enable data analysis",
                value=False,
            )
            with gr.Group(visible=False) as analysis_panel:
                backend_selector = gr.Radio(
                    choices=["Docker (local)", "Colab GPU"],
                    value="Docker (local)",
                    label="Execution backend",
                )
                colab_url_input = gr.Textbox(
                    label="Colab connection URL",
                    placeholder="host:port:kernel_id:token",
                    visible=False,
                )
                data_description_input = gr.Textbox(
                    label="Data description",
                    placeholder=(
                        "Describe the DataFrame: column names, types, and "
                        "what the data represents. E.g. 'DataFrame df with "
                        "columns: name (str), sales (int)'"
                    ),
                    lines=3,
                )
                data_csv_input = gr.Textbox(
                    label="Paste CSV data (optional — paste raw CSV here)",
                    placeholder="name,sales\nAlice,150\nBob,200",
                    lines=5,
                )

        ask_btn = gr.Button("Ask", variant="primary")
        error_output = gr.Markdown("")

        # ── Response section ─────────────────────────────────────────────
        gr.Markdown("## Answer")
        cached_indicator = gr.Markdown("")

        answer_md = gr.Markdown("")
        answer_json = gr.Code("", language="json", label="Answer (JSON)", visible=False)
        answer_df = gr.Dataframe(label="Answer (Table)", visible=False)

        # Chart output (Option 2)
        chart_output = gr.Image(
            label="Chart",
            visible=False,
            type="filepath",
        )

        # Code output accordion (Option 2)
        with gr.Accordion("Code Output", open=False):
            gr.Markdown("*Generated Python code and execution result.*")
            with gr.Row():
                with gr.Column(scale=1):
                    code_output = gr.Code(
                        "",
                        language="python",
                        label="Generated code",
                        visible=False,
                    )
                with gr.Column(scale=1):
                    stdout_output = gr.Code(
                        "",
                        label="Execution output",
                        visible=False,
                    )

        with gr.Row():
            with gr.Column(scale=1):
                verification_output = gr.Markdown("")
            with gr.Column(scale=1):
                budget_output = gr.Markdown("")

        with gr.Accordion("Decision Trace", open=False):
            trace_output = gr.Markdown("")

        # ── Wire events ──────────────────────────────────────────────────
        new_session_btn.click(
            fn=handle_new_session,
            inputs=[],
            outputs=[session_state, session_label],
        )

        format_selector.change(
            fn=handle_format_change,
            inputs=[format_selector],
            outputs=[template_input],
        )

        enable_analysis_toggle.change(
            fn=handle_analysis_toggle,
            inputs=[enable_analysis_toggle],
            outputs=[analysis_panel],
        )

        backend_selector.change(
            fn=handle_backend_change,
            inputs=[backend_selector],
            outputs=[colab_url_input],
        )

        upload_btn.click(
            fn=handle_upload,
            inputs=[file_input, url_input, session_state],
            outputs=[upload_status, session_state, file_input],
        )

        ask_outputs = [
            answer_md,
            answer_json,
            answer_df,
            chart_output,
            code_output,
            stdout_output,
            cached_indicator,
            verification_output,
            budget_output,
            trace_output,
            error_output,
        ]

        ask_inputs = [
            question_input,
            format_selector,
            template_input,
            enable_analysis_toggle,
            backend_selector,
            colab_url_input,
            data_description_input,
            data_csv_input,
            session_state,
        ]

        ask_btn.click(
            fn=handle_ask,
            inputs=ask_inputs,
            outputs=ask_outputs,
            show_progress="minimal",
        )

        question_input.submit(
            fn=handle_ask,
            inputs=ask_inputs,
            outputs=ask_outputs,
            show_progress="minimal",
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_api=False,
        share=False,
    )