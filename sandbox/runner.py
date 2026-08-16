"""
Trusty sandbox runner — executes inside the Docker container.
Reads a JSON payload from stdin:
    {"code": "<python source>"}
Writes a JSON result to stdout:
    {"stdout": "...", "stderr": "...", "chart_b64": "<base64 png or null>"}
Never imported by the host — only executed as a subprocess inside the container.
"""
import sys
import json
import traceback
import io
import base64
import contextlib

def run():
    try:
        payload = json.loads(sys.stdin.read())
        code = payload["code"]
    except Exception as e:
        print(json.dumps({
            "stdout": "",
            "stderr": f"[runner] Failed to read payload: {e}",
            "chart_b64": None
        }))
        return

    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    chart_b64 = None

    # Matplotlib must use non-interactive backend inside the container
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        with contextlib.redirect_stdout(stdout_buf), \
             contextlib.redirect_stderr(stderr_buf):
            exec(code, {"__builtins__": __builtins__})

        # Capture any open matplotlib figure as a base64 PNG
        if plt.get_fignums():
            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            chart_b64 = base64.b64encode(buf.read()).decode("utf-8")
            plt.close("all")

    except Exception:
        stderr_buf.write(traceback.format_exc())

    print(json.dumps({
        "stdout": stdout_buf.getvalue(),
        "stderr": stderr_buf.getvalue(),
        "chart_b64": chart_b64
    }))

if __name__ == "__main__":
    run()