"""
Trusty sandbox client — host side.
Sends generated Python code to either:
  - "docker"  : a local Docker container (default, §9.5)
  - "colab"   : a user-supplied Colab remote kernel (§9.13)
Returns a SandboxResult regardless of which backend ran.
The code agent and verification agent never know which backend was used.
"""
import subprocess
import json
import base64
import time
from dataclasses import dataclass, field
from typing import Optional

from config.settings import settings
from utils.logging import logger


@dataclass
class SandboxResult:
    stdout: str
    stderr: str
    chart_b64: Optional[str]      # base64 PNG string, or None
    elapsed_seconds: float
    backend_used: str             # "docker" or "colab" — for the trace
    error: Optional[str] = None  # set if the client itself failed, not the code


def run(code: str, backend: str = "docker",
        colab_url: Optional[str] = None) -> SandboxResult:
    """
    Execute `code` in the chosen sandbox backend.
    backend: "docker" or "colab"
    colab_url: required when backend="colab"; the connection string from the
               Colab setup cell.
    """
    start = time.monotonic()

    if backend == "docker":
        result = _run_docker(code)
    elif backend == "colab":
        if not colab_url:
            return SandboxResult(
                stdout="", stderr="", chart_b64=None,
                elapsed_seconds=0.0, backend_used="colab",
                error="colab_url is required when backend='colab'"
            )
        result = _run_colab(code, colab_url)
    else:
        return SandboxResult(
            stdout="", stderr="", chart_b64=None,
            elapsed_seconds=0.0, backend_used=backend,
            error=f"Unknown backend: {backend!r}"
        )

    result.elapsed_seconds = time.monotonic() - start
    logger.info(
        f"[sandbox] backend={result.backend_used} "
        f"elapsed={result.elapsed_seconds:.2f}s "
        f"error={result.error!r}"
    )
    return result


# ---------------------------------------------------------------------------
# Docker backend
# ---------------------------------------------------------------------------

def _run_docker(code: str) -> SandboxResult:
    payload = json.dumps({"code": code})
    try:
        proc = subprocess.run(
            [
                "docker", "run",
                "--rm",                          # remove container after exit
                "-i",                            # accept stdin — required for pipe
                "--network", "none",             # no outbound network
                "--memory", "512m",              # RAM cap
                "--cpus", "1.0",                 # CPU cap
                "--read-only",                   # no filesystem writes
                "--tmpfs", "/tmp:size=64m",      # small writable tmp
                settings.SANDBOX_DOCKER_IMAGE,
            ],
            input=payload,
            capture_output=True,
            text=True,
            timeout=settings.SANDBOX_TIMEOUT_SECONDS,
        )
        raw = proc.stdout.strip()
        if not raw:
            return SandboxResult(
                stdout="", stderr=proc.stderr,
                chart_b64=None, elapsed_seconds=0.0,
                backend_used="docker",
                error="Container produced no output"
            )
        data = json.loads(raw)
        return SandboxResult(
            stdout=data.get("stdout", ""),
            stderr=data.get("stderr", ""),
            chart_b64=data.get("chart_b64"),
            elapsed_seconds=0.0,
            backend_used="docker"
        )
    except subprocess.TimeoutExpired:
        return SandboxResult(
            stdout="", stderr="", chart_b64=None,
            elapsed_seconds=settings.SANDBOX_TIMEOUT_SECONDS,
            backend_used="docker",
            error=f"Sandbox timed out after {settings.SANDBOX_TIMEOUT_SECONDS}s"
        )
    except Exception as e:
        return SandboxResult(
            stdout="", stderr="", chart_b64=None,
            elapsed_seconds=0.0, backend_used="docker",
            error=f"Docker client error: {e}"
        )


# ---------------------------------------------------------------------------
# Colab backend
# ---------------------------------------------------------------------------

def _run_colab(code: str, colab_url: str) -> SandboxResult:
    """
    Connect to a running Colab kernel via jupyter_client.
    colab_url format:  host:port:kernel_id:token
    e.g.  "tcp://colab-runtime.example.com:9999:abc-123:mytoken"
    Exact format is printed by the Colab setup cell (sandbox/colab_setup.ipynb).
    """
    try:
        from jupyter_client import BlockingKernelClient
    except ImportError:
        return SandboxResult(
            stdout="", stderr="", chart_b64=None,
            elapsed_seconds=0.0, backend_used="colab",
            error="jupyter_client is not installed. Run: pip install jupyter_client"
        )

    try:
        host, port, kernel_id, token = colab_url.strip().split(":")
    except ValueError:
        return SandboxResult(
            stdout="", stderr="", chart_b64=None,
            elapsed_seconds=0.0, backend_used="colab",
            error="Invalid colab_url format. Expected host:port:kernel_id:token"
        )

    kc = BlockingKernelClient()
    kc.load_connection_info({
        "shell_port": int(port),
        "iopub_port": int(port) + 1,
        "stdin_port": int(port) + 2,
        "control_port": int(port) + 3,
        "hb_port": int(port) + 4,
        "ip": host,
        "key": token,
        "transport": "tcp",
        "signature_scheme": "hmac-sha256",
        "kernel_name": "",
    })
    kc.start_channels()

    stdout_parts = []
    stderr_parts = []
    chart_b64 = None

    try:
        kc.wait_for_ready(timeout=10)
        msg_id = kc.execute(code)

        while True:
            try:
                msg = kc.get_iopub_msg(timeout=settings.SANDBOX_TIMEOUT_SECONDS)
            except Exception:
                break

            msg_type = msg["msg_type"]
            content = msg.get("content", {})

            if msg_type == "stream":
                if content.get("name") == "stdout":
                    stdout_parts.append(content.get("text", ""))
                elif content.get("name") == "stderr":
                    stderr_parts.append(content.get("text", ""))

            elif msg_type in ("display_data", "execute_result"):
                data = content.get("data", {})
                if "image/png" in data:
                    chart_b64 = data["image/png"]

            elif msg_type == "error":
                stderr_parts.append("\n".join(content.get("traceback", [])))

            elif msg_type == "execute_reply":
                # Final message for this execution — stop reading
                break

        return SandboxResult(
            stdout="".join(stdout_parts),
            stderr="".join(stderr_parts),
            chart_b64=chart_b64,
            elapsed_seconds=0.0,
            backend_used="colab"
        )

    except Exception as e:
        return SandboxResult(
            stdout="", stderr="", chart_b64=None,
            elapsed_seconds=0.0, backend_used="colab",
            error=f"Colab kernel error: {e}"
        )
    finally:
        kc.stop_channels()