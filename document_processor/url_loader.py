import ipaddress
import socket

import os
import tempfile

import requests
import trafilatura

from urllib.parse import urlparse

from document_processor.loaders import LOADER_REGISTRY

ALLOWED_SCHEMES = {"http", "https"}


def _validate_public_url(url: str) -> None:
    """SSRF guard: reject non-http(s) schemes and hostnames resolving to
    private/loopback/link-local/reserved IP ranges (e.g. cloud metadata
    endpoints, internal services). Resolves the hostname once, before
    fetching — does NOT defend against DNS rebinding (the IP changing
    between this check and the actual request). Named simplification, see
    chapter-9.md known issues."""
    parsed = urlparse(url)
    if parsed.scheme not in ALLOWED_SCHEMES:
        raise ValueError(f"URL scheme '{parsed.scheme}' not allowed. Use http or https.")
    if not parsed.hostname:
        raise ValueError(f"Could not determine hostname from URL: {url}")

    try:
        resolved_ip = socket.gethostbyname(parsed.hostname)
    except socket.gaierror as e:
        raise ValueError(f"Could not resolve hostname '{parsed.hostname}': {e}")

    ip = ipaddress.ip_address(resolved_ip)
    if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
        raise ValueError(
            f"URL '{url}' resolves to a non-public address ({resolved_ip}) and was rejected."
        )



MAX_URL_RESPONSE_BYTES = 10 * 1024 * 1024  # 10 MB
FETCH_TIMEOUT_SECONDS = 10
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


def fetch_url_text(url: str) -> str:
    """Fetch url and return its extracted text.

    Detects a PDF response (by Content-Type, falling back to a .pdf URL
    path) and routes it through the existing PDF loader instead of trying
    to decode binary PDF bytes as HTML. Non-PDF responses go through
    trafilatura for main-content extraction.

    Raises ValueError for a rejected/invalid URL (via _validate_public_url),
    RuntimeError for a genuine fetch failure (timeout, non-2xx, oversized
    response) — the same split every other loader (Ch9 Steps 3-5) uses."""
    _validate_public_url(url)

    try:
        response = requests.get(
            url,
            timeout=FETCH_TIMEOUT_SECONDS,
            stream=True,
            headers={"User-Agent": USER_AGENT},
        )
        response.raise_for_status()

        content = b""
        for chunk in response.iter_content(chunk_size=8192):
            content += chunk
            if len(content) > MAX_URL_RESPONSE_BYTES:
                raise RuntimeError(
                    f"Response from '{url}' exceeded {MAX_URL_RESPONSE_BYTES // 1_048_576} MB limit."
                )
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to fetch URL '{url}': {e}") from e

    content_type = response.headers.get("Content-Type", "")
    is_pdf = "application/pdf" in content_type.lower() or urlparse(url).path.lower().endswith(".pdf")

    if is_pdf:
        pdf_loader = LOADER_REGISTRY.get(".pdf")
        if pdf_loader is None:
            raise RuntimeError("PDF loader not registered — cannot process PDF URL.")
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        try:
            return pdf_loader(tmp_path)
        finally:
            os.remove(tmp_path)

    html = content.decode(response.encoding or response.apparent_encoding or "utf-8", errors="replace")
    extracted = trafilatura.extract(html)
    return extracted or ""