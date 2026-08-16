"""
Trusty A2A client — Chapter 21.
Sends structured requests to peer Trusty instances and returns
provenance-tagged A2AResponse objects.

The client speaks Trusty's own /ask HTTP API — no custom protocol
needed since every Trusty instance already has a REST interface.
A2A is the interaction pattern (structured request, tagged response,
trust accounting); HTTP is the transport.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

import requests

from a2a.schemas import A2ARequest, A2AResponse
from a2a.trust import trust_registry
from config.settings import settings
from utils.logging import logger


def ask_peer(
    peer_url: str,
    question: str,
    context: str = "",
    requester: str = "trusty-local",
    session_id: Optional[str] = None,
) -> A2AResponse:
    """
    Send a question to a peer Trusty instance and return a tagged response.

    peer_url    : base URL of the peer (e.g. "http://peer.example.com:8000")
    question    : the question to ask
    context     : optional context hint for the peer
    requester   : identifier of this Trusty instance
    session_id  : peer session to use (minted fresh if None)

    The peer must have a document already uploaded in the given session,
    OR the caller must use a session the peer already has context for.
    For cross-instance queries without pre-uploaded docs, pass context
    in the question itself.
    """
    request_id = uuid.uuid4().hex
    peer_session = session_id or uuid.uuid4().hex
    ts = datetime.now(timezone.utc).isoformat()

    req = A2ARequest(
        question=question,
        requester=requester,
        request_id=request_id,
        context=context,
    )

    logger.info(
        f"[a2a] → peer={peer_url} "
        f"request_id={request_id[:8]} "
        f"question={question[:60]!r}"
    )

    try:
        # Combine question and context into a single text field
        full_question = question
        if context:
            full_question = f"{question}\n\nContext from requesting instance:\n{context}"

        response = requests.post(
            f"{peer_url.rstrip('/')}/ask",
            json={"text": full_question},
            headers={"X-Session-ID": peer_session},
            timeout=settings.A2A_PEER_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        data = response.json()

        parsed = data.get("parsed_report", {})
        supported = parsed.get("Supported", "NO").strip().upper() == "YES"
        relevant = parsed.get("Relevant", "NO").strip().upper() == "YES"
        sources = [
            s.get("source", "unknown")
            for s in data.get("chunk_sources", [])
        ]
        trust_level = trust_registry.trust_level(peer_url)
        trust_registry.record_call(peer_url, peer_url, success=True)

        logger.info(
            f"[a2a] ← peer={peer_url} "
            f"supported={supported} trust={trust_level}"
        )

        return A2AResponse(
            answer=data.get("draft_answer", ""),
            peer_url=peer_url,
            peer_id=peer_url,
            request_id=request_id,
            supported=supported,
            relevant=relevant,
            trust_level=trust_level,
            freshness_ts=ts,
            error=None,
            verification_mode=data.get("verification_mode", "document"),
            sources=sources,
        )

    except requests.Timeout:
        trust_registry.record_call(peer_url, peer_url, success=False)
        error = f"Peer timed out after {settings.A2A_PEER_TIMEOUT_SECONDS}s"
        logger.warning(f"[a2a] peer={peer_url} timeout")
        return _error_response(peer_url, request_id, ts, error)

    except requests.HTTPError as e:
        trust_registry.record_call(peer_url, peer_url, success=False)
        error = f"Peer HTTP error: {e.response.status_code}"
        logger.warning(f"[a2a] peer={peer_url} HTTP {e.response.status_code}")
        return _error_response(peer_url, request_id, ts, error)

    except Exception as e:
        trust_registry.record_call(peer_url, peer_url, success=False)
        error = f"Peer connection error: {e}"
        logger.warning(f"[a2a] peer={peer_url} error: {e}")
        return _error_response(peer_url, request_id, ts, error)


def _error_response(
    peer_url: str, request_id: str, ts: str, error: str
) -> A2AResponse:
    return A2AResponse(
        answer="",
        peer_url=peer_url,
        peer_id=peer_url,
        request_id=request_id,
        supported=False,
        relevant=False,
        trust_level=trust_registry.trust_level(peer_url),
        freshness_ts=ts,
        error=error,
    )