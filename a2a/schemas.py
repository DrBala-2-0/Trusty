"""
Trusty A2A schemas — Chapter 21.
Shared dataclasses for agent-to-agent requests and responses.
All A2A communication uses these structures regardless of transport.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class A2ARequest:
    """
    A structured request from one Trusty instance to a peer.
    question    : the question to ask the peer
    context     : optional context to help the peer answer
                  (e.g. what the requesting instance already knows)
    requester   : identifier of the requesting instance
    request_id  : unique ID for tracing
    """
    question: str
    requester: str
    request_id: str
    context: str = ""


@dataclass
class A2AResponse:
    """
    A structured response from a peer Trusty instance.
    All fields are set by the A2A client after receiving the peer's answer.
    """
    answer: str
    peer_url: str
    peer_id: str
    request_id: str
    supported: bool             # did the peer's verifier support the answer?
    relevant: bool              # did the peer's verifier find it relevant?
    trust_level: str            # "high" / "medium" / "low" / "unknown"
    freshness_ts: str           # ISO timestamp of when the response was received
    error: Optional[str] = None # set if the peer call failed
    verification_mode: str = "document"
    sources: list[str] = field(default_factory=list)