"""
Trusty peer trust registry — Chapter 21.
Tracks known peer Trusty instances and their reliability history.
Trust levels are assigned based on historical success rates and
whether the peer is explicitly registered as known/trusted.

Trust levels:
    high    — registered peer with >=80% success rate over >=5 calls
    medium  — registered peer with <80% success rate, or <5 calls seen
    low     — unregistered peer (responded but not in known list)
    unknown — peer has never successfully responded
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PeerRecord:
    url: str
    peer_id: str
    total_calls: int = 0
    successful_calls: int = 0
    registered: bool = False    # explicitly added by the operator

    @property
    def success_rate(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return self.successful_calls / self.total_calls

    @property
    def trust_level(self) -> str:
        if not self.registered:
            return "low"
        if self.total_calls < 5:
            return "medium"
        return "high" if self.success_rate >= 0.8 else "medium"


class TrustRegistry:
    """
    In-memory peer trust registry.
    Wiped on process restart — same deliberate choice as session state.
    Operators register known peers at startup via register_peer().
    """
    def __init__(self):
        self._peers: dict[str, PeerRecord] = {}

    def register_peer(self, url: str, peer_id: str) -> None:
        """Register a known, trusted peer. Call at startup."""
        key = self._key(url)
        if key not in self._peers:
            self._peers[key] = PeerRecord(url=url, peer_id=peer_id, registered=True)
        else:
            self._peers[key].registered = True

    def record_call(self, url: str, peer_id: str, success: bool) -> None:
        """Record the outcome of a call to a peer."""
        key = self._key(url)
        if key not in self._peers:
            self._peers[key] = PeerRecord(url=url, peer_id=peer_id, registered=False)
        record = self._peers[key]
        record.total_calls += 1
        if success:
            record.successful_calls += 1

    def trust_level(self, url: str) -> str:
        key = self._key(url)
        if key not in self._peers:
            return "unknown"
        return self._peers[key].trust_level

    def get_peer(self, url: str) -> Optional[PeerRecord]:
        return self._peers.get(self._key(url))

    def all_peers(self) -> list[PeerRecord]:
        return list(self._peers.values())

    @staticmethod
    def _key(url: str) -> str:
        return url.rstrip("/").lower()


# Shared singleton — imported by the A2A client and app.py
trust_registry = TrustRegistry()