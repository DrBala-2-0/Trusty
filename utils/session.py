import uuid

from fastapi import Header


def resolve_session_id(x_session_id: str | None = Header(default=None, alias="X-Session-ID")) -> str:
    """Resolve the caller's session id from the X-Session-ID request header.

    No auth exists yet (deliberately deferred — see blueprint §5), so this is a
    lightweight identifier, not an authenticated actor: it isolates concurrent
    users' documents/retrievers from each other, nothing more.

    If the caller doesn't send one, mint a new UUID. The caller is expected to
    read it back from /upload's response body and send it on subsequent
    /upload and /ask calls to stay in the same session.
    """
    return x_session_id or uuid.uuid4().hex