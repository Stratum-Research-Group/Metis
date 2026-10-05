"""Unit tests for local session storage."""

from pathlib import Path
import pytest

from prosopon.storage.session_store import SessionStore
from prosopon.contracts.session import ChatMessage


def test_session_store_lifecycle(tmp_path):
    store = SessionStore(tmp_path)

    # 1. Create
    sess = store.create_session("Metis Architecture Discussion")
    assert sess.session_id is not None
    assert sess.name == "Metis Architecture Discussion"

    # 2. Append Message
    msg = ChatMessage(
        message_id="msg-1",
        role="user",
        content="What is Prosopon?",
    )
    updated = store.append_message(sess.session_id, msg)
    assert updated is not None
    assert len(updated.messages) == 1

    # 3. Retrieve
    retrieved = store.get_session(sess.session_id)
    assert retrieved is not None
    assert retrieved.name == "Metis Architecture Discussion"
    assert len(retrieved.messages) == 1
    assert retrieved.messages[0].content == "What is Prosopon?"

    # 4. List
    sessions = store.list_sessions()
    assert len(sessions) == 1
    assert sessions[0]["session_id"] == sess.session_id

    # 5. Delete
    deleted = store.delete_session(sess.session_id)
    assert deleted is True
    assert store.get_session(sess.session_id) is None


def test_session_id_path_traversal_rejected(tmp_path):
    store = SessionStore(tmp_path)

    # Path traversal attempts
    assert store.get_session("../../../etc/passwd") is None
    assert store.delete_session("../../../etc/passwd") is False
    assert store.append_message("../../escape", ChatMessage(message_id="m", role="user", content="x")) is None

    with pytest.raises(ValueError, match="Invalid session ID format|Path traversal"):
        store._resolve_session_path("../evil_session")

    with pytest.raises(ValueError, match="Invalid session ID format|Path traversal"):
        store._resolve_session_path("session/subpath")

    with pytest.raises(ValueError, match="Invalid session ID format|Path traversal"):
        store._resolve_session_path("invalid id with spaces")


def test_session_store_permissions_and_recovery(tmp_path):
    import os
    store = SessionStore(tmp_path)
    sess = store.create_session("Secure Session")

    file_path = store.storage_dir / f"{sess.session_id}.json"
    assert file_path.exists()
    # Check permissions 0600 on file
    mode = os.stat(file_path).st_mode & 0o777
    assert mode == 0o600

