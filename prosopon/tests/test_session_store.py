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
