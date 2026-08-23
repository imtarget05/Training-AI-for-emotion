"""Database tests for tutor_feedback persistence (Phase 7)."""

import database

TS = "2026-08-22T10:00:00Z"


def _save_one(device="db1", message="hello tutor"):
    database.save_tutor_feedback(
        device_id=device,
        timestamp=TS,
        trigger_emotion="Sadness",
        message=message,
        source="fallback",
    )


def test_round_trip():
    _save_tutor_row = _save_one()
    rows = database.get_tutor_feedback_history(device_id="db1")
    assert len(rows) == 1
    row = rows[0]
    assert row["device_id"] == "db1"
    assert row["trigger_emotion"] == "Sadness"
    assert row["message"] == "hello tutor"
    assert row["source"] == "fallback"
    assert row["id"] >= 1
    assert row["created_at"]  # auto timestamp


def test_filter_by_device():
    _save_one("devA", "msg-a")
    _save_one("devB", "msg-b")
    rows_a = database.get_tutor_feedback_history(device_id="devA")
    assert [r["message"] for r in rows_a] == ["msg-a"]
    assert len(database.get_tutor_feedback_history()) == 2


def test_history_limit_and_ordering():
    for i in range(5):
        database.save_tutor_feedback("lim", f"2026-08-2{i}T00:00:00Z", "Anger", f"m{i}", "llm")
    rows = database.get_tutor_feedback_history(device_id="lim", limit=2)
    assert len(rows) == 2
    # most recent first
    assert rows[0]["message"] == "m4"
    assert rows[1]["message"] == "m3"


def test_empty_history_returns_empty_list():
    assert database.get_tutor_feedback_history(device_id="nobody") == []
    assert database.get_tutor_feedback_history(limit=0) == []