"""Unit tests for the sustained-emotion trigger state machine (Phase 4 cases)."""

import time

from tutor_trigger import (
    TUTOR_COOLDOWN_SECONDS,
    TUTOR_STREAK_THRESHOLD,
    update_streak_and_should_trigger,
    recent_emotion_trend,
)


def _frames(device, emotions):
    """Push a sequence of emotions and return the trigger decision for each."""
    return [update_streak_and_should_trigger(device, e) for e in emotions]


# ── Case A: normal/positive emotions never trigger ─────────────────────────

def test_case_a_normal_emotions_never_trigger():
    for emotion in ("Neutral", "Happiness", "Surprise"):
        assert _frames("dev_a", [emotion] * 10) == [False] * 10


# ── Cases B/C/D: streak threshold behaviour ────────────────────────────────

def test_case_bcd_streak_threshold():
    # threshold is 3 → first two frames no, third frame yes
    decisions = _frames("dev_bcd", ["Sadness"] * 4)
    expected = [False, False, True, False]  # 4th suppressed by counter reset
    assert decisions == expected


def test_streak_requires_same_emotion_consecutive():
    # interleaving a different support emotion restarts the counter
    decisions = _frames("dev_mix", ["Sadness", "Fear", "Sadness", "Sadness"])
    assert decisions == [False, False, False, False]  # reset at index 1


# ── Case E: no repeated LLM after trigger (cooldown) ───────────────────────

def test_case_e_no_repeat_within_cooldown():
    # already triggered once (see bcd test), a new streak within cooldown stays off
    _frames("dev_cd", ["Sadness"] * 3)  # triggers, sets last trigger time
    assert _frames("dev_cd", ["Sadness"] * 3) == [False, False, False]


# ── Case F: within cooldown → no additional trigger ────────────────────────

def test_case_f_trigger_once_then_cooldown_blocks_next():
    # one trigger
    assert _frames("dev_f", ["Sadness"] * 3) == [False, False, True]
    # immediately build a fresh streak → cooldown still blocks
    assert _frames("dev_f", ["Sadness", "Sadness", "Sadness"]) == [False, False, False]


def test_case_g_cooldown_expired_allows_retrigger():
    # record a trigger, then force the stored timestamp back past the cooldown
    _frames("dev_g", ["Sadness"] * 3)  # ensures _last_tutor_trigger["dev_g"] set
    from tutor_trigger import _last_tutor_trigger

    _last_tutor_trigger["dev_g"] = time.time() - (TUTOR_COOLDOWN_SECONDS + 1)
    decisions = _frames("dev_g", ["Sadness"] * 3)
    assert decisions == [False, False, True]


# ── Case H: neutral resets the streak ──────────────────────────────────────

def test_case_h_neutral_after_negative_streak_resets():
    _frames("dev_h", ["Sadness"] * 2)  # counter at 2
    assert update_streak_and_should_trigger("dev_h", "Neutral") is False
    # streak dropped → need 3 fresh Sadness frames again (not 1 more)
    assert _frames("dev_h", ["Sadness"] * 3) == [False, False, True]


# ── per-device isolation ───────────────────────────────────────────────────

def test_streak_state_is_per_device():
    _frames("dev_i", ["Sadness"] * 2)
    assert update_streak_and_should_trigger("dev_j", "Sadness") is False  # own counter
    _frames("dev_i", ["Sadness"])  # third frame for dev_i triggers
    assert update_streak_and_should_trigger("dev_i", "Sadness") is False


# ── tuning constants are sane ──────────────────────────────────────────────

def test_default_tuning_constants():
    assert TUTOR_STREAK_THRESHOLD == 3
    assert TUTOR_COOLDOWN_SECONDS == 45


# ── recent_emotion_trend never raises (DB mocked away) ─────────────────────

def test_recent_emotion_trend_returns_list_on_db_failure(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("db down")

    monkeypatch.setattr("database.get_predictions", boom)
    assert recent_emotion_trend("dev_x") == []