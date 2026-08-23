"""
tutor_trigger.py — sustained-emotion trigger logic (framework-free).

Decides WHEN the AI Tutor should be called. Kept as a pure, unit-testable
module separate from main.py and the CV pipeline so the "do we call the LLM?"
decision can be tested deterministically without a GPU, a webcam, or a server.

Rules (see PROJECT goal):
- Only support-needed emotions (Sadness / Fear / Anger / Disgust) count.
- The same emotion must repeat TUTOR_STREAK_THRESHOLD consecutive frames.
- After a trigger, the same device is suppressed for TUTOR_COOLDOWN_SECONDS.
- Any non-support emotion (Neutral / Happiness / Surprise) resets the streak.

Both knobs are configurable via environment variables for tuning.
"""

import os
import time
from typing import Any, Dict, List

from tutor import needs_tutor_support

TUTOR_STREAK_THRESHOLD = int(os.environ.get("TUTOR_STREAK_THRESHOLD", "3"))
TUTOR_COOLDOWN_SECONDS = int(os.environ.get("TUTOR_COOLDOWN_SECONDS", "45"))

# Per-device in-memory state (fine for a single-process uvicorn worker).
_emotion_streaks: Dict[str, Dict[str, Any]] = {}
_last_tutor_trigger: Dict[str, float] = {}


def update_streak_and_should_trigger(device_id: str, emotion: str) -> bool:
    """
    Feed one prediction into the trigger state machine.

    Returns True exactly when a tutor message should be generated for this
    device right now (support emotion, streak reached threshold, cooldown
    elapsed). The streak counter is reset after a successful trigger so a
    brand-new streak is required before triggering again.
    """
    if not needs_tutor_support(emotion):
        _emotion_streaks.pop(device_id, None)  # neutral / happy / surprise resets streak
        return False

    streak = _emotion_streaks.get(device_id)
    if streak and streak["emotion"] == emotion:
        streak["count"] += 1
    else:
        streak = {"emotion": emotion, "count": 1}
    _emotion_streaks[device_id] = streak

    if streak["count"] < TUTOR_STREAK_THRESHOLD:
        return False

    last_trigger = _last_tutor_trigger.get(device_id, 0.0)
    if time.time() - last_trigger < TUTOR_COOLDOWN_SECONDS:
        return False

    _last_tutor_trigger[device_id] = time.time()
    streak["count"] = 0  # reset so the next trigger needs a fresh streak
    return True


def recent_emotion_trend(device_id: str, limit: int = 5) -> List[str]:
    """
    Most recent emotions for this device (oldest → newest), used as context
    in the tutor prompt. Never raises: returns [] if the DB is unavailable.
    """
    try:
        from database import get_predictions

        recent = get_predictions(device_id=device_id, limit=limit)
        return [r["emotion"] for r in reversed(recent)]
    except Exception:
        return []


def reset_device(device_id: str) -> None:
    """Clear streak + cooldown state for one device (used by tests / admin)."""
    _emotion_streaks.pop(device_id, None)
    _last_tutor_trigger.pop(device_id, None)


def reset_all() -> None:
    """Clear all trigger state (used by tests between cases)."""
    _emotion_streaks.clear()
    _last_tutor_trigger.clear()