"""Unit tests for the AI Tutor layer (prompt + fallback + LLM boundary)."""

import asyncio

import pytest

import tutor


def _feedback(emotion, lang="vi", **kwargs):
    return asyncio.run(tutor.generate_tutor_feedback(emotion, 0.8, lang=lang, **kwargs))


# ── Phase 5: prompt engineering quality ───────────────────────────────────

def test_prompt_contains_all_key_sections():
    prompt = tutor._build_prompt("Sadness", 0.81, ["Neutral", "Sadness"], "vi")
    assert "You are an AI learning companion" in prompt
    assert "Sadness" in prompt
    assert "0.81" in prompt
    assert "Neutral -> Sadness" in prompt
    assert "encourage gently" in prompt  # EMOTION_STRATEGY for Sadness
    assert "max 2 sentences" in prompt
    assert "Do not mention that you are an AI" in prompt
    # New safety + scope guardrails
    assert "diagnose" in prompt


def test_prompt_language_instruction_vi_vs_en():
    vi = tutor._build_prompt("Fear", 0.7, None, "vi")
    en = tutor._build_prompt("Fear", 0.7, None, "en")
    assert "Trả lời bằng tiếng Việt" in vi
    assert "Reply in English" in en


def test_prompt_empty_trend_is_handled():
    prompt = tutor._build_prompt("Anger", 0.9, [], "en")
    assert "no recent history" in prompt


def test_prompt_strategy_differs_per_emotion():
    strategies = {e: tutor._build_prompt(e, 0.8, None, "en") for e in tutor.SUPPORT_EMOTIONS}
    # Each emotion must land in a different strategy hint → prompts differ.
    assert len({s for s in strategies.values()}) == len(tutor.SUPPORT_EMOTIONS)


# ── needs_tutor_support ────────────────────────────────────────────────────

@pytest.mark.parametrize("emotion", ["Sadness", "Fear", "Anger", "Disgust"])
def test_needs_tutor_support_true_for_support_emotions(emotion):
    assert tutor.needs_tutor_support(emotion)


@pytest.mark.parametrize("emotion", ["Neutral", "Happiness", "Surprise", "Confusion", ""])
def test_needs_tutor_support_false_otherwise(emotion):
    assert not tutor.needs_tutor_support(emotion)


# ── fallback path (LLM unavailable) ────────────────────────────────────────

@pytest.mark.parametrize("emotion", ["Sadness", "Fear", "Anger", "Disgust"])
@pytest.mark.parametrize("lang, table", [("vi", tutor.FALLBACK_MESSAGES_VI), ("en", tutor.FALLBACK_MESSAGES_EN)])
def test_fallback_uses_emotion_language_table(monkeypatch, emotion, lang, table):
    async def no_llm(*a, **k):
        return None

    monkeypatch.setattr(tutor, "_call_cloudflare", no_llm)
    fb = _feedback(emotion, lang=lang)
    assert fb["source"] == "fallback"
    assert fb["emotion"] == emotion
    assert fb["message"] == table[emotion]
    assert "latency_ms" in fb


def test_fallback_generic_message_for_unknown_emotion(monkeypatch):
    async def no_llm(*a, **k):
        return None

    monkeypatch.setattr(tutor, "_call_cloudflare", no_llm)
    fb = _feedback("Confusion", lang="vi")
    assert fb["source"] == "fallback"
    assert "từ từ" in fb["message"]
    fb_en = _feedback("Confusion", lang="en")
    assert "Take your time" in fb_en["message"]


# ── LLM success path (deterministic, mocked) ───────────────────────────────

def test_llm_success_returns_message_and_source(monkeypatch):
    async def fake_llm(*a, **k):
        return "You've got this — try the easier question first. 💪"

    monkeypatch.setattr(tutor, "_call_cloudflare", fake_llm)
    fb = _feedback("Sadness", lang="vi")
    assert fb["source"] == "llm"
    assert fb["message"] == "You've got this — try the easier question first. 💪"
    assert isinstance(fb["latency_ms"], int)


def test_empty_llm_response_falls_back(monkeypatch):
    async def empty_llm(*a, **k):
        return "   "

    monkeypatch.setattr(tutor, "_call_cloudflare", empty_llm)
    fb = _feedback("Anger", lang="vi")
    assert fb["source"] == "fallback"
    assert "khó chịu" in fb["message"]