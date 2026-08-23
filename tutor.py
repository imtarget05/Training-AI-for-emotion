"""
tutor.py — AI Tutor feedback layer.

Bridges the emotion-recognition pipeline to a cloud-hosted LLM (Cloudflare
Workers AI) to generate short, adaptive coaching messages for a learner in a
digital-learning / e-learning context (e.g., "student looks frustrated ->
suggest a break or an easier next step").

Design notes:
- Uses Cloudflare Workers AI REST API (@cf/meta/llama-3.3-70b-instruct-fp8-fast
  by default, configurable via CLOUDFLARE_AI_MODEL). Credentials are provided
  exclusively through environment variables — never hardcoded.
  NOTE: the foundation model is a Cloudflare-hosted pretrained model; this
  project's contribution is the tutor logic, prompt engineering and
  reliability layer around it — not the model itself.
- Never calls the LLM on every frame. main.py only calls generate_tutor_feedback()
  once an emotion has been *sustained* for a few consecutive frames, to avoid
  spamming the model on noisy single-frame misclassifications.
- If Cloudflare is unreachable, times out, or errors, falls back to a canned,
  rule-based message so the API never breaks because the LLM is down.
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger("tutor")
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)

# ─── Cloudflare Workers AI configuration (env-only) ──────────────────────────
CLOUDFLARE_ACCOUNT_ID = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")
CLOUDFLARE_API_TOKEN = os.environ.get("CLOUDFLARE_API_TOKEN", "")
CLOUDFLARE_AI_MODEL = os.environ.get(
    "CLOUDFLARE_AI_MODEL", "@cf/meta/llama-3.2-3b-instruct"
)
CLOUDFLARE_AI_TIMEOUT_SECONDS = float(
    os.environ.get("CLOUDFLARE_AI_TIMEOUT_SECONDS", "10")
)
_CLOUDFLARE_API_BASE = "https://api.cloudflare.com/client/v4"

# Emotions that should trigger a supportive tutor message.
# (Neutral / Happiness / Surprise are not "needs support" states.)
SUPPORT_EMOTIONS = {"Sadness", "Fear", "Anger", "Disgust"}

# Short pedagogical strategy hint per emotion — used inside the prompt so the
# LLM doesn't have to guess *how* to respond to each state.
EMOTION_STRATEGY = {
    "Sadness": "encourage gently, remind them mistakes are part of learning, offer a smaller next step",
    "Fear": "reduce anxiety, reassure there's no time pressure, offer to slow down or review a prerequisite",
    "Anger": "de-escalate calmly, acknowledge the frustration is valid, suggest a short break before continuing",
    "Disgust": "acknowledge the content may feel unclear or off-putting, offer an alternative explanation or example",
}

# Rule-based fallback messages (Vietnamese) used when the LLM is unavailable.
FALLBACK_MESSAGES_VI = {
    "Sadness": "Có vẻ bạn đang hơi nản. Không sao cả — hãy thử một câu hỏi dễ hơn trước khi quay lại phần này nhé.",
    "Fear": "Bạn không cần vội. Nếu phần này còn mơ hồ, mình có thể ôn lại kiến thức nền trước khi tiếp tục.",
    "Anger": "Có vẻ phần này đang gây khó chịu. Nghỉ 2 phút rồi quay lại thường sẽ giúp ích đấy.",
    "Disgust": "Có thể cách giải thích hiện tại chưa hợp với bạn. Để mình thử một ví dụ khác nhé.",
}

FALLBACK_MESSAGES_EN = {
    "Sadness": "You seem a bit discouraged — that's okay. Let's try an easier question before coming back to this.",
    "Fear": "No rush here. If this part feels unclear, we can review the basics first.",
    "Anger": "This part seems frustrating. A short 2-minute break often helps before continuing.",
    "Disgust": "Maybe this explanation isn't clicking. Let's try a different example.",
}


def _build_prompt(emotion: str, confidence: float, trend: Optional[List[str]], lang: str) -> str:
    """
    Structured, few-shot prompt for the tutor LLM.
    Kept deliberately short: the goal is a 1-2 sentence reply, not an essay.
    """
    strategy = EMOTION_STRATEGY.get(emotion, "respond warmly and encourage the learner")
    trend_str = " -> ".join(trend) if trend else "no recent history"
    lang_instruction = "Trả lời bằng tiếng Việt." if lang == "vi" else "Reply in English."

    return f"""You are an AI learning companion embedded in an e-learning platform.
A webcam-based emotion model just detected the learner's facial expression.

Examples:
- emotion=Sadness, confidence=0.81 -> "It's okay to find this tricky — let's try a simpler example first, then come back to this one."
- emotion=Anger, confidence=0.74 -> "This part looks frustrating. Want to take a short break, or should I explain it a different way?"

Now respond to this real case:
- Detected emotion: {emotion} (confidence={confidence:.2f})
- Recent emotion trend: {trend_str}
- Suggested strategy: {strategy}

Write ONE short, warm, non-repetitive message (max 2 sentences) directly to the learner.
Do not mention that you are an AI, a model, or that emotion was "detected" — just respond naturally, like a supportive tutor.
Do not diagnose or label the learner's mental state, and do not invent facts about specific courses or content.
{lang_instruction}"""


def _cloudflare_configured() -> bool:
    """Whether Cloudflare Workers AI credentials are present."""
    return bool(CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN)


async def _call_cloudflare(prompt: str) -> Optional[str]:
    """
    Call Cloudflare Workers AI text generation via the REST API.
    Returns None on any failure (missing config, timeout, HTTP error,
    malformed body, empty reply) — never raises.
    """
    if not _cloudflare_configured():
        logger.warning(
            "Cloudflare Workers AI not configured "
            "(set CLOUDFLARE_ACCOUNT_ID / CLOUDFLARE_API_TOKEN); using fallback."
        )
        return None

    url = (
        f"{_CLOUDFLARE_API_BASE}/accounts/{CLOUDFLARE_ACCOUNT_ID}"
        f"/ai/run/{CLOUDFLARE_AI_MODEL}"
    )
    headers = {"Authorization": f"Bearer {CLOUDFLARE_API_TOKEN}"}
    payload = {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a supportive learning companion. Follow the "
                    "user's instructions exactly: reply with ONE short warm "
                    "message, max 2 sentences, in the requested language."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 120,
        "temperature": 0.6,
        "stream": False,
    }
    try:
        async with httpx.AsyncClient(timeout=CLOUDFLARE_AI_TIMEOUT_SECONDS) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            # Contract: {"result": {"response": "..."}, "success": true, ...}
            result = data.get("result") or {}
            text = (result.get("response") or "").strip() if isinstance(result, dict) else ""
            return text or None
    except Exception as e:
        logger.warning("Cloudflare Workers AI call failed (%s); falling back to canned message.", e)
        return None


async def generate_tutor_feedback(
    emotion: str,
    confidence: float,
    trend: Optional[List[str]] = None,
    lang: str = "vi",
) -> Dict[str, Any]:
    """
    Main entry point. Returns:
    {
      "message": str,
      "source": "llm" | "fallback",
      "emotion": str,
      "generated_at": float (unix timestamp),
      "latency_ms": int (ttfb of the LLM call; ~0 for fallback)
    }
    """
    prompt = _build_prompt(emotion, confidence, trend, lang)
    start = time.perf_counter()
    message = await _call_cloudflare(prompt)
    latency_ms = int((time.perf_counter() - start) * 1000)
    source = "llm"

    # Defence-in-depth: a blank or whitespace-only LLM reply is treated as a
    # failure too (->_call_cloudflare strips; empty response == unusable).
    if not message or not message.strip():
        message = None

    if message is None:
        fallback_table = FALLBACK_MESSAGES_VI if lang == "vi" else FALLBACK_MESSAGES_EN
        message = fallback_table.get(
            emotion,
            "Cứ từ từ nhé, mình luôn ở đây nếu bạn cần hỗ trợ." if lang == "vi" else "Take your time — I'm here if you need help.",
        )
        source = "fallback"

    return {
        "message": message,
        "source": source,
        "emotion": emotion,
        "generated_at": time.time(),
        "latency_ms": latency_ms,
    }


def needs_tutor_support(emotion: str) -> bool:
    """Whether this emotion class should be eligible to trigger tutor feedback."""
    return emotion in SUPPORT_EMOTIONS
