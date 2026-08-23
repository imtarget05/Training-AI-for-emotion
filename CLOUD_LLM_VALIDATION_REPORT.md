# Cloud LLM Validation Report

## Status: REAL CLOUD LLM = ✅ VERIFIED (2026-08-23)

## 1–4. Provider / Model / Contract / Timestamp

| Item | Value |
|---|---|
| Provider | Cloudflare Workers AI REST API |
| Model ID | `@cf/meta/llama-3.2-3b-instruct` — verified **GA** on official docs (llama-3.1-8b-instruct deprecated 5/30/2026) |
| API contract | `POST /client/v4/accounts/{id}/ai/run/{model}`, Bearer auth, `{messages, max_tokens, temperature, stream:false}` → `result.response` |
| Test timestamp | 2026-08-23 |
| Credentials handling | env-only, never written to disk or repo |

## 5–7. Real test matrix — 8/8 PASS (real inference, not mocks)

Executed through the actual runtime path `tutor.generate_tutor_feedback()`:

| Emotion | Lang | Source | Latency | Response (excerpt) |
|---|---|---|---|---|
| Sadness | vi | llm | 864ms | "Đừng lo, mọi người đều gặp khó khăn khi học một điều gì đó mới…" |
| Sadness | en | llm | 971ms | "…Let's take a deep breath and break it down into smaller steps" |
| Fear | vi | llm | 835ms | "Bạn đang cảm thấy hơi lo lắng, nhưng hãy nhớ rằng không có gì là không thể…" |
| Fear | en | llm | 756ms | "…no rush. Let's review the material at your own pace" |
| Anger | vi | llm | 1024ms | "Tôi hiểu bạn đang cảm thấy khó chịu… nghỉ ngơi một chút" |
| Anger | en | llm | 1050ms | "…frustration is completely valid. Why don't we take a short break" |
| Disgust | vi | llm | 854ms | "…giải thích lại một cách khác hoặc chuyển sang chủ đề khác" |
| Disgust | en | llm | 785ms | "…explain it in a different way" |

Latency range: **755–1050 ms** (cloud round-trip). Smoke test: HTTP 200, correct schema.

## 8. Guardrails — ALL PASS (8/8)

- [x] ≤ 2 sentences (all cases)
- [x] Correct language (vi responses in Vietnamese, en in English)
- [x] Emotion-specific strategy visible (Sadness=encourage, Fear=no-rush, Anger=break, Disgust=alternative explanation)
- [x] No "I am an AI" leakage (automated check)
- [x] No diagnostic/mental-health claims
- [x] No fabricated learner/course facts
- [x] Coaching/educational framing, concise

## Runtime flow + failure regression — REAL

- **Real failure test:** invalid token → Cloudflare 401 → caught, logged → `source="fallback"` (1423ms) → no exception ✅
- **Full regression after real testing:** `55 passed, 7 skipped`; `py_compile` all modules ✅
- Runtime Ollama/local-LLM references: **ZERO**

## Security

- Token used via environment variables only; never committed, never written to any repo file
- `.env` git-ignored; `.env.example` = placeholders; docker-compose uses `${VAR:-}` expansion

## Classification summary

| Evidence type | Result |
|---|---|
| **REAL CLOUD INFERENCE** | **VERIFIED — 8/8 cases + smoke + failure regression** |
| DETERMINISTIC MOCK TESTS | 55 unit/integration tests (CI default, credential-free) |

## PORTFOLIO READY = YES
