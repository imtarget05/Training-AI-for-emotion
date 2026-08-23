# Project Presentation Guide — Bosch AI Engineer Demo

> A tight script to present `Training-AI-for-emotion` as an
> **Emotion-Aware AI Learning Assistant** in 3–5 minutes, with a live or video
> demo. It matches the actual repo evidence; no metric is inflated.

---

## Message in one line

> Real-time facial-emotion **recognition** (ResNet-50, honestly evaluated) +
> **temporal decisions** + a **cloud AI tutor** (Cloudflare Workers AI) +
> **safe fallback** + **persistence**, packaged as a repeatable product.

---

## 60-second headline (walk the architecture)

```
Camera ─▶ OpenCV face detect ─▶ ResNet-50 (7 classes)
                                        │
                                        ▼
                          temporal gate (streak=3, cooldown=45s)
                                        │
                                 (only on sustained negative)
                                        ▼
                     AI Tutor → prompt (vi/en) → Cloudflare Workers AI
                                        │
                              ┌──────────┴──────────┐
                         LLM reply           rule-based fallback
                              └──────────┬──────────┘
                                         ▼
                                Frontend toast + SQLite/Neon
```

Say: *"It's a CV pipeline with a reliability layer and a GenAI layer — not a
chatbot, and not just a label classifier."*

---

## The pitch — section by section

### 1. Problem (30s)
Facial-expression recognition is usually a demo that stops at a label. In
e-learning, the *signal* matters: a learner who looks frustrated for a few
seconds. We intervene only when a negative state is **sustained**.

### 2. ML / evaluation (45s)
- ResNet-50 transfer learning, 7 classes, 224×224.
- Evaluated on **7,178 FER2013 public-test images**: accuracy **49.83%**,
  **macro-F1 0.4210**.
- Why macro-F1? Imbalanced classes; accuracy hides a weak Fear class
  (recall ≈ 0.046) — we measured it and show it in the confusion matrix
  (`image/eval_report.png`).

### 3. GenAI / tutor (45s)
- Cloudflare Workers AI, `@cf/meta/llama-3.2-3b-instruct` (verified current,
  since `llama-3.1-8b-instruct` is deprecated).
- Structured bilingual prompt: emotion strategy, ≤2 sentences, no "I am an AI",
  no diagnosis, no invented course facts.

### 4. Reliability (45s)
- **Never call the LLM per frame** → 3-streak + 45s cooldown per device +
  neutral/emotion reset. Proven with unit + E2E tests.
- **Isolated provider + fallback**: HTTP error / timeout / empty / malformed →
  canned `source="fallback"` message. The CV path **never breaks** when the
  cloud LLM is down (verified 401→fallback, health stays 200).

### 5. MLOps / engineering (45s)
- MLflow (params + metrics + git commit + artifact), model versioning
  (`model_metadata.json` + `GET /info`), Docker, GitHub Actions CI (55 tests),
  7 live E2E tests, Postgres (Neon) / SQLite separation.

---

## Live vs video demo

- **Live (best, if camera + credentials work):** run all 6 scenarios from
  `DEMO_SCRIPT.md` (~4–5 min).
- **Fallback demo (no camera):** `python scripts/demo_e2e.py` against the Docker
  container — same scenarios automated with `PASS/FAIL` printed.
- **Video fallback:** screen-record the live dashboard + `image/eval_report.png`
  + `mlflow ui` so the evidence is visible even if the camera is off.

---

## The honest-limits slide (say this explicitly — it is your strength)

> - Facial **expression ≠ internal emotion**; classification is probabilistic.
> - **Fear recall ≈ 0.046** — the model is weak on Fear; gating reduces false
>   triggers but cannot remove the bias. Fixing it needs class-balanced
>   retraining or new data.
> - Tutor intervention is based on the **detected** state; it is **not**
>   a mental-health diagnostic.
> - Local LLM is not used; Workers AI output quality depends on the model and
>   runtime.
> - Free tiers sleep/suspend → cold starts; "$0/month while within quotas".

---

## Suggested 5-minute talk track

| Minute | Block |
|---|---|
| 0:00–0:20 | Architecture headline + one-liner identity |
| 0:20–1:00 | ML model + honest metrics on FER2013 |
| 1:00–1:45 | GenAI tutor + prompt guardrails |
| 1:45–2:30 | Reliability: streak/cooldown/fallback isolation |
| 2:30–3:10 | MLOps: MLflow, versioning, CI, Docker |
| 3:10–3:50 | Live dashboard demo (or `demo_e2e.py`) |
| 3:50–4:30 | Confusion matrix + limitations (Fear recall, expression≠emotion) |
| 4:30–5:00 | Wrap: what you'd improve next |

---

## Files to show, in order
1. `README.md` (architecture)
2. `image/eval_report.png` (confusion matrix)
3. `MLflow` run page (params/metrics/artifact)
4. Live dashboard (camera) or `demo_e2e.py` output
5. `VALIDATION_REPORT.md` / `PUBLIC_DEPLOYMENT_REPORT.md` (evidence)

## Pitfalls to avoid
- Don't call the model "accurate" or "production emotion recognition".
- Don't present the cloud LLM as "your model" — it's hosted inference; your
  contribution is prompt engineering + tutor policy + reliability.
- Don't claim "personalized adaptive learning" — use **emotion-aware
  intervention**.
- Make sure the live deployment (Pages + Koyeb + Neon) is proven once you
  report live E2E; otherwise present it as "deployment-ready, design proven".