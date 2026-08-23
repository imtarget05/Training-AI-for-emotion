# Emotion-Aware AI Learning Assistant

Real-time facial-emotion recognition (ResNet-50 + OpenCV) that goes one step
further than a classification demo: when it observes a **sustained negative
emotional state**, an **AI Tutor** (cloud-hosted LLM via Cloudflare Workers AI)
generates a short, supportive coaching message for the learner, persists it,
and shows it on the dashboard — all without ever breaking the underlying
emotion pipeline if the LLM is down.

```
TRAIN → EVALUATE → DEPLOY → DETECT → DECIDE → GENERATE → FALLBACK → PERSIST → TEST → DEMONSTRATE
```

## 1. Project overview

Most emotion-recognition demos stop at a predicted label. This project keeps
that CV core and adds the decision layer that makes it useful in an e-learning
context:

* **Computer Vision** — face detection (Haar Cascade) + emotion classification
  (ResNet-50 transfer learning, 7 classes).
* **Real-time inference** — REST (`/predict`) for camera/IoT clients and
  WebSocket (`/ws/camera`) for the browser dashboard.
* **Temporal decision logic** — an emotion only triggers the tutor after it
  repeats **3 consecutive frames** and **45 s** have passed since the last
  trigger for that device. No LLM spam on single noisy frames.
* **Generative AI** — a cloud-hosted LLM (Cloudflare Workers AI, default
  `@cf/meta/llama-3.2-3b-instruct`, multilingual vi/en) writes an
  **emotion-aware**, ≤2-sentence coaching message.
* **Safe fallback** — if Workers AI is unreachable or times out, a rule-based
  canned message is returned (`source="fallback"`); the emotion API stays healthy.
* **Persistence** — every tutor intervention is stored in SQLite and exposed
  via `GET /tutor/history`.

> Exactly speaking: this is an **emotion-aware intervention** system. It reacts
> to the *detected facial expression*, not to an authoritative diagnosis of the
> learner's internal state (see [Limitations](#11-limitations)).

## 2. Architecture

```
                     Learner
                        │  camera
                        ▼
                  Face Detection (OpenCV Haar)
                        │
                        ▼
                ResNet-50 classifier (final_model.pth)
                        │  label + confidence + probs
                        ▼
                FastAPI  POST /predict · WS /ws/camera
                        │
                        ▼
              Temporal aggregation (streak=3, cooldown=45s)
                        │
                  ┌─────┴──────┐
               Neutral      Sadness/Fear/Anger/Disgust
                  │              repeated 3×
                  │                ▼
                  │          AI Tutor trigger
                  │            ┌────┴────┐
                  │        Workers AI  Rule-based
                  │          (llm)     (fallback)
                  │            └────┬────┘
                  └──────────────┬──┘
                                 │  feedback dict (+ latency_ms)
                          ┌──────┴──────┐
                          ▼             ▼
                    Dashboard      SQLite tutor_feedback
                    toast/banner  GET /tutor/history
```

Modules:

| File | Responsibility |
|---|---|
| `main.py` | FastAPI app — REST endpoints, WebSocket, tutor wiring |
| `model.py` | ResNet-50 encoder + classifier, Haar face detection, inference |
| `tutor.py` | GenAI layer — prompt, Cloudflare Workers AI call, fallback messages |
| `tutor_trigger.py` | Streak/cooldown decision logic (pure, unit-testable) |
| `database.py` | SQLite persistence (`predictions`, `tutor_feedback`, …) |
| `automation.py` | Batch inference, HTML reports, sample-data seeding |
| `evaluate.py` | Model evaluation CLI (`sanity` latency, dataset eval) |
| `static/index.html` | Dashboard SPA (camera, charts, tutor toast, Gợi ý button) |
| `scripts/demo_e2e.py` | Printable E2E demo scenario (interview screencast) |
## 3. ML pipeline

* **Backbone**: ResNet-50 (torchvision), feature dim 2048.
* **Classifier**: one linear layer `2048 → 7`
  (`Surprise, Fear, Disgust, Happiness, Sadness, Anger, Neutral`).
* **Preprocessing**: RGB 224×224, ImageNet mean/std normalization
  (`inference_transform` in `model.py`).
* **Face detection**: OpenCV `haarcascade_frontalface_default.xml`.
* **Artifact**: `final_model.pth` (~90 MB).

> ⚠️ The original **training script and evaluation dataset are not part of
> this repo**; the weights came from an earlier training run. `evaluate.py`
> therefore validates inference (`sanity`) and reproduces metrics (`eval`)
> honestly whenever a labeled dataset is supplied — nothing is fabricated
> (see [Testing](#8-testing) and `VALIDATION_REPORT.md`).

## 4. Real-time pipeline

* `POST /predict` — multipart image upload from any HTTP client. Response is
  backward compatible: `{device_id, timestamp, label, confidence, probs, emoji}`
  plus an optional `tutor_feedback` field.
* `WS /ws/camera` — base64 JPEG frames from the browser → JSON
  `{faces: [...], face_count}` per frame + optional `tutor_feedback`.
* Dashboard (`/dashboard`) renders boxes, per-class probability bars, history,
  and the tutor toast.

## 5. AI Tutor

Flow: predicted emotion → support-needed? → streak ≥ 3 consecutive frames →
cooldown ≥ 45 s → `_build_prompt` (few-shot, per-emotion pedagogical strategy,
≤2 sentences, "never mention you are an AI", no mental-health diagnosis) →
Cloudflare Workers AI REST (`messages` + temperature 0.6, `max_tokens` 120) →
fallback if unreachable/timeout → `{message, source, emotion, generated_at, latency_ms}`.

> The foundation model is a **Cloudflare-hosted pretrained model** used as the
> generative language component. This project's contribution is the tutor
> policy, prompt engineering, temporal gating and reliability layer around it —
> the LLM itself was not trained here. No local LLM is required at runtime.

Trigger tuning via env: `TUTOR_STREAK_THRESHOLD` (default 3),
`TUTOR_COOLDOWN_SECONDS` (default 45). Language via `lang="vi"|"en"`.

## 6. API

| Method | Path | Description |
|---|---|---|
| POST | `/predict` | Image → emotion result (+ optional `tutor_feedback`) |
| GET | `/latest/emotion` | Most recent prediction across devices |
| WS | `/ws/camera` | Real-time camera stream |
| POST | `/tutor/feedback` | On-demand tutor feedback (dashboard 💡 button) |
| GET | `/tutor/history?device_id=&limit=` | Persisted tutor interventions |
| GET | `/reports/emotion-distribution` | Emotion counts & percentages |
| GET | `/reports/confidence-stats` | Confidence avg/min/max |
| GET | `/reports/daily-summary` | Daily aggregation (30 days) |
| GET | `/reports/devices` | Per-device stats |
| GET | `/reports/predictions` | Filtered prediction list |
| GET | `/reports/export-csv` | CSV export |
| POST | `/reports/seed` | Seed sample data |
| POST | `/reports/generate-html` | Generate HTML report |

`POST /tutor/feedback` body: `{device_id?, emotion, confidence?, lang?}` returns
`{message, source: "llm"|"fallback", emotion, generated_at, latency_ms}`.

## 7. Database

SQLite (`emotion_data.db`, overridable via `DB_PATH`). Core tables:
`predictions`, `sessions`, `tutor_feedback` (indexed on `device_id`), `reports`.

## 8. Testing

```bash
pip install -r requirements-dev.txt
pytest                       # unit + integration (no GPU/model/credentials needed)
pytest -m e2e                # needs a live server: E2E_BASE_URL=http://localhost:8000
```

The suite covers: trigger state machine (streak, cooldown, reset, per-device
isolation), prompt quality (few-shot, guardrails, vi/en), the LLM/fallback
boundary (mock — no real Cloudflare credentials needed), DB round-trip, and the full API
contract via TestClient — **49 unit/integration tests, deterministic**. It
never requires a developer's GPU or local model weights (they are stubbed at
the import boundary in `tests/conftest.py`).
## 9. ML metrics

Only values that can actually be reproduced/measured are reported here
(full evidence in `VALIDATION_REPORT.md`):

| Metric | Value | Source |
|---|---|---|
| Inference sanity | ✅ valid labels/probs on synthetic inputs | `python evaluate.py sanity` |
| Inference latency (CPU) | median 118.7 ms (measured live in Docker) | `python evaluate.py sanity` |
| Test accuracy (FER2013 test, 7,178 imgs) | **49.83 %** | `python evaluate.py eval --data-dir data/test` |
| Macro-F1 (same run) | **0.4210** | same |
| Strongest class | Happiness F1=0.725; Surprise F1=0.621 | per-class report |
| Weakest class | Fear recall=0.046 (almost never predicted) | per-class report |

Full per-class P/R/F1 table + confusion matrix: `VALIDATION_REPORT.md`,
`image/eval_report.png`. Evaluation protocol: FER2013 public test split,
224×224 RGB center-resize, ImageNet normalization, full-image (pre-cropped
faces), batch of one on CPU. The model is below the FER2013
human-agreement ceiling (~65 %); metrics are reported as measured — see
Limitations.

Historical training curves and a confusion matrix live in `image/`
(`finetune_metrics.png`, `confusion_matrix.png`).

## 10. Run instructions

### Local (Python 3.10+)

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python main.py                        # http://127.0.0.1:8000/dashboard
```

### Docker (compose)

```bash
docker compose up --build             # http://localhost:8000/dashboard
```

`compose` binds `./data` for the SQLite file and forwards the Cloudflare
credentials (`CLOUDFLARE_ACCOUNT_ID` / `CLOUDFLARE_API_TOKEN`) from your
environment — no local LLM server is needed.

### Plain Docker

```bash
docker build -t training-ai-emotion:latest .
docker run -d -p 8080:8080 \
  -e CLOUDFLARE_ACCOUNT_ID=$CLOUDFLARE_ACCOUNT_ID \
  -e CLOUDFLARE_API_TOKEN=$CLOUDFLARE_API_TOKEN \
  training-ai-emotion:latest
```

### LLM provider (Cloudflare Workers AI)

The tutor runs on **Cloudflare Workers AI** — fully cloud-hosted, free-tier
friendly, no local model server. Create an API token in the Cloudflare
dashboard (Workers AI → Use REST API → Create API Token), then:

```bash
export CLOUDFLARE_ACCOUNT_ID=... CLOUDFLARE_API_TOKEN=...
```

Default model: `@cf/meta/llama-3.2-3b-instruct` (GA, multilingual vi/en).
Override with `CLOUDFLARE_AI_MODEL` (e.g.
`@cf/meta/llama-3.3-70b-instruct-fp8-fast`). Without credentials the app
still runs — the tutor falls back to canned messages.

### Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `PORT` | `8000` (Docker `8080`) | uvicorn port |
| `WEIGHTS_PATH` | `final_model.pth` | model weights |
| `DB_PATH` | `emotion_data.db` | SQLite path |
| `CLOUDFLARE_ACCOUNT_ID` | *(empty)* | Workers AI account id |
| `CLOUDFLARE_API_TOKEN` | *(empty)* | Workers AI API token |
| `CLOUDFLARE_AI_MODEL` | `@cf/meta/llama-3.2-3b-instruct` | cloud LLM model id |
| `CLOUDFLARE_AI_TIMEOUT_SECONDS` | `10` | LLM call timeout |
| `TUTOR_STREAK_THRESHOLD` | `3` | consecutive frames before a trigger |
| `TUTOR_COOLDOWN_SECONDS` | `45` | min seconds between triggers/device |

See `.env.example`.

### E2E demo (interview screencast)

```bash
docker compose up --build
python scripts/demo_e2e.py --base http://localhost:8000
```

## 11. Limitations

* Emotion recognition is **probabilistic**; a facial expression is not proof
  of an emotional state.
* The tutor reacts to the *detected* emotion and has **no course or learner
  context**, so personalization is limited — this is emotion-aware
  intervention, not full adaptive learning.
* Haar Cascade works best on frontal faces (angle/light sensitivity).
* LLM quality depends on the selected Workers AI model/runtime; guardrails are
  in the prompt (≤2 sentences, no diagnosis, no "I am an AI" wording).
* Real cloud inference requires Cloudflare credentials; without them the tutor
  deterministically uses rule-based fallbacks (validated live). The LLM path
  itself is unit-tested with a deterministic provider mock.
* Training/eval dataset is not in this repo; `evaluate.py` reproduces metrics
  when data is provided. Nothing is fabricated in these docs.

## Author

**Mai Nguyen Binh Tan** — AI/ML engineering portfolio project.
| `tests/` | pytest suite (unit + integration + E2E) |