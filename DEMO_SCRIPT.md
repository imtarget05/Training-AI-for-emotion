# Demo Script — 3–5 minute interview demo

Prereq (one-time, documented in README): `pip install -r requirements.txt`, Cloudflare env vars set (or omit → fallback demo), `ollama` NOT required.

## Scenario 1 — Normal emotion recognition (~45s)
1. `python main.py` → open http://127.0.0.1:8000/dashboard
2. Allow camera; show live emotion labels updating (smile → Happiness).
3. Point out `/info`: model `emotion-resnet50 v1`, 7 classes — the *trained* ML component.

## Scenario 2 — Sustained negative emotion → AI Tutor (~60s)
1. Frown / look sad for ~3 s.
2. Narrate: "one noisy frame won't trigger anything — the gate needs 3 consecutive support-needed frames."
3. Tutor toast appears, labelled **AI** (cloud) or **Rule-based** (fallback).

## Scenario 3 — Cloud LLM feedback (~30s)
1. With credentials set, trigger again after cooldown: toast shows a fresh generated message (vi or en via the UI toggle).
2. Mention latency ≈ 0.8–1.05 s measured against @cf/meta/llama-3.2-3b-instruct.

## Scenario 4 — Cooldown prevents spam (~20s)
1. Keep the negative expression.
2. Show no second toast within 45 s; explain per-device cooldown + streak reset on neutral.

## Scenario 5 — Failure → fallback (~40s)
1. Unset/revoke the token, restart server, re-trigger.
2. Toast now labelled **Rule-based**; prediction keeps flowing — "LLM failure never breaks the CV path."

## Scenario 6 — Evidence pack (~45s)
1. Open `image/eval_report.png` (confusion matrix) and terminal output:
   `python evaluate.py eval --data-dir data/test` → 49.83% acc / 0.421 macro-F1.
2. `mlflow ui` → show run params/metrics/git commit.
3. Close with limitations slide (Fear recall, expression ≠ emotion).

Fallback (no camera): run `python scripts/demo_e2e.py` against the Docker container — same scenarios automated with printed PASS/FAIL evidence.
