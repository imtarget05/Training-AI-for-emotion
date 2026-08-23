# Interview Preparation — AI Engineer (Bosch) · Training-AI-for-emotion

> Quick-answer bank for the highest-probability technical questions, built from
> **implemented, measured evidence** (see `VALIDATION_REPORT.md`,
> `CLOUD_LLM_VALIDATION_REPORT.md`, `FINAL_REPOSITORY_AUDIT.md`, `PORTFOLIO_SUMMARY.md`).
> Every answer is traceable to source or measurement. Where something is a
> limitation, say so — do not gloss.

---

## 1. Why ResNet-50?
- Depth vs cost trade-off for CPU real-time inference: 50 layers with residual
  connections suit a 7-class affect-classification task without the
  memory/latency of a ResNet-101/152.
- Transfer learning from ImageNet converges fast; we fine-tune the head plus
  late blocks, keeping it small and fast.
- Measured: median CPU inference ≈ 119 ms/image at 224×224 (local container
  baseline; production re-measured separately).

## 2. Why transfer learning on ImageNet?
- FER2013 is small/noisy; training from scratch overfits. ImageNet weights reuse
  low/mid-level visual features so a small dataset only adjusts task-specific
  high-level features.

## 3. Why FER2013?
- Public, widely-used benchmark with **exactly 7 discrete classes** mapping 1:1
  to the app's label space (`Surprise, Fear, Disgust, Happiness, Sadness,
  Anger, Neutral`).
- Public **test split** (7,178 images) = reproducible evaluation; docs in
  `DATASET.md`.

## 4. Why is accuracy insufficient?
- Accuracy hides class imbalance: a model that always predicts `Neutral/Happy`
  can be "accurate" while never detecting rare classes like Fear/Disgust — the
  exact failure an emotion-aware tutor cares about.

## 5. Why Macro-F1 = 0.4210 (and why that's ok to present)?
- Macro-F1 **averages per-class F1 equally**, so rare/weak classes matter — which
  is what we want visible. FER2013 test labels are noisy (human agreement ~65%),
  so even a good model on this protocol lands well below that. 0.4210 is the
  honest, reproducible measured value (reproducible across Docker and venv runs).
- Do NOT say "highly accurate". Say: "Macro-F1 0.42 on the FER2013 public test;
  strong on Happiness, weak on Fear."

## 6. Why Fear recall ≈ 0.046?
- The model almost never labels anything "Fear": under-represented and heavily
  confusable with Surprise → low recall. This is a **measured** failure mode,
  and it's the strongest evidence of honest evaluation.

## 7. Why not claim the model is highly accurate?
- 49.83% / 0.4210 on a noisy public protocol is not a high-performance claim.
- A facial *expression is not the same as internal emotional state*, so "accurate
## 10. Why 45-second cooldown?
- Prevents hammering the LLM when the same negative expression persists.
  `TUTOR_COOLDOWN_SECONDS=45` per device is a reliability/UX trade-off.

## 11. Why Cloudflare Workers AI?
- No local LLM in production (portable, GPU-free); free tier suits a portfolio
  demo; simple REST auth. Model choice **verified against official docs**:
  `@cf/meta/llama-3.2-3b-instruct`.

## 12. Why `llama-3.2-3b-instruct`, not `llama3.1:8b`?
- `llama-3.1-8b-instruct` became **deprecated on Workers AI (5/30/2026)**; we
  verified against the official model catalog and switched to the current stable
  3.2 3B instruct. A real "engineering under external API change" decision.

## 13. Why NO local LLM?
- Keeps deployment **portable / GPU-free**, removes a large runtime dependency,
  and isolates it from the CV path. Short coaching messages don't need a local
  model; cloud inference + fallback gives the same UX with a simpler runtime.

## 14. Why fallback? / What happens when Cloudflare fails?
- Isolated provider with explicit timeout. On HTTP error, timeout, empty, or
  malformed response → return a rule-based canned message with `source="fallback"`.
- Emotion prediction NEVER depends on LLM: a failing tutor call cannot break
  `/predict` / WS camera path. Verified in tests (401→fallback, path stays healthy).

## 15. Why mock tests AND real integration tests?
- Deterministic unit suite (55) mocks/stubs the model + LLM → credential-free on
  any machine, no GPU; used by CI.
- Real integration suite (7 Docker E2E) exercises the actual FastAPI +
  Postgres/SQLite container path.
- Real Cloudflare matrix (8/8 real vi+en cases) is manual / opt-in, never in
  default CI (no free-quota guarantee).

## 16. What does MLflow provide?
- Reproducibility/provenance: run captures parameters, metrics, git commit, and
  the eval artifact. "Which model is live, with which metrics, from which commit?"
  — answerable via MLflow + `/info` + `model_metadata.json`.

## 17. Why Docker?
- Contains training/inference deps in one rebuildable image → local, CI, and
  production share the same build.

## 18. Why Koyeb / Neon / Pages free-tier stack?
- Koyeb runs the Docker backend free tier; Neon gives managed Postgres in prod
  (local stays SQLite); Pages serves the static dashboard in the same Cloudflare
  account. Caveat: free tiers sleep -> cold start; quotas are not "always free".
  Say: "$0/mo while usage stays inside quotas."
  emotion recognition" would over-claim. We position it as **emotion-aware
  intervention**, not mind-reading.

## 8. Why temporal gating before the LLM?
- Single-frame labels are noisy; calling the LLM every frame = cost + latency +
  spam. Gating waits for **3 consecutive** support-needed frames → fewer false
  triggers, fewer wasted LLM calls.

## 9. Why 3 frames?
- Smallest practical window to filter single-frame misclassification; the
  cooldown handles re-triggering. Tunable via `TUTOR_STREAK_THRESHOLD`.
## 10. Why 45-second cooldown?
- Prevents hammering the LLM when the same negative expression persists.
  `TUTOR_COOLDOWN_SECONDS=45` per device is a reliability/UX trade-off.

## 11. Why Cloudflare Workers AI?
- No local LLM in production (portable, GPU-free); free tier suits a simple
  demo; simple REST auth. Model choice **verified against official docs**:
  `@cf/meta/llama-3.2-3b-instruct`.

## 12. Why `llama-3.2-3b-instruct`, not `llama3.1:8b`?
- `llama-3.1-8b-instruct` became **deprecated on Workers AI (5/30/2026)**; we
  verified against the official model catalog and switched to the current stable
  3.2 3B instruct. A real "engineering under external API change" decision.

## 13. Why NO local LLM?
- Keeps deployment **portable / GPU-free**, removes a large runtime dependency,
  and isolates it from the CV path. Short coaching messages don't need a local
  model; cloud inference + fallback gives the same UX with a simpler runtime.

## 14. Why fallback? / What happens when Cloudflare fails?
- Isolated provider with explicit timeout. On HTTP error, timeout, empty, or
  malformed response → return a rule-based canned message with `source="fallback"`.
- Emotion prediction NEVER depends on LLM: a failing tutor call cannot break
  `/predict` / WS camera path. Verified in tests (401→fallback, path stays healthy).

## 15. Why mock tests AND real integration tests?
- Deterministic unit suite (55) mocks/stubs the model + LLM → credential-free,
  no GPU; used by CI. Real integration suite (7 Docker E2E) exercises the real
  container stack. Real Cloudflare matrix (8/8 vi+en) is manual / opt-in.

## 16. What does MLflow provide?
- Reproducibility/provenance: run captures parameters, metrics, git commit, and
  the eval artifact. "Which model is live, with which metrics, from which commit?"
  — answerable via MLflow + `/info` + `model_metadata.json`.

## 17. Why Docker?
- Contains all deps (Torch, CV, FastAPI, PG driver) in one rebuildable image →
  local, CI, and production share the same build.

## 18. Why the Koyeb / Neon / Pages free-tier stack?
- Koyeb runs the Docker backend free tier; Neon gives managed Postgres in prod
  (local stays SQLite); Pages serves the static dashboard in the same Cloudflare
  account. Caveat: free tiers sleep/suspend → cold start; not "always free".

## 19. What are free-tier limitations?
- Koyeb: cold start after idle; 512 MB container (baseline ≈ 368 MiB).
- Neon: compute auto-suspend (~5 min idle) → first query slower.
- Workers AI: free inference-token quota (throttled at scale).
- Pages: free bandwidth quota.
- Say: "$0/mo while usage remains within current quotas — not always free."

## 20. How would you improve the model with more data/compute?
- Class-balanced resampling/augmentation for Fear+Disgust; stronger
  regularization; regression-watch macro-F1; possible multi-task head
  (expression + valence/arousal) to disambiguate confusable pairs.

---

## Behavioral / product framing
- "Tell me about a failure" → Fear recall ≈ 0.046: measured a biased model, did
  not hide it, built gating+fallback so the system still works, documented the
  retraining path.
- "Why AI × learning?" → honest working example where ML + GenAI + reliability
  converge around a learner.
- 20-second pitch → "real-time emotion recognition (ResNet-50, honestly measured)
  plus a cloud AI tutor that intervenes only on sustained negative states,
  degrades safely, is MLflow-tracked and Docker-reproducible."

## Files to re-read before interview
- `VALIDATION_REPORT.md`, `CLOUD_LLM_VALIDATION_REPORT.md`,
  `FINAL_REPOSITORY_AUDIT.md`, `PORTFOLIO_SUMMARY.md`,
  `DEPLOYMENT_RUNBOOK.md`.