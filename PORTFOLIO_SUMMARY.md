# Portfolio Summary — Training-AI-for-emotion

## One-line description
Emotion-aware AI learning assistant: real-time facial emotion recognition (ResNet-50) with a prompt-engineered AI tutor (Cloudflare Workers AI) that intervenes only on sustained negative emotion states.

## Resume bullets
- Built an end-to-end emotion-aware AI tutoring system combining a PyTorch ResNet-50 classifier, real-time FastAPI/WebSocket inference, and a prompt-engineered tutor backed by Cloudflare Workers AI (@cf/meta/llama-3.2-3b-instruct), with rule-based fallback when the cloud LLM fails.
- Evaluated the classifier on 7,178 FER2013 public-test images (49.83% accuracy, 0.421 macro-F1) with per-class metrics and confusion-matrix analysis that identified severe Fear-class weakness (recall 0.046) — reported honestly rather than optimized away.
- Implemented temporal gating (3-frame sustained-emotion streak, 45-second per-device cooldown), bilingual prompt guardrails, MLflow experiment tracking with git-commit provenance, model versioning via a metadata endpoint, and Docker/GitHub Actions CI running 55 credential-free tests plus 7 live E2E tests.

## Technical highlights
- Temporal decision layer between CV output and LLM calls — the LLM is never called per frame
- Provider-isolated GenAI layer: prompt/business logic separated from HTTP infrastructure
- Defence-in-depth fallback: HTTP error, timeout, empty, and malformed responses all degrade safely to canned messages without breaking prediction
- Evaluation gate philosophy: metrics are produced by a reproducible script and blocked (never fabricated) when data is absent

## Key metrics
| Metric | Value |
|---|---|
| Test accuracy (FER2013 test, n=7178) | 49.83% |
| Macro-F1 | 0.4210 |
| Strongest class | Happiness F1=0.725 |
| Weakest class | Fear F1=0.081 (recall 0.046) |
| Emotion inference latency (CPU) | median ≈ 119 ms |
| Real cloud LLM latency | 755–1050 ms (8/8 cases) |
| Tests | 55 passed + 7 E2E passed |

## Architecture explanation (30-second version)
Camera frames go through OpenCV face detection into a fine-tuned ResNet-50 that outputs one of 7 emotions. A temporal gate only lets a *sustained* negative emotion (3 consecutive frames, respecting a 45 s cooldown) reach the AI Tutor, which sends a structured bilingual prompt to Cloudflare Workers AI; any LLM failure falls back to canned supportive messages. Everything is persisted, observable via `/info` + MLflow, and reproducible through Docker and CI.

## Biggest engineering challenge
Deciding *when not to call the LLM*: single-frame emotion predictions are noisy, so naive integration would spam the model. Designing streak + cooldown + reset semantics (and testing device isolation) was the core reliability work.

## Biggest ML limitation
Fear recall ≈ 0.046 — the model almost never detects fear. Gating reduces noisy triggers but cannot correct systematic class bias; fixing it requires class-balanced retraining, which was deliberately out of scope for honest evaluation.

## What I learned
- Honest evaluation (macro-F1, per-class recall) tells a far better engineering story than a headline accuracy number.
- LLM reliability is an integration problem: timeouts, empty replies, and auth failures each need explicit handling.
- Reproducibility (MLflow provenance, pinned deps, Docker) is what turns a demo into evidence.

## Interview talking points
1. Why Macro-F1 over accuracy for imbalanced emotion data
2. Why llama-3.1-8b-instruct was rejected (deprecated 5/30/2026) in favor of llama-3.2-3b-instruct — API verification before implementation
3. Streak/cooldown trade-offs: latency vs. noise vs. UX
4. Mock vs real inference separation in the test suite
5. What "emotion-aware intervention" means — and why I don't call it adaptive learning or diagnosis
Deep-dive answer bank: `INTERVIEW_PREPARATION.md`.
Presentation copy (incl. 5-min talk track): `PROJECT_PRESENTATION.md`.
