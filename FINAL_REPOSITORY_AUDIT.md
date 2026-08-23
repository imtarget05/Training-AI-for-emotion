# Final Repository Audit — Release Candidate (2026-08-23)

## 1. Architecture

```text
Browser/Camera ──WebSocket /ws/camera──┐                    ┌─ MLflow (eval tracking)
                ──POST /predict────────▼                    │
                                        FastAPI (main.py)   ├─ GitHub Actions CI
                                        │        │          │
                            ResNet-50 inference   │         └─ Docker / compose
                            (model.py + Haar)     │
                                        │        │
                                  Emotion     Temporal gating
                                              (tutor_trigger.py:
                                               3-frame streak,
                                               45s cooldown,
                                               per-device)
                                                 │
                                          AI Tutor (tutor.py)
                                            ├─ prompt engineering (vi/en)
                                            ├─ Cloudflare Workers AI
                                            │    @cf/meta/llama-3.2-3b-instruct
                                            └─ rule-based fallback
                                                 │
                                    SQLite (database.py) + Frontend toast
```

## 2–6. Pipelines (traced in source)

| Pipeline | Chain | Files |
|---|---|---|
| Components | model / trigger / tutor / db / api / ui | `model.py` `tutor_trigger.py` `tutor.py` `database.py` `main.py` `static/index.html` |
| ML | FER2013 → ImageNet norm 224×224 → ResNet-50 → argmax→CLASS_NAMES | `model.py`, `evaluate.py`, checkpoint `final_model.pth` |
| GenAI | emotion → streak/cooldown → prompt (few-shot, guardrails) → Cloudflare REST → fallback | `tutor.py`, `tutor_trigger.py` |
| MLOps | eval → accuracy/macro-F1/confusion PNG → MLflow run (params/metrics/git commit/artifact) → `model_metadata.json` → `/info` | `evaluate.py`, `model_metadata.json`, `.github/workflows` |

## 7. Testing
- 55 unit/integration passed, 7 skipped (E2E opt-in) — deterministic, credential-free
- Docker E2E: 7/7 passed (`tests/test_e2e.py -m e2e`)
- Demo scenario script: `scripts/demo_e2e.py` (5/5)

## 8. Deployment
Docker image builds and runs standalone; LLM is cloud-side (Workers AI) — **no local LLM, no GPU, no Ollama required**.

## 9. Known limitations (documented honestly)
- Accuracy 49.83% / Macro-F1 0.4210 on FER2013 test (7,178 imgs); **Fear recall ≈ 0.046**
- Facial expression ≠ guaranteed internal emotional state
- Tutor = emotion-aware intervention, not personalization (no learner/course context)
- SQLite is single-node persistence only
- Real-Cloudflare matrix validated 2026-08-23 (see CLOUD_LLM_VALIDATION_REPORT.md); CI uses deterministic mocks

## 10. Remaining defects
None open at release. Audit fixes applied this pass:

| Problem | Root cause | Fix | Verification |
|---|---|---|---|
| Compiled bytecode tracked in git | `__pycache__/model.cpython-314.pyc` added before ignore rule existed | `git rm -r --cached __pycache__`; confirmed `__pycache__/` in `.gitignore` | `git status` clean of pycache |
| Stray runtime/test artifacts on disk | leftover `test_runtime.db`, `.pytest_cache` | deleted; both gitignored | not in `git status` |

Notes: `final_model.pth` (94 MB) intentionally tracked — core inference artifact, under GitHub's 100 MB limit; documented here rather than silently ignored.
