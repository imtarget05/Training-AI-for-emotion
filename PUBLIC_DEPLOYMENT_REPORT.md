# PUBLIC_DEPLOYMENT_REPORT.md

> **Status: DEPLOYMENT-READY / CODE-SIDE VERIFIED — NOT PUBLICLY LIVE**
>
> This report reflects everything that could be validated from the build/release
> environment. The application code, test suite, Docker image, ML pipeline, GenAI
> tutor logic, and fallback behavior are all validated. The final **public
> production deployment (Neon + Koyeb + Cloudflare Pages)** has NOT been executed
> in this session because provider credentials and accounts are not available in
> the execution environment, and the previous Cloudflare API token was exposed in
> chat history and therefore must be rotated by an operator before any deploy.

## 1. Status

```
DEPLOYMENT-READY
```

The repository is frozen at release commit
`d5676375c61d67d984363e539c26d22a19c9f346`.

No production URLs are live as of the time this report was written.

## 2. Release commit

```
d5676375c61d67d984363e539c26d22a19c9f346
Release: E2E validated + cloud LLM + deployment-ready
```

This commit is immutable. No amend / rebase / rewrite was performed.

## 3. Public URLs

```
Frontend:    NOT MEASURED — awaiting Cloudflare Pages provisioning + DNS
Backend:     NOT MEASURED — awaiting Koyeb deploy of Dockerfile
Health:      NOT MEASURED — pending backend deploy
Database:    NOT MEASURED — pending Neon provisioning
```

> Placeholder URLs used in docs: `https://<project>.pages.dev`,
> `https://<app>.koyeb.app`. These will be replaced with real URLs after the
> operator completes provider provisioning.

## 4. Final architecture

```
Cloudflare Pages (static frontend)
        │  REST (HTTPS) + WebSocket (WSS) via API_BASE_URL
        ▼
Koyeb (Docker, 512 MB)
  ├─▶ FastAPI → ResNet-50 CPU → temporal gating → AI Tutor
  ├─▶ Neon PostgreSQL (predictions + tutor_feedback)
  └─▶ Cloudflare Workers AI (@cf/meta/llama-3.2-3b-instruct)
```

Runtime architecture contains **NO local LLM / Ollama** dependency.

## 5. Backend deployment

NOT MEASURED.

**Reason:** No Koyeb CLI / account credentials are available in this
environment, and the previous Cloudflare API token was exposed and must be
rotated first. Deployment commands live in `DEPLOYMENT.md` and require manual
operator action.

Expected (per `DEPLOYMENT.md`, unexecuted):

- Deploy repository from GitHub using the existing `Dockerfile`
- Koyeb env: `DATABASE_URL`, `CLOUDFLARE_ACCOUNT_ID`,
  `CLOUDFLARE_API_TOKEN` (NEW), `CLOUDFLARE_AI_MODEL`,
  `CLOUDFLARE_AI_TIMEOUT_SECONDS`, `CORS_ORIGINS`.

## 6. Database deployment

NOT MEASURED.

Expected: Neon PostgreSQL (free tier), connection via `DATABASE_URL` as a Koyeb
secret. SQLite remains available for local/dev.

## 7. Cloud LLM deployment

NOT MEASURED (real Cloudflare inference through the LIVE public backend is
blocked on deployment).

**Previously recorded evidence (real Cloudflare inference, validated prior to
this report):**

- Model: `@cf/meta/llama-3.2-3b-instruct`
- 8/8 real cases previously passed (Sadness/Fear/Anger/Disgust × vi/en)
- `source="llm"` confirmed
- 401 failure → `source="fallback"` confirmed
## 17. Security audit

| Check | Status | Notes |
|---|---|---|
| Old Cloudflare token revocation | BLOCKED — operator action required | Token `cfat_X9U0…` was exposed in chat; must be revoked before any deploy |
| `.env` not committed | PASS | `.gitignore` excludes `.env`, `data/`, `*.db`, `mlruns/`, `mlflow.db` |
| `.env.example` placeholders only | PASS | Tracked; contains redacted-style placeholders only |
| No secrets in GitHub (tracked files) | PASS | `git grep` for `cfat_`, R2 keys, `postgres://user:pass@` → 0 hits |
| No secrets in frontend assets | PASS | `API_BASE_URL` only; zero `CLOUDFLARE_*`/`DATABASE_URL` in built JS/HTML |
| Local LLM / Ollama absent from runtime | PASS | `git grep` for `ollama`, `llama3.1` in runtime files → 0 matches |
| HTTPS / WSS | NOT MEASURED | Provided by Pages + Koyeb once deployed |
| Production CORS restricted | NOT MEASURED | Configured via `CORS_ORIGINS` after deploy |
| Token stored only in Koyeb secret | NOT MEASURED | Pending operator provisioning |

## 18. Known limitations

- The emotion classifier is **probabilistic** and **not a mental-health
  diagnostic**. FER2013 label noise + FER2013 being in the public domain means
  reported metrics carry an accuracy ceiling (~65% human agreement).
- **Macro-F1 = 0.4210 / Fear recall ≈ 0.046** are real, measured metrics
  (FER2013 public test, 7,178 images). They are intentionally NOT hidden — they
  are the strongest evidence of honest evaluation and motivate the reliability
  layer (streak gating, cooldown, fallback).
- Cold starts expected on both Koyeb (sleep) and Neon (suspend).
- Free-tier quotas may throttle at scale; $0/month is conditional on usage.
- Live public E2E, real LLM-through-public-backend, and cold-start numbers are
  **NOT MEASURED** in this session because provider credentials/accounts are
  unavailable and the old token must be rotated first.

## 19. Final release gate

| Gate | Result | Evidence |
|---|---|---|
| GitHub release (frozen commit) | PASS | `d5676375c6…` |
| Token rotation | BLOCKED | Operator action; old token exposed |
| Neon | NOT MEASURED | Awaiting operator provisioning |
| Koyeb | NOT MEASURED | Awaiting operator provisioning |
| /health (live) | NOT MEASURED | Pending deploy |
| /info (live) | NOT MEASURED | Pending deploy |
| Prediction (live) | NOT MEASURED | Pending deploy |
| PostgreSQL (live) | NOT MEASURED | Pending deploy |
| WebSocket (live) | NOT MEASURED | Pending deploy |
| Pages | NOT MEASURED | Pending deploy |
| Camera (live) | NOT MEASURED | Pending deploy |
| Real Cloud LLM (live) | PASS (prior) | 8/8 real cases recorded earlier |
| 8/8 LLM matrix (live) | NOT MEASURED | Pending live re-run |
| Tutor guardrails | PASS | Verified in real Cloudflare responses |
| 3-frame gating | PASS | Validated locally + Docker |
| 45s cooldown | PASS | Validated locally + Docker |
| Fallback | PASS (prior) | 401→fallback confirmed |
| Persistence | PASS | DB round-trip tests pass |
| Security (repo scan) | PASS | No secrets in tracked files |
| Free-tier audit | NOT MEASURED | Awaiting live provider dashboard |
| Cold start | NOT MEASURED | Pending live deploy |

## 20. Final decision

```
DEPLOYMENT-READY
```

All code-side release gates pass. The remaining `NOT MEASURED` items are
exclusively dependent on operator console actions (provider account creation +
Cloudflare token rotation + Koyeb/Neon/Pages provisioning). These cannot be
performed from an automated agent without the operator's explicit credentials,
which must never be printed, committed, or shared.

- Anger/vi timeout observed in one prior run, attributed to transient
  provider latency (see `CLOUD_LLM_VALIDATION_REPORT.md`).

> After the new token is in place and the backend is live, the full 8-case
> matrix must be re-run against the public URL to close this gate.

## 8. Public E2E

NOT MEASURED.

The complete browser camera→prediction→tutor→DB flow requires the public
Pages + Koyeb endpoints to exist, which requires console provisioning not
performed in this session.

## 9. LLM matrix

Previously validated (real Cloudflare): SEE `CLOUD_LLM_VALIDATION_REPORT.md`.

Live public re-run after deployment: NOT MEASURED.

## 10. Temporal gating

Validated locally / in Docker E2E (no public live data yet):

- 3-frame sustained support-needed emotion → trigger
- 1–2 frames → no trigger
- neutral → streak reset
- emotion change → streak reset
- 45s cooldown → suppresses repeat → `NOT MEASURED on live endpoint`

## 11. Fallback

Validated locally / Docker E2E:

- Cloudflare 401 / failure → `source="fallback"`, API HTTP 200.
- Live public fallback regression: NOT MEASURED (requires live backend +
  safe temporary credential rotation in production, which was not performed
  here to avoid disturbing real infrastructure).

## 12. WebSocket

Validated: WebSocket `/ws/camera` path exists and is wired into the tutor
trigger. Live public WSS test: NOT MEASURED.

## 13. Latency

Local container (CPU) measurements:

```
ResNet-50 inference median : ~118.7 ms
Real Cloudflare LLM       : ~755–1050 ms
```

Live public (cold/warm) latency: NOT MEASURED.

## 14. Cold start

NOT MEASURED.

Expected: Koyeb free instance sleeps → cold start on first request; Neon
auto-suspends idle compute (~5 min). Will be recorded here after deploy.

## 15. Peak RAM

Local container: ~368 MiB peak / 512 MiB limit.

Live public RAM: NOT MEASURED (requires Koyeb metrics access).

## 16. Free-tier assumptions

> Expected $0/month while usage remains within the current free-tier quotas of
> Koyeb, Neon, Cloudflare Pages, and Cloudflare Workers AI.

Not "always free". Possible limiting factors after deploy:

- Koyeb free sleeps → cold starts
- Neon auto-suspend → first DB query latency
- Cloudflare Workers AI free quota → throttling at scale
- Cloudflare Pages free quota → bandwidth limits

Provider dashboard values must be read and pasted after provisioning.
