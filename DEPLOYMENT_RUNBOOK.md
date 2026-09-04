# OPERATOR DEPLOYMENT RUNBOOK — Training-AI-for-emotion

> Status: **PUBLICLY DEPLOYED — VERIFIED**
>
> Live backend: `https://emotion-tutor-api.onrender.com` (Render Web Service,
> free tier, Docker runtime, region Singapore). Database: Neon PostgreSQL.
> AI Tutor: Cloudflare Workers AI (`@cf/meta/llama-3.1-8b-instruct`).
>
> Deployment automation: **GitHub push → Render auto-deploy** (CD). There is no
> CI pipeline (no GitHub Actions workflow) in this repository; run
> `pytest tests/` manually before pushing.

## 0. Non-negotiable rules

- NEVER paste `DATABASE_URL`, `CLOUDFLARE_API_TOKEN`, or account IDs into chat,
  Git, GitHub, docs, frontend assets, or logs. Redact as `<redacted>`.
- Every variable below that is labelled `<secret>` must exist ONLY in the Render
  environment configuration (Render Dashboard → Service → Environment).

---

## 1. Security action (operator, browser) — must complete FIRST

```
[ ] Revoke old Cloudflare API token (cfat_X9U0…)
[ ] Create a new Cloudflare API token (Workers AI Read + Edit)
[ ] Copy new token to a password manager
[ ] Never place the token in frontend assets
```

---

## 2. Neon PostgreSQL

1. Sign in at `https://console.neon.tech`.
2. Create a project → region, PostgreSQL 16, free tier.
3. Grab the **connection string** (psql / general). This is your `DATABASE_URL`.
4. Do not paste it anywhere except the Render environment configuration.
5. Save the Neon project name/id privately.

**Schema note:** the app runs `init_db()` at startup, so the required tables
(`predictions`, `tutor_feedback`, etc.) are created automatically via
`database.py` when `DATABASE_URL` is in scope. No manual DDL required.

---

## 3. Render backend deploy

The service is provisioned via the Render Blueprint (`render.yaml` in the repo
root) or equivalently via the Render API / Dashboard:

1. Sign in at `https://dashboard.render.com`.
2. New → **Blueprint** → connect repo `imtarget05/Training-AI-for-emotion`, branch `main`
   (build method: **Dockerfile**, top-level; the container serves on `${PORT:-8080}`).
3. Health check path: `/health`. Plan: **free**.
4. Set environment variables (secrets via `sync: false` in `render.yaml`):

```
PORT=8080
DATABASE_URL=<secret from step 2>
CLOUDFLARE_ACCOUNT_ID=<secret>
CLOUDFLARE_API_TOKEN=<secret>
CLOUDFLARE_AI_MODEL=@cf/meta/llama-3.1-8b-instruct
CLOUDFLARE_AI_TIMEOUT_SECONDS=10
CLOUDFLARE_MAX_RETRIES=3
TUTOR_STREAK_THRESHOLD=3
TUTOR_COOLDOWN_SECONDS=45
CORS_ORIGINS=https://<your-pages-domain>.pages.dev   # or * for same-origin dashboard
```

Every push to `main` triggers an automatic redeploy (autoDeploy: yes).

---

## 4. HARD GATE — Render health, before anything else

Run (redact host if you prefer):

```bash
curl -s -o /dev/null -w "%{http_code}" https://emotion-tutor-api.onrender.com/health
curl -s https://emotion-tutor-api.onrender.com/info
```

Require:

```
/health -> HTTP 200
/info   -> HTTP 200, body mentions ResNet-50 / 7 classes / model version matching model_metadata.json
```

If either is not HTTP 200:

```
STOP. Do NOT touch Pages.
Report DEPLOYMENT BLOCKED with: endpoint, HTTP status,
<redacted> response body, likely cause.
```

---

## 5. Backend production smoke (after gate)

```bash
BASE=https://emotion-tutor-api.onrender.com
GET /health
GET /info
POST /predict            # multipart file=@sample.jpg, device_id=demo
POST /tutor/feedback     # {"emotion":"Sadness","confidence":0.8,"lang":"vi"}
GET  /tutor/history
WebSocket /ws/camera     # handshake + one frame message
```

Record actual HTTP status + latency. Never invent latency.

---

## 6. Neon production persistence

After smoke tests, query:

```
GET /tutor/history          # expects persisted tutor message
GET /reports/predictions    # expects persisted predictions
```

Verify rows survive a subsequent request. Confirm via Neon console that
production is truly Postgres (not SQLite).

---

## 7. Cloudflare Pages (frontend)

1. Sign in `https://dash.cloudflare.com`.
2. Pages → connect GitHub → repo `imtarget05/Training-AI-for-emotion`, build dir `static/`.
3. Set env (not secret): `API_BASE_URL=https://emotion-tutor-api.onrender.com`
   (or append `?api=https://emotion-tutor-api.onrender.com` to the Pages URL —
   the dashboard reads the `?api=` query param).
4. Publish → get `https://<project>.pages.dev`.
5. Inspect built assets (JS/HTML): assert ZERO of
   `CLOUDFLARE_API_TOKEN`, `CLOUDFLARE_ACCOUNT_ID`, `DATABASE_URL`.
6. Confirm frontend uses `https://` for REST and `wss://` for WebSocket.

---

## 8. Public browser E2E checklist

```
[ ] Pages URL loads over HTTPS
[ ] camera permission prompt
[ ] camera stream renders
[ ] WebSocket connects (wss)
[ ] face detected
[ ] prediction shown
[ ] sustained negative emotion
[ ] 3-frame temporal gate triggers teacher
[ ] Cloudflare LLM responds -> source="llm"
[ ] toast/banner shows guidance
[ ] Neon persists; /tutor/history shows it
```

If browser/camera cannot be used on this machine, mark camera = NOT MEASURED.

---

## 9. Real Cloudflare LLM matrix (8 cases, no mocks)

POST `/tutor/feedback` with these (only emotion/lang vary):

```
Sadness/vi  Fear/vi  Anger/vi  Disgust/vi
Sadness/en  Fear/en  Anger/en  Disgust/en
```

For each record: HTTP, `source`, latency, output length, language, ≤2
sentences, emotion-presence, no AI self-ref, no diagnosis, no fact invention.

**Anger/vi timeout:** reproduce under timeout=15 before changing anything. If
it times out, record and diagnose provider vs app. Only then adjust `timeout`.

---

## 10. Fallback

Change nothing on app. Temporarily set `CLOUDFLARE_API_TOKEN` to an invalid
value **in Render only**, then call `/tutor/feedback`. Expect:
- `source="fallback"`
- non-empty message
- HTTP 200
- `GET /predict` still 200 (emotion inference unaffected)

RESTORE the real token immediately, then `GET /health` = 200.

---

## 11. Performance

Record actual: cold-start, warm request, prediction ms, LLM ms, E2E ms. If
provider dashboard does not expose them, write `NOT MEASURED`. Local baseline
is NOT production evidence.

---

## 12. Free-tier verification

Read dashboards and record: Render quota/sleep/RAM (free tier sleeps after
15 min idle); Neon plan/compute/suspend;
Pages plan/quota; Workers AI quota/usage.
Statement: **"Expected $0/month while usage remains within current free-tier
quotas."** Never say "always free".

---

## 13. Security final

```
[ ] old token revoked
[ ] new token only in Render
[ ] DATABASE_URL only in Render
[ ] no secrets in GitHub / frontend / docs / logs
[ ] HTTPS / WSS
[ ] restricted CORS
[ ] debug off
[ ] no local LLM / Ollama
```

---

## 14. Final documentation

Update `LIVE_DEPLOYMENT.md` + `PUBLIC_DEPLOYMENT_REPORT.md` with the actual
values/URLs/statuses from the runbook (no secrets). Create a docs-only commit,
push, record the SHA, and declare the freeze.

---

## 15. Redaction-safe reporting (what to come back with)

```
Frontend: https://<project>.pages.dev
Backend:  https://emotion-tutor-api.onrender.com

GET /health -> HTTP 200
GET /info   -> HTTP 200
<redacted body confirming ResNet-50 / 7 classes>

Database:  PASS / FAIL / NOT MEASURED
LLM matrix: X/8
Cold start: <duration or NOT MEASURED>
```

Never include token or connection-string values.
5. Deploy and wait for `Running` / `Healthy`.