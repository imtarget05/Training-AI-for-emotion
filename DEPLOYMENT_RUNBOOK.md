# OPERATOR DEPLOYMENT RUNBOOK — Training-AI-for-emotion

> Status: **DEPLOYMENT-READY / BLOCKED BY OPERATOR ACTION**
>
> This runbook is the exact, step-by-step operator procedure. It is the ONLY
> person-in-the-loop path to `PUBLICLY DEPLOYED — VERIFIED`. The application code
> is frozen at `d5676375c61d67d984363e539c26d22a19c9f346`.
> No feature/refactor work is permitted.

## 0. Non-negotiable rules

- The old Cloudflare token (`cfat_X9U0…`) was leaked in an earlier chat. **Revoke
  it now** at `https://dash.cloudflare.com` → *My Profile* → *API Tokens*, before
  any provider provisioning.
- NEVER paste `DATABASE_URL`, `CLOUDFLARE_API_TOKEN`, or account IDs into chat,
  Git, GitHub, docs, frontend assets, or logs. Redact as `<redacted>`.
- Every variable below that is labelled `<secret>` must exist ONLY in the Koyeb
  environment configuration.

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
4. Do not paste it anywhere except the Koyeb secret field.
5. Save the Neon project name/id privately.

**Schema note:** the app runs `init_db()` at startup, so the required tables
(`predictions`, `tutor_feedback`, etc.) are created automatically via
`database.py` when `DATABASE_URL` is in scope. No manual DDL required.

---

## 3. Koyeb backend deploy

1. Sign in at `https://app.koyeb.com`.
2. Create App → **GitHub** → select repo `imtarget05/Training-AI-for-emotion`, branch `main`.
3. Build method: **Dockerfile** (top-level; the container serves on `${PORT:-8080}`).
4. Set environment variables:

```
PORT=8080
DATABASE_URL=<secret from step 2>
CLOUDFLARE_ACCOUNT_ID=<secret>
CLOUDFLARE_API_TOKEN=<secret - the NEW token>
CLOUDFLARE_AI_MODEL=@cf/meta/llama-3.2-3b-instruct
CLOUDFLARE_AI_TIMEOUT_SECONDS=15
CORS_ORIGINS=https://<your-pages-domain>.pages.dev
```

---

## 4. HARD GATE — Koyeb health, before anything else

Run (redact host if you prefer):

```bash
curl -s -o /dev/null -w "%{http_code}" https://<app>.koyeb.app/health
curl -s https://<app>.koyeb.app/info
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
BASE=https://<app>.koyeb.app
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
3. Set env (not secret): `API_BASE_URL=https://<app>.koyeb.app`.
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
value **in Koyeb only**, then call `/tutor/feedback`. Expect:
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

Read dashboards and record: Koyeb quota/sleep/RAM; Neon plan/compute/suspend;
Pages plan/quota; Workers AI quota/usage.
Statement: **"Expected $0/month while usage remains within current free-tier
quotas."** Never say "always free".

---

## 13. Security final

```
[ ] old token revoked
[ ] new token only in Koyeb
[ ] DATABASE_URL only in Koyeb
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
Backend:  https://<app>.koyeb.app

GET /health -> HTTP 200
GET /info   -> HTTP 200
<redacted body confirming ResNet-50 / 7 classes>

Database:  PASS / FAIL / NOT MEASURED
LLM matrix: X/8
Cold start: <duration or NOT MEASURED>
```

Never include token or connection-string values.
5. Deploy and wait for `Running` / `Healthy`.