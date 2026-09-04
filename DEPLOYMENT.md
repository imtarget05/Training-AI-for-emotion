# DEPLOYMENT — Free-Tier Production Runbook

Architecture selected (see `COST_MODEL.md`, `DEPLOYMENT_RUNBOOK.md`):

```
Browser ──► Cloudflare Pages (static frontend, optional)
                 │  API_BASE_URL / ?api=
                 ▼
            Render free Web Service (Docker backend)  ← LIVE
                 ├─ FastAPI / WebSocket
                 ├─ ResNet-50 CPU inference
                 ├─ Cloudflare Workers AI (@cf/meta/llama-3.1-8b-instruct)  [token backend-only]
                 └─ Neon Postgres free (DATABASE_URL)                        ← LIVE
```

Live backend: `https://emotion-tutor-api.onrender.com`. The backend also serves
the dashboard at `/dashboard` (same-origin, no Pages needed).

Local dev keeps SQLite; production uses Postgres via `DATABASE_URL`.

## Prerequisites
GitHub repo · Render account (no card) · Cloudflare account · Neon account (no card).

## Environment variables (backend — set in Render dashboard, or `render.yaml` with `sync: false`)

```
CLOUDFLARE_ACCOUNT_ID=<account id>
CLOUDFLARE_API_TOKEN=<workers-ai token>     # NEVER in frontend/git
CLOUDFLARE_AI_MODEL=@cf/meta/llama-3.1-8b-instruct
CLOUDFLARE_AI_TIMEOUT_SECONDS=10
DATABASE_URL=postgresql://<user>:<pass>@<host>/<db>?sslmode=require
PORT=8080
CORS_ORIGINS=https://<your-frontend>.pages.dev
```

`.env.example` contains placeholders only.

## Database setup (Neon)
1. Create project → copy pooled connection string.
2. Schema is created automatically on startup (`init_db()`); for Postgres,
   `DB_PATH` is ignored when `DATABASE_URL` is set (see `database.py`).
3. Verify: `curl $BACKEND/health` then `POST /tutor/feedback` once and check
   `GET /tutor/history`.

## Backend deploy (Render)
1. Render Dashboard → New → Blueprint → connect GitHub repo (`render.yaml`).
2. Free instance (512 MB RAM — Docker runtime, region Singapore).
3. Health check path: `/health` (returns `{"status":"ok"}`, no model load).
4. Port: container reads `PORT` (default 8080).
5. Every push to `main` auto-deploys. First build builds a ~1.9 GB image
   (~5–10 min). Free tier sleeps after 15 min idle → first request after
   idle takes ~30–60 s (instance start + model load).

## Frontend deploy (Cloudflare Pages)
1. Pages → Connect to Git → build command: none; output dir: `static`.
2. Set production env `API_BASE_URL=https://emotion-tutor-api.onrender.com`
   (or open the Pages URL with `?api=https://emotion-tutor-api.onrender.com` —
   the frontend reads this at runtime for REST + `wss://` WebSocket).
3. Backend CORS already allows configurable origins — set allowed origin to
   the Pages URL instead of `*` before going public.

## Security rules
- `CLOUDFLARE_API_TOKEN` exists ONLY as a Render env var.
- Browser receives only the backend URL — never provider tokens.
- No secrets in git; `.env` git-ignored; CI uses GitHub Secrets if wired.

## Cold starts
Render free sleeps after 15 min idle. Expected: first request after idle takes
~30–60 s (instance start + 3 s app boot + model warm-up). Documented,
not hidden. Keep-alive pings can be added later if annoying.

## Troubleshooting
| Symptom | Cause | Fix |
|---|---|---|
| `/health` 502 during deploy | image still building | wait for build |
| tutor always `source=fallback` | missing CF env vars on instance | re-check dashboard |
| DB errors at startup | bad `DATABASE_URL` / SSL | append `?sslmode=require` |
| camera blocked | browser requires HTTPS for getUserMedia | use the HTTPS Pages URL |

## Rollback
Backend: Render → Suspend/Resume or redeploy previous commit.
Frontend: Pages → instant rollback to previous build. DB: additive schema
only (`CREATE TABLE IF NOT EXISTS`) — old images stay compatible.

## Cost
See `COST_MODEL.md`: expected **$0/month within current free quotas**;
no card on file anywhere → no surprise-bill vector.
