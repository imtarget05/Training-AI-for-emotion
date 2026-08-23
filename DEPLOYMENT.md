# DEPLOYMENT — Free-Tier Production Runbook

Architecture selected (see `PROVIDER_COMPATIBILITY.md`, `COST_MODEL.md`):

```
Browser ──► Cloudflare Pages (static frontend)
                 │  API_BASE_URL
                 ▼
            Koyeb free instance (Docker backend)
                 ├─ FastAPI / WebSocket
                 ├─ ResNet-50 CPU inference
                 ├─ Cloudflare Workers AI (@cf/meta/llama-3.2-3b-instruct)  [token backend-only]
                 └─ Neon Postgres free (DATABASE_URL)
```

Local dev keeps SQLite; production uses Postgres via `DATABASE_URL`.

## Prerequisites
GitHub repo · Koyeb account (no card) · Cloudflare account · Neon account (no card).

## Environment variables (backend — set in Koyeb dashboard)

```
CLOUDFLARE_ACCOUNT_ID=<account id>
CLOUDFLARE_API_TOKEN=<workers-ai token>     # NEVER in frontend/git
CLOUDFLARE_AI_MODEL=@cf/meta/llama-3.2-3b-instruct
CLOUDFLARE_AI_TIMEOUT_SECONDS=8
DATABASE_URL=postgresql://<user>:<pass>@<host>/<db>?sslmode=require
PORT=8000
CORS_ORIGINS=https://<your-frontend>.pages.dev
```

`.env.example` contains placeholders only.

## Database setup (Neon)
1. Create project → copy pooled connection string.
2. Schema is created automatically on startup (`init_db()`); for Postgres,
   `DB_PATH` is ignored when `DATABASE_URL` is set (see `database.py`).
3. Verify: `curl $BACKEND/health` then `POST /tutor/feedback` once and check
   `GET /tutor/history`.

## Backend deploy (Koyeb)
1. Create Service → Deploy from GitHub → Dockerfile detected.
2. Free instance (≈0.1 vCPU / 512 MB — measured peak RAM 368 MiB ✓).
3. Health check path: `/health` (returns `{"status":"ok"}`, no model load).
4. Port: expose `PORT` env value.
5. First deploy builds a 1.85 GB image (~5–10 min). Startup ≈3 s; first
   predict ≈1.3 s, warm ≈0.3–0.6 s.

## Frontend deploy (Cloudflare Pages)
1. Pages → Connect to Git → build command: none; output dir: `static`.
2. Set production env `API_BASE_URL=https://<koyeb-app>.koyeb.app`
   (the frontend reads this at runtime for REST + `wss://` WebSocket).
3. Backend CORS already allows configurable origins — set allowed origin to
   the Pages URL instead of `*` before going public.

## Security rules
- `CLOUDFLARE_API_TOKEN` exists ONLY as a Koyeb env var.
- Browser receives only the backend URL — never provider tokens.
- No secrets in git; `.env` git-ignored; CI uses GitHub Secrets if wired.

## Cold starts
Koyeb free scales to zero. Expected: first request after idle takes a few
seconds (instance start) + 3 s app boot + 1.3 s model warm-up. Documented,
not hidden. Keep-alive pings can be added later if annoying.

## Troubleshooting
| Symptom | Cause | Fix |
|---|---|---|
| `/health` 502 during deploy | image still building | wait for build |
| tutor always `source=fallback` | missing CF env vars on instance | re-check dashboard |
| DB errors at startup | bad `DATABASE_URL` / SSL | append `?sslmode=require` |
| camera blocked | browser requires HTTPS for getUserMedia | use the HTTPS Pages URL |

## Rollback
Backend: Koyeb → previous deployment (one click) or redeploy previous commit.
Frontend: Pages → instant rollback to previous build. DB: additive schema
only (`CREATE TABLE IF NOT EXISTS`) — old images stay compatible.

## Cost
See `COST_MODEL.md`: expected **$0/month within current free quotas**;
no card on file anywhere → no surprise-bill vector.
