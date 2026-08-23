# LIVE_DEPLOYMENT.md

> Status: **DEPLOYMENT-READY / CODE-SIDE VERIFIED — NOT YET PUBLIC LIVE**
> Fill in the URLs below after completing the provider setup steps in `DEPLOYMENT.md`.

## Public URLs (fill after deployment)

| Component | Provider | URL |
|---|---|---|
| Frontend | Cloudflare Pages | `https://<project>.pages.dev` |
| Backend | Koyeb Free | `https://<app>.koyeb.app` |
| Health | Koyeb | `https://<app>.koyeb.app/health` |
| Model info | Koyeb | `https://<app>.koyeb.app/info` |
| Database | Neon Postgres Free | backend-only (`DATABASE_URL`, never public) |
| LLM | Cloudflare Workers AI `@cf/meta/llama-3.2-3b-instruct` | backend-only |

## Architecture
```
Cloudflare Pages (static frontend)
        │  REST + wss via API_BASE_URL
        ▼
Koyeb (Docker, 512 MB) → FastAPI → ResNet-50 CPU → temporal gating
        ├─▶ Neon PostgreSQL (predictions + tutor_feedback)
        └─▶ Cloudflare Workers AI (tutor LLM)
```

## Free-tier assumptions
- Expected $0/month while usage stays within current Koyeb / Neon / Pages /
  Workers AI free quotas. NOT "always free".
- Koyeb free instance sleeps → cold start expected on first request.
- Neon auto-suspends idle compute (~5 min); first DB query after sleep is slower.

## Cold-start behavior
NOT YET MEASURED — measure after first public deploy and record here.

## Known limitations
- Cold start on both backend and database sleep.
- Fear-class recall ≈ 0.046 in the emotion model (documented in ML evaluation).
- Camera requires HTTPS origin (Pages provides this).

## Secrets policy
`CLOUDFLARE_API_TOKEN` and `DATABASE_URL` exist ONLY in Koyeb environment
config. Never in Git, never in frontend assets.
