# PROVIDER COMPATIBILITY MATRIX

Measured app requirements (Phase 2, Docker, this machine):

| Requirement | Measured value |
|---|---|
| Runtime | Python 3.11-slim container |
| PyTorch | 2.5.1 (CPU-only wheels) |
| Docker image size | **1.85 GB** |
| Model weights | 90 MB (`final_model.pth`) |
| Idle RAM | ~219 MiB |
| Peak RAM under inference (512 MB cgroup limit) | **368 MiB — no OOM** |
| Startup to first response | **3 s** (lazy model load) |
| First predict latency | ~1.3 s; warm 0.3–0.6 s |
| WebSocket required | Yes (`/ws/camera`) |
| Persistent storage | SQLite file (to be replaced by managed Postgres) |
| Outbound LLM call | Cloudflare Workers AI REST (token server-side only) |

## Matrix

| Requirement | Render free | Koyeb free | Fly.io | Railway trial | Cloudflare Workers/Pages | Neon free | Supabase free |
|---|---|---|---|---|---|---|---|
| Docker deploy | PASS | PASS | PARTIAL¹ | PASS | FAIL (no Docker) | n/a | n/a |
| FastAPI/uvicorn | PASS | PASS | PASS | PASS | PARTIAL² | n/a | n/a |
| 512 MB RAM enough | PARTIAL³ | PASS (up to 2 vCPU instances vary) | PARTIAL⁴ | PASS | FAIL⁵ | n/a | n/a |
| WebSocket | PARTIAL⁶ | PASS | PASS | PASS | PASS | n/a | n/a |
| Persistent DB via env URL | n/a | n/a | n/a | n/a | D1 only (SQLite dialect) | PASS | PASS |
| Always-on without card | FAIL⁷ | PASS | FAIL⁸ | FAIL | PASS | PASS | PASS |
| Cold start acceptable for demo | PARTIAL (~50 s spin-up) | PARTIAL (scale-to-zero) | none (kept-alive VM, may stop) | n/a | none | n/a | n/a |
| Free monthly cost at demo usage | $0 within limits | $0 within limits | $0 only while allowance lasts | $5 one-time credit only | $0 | $0 (0.5 GB) | $0 (0.5 GB) |

Notes:
1. Fly.io uses `flyctl` deploys of Docker images — works, but requires a credit card on file and machines auto-stop after inactivity.
2. Cloudflare Workers cannot host a full Python/PyTorch process; only the static frontend belongs there.
3. Render free web service = 512 MB RAM. Our measured peak is 368 MiB → fits, but headroom is thin under multi-face frames; acceptable for portfolio traffic.
4. Fly free allowances changed repeatedly; machines bill against a pay-as-you-go account even with small usage.
5. Workers memory limit (128 MB) and Python-in-Workers constraints make ResNet inference impractical.
6. Render free tier supports WebSockets but the service sleeps after ~15 min idle and free instances have limited bandwidth (100 GB/mo historically — verify current).
7. Render free web services require verifying ownership; card not strictly required for free web services but required for most paid/persistent options; spins down when idle.
8. Fly.io requires adding a payment method before deploying.

## Decision

**Backend: Koyeb free instance** — native Docker deploy, no credit card, WebSockets supported, scale-to-zero cold start acceptable for a portfolio demo, RAM sufficient (measured peak 368 MiB).
**Runner-up: Render free** — simplest UX, but 512 MB limit leaves little headroom plus long spin-ups; kept as documented fallback.
**Frontend: Cloudflare Pages** (static `static/`, zero cost, global HTTPS).
**Database: Neon Postgres free tier** (0.5 GB >> our row sizes; serverless driver over HTTPS works from any host) with SQLite retained for local dev.
**LLM: Cloudflare Workers AI** (already integrated; free daily Neuron allocation far exceeds tutor volume).

Why not the alternatives: Cloudflare Workers cannot run PyTorch; Railway's trial expires; Fly.io needs a card; Supabase adds auth/storage overhead we don't need vs plain Neon Postgres.

*Free tiers change frequently — every figure above must be re-checked against official pricing pages at deploy time.*
