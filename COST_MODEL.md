# COST MODEL — Free-Tier Deployment

Assumptions: portfolio/demo traffic only. All figures must be re-verified
against official pricing pages before each deployment — free tiers change.

## Usage profiles

| Profile | Visitors/day | Frames/predicts | Tutor triggers/day |
|---|---|---|---|
| Quiet | 10 | ~500 | ~5 |
| Normal demo | 50 | ~2,500 | ~25 |
| Busy day (e.g. after sharing) | 100 | ~5,000 | ~50 |

## Per-layer cost

### Frontend — Cloudflare Pages
Free: unlimited static requests, unlimited bandwidth on free tier.
Cost at all profiles: **$0**.

### Backend — Render free Web Service (current)
The backend runs on Render (free Web Service, Docker runtime, region
Singapore) via the `render.yaml` blueprint. Free-tier instances sleep after
15 min idle → cold start ~30–60 s (model load). Koyeb was the previous
provider and has been replaced.
Cost at all profiles: **$0 within current free-instance quota**.
⚠️ If usage kept the instance awake continuously 24/7 it could exceed the
free quota → verify dashboard monthly.

### Database — Neon Postgres free
Free: 0.5 GB storage. Our rows are tiny (<100 bytes); even 100k predictions ≈ tens of MB.
Compute: serverless compute-hours allowance far exceeds demo load.
Cost at all profiles: **$0**.

### LLM — Cloudflare Workers AI free
Free allocation: daily Neuron allowance. @cf/meta/llama-3.2-3b-instruct pricing
≈ $0.05/M input + $0.45/M output tokens beyond free tier.
Per tutor call: ~450 in / ~40 out tokens → ~0.00002¢ paid-equivalent.
Even 1,000 triggers/month ≪ free tier.
Cost at all profiles: **$0**.

## Monthly estimate

| Profile | Frontend | Backend | DB | LLM | Total |
|---|---|---|---|---|---|
| 10 visitors/day | $0 | $0 | $0 | $0 | **$0** |
| 50 visitors/day | $0 | $0 | $0 | $0 | **$0** |
| 100 visitors/day | $0 | $0 (verify hours) | $0 | $0 | **$0** |

Expected **$0/month while usage remains within each provider's current
free-tier limits**. This is NOT "always free": quotas, sleep policies and
terms can change at any time.

## Payment-method requirements

| Provider | Card required? |
|---|---|
| Cloudflare Pages / Workers AI | No (Workers AI needs no card on free) |
| Render | No for free web services |
| Neon | No |

## Worst-case boundary

The first layer to break at scale is backend instance-hours (Render free),
then
Neon compute-hours. LLM cost stays negligible until ~10⁵ tutor calls/month.
No auto-scaling is configured anywhere → **no surprise bill vector exists**
(no provider has a card on file).
