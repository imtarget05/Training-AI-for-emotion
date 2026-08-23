#!/usr/bin/env python3
"""
deploy_healthcheck.py — Secrets-safe production pre-flight checker.

Usage (operator machine, after Koyeb + Neon are provisioned):

    export BASE_URL=https://<app>.koyeb.app
    python3 scripts/deploy_healthcheck.py

Checks, WITHOUT printing any secret:
  - env var PRESENCE (bool only) for CLOUDFLARE_ACCOUNT_ID,
    CLOUDFLARE_API_TOKEN, DATABASE_URL
  - GET <BASE>/health -> HTTP 200, non-empty body
  - GET <BASE>/info   -> HTTP 200, mentions ResNet-50 / 7 class signals
  - PTFE exporter? No. This is a reporting/redaction aid.

Real Cloudflare / Postgres credential VALUES are never logged.
Only booleans, HTTP statuses, and redacted response snippets.
"""

import os
import sys

import httpx

BASE_URL = os.environ.get("BASE_URL", "").rstrip("/")
REQUIRED = ("DATABASE_URL", "CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_API_TOKEN")
TIMEOUT = float(os.environ.get("HEALTH_TIMEOUT_SECONDS", "20"))


def redact(text: str, words: tuple = ("token", "key", "secret", "pass")) -> str:
    """Replace the value of any <key><sep><value> pair with <redacted>."""
    lowered = text.lower()
    for word in words:
        # heuristics for JSON-ish key/value pairs
        marker = f'"{word}":'
        if marker in lowered:
            start = text.lower().index(marker) + len(marker)
            # find the quoted value start
            q = text.find('"', start)
            q2 = text.find('"', q + 1) if q != -1 else -1
            if q != -1 and q2 != -1:
                text = text[:q + 1] + "<redacted>" + text[q2:]
    return text[:400]


def main() -> int:
    failures = 0

    print("== 1. Env-var PRESENCE (booleans only; never values) ==")
    for var in ("CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_API_TOKEN",
                "CLOUDFLARE_AI_MODEL", "DATABASE_URL"):
        present = bool(os.environ.get(var))
        print(f"   {var:<24} {'PRESENT' if present else 'MISSING'}")

    print("\n== 2. Backend health ==")
    if not BASE_URL:
        print("   BASE_URL not set -> SKIP (set BASE_URL env).")
        return 2

    try:
        with httpx.Client(timeout=TIMEOUT) as c:
            r_health = c.get(f"{BASE_URL}/health")
            r_info = c.get(f"{BASE_URL}/info")
    except Exception as e:  # connection / timeout / dns
        print(f"   BLOCKED: {type(e).__name__}: {str(e)[:300]}")
        return 1

    print(f"   /health status={r_health.status_code} "
          f"body={redact(r_health.text)}")
    print(f"   /info   status={r_info.status_code} "
          f"body={redact(r_info.text)}")
    if r_health.status_code != 200:
        print("   HEALTH GATE::FAIL")
        failures = 1
    else:
        print("   HEALTH GATE::PASS")

    info_ok = r_info.status_code == 200 and "resnet" in r_info.text.lower()
    print(f"   INFO   GATE::{'PASS' if info_ok else 'FAIL'} "
          f"(expects ResNet-50 mention)")
    if not info_ok:
        failures = 1

    print("\n== 3. Cloudflare LLM config (presence only) ==")
    if not os.environ.get("CLOUDFLARE_API_TOKEN"):
        print("   CLOUDFLARE_API_TOKEN not set in environment - real LLM not run.")
    else:
        print("   CLOUDFLARE_API_TOKEN set -> run the 8-case matrix next.")

    return failures if failures else 0


if __name__ == "__main__":
    sys.exit(main())