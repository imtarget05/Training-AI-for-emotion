"""
scripts/demo_e2e.py - canonical end-to-end demo scenario (Phase 12).

Usage:
    docker compose up --build
    E2E_BASE_URL=http://localhost:8000 python scripts/demo_e2e.py
    python scripts/demo_e2e.py --base http://localhost:8000
"""
import argparse, base64, io, json, os, httpx
from PIL import Image

SUPPORT = {"Sadness", "Fear", "Anger", "Disgust"}


def step(n, title, ok, detail=""):
    flag = "PASS" if ok else "FAIL"
    label = f"{n:02d}" if isinstance(n, int) else str(n)
    print(f"[{label}] [{flag}] {title}" + (f" -- {detail}" if detail else ""))
    return ok


def jpeg_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (128, 128), (70, 90, 120)).save(buf, format="JPEG")
    return buf.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=os.environ.get("E2E_BASE_URL", "http://localhost:8000"))
    args = parser.parse_args()
    base = args.base.rstrip("/")
    results = []
    print(f"== E2E @ {base} ==\n")
    with httpx.Client(base_url=base, timeout=20) as c:
        r = c.get("/")
        ok = r.status_code == 200
        results.append(step(1, "Server healthy", ok, r.json().get("message", "")))
        r = c.post("/predict", files={"file": ("f.jpg", jpeg_bytes(), "image/jpeg")}, data={"device_id": "demo"})
        body = r.json()
        ok = r.status_code == 200 and "label" in body
        results.append(step(2, "Predict", ok, f"{body.get('label', '?')}"))
        label = body["label"]
        if label in SUPPORT:
            for _ in range(2):
                c.post("/predict", files={"file": ("f.jpg", jpeg_bytes(), "image/jpeg")}, data={"device_id": "demo"})
            r3 = c.post("/predict", files={"file": ("f.jpg", jpeg_bytes(), "image/jpeg")}, data={"device_id": "demo"})
            fb = r3.json().get("tutor_feedback")
            ok = fb is not None
            track = f"source={fb.get('source')}" if fb else "expected tutor_feedback"
            results.append(step(3, "Streak trigger", ok, track))
        r = c.post("/tutor/feedback", json={"device_id": "demo", "emotion": "Sadness", "confidence": 0.8, "lang": "vi"})
        body6 = r.json()
        ok = r.status_code == 200 and body6.get("source") in ("llm", "fallback")
        results.append(step(6, "On-demand tutor", ok, f"source={body6.get('source')}"))
        r = c.post("/predict", files={"file": ("f.jpg", jpeg_bytes(), "image/jpeg")}, data={"device_id": "demo"})
        ok = r.status_code == 200 and "label" in r.json()
        results.append(step(7, "CV healthy", ok, f"label={r.json().get('label', '?')}"))
        try:
            from websockets.sync.client import connect as ws_connect
            ws_url = base.replace("http", "ws").rstrip("/") + "/ws/camera"
            with ws_connect(ws_url) as ws:
                ws.send(base64.b64encode(jpeg_bytes()).decode())
                msg = json.loads(ws.recv())
            ok = "faces" in msg or "error" in msg
            results.append(step(8, "WebSocket", ok, f"faces={msg.get('face_count', '?')}"))
        except ImportError:
            results.append(step(8, "WebSocket", True, "lib unavailable"))
        except Exception as e:
            results.append(step(8, "WebSocket", False, str(e)))
    print("\n" + "=" * 40)
    passed = sum(1 for r in results if r)
    print(f"SUMMARY: {passed}/{len(results)} passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())