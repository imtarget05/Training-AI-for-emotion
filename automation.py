"""
Automation module: batch image processing, automated HTML report generation,
and sample data seeding for testing.

Usage:
  python automation.py report            # Generate daily HTML report
  python automation.py seed              # Seed sample data for testing
  python automation.py batch <folder>   # Batch-process images in folder
"""
import os
import sys
import json
from datetime import datetime, timedelta
from pathlib import Path
from database import init_db, save_batch_predictions, get_connection

WEIGHTS_PATH = "final_model.pth"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
REPORTS_DIR = "reports"


# ─────────────────────────────────────────────────────────────
# 1. Batch image processing
# ─────────────────────────────────────────────────────────────

def process_batch_folder(folder_path: str, device_id: str = "batch_automation") -> int:
    """
    Scan a folder for images, run emotion inference on each, and persist
    all results to the SQLite database in one batch transaction.
    """
    from PIL import Image
    from model import load_model, predict_image

    folder = Path(folder_path)
    if not folder.is_dir():
        print(f"⚠️  Folder not found: {folder_path}")
        return 0

    images = [p for p in folder.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS]
    if not images:
        print("ℹ️  No supported images found.")
        return 0

    print(f"⏳ Loading model from {WEIGHTS_PATH}...")
    model = load_model(WEIGHTS_PATH)
    print(f"🚀 Processing {len(images)} images (device_id={device_id})...")

    batch = []
    for i, img_path in enumerate(images, 1):
        try:
            img = Image.open(img_path)
            result = predict_image(model, img)
            batch.append({
                "device_id": device_id,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "emotion": result["label"],
                "confidence": result["confidence"],
                "face_detected": True,
                "probs": result.get("probs", {}),
            })
            print(f"  [{i}/{len(images)}] {img_path.name} → {result['label']} "
                  f"({result['confidence']*100:.1f}%)")
        except Exception as e:
            print(f"  [{i}/{len(images)}] ⚠️  Skip {img_path.name}: {e}")

    saved = save_batch_predictions(batch)
    print(f"\n✅ Saved {saved}/{len(images)} predictions to database.")
    return saved


# ─────────────────────────────────────────────────────────────
# 2. Automated HTML report
# ─────────────────────────────────────────────────────────────

def generate_daily_report(output_dir: str = REPORTS_DIR) -> str:
    """
    Query SQLite, aggregate emotion metrics for the last 7 days,
    and write a self-contained HTML report to output_dir/.
    """
    os.makedirs(output_dir, exist_ok=True)
    conn = get_connection()
    c = conn.cursor()

    c.execute("SELECT COUNT(*), AVG(confidence) FROM predictions")
    total_count, avg_conf = c.fetchone()
    total_count = total_count or 0
    avg_conf = round(avg_conf or 0.0, 4)

    c.execute("SELECT emotion, COUNT(*), AVG(confidence) FROM predictions GROUP BY emotion ORDER BY COUNT(*) DESC")
    distribution = c.fetchall()

    c.execute("""
        SELECT DATE(timestamp) as d, COUNT(*), AVG(confidence)
        FROM predictions GROUP BY d ORDER BY d DESC LIMIT 7
    """)
    daily = c.fetchall()

    c.execute("SELECT device_id, COUNT(*), MAX(timestamp) FROM predictions GROUP BY device_id")
    devices = c.fetchall()

    # Log report metadata
    today_str = datetime.utcnow().strftime("%Y-%m-%d")
    report_path = os.path.join(output_dir, f"report_{today_str}.html")
    try:
        c.execute(
            "INSERT INTO reports (report_type, generated_at, file_path, metadata_json) VALUES (?,?,?,?)",
            (
                "daily_summary",
                datetime.utcnow().isoformat() + "Z",
                report_path,
                json.dumps({"total": total_count, "avg_conf": avg_conf}),
            ),
        )
        conn.commit()
    except Exception:
        pass
    conn.close()

    def rows_html(rows, cols):
        out = ""
        for row in rows:
            out += "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>\n"
        return out

    dist_rows = [
        (r[0], r[1], f"{r[1]/total_count*100:.1f}%" if total_count else "0%", f"{r[2]*100:.1f}%")
        for r in distribution
    ]

    html = f"""<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8"/>
  <title>Emotion Report {today_str}</title>
  <style>
    body{{font-family:'Segoe UI',sans-serif;background:#f3f4f6;color:#1f2937;margin:0;padding:32px}}
    .wrap{{max-width:960px;margin:0 auto;background:#fff;border-radius:12px;padding:32px;
           box-shadow:0 4px 12px rgba(0,0,0,.08)}}
    h1{{color:#4f46e5;border-bottom:2px solid #e5e7eb;padding-bottom:12px}}
    h2{{color:#374151;margin-top:28px}}
    .meta{{color:#6b7280;font-size:.9rem;margin-bottom:24px}}
    .kpi-row{{display:flex;gap:16px;flex-wrap:wrap;margin-bottom:8px}}
    .kpi{{flex:1;min-width:160px;background:#f9fafb;border:1px solid #e5e7eb;
          border-radius:10px;padding:16px}}
    .kpi-title{{font-size:.8rem;text-transform:uppercase;color:#6b7280;font-weight:600}}
    .kpi-val{{font-size:2rem;font-weight:700;color:#111827;margin-top:4px}}
    table{{width:100%;border-collapse:collapse;margin-top:12px;font-size:.9rem}}
    th{{background:#f3f4f6;text-align:left;padding:10px;border:1px solid #e5e7eb;color:#4b5563;font-weight:600}}
    td{{padding:9px 10px;border:1px solid #e5e7eb}}
    tr:nth-child(even){{background:#f9fafb}}
    .footer{{margin-top:40px;text-align:center;font-size:.8rem;color:#9ca3af}}
  </style>
</head>
<body>
<div class="wrap">
  <h1>📊 AI Emotion Recognition — Daily Report</h1>
  <p class="meta">Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC</p>

  <div class="kpi-row">
    <div class="kpi"><div class="kpi-title">Total Predictions</div>
      <div class="kpi-val">{total_count}</div></div>
    <div class="kpi"><div class="kpi-title">Avg Confidence</div>
      <div class="kpi-val">{avg_conf*100:.1f}%</div></div>
    <div class="kpi"><div class="kpi-title">Active Devices</div>
      <div class="kpi-val">{len(devices)}</div></div>
    <div class="kpi"><div class="kpi-title">Emotion Types</div>
      <div class="kpi-val">{len(distribution)}</div></div>
  </div>

  <h2>🏷️ Emotion Distribution</h2>
  <table>
    <thead><tr><th>Emotion</th><th>Count</th><th>%</th><th>Avg Confidence</th></tr></thead>
    <tbody>{rows_html(dist_rows, 4)}</tbody>
  </table>

  <h2>📅 Last 7 Days</h2>
  <table>
    <thead><tr><th>Date</th><th>Detections</th><th>Avg Confidence</th></tr></thead>
    <tbody>{rows_html([(r[0], r[1], f"{r[2]*100:.1f}%") for r in daily], 3)}</tbody>
  </table>

  <h2>🔌 Devices</h2>
  <table>
    <thead><tr><th>Device ID</th><th>Total</th><th>Last Active</th></tr></thead>
    <tbody>{rows_html(devices, 3)}</tbody>
  </table>

  <div class="footer">Auto-generated by AI Data Automation Pipeline</div>
</div>
</body>
</html>
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"✅ HTML report saved: {report_path}")
    return report_path


# ─────────────────────────────────────────────────────────────
# 3. Seed sample data for testing
# ─────────────────────────────────────────────────────────────

def generate_sample_data(num_records: int = 100) -> int:
    """Insert synthetic predictions spanning the last 5 days for local testing."""
    import random
    emotions = ["Happiness", "Neutral", "Surprise", "Sadness", "Anger", "Fear", "Disgust"]
    devices = ["esp32_cam_1", "esp32_cam_2", "webcam"]
    base = datetime.utcnow() - timedelta(days=5)

    batch = []
    for i in range(num_records):
        emotion = random.choice(emotions)
        confidence = round(random.uniform(0.55, 0.98), 4)
        others = [e for e in emotions if e != emotion]
        probs = {e: round((1 - confidence) / len(others), 4) for e in others}
        probs[emotion] = confidence

        batch.append({
            "device_id": random.choice(devices),
            "timestamp": (base + timedelta(minutes=i * 72)).isoformat() + "Z",
            "emotion": emotion,
            "confidence": confidence,
            "face_detected": True,
            "probs": probs,
        })

    saved = save_batch_predictions(batch)
    print(f"✅ Seeded {saved} sample records into the database.")
    return saved


# ─────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    init_db()

    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"

    if cmd == "report":
        generate_daily_report()
    elif cmd == "seed":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 100
        generate_sample_data(n)
    elif cmd == "batch":
        if len(sys.argv) < 3:
            print("Usage: python automation.py batch <image_folder> [device_id]")
            sys.exit(1)
        folder = sys.argv[2]
        dev_id = sys.argv[3] if len(sys.argv) > 3 else "batch_automation"
        process_batch_folder(folder, dev_id)
    else:
        print("Commands: report | seed [n] | batch <folder> [device_id]")
        sys.exit(1)