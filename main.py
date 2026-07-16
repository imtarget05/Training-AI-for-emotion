import asyncio
import base64
import json
import time
from datetime import datetime
from io import BytesIO
from typing import Any, Dict

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import uvicorn

from database import (
    init_db,
    save_prediction,
    get_emotion_distribution,
    get_confidence_statistics,
    get_daily_summary,
    get_device_stats,
    get_predictions,
)
from model import load_model, predict_image, predict_frame

WEIGHTS_PATH = "final_model.pth"

app = FastAPI(
    title="Emotion Recognition API",
    description="API nhận diện cảm xúc khuôn mặt từ ảnh & camera real-time.",
    version="2.0.0",
)

# Cho phép call từ dashboard (domain khác)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # sau này có thể giới hạn domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lưu kết quả mới nhất của từng thiết bị (esp32_cam_1, esp32_cam_2, ...)
device_results: Dict[str, Dict[str, Any]] = {}

# Lazy-load model để /docs mở nhanh, chỉ load khi gọi /predict lần đầu
model = None


# Initialize database on startup
@app.on_event("startup")
def startup_event():
    init_db()


def get_model():
    global model
    if model is None:
        print("⏳ Loading model weights from:", WEIGHTS_PATH)
        model_local = load_model(WEIGHTS_PATH)
        model = model_local
        print("✅ Model loaded successfully.")
    return model


# ============== REST API Endpoints ==============


@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Emotion Recognition API v2.0",
        "endpoints": {
            "dashboard": "/dashboard",
            "predict_image": "POST /predict",
            "latest_emotion": "GET /latest/emotion",
            "websocket": "WS /ws/camera",
            "docs": "/docs",
        },
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    device_id: str = Form("esp32_cam_1"),
):
    """
    ESP32-CAM (hoặc bất kỳ client nào) gửi ảnh qua multipart/form-data:
      - file: ảnh jpg/png
      - device_id: id thiết bị (esp32_cam_1, ...)
    Trả về JSON: {device_id, timestamp, label, confidence, probs}
    """
    try:
        contents = await file.read()
        img = Image.open(BytesIO(contents))
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": f"Cannot read image: {str(e)}"},
        )

    current_model = get_model()
    result = predict_image(current_model, img)

    wrapped = {
        "device_id": device_id,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        **result,
    }

    # Lưu kết quả mới nhất của device cho dashboard đọc
    device_results[device_id] = wrapped

    # Lưu vào database
    try:
        save_prediction(
            device_id=device_id,
            timestamp=wrapped["timestamp"],
            emotion=result["label"],
            confidence=result["confidence"],
            probs=result.get("probs", {}),
            face_detected=True,
        )
    except Exception as e:
        print("⚠️ Failed to save prediction to DB:", e)

    return JSONResponse(wrapped)


@app.get("/latest/emotion")
def get_latest_emotion():
    """
    Dashboard gọi để lấy kết quả cảm xúc mới nhất (không cần device_id).
    Nếu có nhiều ESP32 thì sẽ lấy kết quả cuối cùng hệ thống nhận được.
    """
    if not device_results:
        raise HTTPException(status_code=404, detail="No result yet")

    # Lấy kết quả mới nhất theo timestamp
    latest = max(device_results.values(), key=lambda x: x["timestamp"])
    return latest


# ============== WebSocket Endpoint cho Camera Real-time ==============


@app.websocket("/ws/camera")
async def websocket_camera(websocket: WebSocket):
    """
    WebSocket endpoint nhận frame từ browser camera (base64 JPEG)
    và trả về kết quả emotion detection real-time.
    
    Client gửi: base64 string của JPEG image
    Server trả về: JSON {faces: [{face, label, confidence, emoji, probs}]}
    """
    await websocket.accept()
    print("📷 WebSocket client connected")

    current_model = get_model()

    try:
        while True:
            # Nhận base64 frame từ client
            data = await websocket.receive_text()

            try:
                # Decode base64 → image
                # Client có thể gửi data:image/jpeg;base64,... hoặc chỉ base64
                if "," in data:
                    data = data.split(",", 1)[1]

                img_bytes = base64.b64decode(data)
                img_np = np.frombuffer(img_bytes, dtype=np.uint8)
                img_np = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

                if img_np is None:
                    await websocket.send_json({"error": "Cannot decode image"})
                    continue

                # BGR → RGB
                img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

                # Predict
                results = predict_frame(current_model, img_rgb)

                response = {
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "faces": results,
                    "face_count": len(results),
                }

                # Cập nhật kết quả mới nhất
                if results:
                    device_results["webcam"] = {
                        "device_id": "webcam",
                        "timestamp": response["timestamp"],
                        **results[0],  # Lấy kết quả face đầu tiên
                    }

                    # Lưu mỗi face vào database (giới hạn 3 để tránh quá tải)
                    for r in results[:3]:
                        try:
                            save_prediction(
                                device_id="webcam",
                                timestamp=response["timestamp"],
                                emotion=r["label"],
                                confidence=r["confidence"],
                                probs=r.get("probs", {}),
                                face_detected=True,
                            )
                        except Exception:
                            pass

                await websocket.send_json(response)

            except Exception as e:
                await websocket.send_json({"error": str(e)})

    except WebSocketDisconnect:
        print("📷 WebSocket client disconnected")


# ============== Reporting & Statistics API ==============


@app.get("/reports/emotion-distribution")
def report_emotion_distribution(
    device_id: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
):
    """Báo cáo phân bố cảm xúc theo số lượng và phần trăm."""
    data = get_emotion_distribution(
        device_id=device_id,
        start_date=start_date,
        end_date=end_date,
    )
    total = sum(data.values()) if data else 0
    percentages = (
        {k: {"count": v, "percentage": round(v / total * 100, 2)} for k, v in data.items()}
        if total > 0
        else {}
    )
    return {"distribution": data, "total_predictions": total, "percentages": percentages}


@app.get("/reports/confidence-stats")
def report_confidence_stats(
    emotion: str | None = None,
    device_id: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
):
    """Thống kê độ tin cậy: avg/min/max."""
    return get_confidence_statistics(
        emotion=emotion,
        device_id=device_id,
        start_date=start_date,
        end_date=end_date,
    )


@app.get("/reports/daily-summary")
def report_daily_summary(
    device_id: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
):
    """Báo cáo tổng hợp theo ngày (30 ngày gần nhất)."""
    data = get_daily_summary(device_id=device_id, start_date=start_date, end_date=end_date)
    return {"daily_summary": data, "days": len(data)}


@app.get("/reports/devices")
def report_devices():
    """Thống kê theo từng thiết bị."""
    return {"devices": get_device_stats()}


@app.get("/reports/predictions")
def report_predictions(
    device_id: str | None = None,
    emotion: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int = 100,
    offset: int = 0,
):
    """Danh sách các lần nhận diện với bộ lọc."""
    data = get_predictions(
        device_id=device_id,
        emotion=emotion,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
        offset=offset,
    )
    return {"predictions": data, "count": len(data), "limit": limit, "offset": offset}


@app.get("/reports/export-csv")
def export_csv(
    device_id: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
):
    """Xuất dữ liệu CSV để phân tích trong Excel/BI tool."""
    import csv
    import io
    from fastapi.responses import StreamingResponse

    data = get_predictions(device_id=device_id, start_date=start_date, end_date=end_date, limit=10000)
    if not data:
        raise HTTPException(status_code=404, detail="No data to export")

    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=["id", "device_id", "timestamp", "emotion", "confidence", "created_at"],
    )
    writer.writeheader()
    for row in data:
        writer.writerow({k: row[k] for k in ["id", "device_id", "timestamp", "emotion", "confidence", "created_at"]})

    output.seek(0)
    filename = f"emotion_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.post("/reports/seed")
def seed_sample_data(num_records: int = 100):
    """Tự động thêm dữ liệu mẫu để demo báo cáo/bảng điều khiển."""
    from automation import generate_sample_data
    saved = generate_sample_data(num_records)
    return {"status": "ok", "seeded_records": saved}


@app.post("/reports/generate-html")
def trigger_daily_report():
    """Kích hoạt tự động hóa xuất báo cáo HTML."""
    from automation import generate_daily_report
    path = generate_daily_report()
    return {"status": "ok", "report_path": path}


# ============== Serve Dashboard ==============

# Đảm bảo thư mục reports tồn tại và serve HTML reports
import os as _os
if not _os.path.exists("reports"):
    _os.makedirs("reports")
app.mount("/reports-html", StaticFiles(directory="reports"), name="reports_html")


# Mount static files SAU tất cả routes để không override
app.mount("/dashboard", StaticFiles(directory="static", html=True), name="dashboard")


if __name__ == "__main__":
    # Chạy local: python main.py
    # Dashboard: http://127.0.0.1:8000/dashboard
    # API Docs: http://127.0.0.1:8000/docs
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
