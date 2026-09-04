import base64
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
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

from pydantic import BaseModel

from database import (
    init_db,
    save_prediction,
    get_emotion_distribution,
    get_confidence_statistics,
    get_daily_summary,
    get_device_stats,
    get_predictions,
    save_tutor_feedback,
    get_tutor_feedback_history,
)
from model import get_model_info, load_model, predict_image, predict_frame
from tutor import (
    CloudflareRateLimitExhausted,
    FALLBACK_MESSAGES_VI,
    generate_tutor_feedback,
)
from tutor_trigger import (
    update_streak_and_should_trigger,
    recent_emotion_trend,
)

# Private aliases (kept for call-site readability) — logic lives in tutor_trigger.py
_update_streak_and_should_trigger = update_streak_and_should_trigger
_recent_emotion_trend = recent_emotion_trend

WEIGHTS_PATH = os.environ.get("WEIGHTS_PATH", "final_model.pth")

@asynccontextmanager
async def lifespan(_: FastAPI):
    """Initialize the SQLite database (schema + indexes) on startup."""
    init_db()
    yield


app = FastAPI(
    title="Emotion-Aware AI Learning Assistant API",
    description=(
        "Nhận diện cảm xúc khuôn mặt real-time (ResNet50 + OpenCV Haar) kết hợp "
        "AI Tutor (Cloudflare Workers AI) sinh phản hồi học tập khi phát hiện cảm xúc tiêu cực kéo dài."
    ),
    version="2.1.0",
    lifespan=lifespan,
)

# Cho phép call từ dashboard (domain khác). Giới hạn origin qua CORS_ORIGINS
# khi deploy production (vd: https://your-frontend.pages.dev).
_cors_origins = [o.strip() for o in os.environ.get("CORS_ORIGINS", "*").split(",") if o.strip()] or ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lưu kết quả mới nhất của từng thiết bị (esp32_cam_1, esp32_cam_2, ...)
device_results: Dict[str, Dict[str, Any]] = {}

# Lazy-load model để /docs mở nhanh, chỉ load khi gọi /predict lần đầu
model = None

# ====== AI Tutor: sustained-emotion tracking ======
# Trigger logic (streak + cooldown, per device) lives in tutor_trigger.py
# so it can be unit-tested without the CV pipeline. Only support-needed
# emotions that repeat TUTOR_STREAK_THRESHOLD frames trigger the LLM.


# Initialize database on startup
# (handled by `lifespan` above — keeps uvicorn + TestClient in sync)


def get_model():
    global model
    if model is None:
        _info = get_model_info()
        print(f"⏳ Loading model {_info['model_name']} version {_info['model_version']} "
              f"({_info['architecture']}, git {_info['git_commit'] or 'n/a'}) from {WEIGHTS_PATH}")
        model_local = load_model(WEIGHTS_PATH)
        model = model_local
        print("✅ Model loaded successfully.")
    return model


# ============== REST API Endpoints ==============


@app.get("/health")
def health():
    """Liveness probe cho deployment (Render/Koyeb/...): không load model, luôn nhanh."""
    return {"status": "ok"}


@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Emotion-Aware AI Learning Assistant API v2.1.0",
        "endpoints": {
            "dashboard": "/dashboard",
            "predict_image": "POST /predict",
            "latest_emotion": "GET /latest/emotion",
            "websocket": "WS /ws/camera",
            "tutor_feedback_on_demand": "POST /tutor/feedback",
            "tutor_history": "GET /tutor/history",
            "model_info": "GET /info",
            "docs": "/docs",
        },
    }


@app.get("/info")
def model_info_endpoint():
    """Non-sensitive identity of the active model (MLOps / model versioning)."""
    return get_model_info()


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
        "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
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

    # AI Tutor: chỉ trigger khi cảm xúc "cần hỗ trợ" lặp lại liên tiếp.
    # Reliability (§6): prediction luôn 200 — exhausted-429 ở background path
    # degrade thành fallback rate_limited (không vỡ prediction); chỉ endpoint
    # on-demand /tutor/feedback mới trả 503.
    if _update_streak_and_should_trigger(device_id, result["label"]):
        try:
            trend = _recent_emotion_trend(device_id)
            feedback = await generate_tutor_feedback(
                emotion=result["label"],
                confidence=result["confidence"],
                trend=trend,
            )
            save_tutor_feedback(
                device_id=device_id,
                timestamp=wrapped["timestamp"],
                trigger_emotion=result["label"],
                message=feedback["message"],
                source=feedback["source"],
            )
            wrapped["tutor_feedback"] = feedback
            print(f"🤖 AI Tutor triggered [{device_id}] "
                  f"emotion={feedback['emotion']} source={feedback['source']} "
                  f"latency={feedback.get('latency_ms', '?')}ms")
        except CloudflareRateLimitExhausted:
            # Safe degradation: prediction stays 200, tutor part signals retryable.
            fallback_msg = FALLBACK_MESSAGES_VI.get(
                result["label"],
                "Cứ từ từ nhé, mình luôn ở đây nếu bạn cần hỗ trợ.",
            )
            wrapped["tutor_feedback"] = {
                "message": fallback_msg,
                "source": "fallback",
                "emotion": result["label"],
                "rate_limited": True,
                "retryable": True,
            }
            print(f"⏳ Tutor rate-limited [{device_id}] emotion={result['label']} "
                  f"→ fallback (prediction unaffected).")
        except Exception as e:
            print("⚠️ Tutor feedback generation failed:", e)

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
                    "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
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

                    # AI Tutor: dựa trên face đầu tiên, chỉ trigger khi cảm
                    # xúc tiêu cực lặp lại liên tiếp (xem _update_streak_and_should_trigger)
                    top_emotion = results[0]["label"]
                    if _update_streak_and_should_trigger("webcam", top_emotion):
                        try:
                            trend = _recent_emotion_trend("webcam")
                            feedback = await generate_tutor_feedback(
                                emotion=top_emotion,
                                confidence=results[0]["confidence"],
                                trend=trend,
                            )
                            save_tutor_feedback(
                                device_id="webcam",
                                timestamp=response["timestamp"],
                                trigger_emotion=top_emotion,
                                message=feedback["message"],
                                source=feedback["source"],
                            )
                            response["tutor_feedback"] = feedback
                            print(
                                f"🤖 AI Tutor triggered [webcam] "
                                f"emotion={feedback['emotion']} "
                                f"source={feedback['source']} "
                                f"latency={feedback.get('latency_ms', '?')}ms"
                            )
                        except CloudflareRateLimitExhausted:
                            response["tutor_feedback"] = {
                                "message": FALLBACK_MESSAGES_VI.get(
                                    top_emotion,
                                    "Cứ từ từ nhé, mình luôn ở đây nếu bạn cần hỗ trợ.",
                                ),
                                "source": "fallback",
                                "emotion": top_emotion,
                                "rate_limited": True,
                                "retryable": True,
                            }
                            print("⏳ Tutor rate-limited [webcam] → fallback (stream unaffected).")
                        except Exception as e:
                                print("⚠️ Tutor feedback generation failed:", e)

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
    filename = f"emotion_report_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv"
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


class TutorFeedbackRequest(BaseModel):
    device_id: str = "webcam"
    emotion: str
    confidence: float = 1.0
    lang: str = "vi"


@app.post("/tutor/feedback")
async def tutor_feedback_on_demand(req: TutorFeedbackRequest):
    """
    Sinh feedback AI Tutor theo yêu cầu (vd: nút "Gợi ý" trên dashboard),
    độc lập với cơ chế sustained-streak dùng trong /predict và WebSocket.

    Reliability (§6): exhausted-429 sau retry được map thành HTTP 503 safe
    (không lộ provider internals); các lỗi provider khác degrade về
    fallback 200.
    """
    trend = _recent_emotion_trend(req.device_id)
    try:
        feedback = await generate_tutor_feedback(
            emotion=req.emotion,
            confidence=req.confidence,
            trend=trend,
            lang=req.lang,
        )
    except CloudflareRateLimitExhausted:
        # Safe 503: retryable, no provider internals in the body.
        raise HTTPException(
            status_code=503,
            detail="Tutor temporarily unavailable due to provider rate limit. Please retry shortly.",
        )
    # DB write must never break feedback delivery (same isolation principle as
    # the CV pipeline: a downstream failure never fails the request).
    try:
        save_tutor_feedback(
            device_id=req.device_id,
            timestamp=datetime.now(timezone.utc).isoformat() + "Z",
            trigger_emotion=req.emotion,
            message=feedback["message"],
            source=feedback["source"],
        )
    except Exception as e:
        print("⚠️ Failed to save tutor feedback to DB:", e)
    print(f"✅ Tutor on-demand [{req.device_id}] emotion={feedback['emotion']} "
          f"source={feedback['source']} latency={feedback.get('latency_ms', '?')}ms")
    return feedback


@app.get("/tutor/history")
def tutor_feedback_history(device_id: str | None = None, limit: int = 50):
    """Lịch sử các message AI Tutor đã sinh ra, để hiện trên dashboard hoặc phân tích."""
    return {"history": get_tutor_feedback_history(device_id=device_id, limit=limit)}


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
