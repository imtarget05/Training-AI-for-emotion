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

                await websocket.send_json(response)

            except Exception as e:
                await websocket.send_json({"error": str(e)})

    except WebSocketDisconnect:
        print("📷 WebSocket client disconnected")


# ============== Serve Dashboard ==============

# Mount static files SAU tất cả routes để không override
app.mount("/dashboard", StaticFiles(directory="static", html=True), name="dashboard")


if __name__ == "__main__":
    # Chạy local: python main.py
    # Dashboard: http://127.0.0.1:8000/dashboard
    # API Docs: http://127.0.0.1:8000/docs
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
