from datetime import datetime
from io import BytesIO
from typing import Any, Dict

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import uvicorn

from model import load_model, predict_image

WEIGHTS_PATH = "final_model.pth"

app = FastAPI(
    title="Emotion Recognition API",
    description="API nhận diện cảm xúc khuôn mặt từ ảnh (ESP32-CAM, Dashboard, ...).",
    version="1.0.0",
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
        print("Loading model weights from:", WEIGHTS_PATH)
        model_local = load_model(WEIGHTS_PATH)
        model = model_local
        print("Model loaded successfully.")
    return model


@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Emotion API is running",
        "detail": "Use POST /predict to send an image.",
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



if __name__ == "__main__":
    # Chạy local: python main.py
    # Sau đó vào: http://127.0.0.1:8000/docs
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
