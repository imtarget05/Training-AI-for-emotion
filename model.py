import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
import os

# ================== Kiến trúc Encoder & Classifier ==================


class ResNetEncoder(nn.Module):
    def __init__(self, architecture: str = "resnet50", pretrained: bool = False):
        super(ResNetEncoder, self).__init__()
        self.architecture = architecture

        if architecture == "resnet18":
            # pretrained=False -> weights=None để tránh warning
            self.resnet = models.resnet18(weights=None)
            self.feature_dim = 512

        elif architecture == "resnet34":
            self.resnet = models.resnet34(weights=None)
            self.feature_dim = 512

        elif architecture == "resnet50":
            self.resnet = models.resnet50(weights=None)
            self.feature_dim = 2048

        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

        # Bỏ FC cuối, giữ lại feature map [B, C, 1, 1]
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

    def forward(self, x):
        features = self.resnet(x)  # [B, C, 1, 1]
        features = torch.flatten(features, 1)  # [B, C]
        return features


class FinetuneClassifier(nn.Module):
    def __init__(self, encoder: ResNetEncoder, num_classes: int = 7):
        super(FinetuneClassifier, self).__init__()
        self.encoder = encoder
        feature_dim = encoder.feature_dim
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        features = self.encoder(x)
        logits = self.classifier(features)
        return logits


# ================== Cấu hình inference ==================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = [
    "Surprise",
    "Fear",
    "Disgust",
    "Happiness",
    "Sadness",
    "Anger",
    "Neutral",
]

# Emoji mapping cho từng cảm xúc
EMOTION_EMOJIS = {
    "Surprise": "😲",
    "Fear": "😨",
    "Disgust": "🤢",
    "Happiness": "😊",
    "Sadness": "😢",
    "Anger": "😠",
    "Neutral": "😐",
}

# Chỉnh lại cho đúng với lúc train (nếu bạn dùng resize/normalize khác thì sửa ở đây)
inference_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

# Load Haar Cascade cho face detection
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)


def load_model(weights_path: str) -> FinetuneClassifier:
    """
    Load model ResNet50 + classifier từ file .pth
    """
    encoder = ResNetEncoder(architecture="resnet50", pretrained=False)
    model = FinetuneClassifier(encoder, num_classes=len(CLASS_NAMES))

    state_dict = torch.load(weights_path, map_location=device, weights_only=True)

    # Xử lý trường hợp state_dict được lưu với DataParallel (module.xxx)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_key = k[len("module.") :]
        else:
            new_key = k
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    return model


def detect_faces(img_np: np.ndarray):
    """
    Detect faces using OpenCV Haar Cascade.
    Input: numpy array BGR (from cv2) or RGB
    Returns: list of (x, y, w, h) tuples
    """
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(48, 48),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    return faces if len(faces) > 0 else []


def predict_image(model: FinetuneClassifier, img: Image.Image):
    """
    Nhận PIL Image, trả về dict JSON:
    {
      "label": ...,
      "confidence": ...,
      "probs": {label: prob, ...}
    }
    """
    # đảm bảo ảnh 3 kênh
    if img.mode != "RGB":
        img = img.convert("RGB")

    tensor = inference_transform(img).unsqueeze(0).to(device)  # [1, C, H, W]

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]  # [num_classes]

    idx = torch.argmax(probs).item()
    label = CLASS_NAMES[idx]
    confidence = float(probs[idx])

    return {
        "label": label,
        "confidence": confidence,
        "emoji": EMOTION_EMOJIS.get(label, ""),
        "probs": {
            CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))
        },
    }


def predict_frame(model: FinetuneClassifier, img_np: np.ndarray):
    """
    Nhận numpy array (RGB), detect faces, predict emotion cho mỗi face.
    Trả về list các kết quả, mỗi kết quả gồm:
    {
      "face": {"x": int, "y": int, "w": int, "h": int},
      "label": str,
      "confidence": float,
      "emoji": str,
      "probs": {label: prob, ...}
    }
    """
    faces = detect_faces(img_np)
    results = []

    if len(faces) == 0:
        # Không tìm thấy face → predict toàn ảnh
        pil_img = Image.fromarray(img_np)
        result = predict_image(model, pil_img)
        h, w = img_np.shape[:2]
        result["face"] = {"x": 0, "y": 0, "w": w, "h": h}
        results.append(result)
    else:
        for (x, y, w, h) in faces:
            # Mở rộng bounding box 1 chút cho tự nhiên hơn
            pad = int(0.1 * max(w, h))
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(img_np.shape[1], x + w + pad)
            y2 = min(img_np.shape[0], y + h + pad)

            face_crop = img_np[y1:y2, x1:x2]
            pil_face = Image.fromarray(face_crop)
            result = predict_image(model, pil_face)
            result["face"] = {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}
            results.append(result)

    return results
