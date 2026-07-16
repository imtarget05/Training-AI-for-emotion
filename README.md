# AI Emotion Recognition

Real-time facial emotion recognition system powered by Deep Learning. The project provides a web dashboard with live camera analysis, a REST API for IoT devices, persistent data storage, automated reporting, and CSV export.

## Features

- **Real-time Emotion Detection** via WebSocket — webcam frames are processed in real-time with bounding boxes and probability charts
- **REST API** — ESP32-CAM or any HTTP client can POST images and receive emotion results
- **7 Emotion Classes** — Surprise, Fear, Disgust, Happiness, Sadness, Anger, Neutral
- **SQLite Persistence** — every prediction is logged with timestamps, device IDs, and confidence scores
- **Reporting & Analytics API** — emotion distribution, confidence statistics, daily summaries, per-device stats
- **CSV Export** — download filtered prediction data as CSV for Excel/BI tools
- **Automated HTML Report** — one-click generated summary with KPIs, tables, and charts
- **Batch Image Processing** — process an entire folder of images via CLI and store results
- **Seed Sample Data** — generate synthetic records for quick demo/testing
- **Interactive Dashboard** — modern SPA with WebSocket camera, emotion charts, and reporting UI

## Model Architecture

| Component | Detail |
|---|---|
| Backbone | ResNet50 (transfer learning, 2048-dim features) |
| Classifier | Linear layer (2048 → 7 classes) |
| Face Detection | OpenCV Haar Cascade (`haarcascade_frontalface_default.xml`) |
| Input | RGB 224×224, normalized with ImageNet mean/std |
| Output | Emotion label + confidence + per-class probabilities |

## Project Structure

```
.
├── main.py               # FastAPI app — REST endpoints, WebSocket, reporting routes
├── model.py              # ResNet50 encoder, classifier, face detection, inference helpers
├── database.py           # SQLite persistence — predictions, queries, statistics
├── automation.py         # Batch processing, HTML report generation, sample data seeding
├── final_model.pth       # Pre-trained model weights (~90 MB)
├── requirements.txt      # Python dependencies
├── Dockerfile            # Docker build configuration
├── static/
│   └── index.html        # Single-page dashboard (HTML/CSS/JS)
├── image/                # Architecture diagrams and evaluation charts
└── reports/              # Generated HTML reports (auto-created)
```

## Quick Start

### Prerequisites

- Python 3.10+
- pip

### Install & Run

```bash
cd Training-AI-for-emotion

# Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start server
python main.py
```

The server runs at `http://127.0.0.1:8000`.

| URL | Description |
|---|---|
| `http://127.0.0.1:8000/dashboard` | Web Dashboard |
| `http://127.0.0.1:8000/docs` | Swagger API Docs |

### Docker

```bash
docker build -t ai-emotion-app .
docker run -d -p 8080:8080 ai-emotion-app
```

## API Reference

### Prediction

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/predict` | Upload image (multipart) → emotion result |
| `GET` | `/latest/emotion` | Latest prediction across all devices |
| `WS` | `/ws/camera` | Real-time camera stream (base64 JPEG → JSON) |

### Reports & Analytics

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/reports/emotion-distribution` | Emotion counts & percentages |
| `GET` | `/reports/confidence-stats` | Avg/min/max confidence |
| `GET` | `/reports/daily-summary` | Daily aggregated metrics (30 days) |
| `GET` | `/reports/devices` | Per-device statistics |
| `GET` | `/reports/predictions` | Filtered prediction list |
| `GET` | `/reports/export-csv` | Download data as CSV |
| `POST` | `/reports/seed` | Seed sample data for testing |
| `POST` | `/reports/generate-html` | Generate HTML summary report |

All report endpoints support query filters: `device_id`, `emotion`, `start_date`, `end_date`.

### Automation CLI

```bash
# Seed 100 sample records
python automation.py seed 100

# Generate HTML report
python automation.py report

# Batch-process images in a folder
python automation.py batch ./my_images device_id_1
```

## Experimental Results

| Metric | Value |
|---|---|
| Architecture | ResNet50 + Linear Classifier |
| Emotion Classes | 7 |
| Input Size | 224 × 224 |
| Face Detector | Haar Cascade |

Evaluation charts are available in the `image/` directory:
- **Confusion Matrix** (`image/confusion_matrix.png`) — per-class accuracy analysis
- **Fine-tuning Metrics** (`image/finetune_metrics.png`) — train/val loss and accuracy curves

## Limitations & Future Improvements

**Current limitations:**
- Haar Cascade face detector works best with frontal faces; sensitivity to angles and low light
- CPU-only inference may cause frame drops with multiple faces

**Planned improvements:**
- Replace Haar Cascade with MTCNN / RetinaFace / YOLO Face for robust detection
- Swap ResNet50 for lightweight backbones (MobileNetV2, EfficientNet-B0) for edge deployment
- Add time-series database support (InfluxDB) for large-scale analytics
- WebSocket authentication and device registration

## Author

**Mai Nguyen Binh Tan**
