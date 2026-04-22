# 🧠 AI Emotion Recognition

## 1. Project Overview
Ứng dụng nhận diện cảm xúc khuôn mặt (Real-time Emotion Recognition) sử dụng mô hình Deep Learning. Dự án cung cấp:
- **Giao diện Dashboard** trực quan chạy trên trình duyệt web, lấy luồng video trực tiếp từ Webcam và nhận diện cảm xúc theo thời gian thực qua giao thức WebSocket.
- **REST API** cho phép các thiết bị ngoại vi (như IoT ESP32-CAM hoặc các hệ thống khác) chụp và gửi ảnh lên server để nhận lại kết quả phân tích cảm xúc.

Hệ thống hỗ trợ phân loại khuôn mặt thành 7 lớp cảm xúc chính: **Surprise (Ngạc nhiên), Fear (Sợ hãi), Disgust (Kinh tởm), Happiness (Hạnh phúc), Sadness (Buồn bã), Anger (Tức giận), Neutral (Bình thường)**.

## 2. Model Architecture
Mô hình sử dụng phương pháp **Transfer Learning** dựa trên kiến trúc mạnh mẽ **ResNet50**:
- **Face Detection**: Sử dụng OpenCV Haar Cascade (`haarcascade_frontalface_default.xml`) để phát hiện và tự động cắt vùng chứa khuôn mặt.
- **Backbone (Encoder)**: ResNet50 (pre-trained, loại bỏ lớp Fully Connected cuối cùng để trích xuất đặc trưng với chiều 2048).
- **Classifier**: Lớp Linear Classifier (FinetuneClassifier) thực hiện ánh xạ từ 2048 chiều đặc trưng về 7 lớp cảm xúc tương ứng.
- **Tiền xử lý đầu vào (Preprocessing)**: Ảnh RGB được resize về chuẩn `224x224` pixel và chuẩn hóa (Normalize) theo tham số phân phối chuẩn của ImageNet.

## 3. Project Architecture (Kiến trúc Hệ thống)
Hệ thống hoạt động theo mô hình Client-Server monolith nhỏ gọn nhưng hiệu quả:
- **Backend (API Server)**: Sử dụng Python + framework **FastAPI**. Chịu trách nhiệm cung cấp REST API và dịch vụ WebSocket Server. Tích hợp PyTorch để thực hiện inference nhận diện ảnh.
- **Frontend (Dashboard)**: SPA (Single Page Application) sử dụng Vanilla HTML/CSS/JS (nằm gọn trong file `static/index.html`). Đảm nhiệm việc kết nối WebSocket, lấy camera frame, render biểu đồ xác suất và vẽ bounding box nhận diện trực tiếp.

### Luồng xử lý Real-time (WebSocket):
1. Trình duyệt (Browser Client) lấy frame từ Webcam, encode thành dạng Base64 JPEG.
2. Gửi ảnh liên tục lên Backend thông qua kết nối **WebSocket** (để giảm độ trễ so với HTTP).
3. Backend nhận frame, decode ảnh, sử dụng Haar Cascade phát hiện mặt, crop và đưa vào mô hình AI xử lý.
4. JSON kết quả bao gồm nhãn cảm xúc, độ tin cậy, bounding box tọa độ mặt được trả ngược về cho Client để cập nhật giao diện ngay lập tức.

## 4. How to Run (Hướng dẫn cài đặt & khởi chạy)

### Cách 1: Chạy trực tiếp trên máy host (Khuyên dùng cho Development)
**Yêu cầu hệ thống:** Có cài đặt sẵn Python 3.10+

1. **Di chuyển vào thư mục dự án:**
   ```bash
   cd "Training-AI-for-emotion"
   ```

2. **Tạo môi trường ảo (Virtual Environment) và cài đặt thư viện:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # (Trên Mac/Linux) hoặc venv\Scripts\activate (Trên Windows)
   pip install -r requirements.txt
   ```

3. **Khởi chạy API Server:**
   ```bash
   python main.py
   ```
   *Lưu ý: Mặc định server Uvicorn sẽ chạy ở cổng (port) 8000.*

4. **Truy cập Giao diện:**
   - Mở trình duyệt và vào Dashboard: [http://127.0.0.1:8000/dashboard](http://127.0.0.1:8000/dashboard)
   - Truy cập Swagger API Docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### Cách 2: Chạy thông qua Docker
1. **Build Docker image:**
   ```bash
   docker build -t ai-emotion-app .
   ```

2. **Khởi chạy Docker container:**
   ```bash
   docker run -d -p 8080:8080 ai-emotion-app
   ```
   *Truy cập ứng dụng tại địa chỉ: http://localhost:8080/dashboard*

## 5. Experimental Results (Đánh giá hiệu năng AI)
Dự án có đi kèm các biểu đồ đánh giá nằm trong thư mục `image/`:
- **Confusion Matrix** (`image/confusion_matrix.png`): Ma trận nhầm lẫn biểu diễn độ chính xác của dự đoán trên tập Test. Giúp phân tích chi tiết xem các lớp cảm xúc nào dễ bị nhầm lẫn với nhau nhất (VD: Sadness vs Neutral).
- **Fine-tuning Metrics** (`image/finetune_metrics.png`): Các đồ thị biểu diễn Loss/Accuracy thay đổi qua các epochs giữa tập Train và tập Validation, đóng vai trò quan trọng trong việc theo dõi tình trạng Overfitting hoặc Underfitting.

## 6. Project Structure (Cấu trúc thư mục)
```text
.
├── main.py                  # Entry point - Chứa toàn bộ API Endpoints & WebSocket Server
├── model.py                 # File code cấu trúc mạng ResNet50 và các hàm inference (detect_faces, predict_frame)
├── final_model.pth          # Trọng số pre-trained của mô hình AI (~90MB)
├── requirements.txt         # Khai báo các package Python (FastAPI, PyTorch, OpenCV, v.v...)
├── Dockerfile               # File script cấu hình để build ảnh Docker
├── README.md                # File tài liệu kỹ thuật của dự án (Bạn đang đọc)
├── static/
│   └── index.html           # Giao diện Frontend Web Dashboard
└── image/
    ├── confusion_matrix.png # Biểu đồ đánh giá: Ma trận nhầm lẫn
    └── finetune_metrics.png # Biểu đồ đánh giá: Loss / Accuracy
```

## 7. Limitations & Future Improvements
**Giới hạn hiện tại:**
- Module nhận diện khuôn mặt (Haar Cascade) hoạt động hiệu quả khi nhìn thẳng trực diện nhưng khá nhạy cảm với các góc nhìn nghiêng hoặc điều kiện thiếu sáng.
- Việc inference liên tục nhiều khuôn mặt trên môi trường chỉ chạy CPU có thể gây tụt FPS.

**Hướng cải thiện:**
- Nâng cấp mô hình Object Detection mạnh mẽ hơn thay cho Haar Cascade (như MTCNN, RetinaFace, hoặc YOLO Face).
- Sử dụng các mô hình gọn nhẹ (Lightweight Backbone) thay thế ResNet50 như MobileNetV2 / EfficientNet B0 để tăng tốc độ inference đối với các thiết bị Edge-AI.
- Lưu trữ lịch sử dạng Time-Series Data vào cơ sở dữ liệu (vd: InfluxDB/MongoDB) để phục vụ cho việc thống kê sau này thay vì lưu in-memory.

## 8. Author
- **Mai Nguyen Binh Tan**
