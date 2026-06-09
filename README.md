# 📝 MLOps OMR Grading Engine


## 🎯 Tổng quan

- **Nhận dạng OMR**: Phát hiện và nhận diện chính xác các ô trắc nghiệm được tô trên phiếu thi.
- **FastAPI Backend**: API suy luận hiệu năng cao, tích hợp xử lý bất đồng bộ và tiếp nhận phản hồi.
- **Streamlit Frontend**: Giao diện web thân thiện để người dùng tải ảnh lên, kiểm thử và hiệu chỉnh kết quả.
- **Giám sát & Báo động**: Giám sát hệ thống thời gian thực (Độ trễ, Data Drift, Phần cứng) với Prometheus & Grafana.
- **Huấn luyện tự động**: Tiến trình chạy ngầm (Worker) tự động thu thập dữ liệu lỗi và huấn luyện lại mô hình.
- **Tích hợp MLflow**: Quản lý phiên bản mô hình, theo dõi thử nghiệm và tự động chuyển trạng thái (Staging/Production).
- **DVC Pipeline**: Phiên bản hóa dữ liệu lớn và tự động hóa quy trình tái huấn luyện.
- **Lưu trữ MinIO S3**: Kho lưu trữ Object Storage cho file trọng số mô hình và tập dữ liệu thô.

### Ứng dụng thực tế
✅ Tự động hóa chấm thi nhanh chóng cho các trường học, cơ sở giáo dục.
✅ Xử lý bài thi trắc nghiệm quy mô lớn với độ ổn định cao.
✅ Cải thiện độ chính xác liên tục nhờ vòng lặp phản hồi từ con người. 

---

## 🏗️ Kiến trúc Hệ thống

```text
┌─────────────────────────────────────────────────────────────┐
│                    System Architecture                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │  Streamlit   │  │   FastAPI    │  │ Monitoring   │       │
│  │  Frontend    │  │   Backend    │  │ Prometheus   │       │
│  │    :8888     │  │    :8001     │  │    :9090     │       │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘       │
│         │                 │                 │               │
│         └─────────────────┼─────────────────┘               │
│                           │                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │ PostgreSQL   │  │    MinIO     │  │    MLflow    │       │
│  │    :5432     │  │    :9000     │  │    :5000     │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │ Retrain      │  │   Grafana    │  │ Node Exporter│       │
│  │ Worker       │  │    :3000     │  │    :9100     │       │
│  │ DVC+MLflow   │  └──────────────┘  └──────────────┘       │
│  └──────────────┘                                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```


---

## 📁 Cấu trúc Dự án

```text
mlops-omr-grading/
│
├── src/                          # Mã nguồn ML cốt lõi
│   ├── train.py                  # Script huấn luyện mô hình PyTorch
│   ├── evaluate.py               # Kiểm định & tự động deploy lên MLflow
│   ├── crop_bubbles.py           # Tiền xử lý và cắt lọc ô trắc nghiệm
│   ├── split_data.py             # Chia tập dữ liệu (Train/Val/Test)
│   └── generate_template_json.py # Khởi tạo tọa độ khuôn mẫu form thi
│
├── monitoring/                   # Cấu hình Giám sát & Cảnh báo
│   ├── Dockerfile
│   ├── monitor.py                # Bot theo dõi API & bắn cảnh báo Discord
│   ├── prometheus.yml
│   └── grafana/
│       └── dashboards/
│
├── retrain_worker/               # Vòng lặp huấn luyện liên tục
│   ├── Dockerfile
│   └── worker.py                 # Bot kiểm tra dữ liệu mới và chạy DVC Pipeline
│
├── config/
│   └── config.yaml               # Cấu hình siêu tham số mô hình
│
├── data/
│   ├── synthetic_images/         # Ảnh phiếu thi đầu vào
│   ├── bubbles/                  # Kho ảnh nhị phân đã cắt (32x32)
│   ├── processed/                # File chỉ mục CSV đã phân chia
│   ├── template_config.json      # Tọa độ gốc của 160 ô trắc nghiệm
│   ├── labels.csv                # Nhãn chuẩn (Ground truth) gốc
│   └── feedback_labels.csv       # Nhãn chuẩn do người dùng phản hồi
│
├── main.py                       # Ứng dụng API lõi (FastAPI)
├── app_api.py                    # Giao diện người dùng (Streamlit)
├── generate_data.py              # Script tạo dữ liệu ảo ban đầu
│
├── Dockerfile                    # Container chính của API
├── docker-compose.yml            # Điều phối toàn bộ hạ tầng Microservices
│
├── requirements.txt
├── requirements-lock.txt
│
├── dvc.yaml                      # Cấu hình DAG Pipeline của DVC
├── dvc.lock
├── .dvc/
│
├── .env.example
├── .gitignore
└── README.md
```

## Access the web interfaces


| Service | URL | Purpose |
|----------|----------|----------|
| 🎨 Streamlit Demo | http://localhost:8888 | Giao diện kiểm thử OMR & Gửi phản hồi |
| ⚡ FastAPI Docs | http://localhost:8001/docs | Tài liệu API để lập trình viên sử dụng |
| 📊 Grafana | http://localhost:3000 | Bảng điều khiển trực quan hóa hệ thống |
| 📈 Prometheus | http://localhost:9090 | Thu thập dữ liệu chuỗi thời gian |
| 🧪 MLflow | http://localhost:5000 | Sổ theo dõi thử nghiệm & Kho chứa mô hình |
| 🗂️ MinIO | http://localhost:9001 | Lưu trữ tài nguyên (Data, File trọng số .pth) |


## Khởi động Hệ thống

Để chạy toàn bộ hệ sinh thái MLOps này, đảm bảo bạn đã cài đặt Docker và Docker Compose, sau đó chạy các lệnh sau:
docker-compose down
docker-compose up -d --build
