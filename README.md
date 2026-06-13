# 📝 MLOps OMR Grading Engine

## Giới thiệu Dự án

- **Topic:** Hệ thống MLOps cho chấm bài OMR tự động.
- **Mục tiêu:** Xây dựng giải pháp hoàn chỉnh từ thu thập dữ liệu, huấn luyện, deploy API, giám sát, log tập trung và tự động retrain.
- **Công nghệ chính:** Docker Compose V2, FastAPI, Streamlit, MLflow, DVC, Prometheus, Grafana, Loki/Promtail, MinIO.

## Giới thiệu Thành viên

- Nguyễn Chí Vỹ - 23521829
- Nguyễn Hà Công Toàn - 23521607
- Vũ Thanh Sơn - 23521365
- Đỗ Lê Duy Tín - 23521592

## 🎯 Tổng quan

Dự án xây dựng một hệ thống chấm bài OMR MLOps đầy đủ bao gồm:

- Nhận dạng và phân tích phiếu thi trắc nghiệm tự động.
- API FastAPI để phục vụ suy luận và nhận phản hồi người dùng.
- Streamlit frontend để kiểm thử, upload ảnh và xem kết quả.
- Worker thu thập feedback, tái huấn luyện và cập nhật mô hình tự động.
- Dashboard giám sát với Metrics, Logs và cảnh báo.

## 🌟 PHẦN BỔ SUNG CHO HỌC PHẦN THỰC HÀNH

Nhằm đáp ứng toàn diện vòng đời MLOps trên môi trường thực tế, nhóm đã bổ sung và điều chỉnh các thành phần sau:

1. **☁️ Cloud Deployment (CI/CD to Cloud):** Chuyển đổi hạ tầng từ Localhost sang máy chủ đám mây **Google Cloud VM**. Tích hợp GitHub Actions self-hosted runner để tự động deploy hệ thống không gián đoạn.
2. **📜 Centralized Logging:** Tích hợp thêm stack **Grafana Loki & Promtail** vào hệ sinh thái Docker Compose, giúp thu thập, truy vấn và quản lý toàn bộ logs của các container tại một giao diện duy nhất trên Grafana.
3. **🧪 Automated Testing:** Xây dựng bộ Unit Test với `pytest` để kiểm thử độ ổn định của API (Data validation, Error handling, Timeout) trước khi deploy.

### Tiêu chí nổi bật cho phần thực hành

- **Hệ thống MLOps OMR Grading:** Pipeline 4 bước (Preprocess, Train, Val, Eval) hoàn thiện 100% thông qua cấu trúc DAG 5 bước của DVC (`dvc.yaml`).
- **Experiment Tracking:** Triển khai **MLflow Tracking** Server nội bộ, tự động log các tham số (Hyperparams), Precision, Recall và lưu Artifact trọng số.
- **Dễ thay đổi Input/Training:** File cấu hình tập trung `config.yaml` và cơ chế caching của DVC giúp chạy lại tập dữ liệu mới chỉ trong vài giây.
- **Hyper-parameter Tuning (Tiêu chí phụ):** Ứng dụng thuật toán **ReduceLROnPlateau** để tự động tinh chỉnh learning rate dựa trên Validation Loss, chống over-fitting.
- **Serving API (FastAPI) & Docker Compose:** Hoàn thiện 100%. Đóng gói toàn bộ 12 vi dịch vụ bằng Docker Compose, thiết lập API async, resource limits và auto-healing.
- **Deploy trên Server/Cloud (Tiêu chí phụ):** Đã triển khai thành công toàn bộ hệ thống lên **Google Cloud Platform (Compute Engine)**.
- **Thêm Logging & Monitoring Stack:** Stack giám sát gồm **Prometheus** (Metrics), **Grafana** (Visualization), **Node Exporter** (Hardware), và **Loki/Promtail** (Logging).
- **Server, Model, API, App, System Metrics/Logs:** Giám sát CPU/RAM (Node Exporter), độ trễ API (Prometheus Histogram), Data Drift Score (Model), log hệ thống và API (Loki).
- **Setup Alerting:** Viết bot Python ngầm và Alertmanager bắn thẳng Webhook cảnh báo về kênh quản trị **Discord** khi API sập, data mờ hoặc phần cứng quá tải.
- **Build Automatic Retrain Pipeline:** Xây dựng tiến trình `worker.py` gom dữ liệu phản hồi (Feedback Loop). Khi đủ 50 mẫu, tự động kích hoạt DVC, so sánh MLflow Model Registry và nâng cấp hot-reload.
- **Hướng dẫn cài đặt & Kiểm thử (Testing):** Cung cấp đầy đủ hướng dẫn setup. Bổ sung bộ **Unit Test với Pytest** rà soát luồng API.

---

## 🚀 Kiến trúc Hệ thống

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
├── docker-compose.yml            # Docker Compose file (Compose V2 compatible)
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

## 🔧 Phiên bản và Yêu cầu môi trường

- Python: **3.10+**
- Docker: **23+** thuần, tương thích `docker compose`
- Docker Compose: **V2** (plugin `docker compose`)
- PyTorch: bản cài từ `requirements.txt` qua `--extra-index-url https://download.pytorch.org/whl/cpu`
- Prometheus/grafana/loki/promtail: dùng image chính thức gần nhất (`prom/prometheus`, `grafana/grafana`, `grafana/loki`, `grafana/promtail`)
- GitHub Actions: CI/CD lint, scan, self-hosted deploy
- pytest: để chạy unit test API

### Yêu cầu phần cứng tối thiểu (Local)

- CPU: **2 cores**
- RAM: **8 GB**
- Disk: **20 GB**
- Mạng: kết nối Internet để kéo image và tải packages

### Đề xuất GCP cho triển khai học phần

- **Machine type:** `e2-standard-2`
- **CPU:** 2 vCPU
- **RAM:** 8 GB
- **OS:** Ubuntu 22.04 LTS
- **Docker:** Docker Engine mới và Compose V2

---

## 🌐 Ports và dịch vụ

| Service | Port | Notes |
|---------|------|-------|
| PostgreSQL | 5432 | Database cho MLflow và metadata |
| MinIO | 9000 | Object storage |
| MinIO Console | 9001 | MinIO quản trị |
| MLflow | 5000 | Experiment tracking |
| FastAPI | 8001 | API chính |
| Streamlit | 8888 | Demo UI |
| Prometheus | 9090 | Metrics |
| Grafana | 3000 | Visualization |
| Loki | 3100 | Centralized logs |
| Node Exporter | 9100 | Host metrics |

---

## 🌟 Tính năng chính

- **DVC Pipeline**: Quản lý dữ liệu, training, validation, evaluation qua `dvc.yaml`.
- **MLflow**: Lưu trữ tham số, chỉ số, và artifact mô hình.
- **Docker Compose V2**: Điều phối 12 service, build local, và phối hợp monitor/log.
- **Loki/Promtail**: Log tập trung, hỗ trợ truy vấn log trên Grafana.
- **Alerting Discord**: Bot monitor gửi cảnh báo khi dịch vụ offline hoặc quá tải.
- **Feedback Loop**: `worker.py` gom dữ liệu phản hồi và kích hoạt retrain khi đủ mẫu.
- **Unit Test**: Dùng `pytest` để kiểm thử API trước deploy.

---

## 💾 Cụm service chính

- `omr_api`: FastAPI backend, port `8001`
- `omr_api_demo`: Streamlit frontend, port `8888`
- `monitor`: monitor bot, `monitoring/Dockerfile`
- `retrain_worker`: worker tự động retrain
- `minio`: S3-compatible storage
- `mlflow`: tracking server
- `postgres`: metadata database
- `grafana`: dashboard visualization
- `prometheus`: metrics collection
- `node-exporter`: hardware metrics
- `loki`: logs storage
- `promtail`: log shipping

---

## 📥 Cài đặt và Chạy

### 1. Local setup

```bash
git clone https://github.com/chivy2508/mlops-omr-grading.git
cd mlops-omr-grading

# Khởi động toàn bộ hệ thống
docker compose up -d --build

# Kiểm tra
docker compose ps
```

### 2. Dừng hệ thống

```bash
docker compose down
```

### 3. Chạy unit test

```bash
pytest -q
```


---

## 📌 Ghi chú

- `config/config.yaml` chứa cấu hình model và tham số.
- `dvc.yaml` là DAG pipeline, gồm các bước preprocess, train, val, eval.
- `monitoring/prometheus.yml` và `monitoring/grafana/provisioning` cấu hình giám sát.
- `monitoring/loki-config.yaml` và `monitoring/promtail-config.yml` quản lý logging tập trung.

---

## 🧠 Tổ chức triển khai Cloud

- Triển khai lên **Google Cloud VM** với cấu hình `e2-standard-2`.
- Dùng GitHub Actions để pull code và chạy `docker compose` trên self-hosted runner.

---

## 🌱 Lưu ý về Phiên bản

- `python` nên dùng **3.10+**
- `docker` nên dùng bản **24.x** hoặc tương đương
- `docker compose` sử dụng plugin V2
- `pytest` dùng để kiểm thử tự động
- `prometheus-client` cố định `0.17.1` trong requirements

---

## 🧾 Kết luận

Project này là một hệ thống MLOps End-to-End cho OMR Grading, từ training và versioning đến deploy, giám sát, logging và retrain tự động.

Mỗi thành viên đã đóng góp vào cả phần data pipeline, model, API, monitoring và CI/CD để hoàn thiện bộ đồ án thực hành.
