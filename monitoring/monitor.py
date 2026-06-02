import time
import requests
import numpy as np
import cv2
import tempfile
import os
from prometheus_client import start_http_server, Counter, Histogram, Gauge
import random
import shutil

# === CẤU HÌNH ===
DRIFT_THRESHOLD = 0.15
RETRAIN_THRESHOLD = 50
DISCORD_WEBHOOK = "https://discordapp.com/api/webhooks/1510835063229513749/Q9IL2feBSAqrteKmzKIOQCmp8e3BYH7ABjQW27so7uQDrNFKaGk4pKA0rqudMSGKMvyo"
API_URL = "http://omr_api:8000/predict"

RETRAIN_DIR = "/app/retrain_queue"
os.makedirs(RETRAIN_DIR, exist_ok=True)

DEMO_IMAGES_DIR = "./demo_images"

# === METRICS ===
TOTAL_REQUESTS = Counter('omr_total_requests', 'Tong so request')
SUCCESS_REQUESTS = Counter('omr_success_requests', 'So request thanh cong')
FAILED_REQUESTS = Counter('omr_failed_requests', 'So request that bai')
RESPONSE_TIME = Histogram('omr_response_time_seconds', 'Thoi gian xu ly')
ANSWER_A = Counter('omr_answer_A', 'So lan dap an A')
ANSWER_B = Counter('omr_answer_B', 'So lan dap an B')
ANSWER_C = Counter('omr_answer_C', 'So lan dap an C')
ANSWER_D = Counter('omr_answer_D', 'So lan dap an D')
DRIFT_SCORE = Gauge('omr_drift_score', 'Data drift score')
API_HEALTH = Gauge('omr_api_health', 'API health')
RETRAIN_QUEUE = Gauge('omr_retrain_queue_size', 'So anh cho retrain')
BAD_DATA_COUNT = Gauge('omr_bad_data_count', 'So anh bi loai')

ANSWER_COUNTERS = {'A': ANSWER_A, 'B': ANSWER_B, 'C': ANSWER_C, 'D': ANSWER_D}
REFERENCE_DIST = {'A': 0.25, 'B': 0.25, 'C': 0.25, 'D': 0.25}
answer_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
total_predictions = 0

def send_discord(message, color=16711680):
    try:
        requests.post(DISCORD_WEBHOOK, json={
            "embeds": [{"title": "OMR Monitoring Alert", "description": message, "color": color}]
        }, timeout=5)
    except Exception as e:
        print(f"[DISCORD ERROR] {e}", flush=True)

def calculate_drift():
    if total_predictions == 0:
        return 0.0
    current_dist = {k: v / total_predictions for k, v in answer_counts.items()}
    return sum(abs(current_dist[k] - REFERENCE_DIST[k]) for k in REFERENCE_DIST) / 2

def check_api_health():
    try:
        r = requests.get("http://omr_engine:8000/docs", timeout=5)
        return 1 if r.status_code == 200 else 0
    except:
        return 0

def get_queue_size(folder):
    return len([f for f in os.listdir(folder) if f.endswith('.jpg')])

def simulate_and_monitor():
    global total_predictions

    # 1. Lấy danh sách tất cả file ảnh trong thư mục
    try:
        available_images = [f for f in os.listdir(DEMO_IMAGES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not available_images:
            print("[WARN] Thư mục demo_images đang trống! Vui lòng thêm ảnh.", flush=True)
            return
            
        # 2. Bốc ngẫu nhiên 1 file ảnh
        random_image_name = random.choice(available_images)
        image_path = os.path.join(DEMO_IMAGES_DIR, random_image_name)
    except FileNotFoundError:
        print(f"[ERROR] Không tìm thấy thư mục {DEMO_IMAGES_DIR}", flush=True)
        return

    start = time.time()
    try:
        # 3. Gửi thẳng file ảnh thật sang FastAPI
        with open(image_path, 'rb') as f:
            response = requests.post(API_URL, files={'file': (random_image_name, f, 'image/jpeg')}, timeout=30)
            
        elapsed = time.time() - start
        RESPONSE_TIME.observe(elapsed)
        TOTAL_REQUESTS.inc()

        if response.status_code == 200:
            SUCCESS_REQUESTS.inc()
            data = response.json()
            
            # Kiểm tra xem API có trả về trạng thái từ chối không (VD: ảnh thật nhưng bị nhòe)
            if data.get("trang_thai") == "từ chối":
                print(f"[REJECTED] File {random_image_name} bị từ chối: {data.get('chi_tiet')}", flush=True)
                return

            answers = data.get('ket_qua_cham_diem', [])
            for ans in answers:
                if ans in ANSWER_COUNTERS:
                    ANSWER_COUNTERS[ans].inc()
                    answer_counts[ans] += 1
                    total_predictions += 1

            drift = calculate_drift()
            DRIFT_SCORE.set(drift)
            print(f"[OK] {random_image_name} -> {len(answers)} đáp án | drift={drift:.3f} | time={elapsed:.2f}s", flush=True)

            if drift > DRIFT_THRESHOLD:
                # Nếu drift cao, copy ảnh thật này sang thư mục chờ retrain
                
                filename = f"drift_{int(time.time())}.jpg"
                shutil.copy(image_path, os.path.join(RETRAIN_DIR, filename))
                
                queue_size = get_queue_size(RETRAIN_DIR)
                RETRAIN_QUEUE.set(queue_size)
                print(f"[QUEUE] Đã lưu ảnh retrain: {filename} | Tổng: {queue_size}", flush=True)
                
                send_discord(
                    f"🔴 **Drift cao!**\nĐã đưa `{random_image_name}` vào kho Retrain.\nDrift: {drift:.3f} | Queue: {queue_size}/{RETRAIN_THRESHOLD}",
                    color=16711680
                )
                
        else:
            FAILED_REQUESTS.inc()
            print(f"[FAIL] Status: {response.status_code}", flush=True)
            
    except Exception as e:
        FAILED_REQUESTS.inc()
        TOTAL_REQUESTS.inc()
        print(f"[ERROR] {e}", flush=True)

if __name__ == '__main__':
    port = 8890
    print(f"=== Khoi dong Monitoring Service tai port {port} ===", flush=True)
    start_http_server(port)
    send_discord("✅ **Monitoring Service khởi động thành công!**\nĐang theo dõi model OMR...", color=65280)
    while True:
        health = check_api_health()
        API_HEALTH.set(health)
        RETRAIN_QUEUE.set(get_queue_size(RETRAIN_DIR))
        print(f"[HEALTH] API={'OK' if health else 'DOWN'}", flush=True)
        if health:
            simulate_and_monitor()
        else:
            print("[WARN] API chua san sang, cho...", flush=True)
        time.sleep(30)
