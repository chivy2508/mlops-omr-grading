import time
import requests
import numpy as np
import cv2
import tempfile
import os
import shutil
from prometheus_client import start_http_server, Counter, Histogram, Gauge

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

API_URL = "http://omr_engine:8000/predict"
ANSWER_COUNTERS = {'A': ANSWER_A, 'B': ANSWER_B, 'C': ANSWER_C, 'D': ANSWER_D}
REFERENCE_DIST = {'A': 0.25, 'B': 0.25, 'C': 0.25, 'D': 0.25}
answer_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
total_predictions = 0

DRIFT_THRESHOLD = 0.15
RETRAIN_THRESHOLD = 50
RETRAIN_DIR = "/app/retrain_queue"
os.makedirs(RETRAIN_DIR, exist_ok=True)

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

def get_retrain_queue_size():
    return len([f for f in os.listdir(RETRAIN_DIR) if f.endswith('.jpg')])

def save_to_retrain_queue(img, filename):
    path = os.path.join(RETRAIN_DIR, filename)
    cv2.imwrite(path, img)
    size = get_retrain_queue_size()
    RETRAIN_QUEUE.set(size)
    print(f"[QUEUE] Luu anh vao retrain queue: {filename} | Tong: {size}", flush=True)
    if size >= RETRAIN_THRESHOLD:
        print(f"[RETRAIN] Du {RETRAIN_THRESHOLD} anh! Can retrain model!", flush=True)
        print(f"[RETRAIN] Chay: cd /app && dvc add data/retrain_queue && dvc push", flush=True)

def simulate_and_monitor():
    global total_predictions
    img = np.random.randint(200, 255, (1200, 800), dtype=np.uint8)
    tmp = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    cv2.imwrite(tmp.name, img)
    start = time.time()
    try:
        with open(tmp.name, 'rb') as f:
            response = requests.post(API_URL, files={'file': ('test.jpg', f, 'image/jpeg')}, timeout=30)
        elapsed = time.time() - start
        RESPONSE_TIME.observe(elapsed)
        TOTAL_REQUESTS.inc()
        if response.status_code == 200:
            SUCCESS_REQUESTS.inc()
            data = response.json()
            answers = data.get('ket_qua_cham_diem', [])
            for ans in answers:
                if ans in ANSWER_COUNTERS:
                    ANSWER_COUNTERS[ans].inc()
                    answer_counts[ans] += 1
                    total_predictions += 1
            drift = calculate_drift()
            DRIFT_SCORE.set(drift)
            print(f"[OK] {len(answers)} dap an | drift={drift:.3f} | time={elapsed:.2f}s", flush=True)

            # Neu drift cao thi luu anh vao retrain queue
            if drift > DRIFT_THRESHOLD:
                filename = f"drift_{int(time.time())}.jpg"
                save_to_retrain_queue(img, filename)
        else:
            FAILED_REQUESTS.inc()
            print(f"[FAIL] Status: {response.status_code}", flush=True)
    except Exception as e:
        FAILED_REQUESTS.inc()
        TOTAL_REQUESTS.inc()
        print(f"[ERROR] {e}", flush=True)
    finally:
        os.unlink(tmp.name)

if __name__ == '__main__':
    port = 8890
    print(f"=== Khoi dong Monitoring Service tai port {port} ===", flush=True)
    print(f"=== Drift threshold: {DRIFT_THRESHOLD} | Retrain threshold: {RETRAIN_THRESHOLD} anh ===", flush=True)
    start_http_server(port)
    while True:
        health = check_api_health()
        API_HEALTH.set(health)
        RETRAIN_QUEUE.set(get_retrain_queue_size())
        print(f"[HEALTH] API={'OK' if health else 'DOWN'}", flush=True)
        if health:
            simulate_and_monitor()
        else:
            print("[WARN] API chua san sang, cho...", flush=True)
        time.sleep(30)
