import time
import requests
import os
import random

RETRAIN_THRESHOLD = 10
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")
API_URL = "http://omr_api:8000/predict"
RETRAIN_DIR = "./retrain_dataset"
DEMO_IMAGES_DIR = "./demo_images"

os.makedirs(RETRAIN_DIR, exist_ok=True)

def send_discord(message, color=16711680):
    try:
        requests.post(DISCORD_WEBHOOK_URL, json={
            "embeds": [{"title": "OMR Bot", "description": message, "color": color}]
        }, timeout=5)
    except Exception as e:
        print(f"[DISCORD ERROR] {e}", flush=True)

def check_api_health():
    try:
        r = requests.get("http://omr_api:8000/docs", timeout=5)
        return True if r.status_code == 200 else False
    except:
        return False

def simulate_traffic():
    """Bắn ảnh để duy trì biểu đồ (Không đụng chạm logic Drift nữa)"""
    try:
        available_images = [f for f in os.listdir(DEMO_IMAGES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not available_images:
            return
            
        random_image_name = random.choice(available_images)
        image_path = os.path.join(DEMO_IMAGES_DIR, random_image_name)
        
        with open(image_path, 'rb') as f:
            requests.post(API_URL, files={'file': (random_image_name, f, 'image/jpeg')}, timeout=10)
    except Exception:
        pass

if __name__ == '__main__':
    print("=== Bot Giám Sát OMR Đã Khởi Động ===", flush=True)
    send_discord("✅ **Bot Giám Sát đã thức dậy!** Đang theo dõi kho data để Retrain...", color=65280)
    
    previous_health_status = True 
    
    while True:
        current_health = check_api_health()
        
        if current_health != previous_health_status:
            if current_health == True:
                send_discord("🟩 **HỆ THỐNG PHỤC HỒI:** API OMR đã hoạt động trở lại! Mọi thứ đang chạy bình thường.", color=65280)
            else:
                send_discord("🚨 **CẢNH BÁO MẤT KẾT NỐI:** API OMR vừa bị sập hoặc không phản hồi! Vui lòng kiểm tra Docker ngay lập tức.", color=16711680)
            
            previous_health_status = current_health

        if current_health:
            simulate_traffic()
        else:
            print("[WARN] API đang sập, tạm dừng các hoạt động tuần tra...", flush=True)
            
        time.sleep(30)