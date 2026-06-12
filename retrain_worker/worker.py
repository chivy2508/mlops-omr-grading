import schedule
import time
import os
import glob
import requests
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.integrate_feedback import process_and_trigger_pipeline

DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")
API_RELOAD_URL = os.getenv("API_RELOAD_URL", "http://omr_api:8000/reload-model")
RETRAIN_DIR = "./retrain_dataset"
RETRAIN_THRESHOLD = 50

def send_discord_alert(message: str):
    if not DISCORD_WEBHOOK_URL:
        return
    try:
        requests.post(DISCORD_WEBHOOK_URL, json={"content": message}, timeout=3)
    except requests.Timeout:
        print(f"Discord timeout: {message}")
    except Exception as e:
        print(f"Discord error: {e}")

def retrain_job():
    ready_samples = len(glob.glob(os.path.join(RETRAIN_DIR, "*_label.json")))
    
    if ready_samples >= RETRAIN_THRESHOLD:
        print(f"🚀 Retrain job started with {ready_samples} new samples...")
        send_discord_alert(f"⏳ **Bắt đầu Retrain Pipeline ({ready_samples} mẫu mới)...**")
        
        try:
            success = process_and_trigger_pipeline()
            
            if success:
                send_discord_alert("✅ **Retrain hoàn tất! Đang báo API tải lại Model...**")
                try:
                    requests.post(API_RELOAD_URL, timeout=20)
                    send_discord_alert("🔄 **Model mới nhất đã được áp dụng vào API (Hot-Reload)!**")
                except requests.exceptions.Timeout:
                    send_discord_alert("⚠️ **API đang tải model nhưng phản hồi chậm (Timeout).**")
                except Exception as e:
                    send_discord_alert(f"⚠️ **Lỗi kết nối khi gọi API reload model:** {e}")
                    
        except Exception as e:
            print(f"❌ Retrain failed: {e}")
            send_discord_alert(f"🚨 **Tiến trình Retrain bị hủy hoặc gặp lỗi:** {e}")

# Chạy kiểm tra mỗi 1 giờ
schedule.every(1).hours.do(retrain_job)

if __name__ == "__main__":
    print("🚀 Retrain Worker started and waiting...")
    while True:
        schedule.run_pending()
        time.sleep(60)