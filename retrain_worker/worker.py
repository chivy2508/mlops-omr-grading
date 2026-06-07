import schedule
import time
import os
import glob
import requests
from src.integrate_feedback import process_and_trigger_pipeline

DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")
RETRAIN_DIR = "./retrain_dataset"
RETRAIN_THRESHOLD = 10

def send_discord_alert(message: str):
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
                send_discord_alert("✅ **Retrain thành công! Đang báo API tải lại Model...**")
                try:
                    # Bắn tín hiệu sang API container yêu cầu reload
                    requests.post("http://omr_api:8000/reload-model", timeout=10)
                    send_discord_alert("🔄 **Model mới nhất trên Cloud đã được áp dụng vào API!**")
                except Exception as e:
                    send_discord_alert(f"⚠️ **Không thể gọi API reload model:** {e}")
        except Exception as e:
            print(f"❌ Retrain failed: {e}")
            send_discord_alert(f"🚨 **Lỗi Retrain Pipeline:** {e}")

# Chạy kiểm tra mỗi 1 giờ
schedule.every(1).hours.do(retrain_job)

if __name__ == "__main__":
    print("🚀 Retrain Worker started and waiting...")
    while True:
        schedule.run_pending()
        time.sleep(60)