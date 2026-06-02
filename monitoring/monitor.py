import time
import requests
import os
import random
import subprocess
import glob

# === CẤU HÌNH ===
RETRAIN_THRESHOLD = 500
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
        r = requests.get("http://omr_engine:8000/docs", timeout=5)
        return True if r.status_code == 200 else False
    except:
        return False

def check_and_trigger():
    """Hàm kiểm tra kho data và kích hoạt Retrain Pipeline"""
    # Đếm cặp file hoàn chỉnh (có cả JSON) thay vì chỉ đếm ảnh
    ready_samples = len(glob.glob(os.path.join(RETRAIN_DIR, "*_label.json")))
    
    if ready_samples > 0:
        print(f"[KIỂM TRA DATA] Kho đang có {ready_samples}/{RETRAIN_THRESHOLD} mẫu sẵn sàng...", flush=True)
    
    if ready_samples >= RETRAIN_THRESHOLD:
        msg = f"🚀 Đã gom đủ {ready_samples} mẫu Ground Truth từ khách hàng. Bắt đầu kích hoạt Pipeline MLOps!"
        print(msg, flush=True)
        send_discord(msg, color=3447003) # Màu xanh dương báo hiệu Pipeline
        
        # Kéo script gom data và bắn lên DVC
        result = subprocess.run(["python", "src/integrate_feedback.py"])
        
        if result.returncode == 0:
            print("✅ DVC Pipeline hoàn tất. Kích hoạt MLflow...", flush=True)
            send_discord("⏳ **Dữ liệu chuẩn đã lên nòng.** Đang tiến hành Retrain mô hình...", color=16776960) # Màu vàng chờ đợi
            
            # Bắt đầu Train và "nghe ngóng" kết quả (capture_output)
            train_result = subprocess.run(["mlflow", "run", "."], capture_output=True, text=True)
            
            if train_result.returncode == 0:
                # Nếu train mượt mà
                send_discord("✅ **Retrain thành công xuất sắc!** Mô hình mới đã được cập nhật.", color=65280)
            else:
                # Nếu quá trình train bị lỗi (Tràn RAM, sai shape tensor, v.v.)
                # Trích xuất 500 ký tự cuối cùng của log lỗi để gửi lên Discord
                error_log = train_result.stderr[-500:] if train_result.stderr else "Lỗi không xác định."
                send_discord(f"🚨 **Quá trình Retrain thất bại!**\nChi tiết lỗi:\n```text\n{error_log}\n```", color=16711680)
                
        else:
            # Lỗi ở khâu trộn Data hoặc đẩy DVC
            send_discord("❌ **Lỗi ở khâu chuẩn bị Dữ liệu/DVC!** Vui lòng kiểm tra lại data đầu vào.", color=16711680)

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
        
        # 1. LOGIC BÁO ĐỘNG SỐNG/CHẾT (Chỉ báo khi trạng thái thay đổi)
        if current_health != previous_health_status:
            if current_health == True:
                # API từ trạng thái Chết -> Sống lại
                send_discord("🟩 **HỆ THỐNG PHỤC HỒI:** API OMR đã hoạt động trở lại! Mọi thứ đang chạy bình thường.", color=65280)
            else:
                # API từ trạng thái Sống -> Lăn ra chết
                send_discord("🚨 **CẢNH BÁO MẤT KẾT NỐI:** API OMR vừa bị sập hoặc không phản hồi! Vui lòng kiểm tra Docker ngay lập tức.", color=16711680)
            
            # Cập nhật lại bộ nhớ của Bot
            previous_health_status = current_health

        # 2. LOGIC ĐI TUẦN TRA & GIẢ LẬP
        if current_health:
            # Chỉ đi tuần tra retrain và bắn data giả khi API đang sống
            check_and_trigger()
            simulate_traffic()
        else:
            print("[WARN] API đang sập, tạm dừng các hoạt động tuần tra...", flush=True)
            
        time.sleep(30)