from fastapi import FastAPI, File, UploadFile
import mlflow
import os
import cv2
import numpy as np
import torch
import requests
from prometheus_client import make_asgi_app, Counter, Gauge

# ==========================================
# 1. CẤU HÌNH MLFLOW & MINIO 
# ==========================================
os.environ["AWS_ACCESS_KEY_ID"] = "admin"
os.environ["AWS_SECRET_ACCESS_KEY"] = "password123"

os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://minio:9000"
mlflow.set_tracking_uri("http://mlflow:5000")

app = FastAPI(title="OMR Grading Engine API")

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# --- 3. ĐỊNH NGHĨA QUYỂN SỔ GHI CHÉP (METRICS) ---
TOTAL_REQUESTS = Counter('omr_total_requests', 'Tong so request')
SUCCESS_REQUESTS = Counter('omr_success_requests', 'So request thanh cong')
FAILED_REQUESTS = Counter('omr_failed_requests', 'So request that bai')
ANSWER_A = Counter('omr_answer_A', 'So lan dap an A')
ANSWER_B = Counter('omr_answer_B', 'So lan dap an B')
ANSWER_C = Counter('omr_answer_C', 'So lan dap an C')
ANSWER_D = Counter('omr_answer_D', 'So lan dap an D')
DRIFT_SCORE = Gauge('omr_drift_score', 'Data drift score')

ANSWER_COUNTERS = {'A': ANSWER_A, 'B': ANSWER_B, 'C': ANSWER_C, 'D': ANSWER_D}
REFERENCE_DIST = {'A': 0.25, 'B': 0.25, 'C': 0.25, 'D': 0.25}
answer_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
total_predictions = 0

def update_drift_score():
    """Hàm tự động tính toán lại độ lệch khi có đáp án mới"""
    global total_predictions
    if total_predictions == 0:
        return
    current_dist = {k: v / total_predictions for k, v in answer_counts.items()}
    drift = sum(abs(current_dist[k] - REFERENCE_DIST[k]) for k in REFERENCE_DIST) / 2
    DRIFT_SCORE.set(drift)

def send_discord_alert(message: str):
    """Gửi cảnh báo rác về Discord"""
    try:
        requests.post(DISCORD_WEBHOOK_URL, json={"content": message}, timeout=3)
    except:
        pass

# ==========================================
# 2. TẢI MÔ HÌNH TỰ ĐỘNG TỪ MLFLOW
# ==========================================
print("🔄 Đang tải mô hình OMR_Grading_Engine từ hệ thống...")
MODEL_URI = "models:/OMR_Grading_Engine@production"

try:
    model = mlflow.pytorch.load_model(MODEL_URI)
    model.eval()
    print(" Nạp mô hình thành công!")
except Exception as e:
    print(f" Lỗi tải mô hình: {e}")

def check_blur_and_brightness(image_gray: np.ndarray):
    laplacian_var = cv2.Laplacian(image_gray, cv2.CV_64F).var()
    if laplacian_var < 30:  
        return False, f"Ảnh quá mờ (độ nét: {laplacian_var:.1f})"      

    mean_brightness = np.mean(image_gray)
    if mean_brightness < 30: 
        return False, f"Ảnh quá tối (độ sáng: {mean_brightness:.1f})"
    if mean_brightness > 245: 
        return False, f"Ảnh quá sáng/chói (độ sáng: {mean_brightness:.1f})"
    return True, "OK"



def get_aligned_paper(image: np.ndarray, target_w: int = 800, target_h: int = 1200):
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    edged = cv2.Canny(blur, 75, 200)
    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None 
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
    paper_contour = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            paper_contour = approx
            break
    if paper_contour is None: return None 
    rect = np.zeros((4, 2), dtype="float32")
    pts = paper_contour.reshape(4, 2)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]      
    rect[2] = pts[np.argmax(s)]       
    rect[1] = pts[np.argmin(diff)]    
    rect[3] = pts[np.argmax(diff)]    
    dst = np.array([[0, 0], [target_w - 1, 0], [target_w - 1, target_h - 1], [0, target_h - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    aligned = cv2.warpPerspective(image, M, (target_w, target_h))
    return aligned



def clean_and_binarize(aligned_image: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(aligned_image, (3, 3), 0)
    cleaned = cv2.adaptiveThreshold(

        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 

        cv2.THRESH_BINARY, 11, 2
    )
    return cleaned

# ==========================================
# 3. GIAO THỨC API
# ==========================================
@app.post("/predict")
async def predict_omr(file: UploadFile = File(...)):
    TOTAL_REQUESTS.inc()
    if model is None:
        return {"trang_thai": "lỗi", "chi_tiet": "Mô hình chưa được nạp, vui lòng kiểm tra server"}
        
    try:
        image_bytes = await file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        
        # 1. Cắt nắn khung giấy (Kiêm luôn bộ lọc rác)
        is_clear, reason_clear = check_blur_and_brightness(image)
        if not is_clear:
            send_discord_alert(f"🚨 Từ chối `{file.filename}`: {reason_clear}")
            return {"trang_thai": "từ chối", "chi_tiet": reason_clear}
            
        # --- CHỐT CHẶN 2: Có phải giấy thi không? ---
        aligned_img = get_aligned_paper(image, target_w=800, target_h=1200)
        if aligned_img is None:
            reason_paper = "Không tìm thấy tờ phiếu thi hợp lệ."
            send_discord_alert(f"🚨 Từ chối `{file.filename}`: {reason_paper}")
            return {"trang_thai": "từ chối", "chi_tiet": reason_paper}
        
        if aligned_img is None:
            # 🚨 BẮN CẢNH BÁO DISCORD Ở ĐÂY
            reason = "Không tìm thấy tờ phiếu thi hợp lệ."
            send_discord_alert(f"🚨 Đã từ chối file `{file.filename}`. Lý do: {reason}")
            return {"trang_thai": "từ chối", "chi_tiet": f"{reason} Chụp lại rõ hơn, đủ 4 góc, dưới ánh sáng đều."}
            
        # 2. Làm sạch nền
        final_processed_image = clean_and_binarize(aligned_img)
        
        # 3. Chuẩn bị tensor cho PyTorch
        input_image = cv2.resize(final_processed_image, (800, 1200))
        input_image = input_image.astype(np.float32) / 255.0
        input_tensor = torch.tensor(input_image).unsqueeze(0).unsqueeze(0)
        
        # 4. Dự đoán
        with torch.no_grad(): 
            result_tensor = model(input_tensor)
        
        predictions = torch.argmax(result_tensor, dim=2)[0] 
        label_map_reverse = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
        final_answers = [label_map_reverse[ans.item()] for ans in predictions]

        SUCCESS_REQUESTS.inc()
        global total_predictions
        
        # Sửa chữ answers thành final_answers
        for ans in final_answers: 
            if ans in ANSWER_COUNTERS:
                ANSWER_COUNTERS[ans].inc()
                answer_counts[ans] += 1
                total_predictions += 1
                
        update_drift_score() # Tính lại điểm Drift
        
        # Sửa chữ answers thành final_answers
        return {"trang_thai": "thành công", "ket_qua_cham_diem": final_answers}
    except Exception as e:
        return {"trang_thai": "lỗi", "chi_tiet": str(e)}