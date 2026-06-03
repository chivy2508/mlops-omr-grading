from fastapi import FastAPI, File, UploadFile, Form
import mlflow
import os
import cv2
import numpy as np
import torch
import requests
import json
import uuid
from prometheus_client import make_asgi_app, Counter, Gauge
import base64

# ==========================================
# 1. CẤU HÌNH MLFLOW & MINIO 
# ==========================================

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


app = FastAPI(title="OMR Grading Engine API")
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")

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
    
    if total_predictions == 0:
        return 0.0
    current_dist = {k: v / total_predictions for k, v in answer_counts.items()}
    drift = sum(abs(current_dist[k] - REFERENCE_DIST[k]) for k in REFERENCE_DIST) / 2
    DRIFT_SCORE.set(drift)
    return drift

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
    # Lấy diện tích ảnh gốc để làm chuẩn so sánh
    orig_h, orig_w = image.shape[:2]
    total_area = orig_h * orig_w
    
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    edged = cv2.Canny(blur, 75, 200)
    
    # Dùng RETR_LIST thay vì EXTERNAL để quét được cả viền giấy bên ngoài lẫn khung in bên trong
    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    paper_contour = None
    if cnts:
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            
            if len(approx) == 4:
                area = cv2.contourArea(approx)
                # ĐIỀU KIỆN SỐNG CÒN: Khung tìm được phải chiếm ít nhất 30% diện tích ảnh gốc.
                # Nếu bé hơn, nó chỉ là một cái khung phụ (như khung điền tên) -> Bỏ qua ngay!
                if area > 0.3 * total_area:
                    paper_contour = approx
                    break
                    
    if paper_contour is None:
        # FALLBACK (Dự phòng): Nếu không tìm thấy khung giấy nào đủ to (do chụp trên nền trắng, thiếu tương phản), 
        # thì không cắt xén gì cả. Ép thẳng tấm ảnh gốc của bạn về 800x1200.
        return cv2.resize(image, (target_w, target_h))
        
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

def get_dynamic_start(binary_image: np.ndarray, default_x=141.0, default_y=343.0):
    """
    Tìm ô vuông đen (Anchor) ở gần Câu 1 để xác định lại START_X và START_Y
    """
    # cv2.findContours tìm vật thể trắng trên nền đen.
    # Ảnh binary hiện tại của bạn thường là nền trắng, ô đen -> Cần đảo ngược màu (Invert)
    inv = cv2.bitwise_not(binary_image)
    
    # Khoanh vùng tìm kiếm: Chỉ tìm ở 1/4 góc trên - bên trái tờ giấy để code chạy nhanh và không bắt nhầm rác
    h, w = inv.shape
    roi = inv[0:int(h/2), 0:int(w/2)]
    
    cnts, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for c in cnts:
        x, y, w_box, h_box = cv2.boundingRect(c)
        aspect_ratio = w_box / float(h_box)
        area = cv2.contourArea(c)
        
        # Điều kiện nhận dạng "Ô vuông đen":
        # 1. Tỷ lệ dài/rộng xấp xỉ 1.0 (Hình vuông)
        # 2. Diện tích đủ lớn (không phải hạt bụi) nhưng không quá to (không phải cả cái bảng)
        if 0.8 <= aspect_ratio <= 1.2 and 150 < area < 900:
            
            # KHOẢNG CÁCH TỪ Ô VUÔNG ĐẾN TÂM ĐÁP ÁN A (BẠN CẦN ĐO LẠI 1 LẦN DUY NHẤT)
            # Ví dụ: Từ mép trái ô vuông dịch sang phải 40px là tâm đáp án A
            OFFSET_X = 40.0 
            # Ví dụ: Tâm đáp án A nằm ngang hàng với mép trên ô vuông
            OFFSET_Y = 0.0  
            
            dynamic_start_x = float(x + OFFSET_X)
            dynamic_start_y = float(y + OFFSET_Y)
            
            return dynamic_start_x, dynamic_start_y
            
    # Nếu xui xẻo giấy bị rách/mất ô vuông đen, trả về tọa độ cứng dự phòng để API không sập
    return default_x, default_y

RETRAIN_DIR = "./retrain_dataset"
os.makedirs(RETRAIN_DIR, exist_ok=True)

@app.post("/feedback")
async def save_human_feedback(
    file: UploadFile = File(...), 
    correct_labels: str = Form(...) 
):
    try:
        unique_id = str(uuid.uuid4())[:8]
        
        image_extension = file.filename.split(".")[-1]
        image_filename = f"{unique_id}_image.{image_extension}"
        image_path = os.path.join(RETRAIN_DIR, image_filename)
        
        with open(image_path, "wb") as buffer:
            buffer.write(await file.read())
            
        # 2. Lưu đáp án chuẩn (Ground Truth) ra file JSON
        label_filename = f"{unique_id}_label.json"
        label_path = os.path.join(RETRAIN_DIR, label_filename)
        
        # Chuyển chuỗi JSON từ Streamlit gửi lên thành dạng dictionary
        labels_data = json.loads(correct_labels) 
        
        with open(label_path, "w", encoding="utf-8") as f:
            json.dump(labels_data, f, ensure_ascii=False, indent=4)
        
        send_discord_alert(f"👤 Có người dùng vừa sửa nhãn cho ảnh `{image_filename}`. Đã đưa vào kho chờ.")
            
        return {"status": "success", "message": f"Đã lưu data vào {RETRAIN_DIR}"}
        
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/reload-model")
async def reload_model():
    global model
    try:
        print("Đang kéo mô hình Production mới nhất về...")
        # Kéo model mới đè lên biến model cũ
        model = mlflow.pytorch.load_model("models:/OMR_Grading_Engine@production")
        model.eval()
        return {"status": "thành công", "message": "Đã cập nhật model Production mới nhất lên RAM!"}
    except Exception as e:
        return {"status": "lỗi", "message": str(e)}
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

        #START_X, START_Y = get_dynamic_start(final_processed_image, default_x=141.0, default_y=343.0)

        START_X = 135.0  # Chỉnh số này cho đến khi cột 1 khớp (Tăng -> Dịch phải, Giảm -> Dịch trái)
        START_Y = 665.0

        X_STRIDE = 30.5 # Khoảng cách giữa A, B, C, D
        Y_STRIDE = 35  # Khoảng cách giữa Câu 1, Câu 2... dọc xuống
        COL_STRIDE = 150 # Khoảng cách sang cột mới
        BLOCK_GAP = 0    # Khoảng trống giữa các block (nếu có)
        
        predictions_detail = []
        for i, ans in enumerate(final_answers):
            cau_hoi = i + 1
            q = i  # q chạy từ 0 đến 39 (chuẩn array index)
            
            # Ánh xạ chữ sang số: A=0, B=1, C=2, D=3
            offset_x_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
            ans_idx = offset_x_map.get(ans, 0)
            
            # Logic chia cột và dòng y hệt code cũ của bạn
            col_index = q // 10  
            row_index = q % 10   
            
            # Tính tọa độ bằng float
            center_x = START_X + (ans_idx * X_STRIDE) + (col_index * COL_STRIDE)
            center_y = START_Y + (row_index * Y_STRIDE)
            
            # Bù trừ đường kẻ ngang
            if row_index >= 5:
                center_y += BLOCK_GAP
                
            # Đóng gói JSON trả về Streamlit
            predictions_detail.append({
                "cau": cau_hoi,
                "dap_an": ans,
                "x": center_x,
                "y": center_y
            })

        SUCCESS_REQUESTS.inc()
        global total_predictions
        
        # Sửa chữ answers thành final_answers
        for ans in final_answers: 
            if ans in ANSWER_COUNTERS:
                ANSWER_COUNTERS[ans].inc()
                answer_counts[ans] += 1
                total_predictions += 1
                
        current_drift = update_drift_score()
        
        _, buffer = cv2.imencode('.jpg', aligned_img)
        aligned_base64 = base64.b64encode(buffer).decode('utf-8')

        # Sửa chữ answers thành final_answers
        return {
            "trang_thai": "thành công", 
            "predictions": predictions_detail,
            "drift_score": current_drift,
            "aligned_image": aligned_base64
        }
    except Exception as e:
        return {"trang_thai": "lỗi", "chi_tiet": str(e)}