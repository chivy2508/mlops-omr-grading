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



RETRAIN_DIR = "./retrain_dataset"
os.makedirs(RETRAIN_DIR, exist_ok=True)

STATE_FILE = os.path.join(RETRAIN_DIR, "drift_state.json")

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


# --- QUẢN LÝ TRẠNG THÁI DRIFT ---
REFERENCE_DIST = {'A': 0.25, 'B': 0.25, 'C': 0.25, 'D': 0.25}

def load_drift_state():
    """Tải dữ liệu cũ từ ổ cứng lên RAM khi API khởi động"""
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    # Mặc định nếu chưa có file
    return {"total_predictions": 0, "answer_counts": {"A": 0, "B": 0, "C": 0, "D": 0}}

def save_drift_state(state):
    """Ghi đè dữ liệu mới xuống ổ cứng"""
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)

# Khởi tạo state toàn cục
drift_state = load_drift_state()

def update_drift_score(final_answers):
    """Hàm tự động tính toán lại độ lệch và ghi xuống đĩa"""
    
    
    # 1. Cập nhật số liệu mới
    for ans in final_answers:
        if ans in drift_state["answer_counts"]:
            drift_state["answer_counts"][ans] += 1
            drift_state["total_predictions"] += 1
            # Cập nhật luôn metrics cho Prometheus
            ANSWER_COUNTERS[ans].inc()
            
    # 2. Lưu ngay xuống file JSON bảo toàn tính mạng
    save_drift_state(drift_state)
    
    # 3. Tính toán độ lệch
    total = drift_state["total_predictions"]
    if total == 0:
        return 0.0
        
    current_dist = {k: v / total for k, v in drift_state["answer_counts"].items()}
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
    print("✅ Nạp mô hình thành công!")
except Exception as e:
    print(f"🚨 Lỗi tải mô hình chí mạng: {e}")
    import sys
    sys.exit(1)

def check_blur_and_brightness(image_gray: np.ndarray):
    laplacian_var = cv2.Laplacian(image_gray, cv2.CV_64F).var()
    if laplacian_var < 30:  
        return False, f"Ảnh quá mờ (độ nét: {laplacian_var:.1f})"      

    mean_brightness = np.mean(image_gray)
    if mean_brightness < 30: 
        return False, f"Ảnh quá tối (độ sáng: {mean_brightness:.1f})"
    if mean_brightness > 235: 
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
        
        # 1. Tiền xử lý (Cắt nắn khung giấy)
        is_clear, reason_clear = check_blur_and_brightness(image)
        if not is_clear:
            send_discord_alert(f"🚨 Từ chối `{file.filename}`: {reason_clear}")
            return {"trang_thai": "từ chối", "chi_tiet": reason_clear}
            
        aligned_img = get_aligned_paper(image, target_w=800, target_h=1200)
        if aligned_img is None:
            reason_paper = "Không tìm thấy tờ phiếu thi hợp lệ."
            send_discord_alert(f"🚨 Từ chối `{file.filename}`: {reason_paper}")
            return {"trang_thai": "từ chối", "chi_tiet": reason_paper}
            
        # 2. Làm sạch nền & Đảm bảo kích thước chuẩn để đo tọa độ
        final_processed_image = clean_and_binarize(aligned_img)
        final_processed_image = cv2.resize(final_processed_image, (800, 1200))

        with open("data/template_config.json", "r") as f:
            template_config = json.load(f)
            
        patches = []
        coords_info = []
        
        for bubble in template_config["bubbles"]:
            center_x = bubble["x"]
            center_y = bubble["y"]
            half_w = bubble["w"] // 2
            half_h = bubble["h"] // 2
            
            # Tính Bounding Box
            x1, y1 = center_x - half_w, center_y - half_h
            x2, y2 = center_x + half_w, center_y + half_h
            
            # Cắt ảnh
            patch = final_processed_image[y1:y2, x1:x2]
            
            # Rủi ro an toàn: Đảm bảo patch luôn là 32x32
            if patch.shape != (32, 32):
                patch = cv2.resize(patch, (32, 32))
                
            patch = patch.astype(np.float32) / 255.0
            patches.append(patch)
            
            coords_info.append({
                "cau": bubble["question"],
                "char": bubble["option"],
                "x": center_x,
                "y": center_y
            })

        # =========================================================
        # PIPELINE 2 (CLASSIFICATION): MobileNet Nhị phân
        # =========================================================
        # Chuyển list 160 ảnh thành Tensor có shape [160, 1, 32, 32]
        input_tensor = torch.tensor(np.array(patches)).unsqueeze(1) 
        
        with torch.no_grad(): 
            # Model trả ra shape [160, 2] (Xác suất cho class 0 và class 1)
            outputs = model(input_tensor)
            
            # Lấy xác suất của class 1 (Ô đã bị tô đen) dùng hàm Softmax
            probs = torch.softmax(outputs, dim=1)[:, 1]

        # =========================================================
        # TỔNG HỢP KẾT QUẢ VÀ TÌM ĐÁP ÁN ĐÚNG
        # =========================================================
        final_answers = []
        predictions_detail = []
        
        for q in range(40):
            # Lấy 4 mức độ tự tin (Confidence) của 4 đáp án A, B, C, D trong câu q
            q_probs = probs[q*4 : (q+1)*4]
            
            # Chọn ô có điểm "tô đen" cao nhất
            best_ans_idx = torch.argmax(q_probs).item()
            best_prob = q_probs[best_ans_idx].item()
            
            # Thiết lập ngưỡng: Nếu không ô nào đạt độ đen > 0.5 thì coi như bỏ trống (N)
            if best_prob > 0.5:
                chosen_char = ['A', 'B', 'C', 'D'][best_ans_idx]
            else:
                chosen_char = 'N' 
                
            final_answers.append(chosen_char)
            
            # Cấu trúc JSON trả về Streamlit
            chosen_coord = coords_info[q*4 + best_ans_idx]
            predictions_detail.append({
                "cau": q + 1,
                "dap_an": chosen_char,
                "x": chosen_coord["x"],
                "y": chosen_coord["y"],
                "confidence": float(best_prob)
            })

        # --- Ghi nhận Metrics ---
        SUCCESS_REQUESTS.inc()
        global total_predictions
        
        # Chỉ đếm số liệu cho đáp án A, B, C, D hợp lệ (bỏ qua N)
        valid_answers = [ans for ans in final_answers if ans in ANSWER_COUNTERS]
        for ans in valid_answers: 
            ANSWER_COUNTERS[ans].inc()
            answer_counts[ans] += 1
            total_predictions += 1
                
        current_drift = update_drift_score(valid_answers)
        
        _, buffer = cv2.imencode('.jpg', aligned_img)
        aligned_base64 = base64.b64encode(buffer).decode('utf-8')

        return {
            "trang_thai": "thành công", 
            "predictions": predictions_detail,
            "drift_score": current_drift,
            "aligned_image": aligned_base64
        }
    except Exception as e:
        return {"trang_thai": "lỗi", "chi_tiet": str(e)}