from fastapi import FastAPI, File, UploadFile, Form, HTTPException
import mlflow
import os
import cv2
import numpy as np
import torch
import requests
import json
import uuid
from prometheus_client import make_asgi_app, Counter, Gauge, Histogram
import base64
import asyncio
from dotenv import load_dotenv
from src.align_document import get_aligned_paper

load_dotenv()

RETRAIN_DIR = "./retrain_dataset"
os.makedirs(RETRAIN_DIR, exist_ok=True)

STATE_FILE = os.path.join(RETRAIN_DIR, "drift_state.json")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
if not MLFLOW_TRACKING_URI:
    raise ValueError("MLFLOW_TRACKING_URI must be set")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


app = FastAPI(title="OMR Grading Engine API")
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

TOTAL_REQUESTS = Counter('omr_total_requests', 'Tong so request')
SUCCESS_REQUESTS = Counter('omr_success_requests', 'So request thanh cong')
FAILED_REQUESTS = Counter('omr_failed_requests', 'So request that bai')
ANSWER_A = Counter('omr_answer_A', 'So lan dap an A')
ANSWER_B = Counter('omr_answer_B', 'So lan dap an B')
ANSWER_C = Counter('omr_answer_C', 'So lan dap an C')
ANSWER_D = Counter('omr_answer_D', 'So lan dap an D')
DRIFT_SCORE = Gauge('omr_drift_score', 'Data drift score')
INFERENCE_TIME = Histogram('omr_inference_seconds', 'Model inference time')

ANSWER_COUNTERS = {'A': ANSWER_A, 'B': ANSWER_B, 'C': ANSWER_C, 'D': ANSWER_D}

REFERENCE_DIST = {'A': 0.25, 'B': 0.25, 'C': 0.25, 'D': 0.25}

def load_drift_state():
    """Tải dữ liệu cũ từ ổ cứng lên RAM khi API khởi động"""
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass

    return {"total_predictions": 0, "answer_counts": {"A": 0, "B": 0, "C": 0, "D": 0}}

def save_drift_state(state):
    """Ghi đè dữ liệu mới xuống ổ cứng"""
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)

drift_state = load_drift_state()

def update_drift_score(final_answers):
    """Hàm tự động tính toán lại độ lệch và ghi xuống đĩa"""
    
    
    for ans in final_answers:
        if ans in drift_state["answer_counts"]:
            drift_state["answer_counts"][ans] += 1
            drift_state["total_predictions"] += 1
           
            ANSWER_COUNTERS[ans].inc()
            
    save_drift_state(drift_state)
    
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

print("🔄 Đang tải mô hình OMR_Grading_Engine từ hệ thống...")
MODEL_URI = "models:/OMR_Grading_Engine@production"
current_model_version = "unknown"

try:
    model = mlflow.pytorch.load_model(MODEL_URI)
    model.eval()
    client = mlflow.tracking.MlflowClient()
    
    prod_model = client.get_model_version_by_alias("OMR_Grading_Engine", "production")
    current_model_version = prod_model.version
    
    print(f"✅ Nạp mô hình thành công! (Version: {current_model_version})")
except Exception as e:
    print(f"🚨 Lỗi tải mô hình chí mạng: {e}")
    import sys
    sys.exit(1)

async def watch_mlflow_registry():
    global model, current_model_version
    client = mlflow.tracking.MlflowClient()
    while True:
        try:
            prod_model = await asyncio.to_thread(client.get_model_version_by_alias, "OMR_Grading_Engine", "production")
            latest_prod_version = prod_model.version
            
            if latest_prod_version != current_model_version:
                print(f"🔄 Phát hiện phiên bản mới (v{latest_prod_version}), đang tải lại model...")
                new_model = await asyncio.to_thread(mlflow.pytorch.load_model, "models:/OMR_Grading_Engine@production")
                new_model.eval()
                model = new_model
                current_model_version = latest_prod_version
                print(f"✅ Đã cập nhật mô hình lên phiên bản v{latest_prod_version} thành công!")
        except Exception:
            pass 
        await asyncio.sleep(300)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(watch_mlflow_registry())

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


def clean_and_binarize(aligned_image: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(aligned_image, (3, 3), 0)
    cleaned = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    return cleaned

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
            
        label_filename = f"{unique_id}_label.json"
        label_path = os.path.join(RETRAIN_DIR, label_filename)
        
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
        model = mlflow.pytorch.load_model("models:/OMR_Grading_Engine@production")
        model.eval()
        return {"status": "thành công", "message": "Đã cập nhật model Production mới nhất lên RAM!"}
    except Exception as e:
        return {"status": "lỗi", "message": str(e)}

def load_template_config():
    config_path = "/app/data/template_config.json"
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return json.load(f)
    return None

@app.post("/predict")
async def predict_omr(file: UploadFile = File(...)):
    TOTAL_REQUESTS.inc()
    if model is None:
        return {"trang_thai": "lỗi", "chi_tiet": "Mô hình chưa được nạp, vui lòng kiểm tra server"}

    if file.size is not None and file.size > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File quá lớn (Tối đa 10MB)")
    
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Chỉ chấp nhận định dạng JPG/PNG")
        
    try:
        image_bytes = await file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        
        is_clear, reason_clear = check_blur_and_brightness(image)
        if not is_clear:
            send_discord_alert(f"🚨 Từ chối `{file.filename}`: {reason_clear}")
            return {"trang_thai": "từ chối", "chi_tiet": reason_clear}
            
        aligned_img = get_aligned_paper(image, target_w=800, target_h=1200)
        
        if aligned_img is None:
            reason_paper = "Không tìm thấy tờ phiếu thi hợp lệ."
            send_discord_alert(f"🚨 Từ chối `{file.filename}`: {reason_paper}")
            return {"trang_thai": "từ chối", "chi_tiet": reason_paper}
            
        final_processed_image = clean_and_binarize(aligned_img)

        TEMPLATE_CONFIG = load_template_config()
    
        if TEMPLATE_CONFIG is None:
            return {"trang_thai": "lỗi", "chi_tiet": "Server chưa khởi tạo lưới tọa độ. Hệ thống đang chờ sinh file template_config.json"}

        patches = []
        coords_info = []
        
        for bubble in template_data["bubbles"]:
            center_x = bubble["x"]
            center_y = bubble["y"]
            half_w = bubble["w"] // 2
            half_h = bubble["h"] // 2
            
            x1, y1 = center_x - half_w, center_y - half_h
            x2, y2 = center_x + half_w, center_y + half_h
            
            patch = final_processed_image[y1:y2, x1:x2]
            
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

        input_tensor = torch.tensor(np.array(patches)).unsqueeze(1) 

        with INFERENCE_TIME.time():
            with torch.no_grad(): 
                outputs = model(input_tensor)
                probs = torch.softmax(outputs, dim=1)[:, 1]

        final_answers = []
        predictions_detail = []
        
        for q in range(40):
            q_probs = probs[q*4 : (q+1)*4]
            
            best_ans_idx = torch.argmax(q_probs).item()
            best_prob = q_probs[best_ans_idx].item()
            
            if best_prob > 0.5:
                chosen_char = ['A', 'B', 'C', 'D'][best_ans_idx]
            else:
                chosen_char = 'N' 
                
            final_answers.append(chosen_char)
            
            chosen_coord = coords_info[q*4 + best_ans_idx]
            predictions_detail.append({
                "cau": q + 1,
                "dap_an": chosen_char,
                "x": chosen_coord["x"],
                "y": chosen_coord["y"],
                "confidence": float(best_prob)
            })

        SUCCESS_REQUESTS.inc()
        
        # Chỉ đếm số liệu cho đáp án A, B, C, D hợp lệ (bỏ qua N)
        valid_answers = [ans for ans in final_answers if ans in ANSWER_COUNTERS]
                
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