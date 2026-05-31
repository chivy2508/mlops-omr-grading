from fastapi import FastAPI, File, UploadFile
import mlflow
import os
import cv2
import numpy as np
import torch

# ==========================================
# 1. CẤU HÌNH MLFLOW & MINIO 
# ==========================================
os.environ["AWS_ACCESS_KEY_ID"] = "admin"
os.environ["AWS_SECRET_ACCESS_KEY"] = "password123"

os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://minio:9000"
mlflow.set_tracking_uri("http://mlflow:5000")

app = FastAPI(title="OMR Grading Engine API v1")

# ==========================================
# 2. TẢI MÔ HÌNH TỰ ĐỘNG TỪ MLFLOW
# ==========================================
print("🔄 Đang tải mô hình OMR_Grading_Engine từ hệ thống...")
MODEL_URI = "models:/OMR_Grading_Engine@production"

try:
    model = mlflow.pyfunc.load_model(MODEL_URI)
    print(" Nạp mô hình thành công!")
except Exception as e:
    print(f" Lỗi tải mô hình: {e}")

# ==========================================
# 3. GIAO THỨC API
# ==========================================
@app.post("/predict")
async def predict_omr(file: UploadFile = File(...)):
    try:
        # 1. Đọc dữ liệu byte từ mạng 
        image_bytes = await file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        
        # 2. Giải mã trực tiếp thành ảnh xám 
        image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        
        # 3. Resize về chuẩn 800x1200 
        image = cv2.resize(image, (800, 1200))
        
        # 4. Chuẩn hóa pixel / 255.0
        image = image.astype(np.float32) / 255.0
        
        # 5. Ép kiểu Tensor và thêm chiều 
        input_tensor = torch.tensor(image).unsqueeze(0).unsqueeze(0)
        
        # Gọi mô hình dự đoán 
        result_tensor = model.predict(input_tensor.numpy())
        
        # Result đang ở dạng ma trận, ta cần tìm đáp án (0,1,2,3) -> (A,B,C,D)
        # Biến đổi thành Tensor để xài hàm argmax như trong train.py
        result_tensor = torch.tensor(result_tensor)
        predictions = torch.argmax(result_tensor, dim=2)[0] # Lấy kết quả của ảnh đầu tiên trong batch
        
        label_map_reverse = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
        final_answers = [label_map_reverse[ans.item()] for ans in predictions]

        return {
            "trang_thai": "thành công",
            "ten_file": file.filename,
            "ket_qua_cham_diem": final_answers
        }
    except Exception as e:
        return {"trang_thai": "lỗi", "chi_tiet": str(e)}