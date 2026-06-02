import os
import json
import glob
import shutil
import pandas as pd
import subprocess

# --- CẤU HÌNH ĐƯỜNG DẪN ---
RETRAIN_DIR = "./retrain_dataset"
TARGET_IMG_DIR = "data/synthetic_images"
TARGET_CSV = "data/labels.csv"

def process_and_trigger_pipeline():
    # Lấy danh sách các file JSON đáp án từ Streamlit gửi về
    json_files = glob.glob(os.path.join(RETRAIN_DIR, "*_label.json"))
    
    if not json_files:
        print("Kho dữ liệu trống. Không có gì để cập nhật.")
        return False

    new_rows = []
    
    print(f"🔄 Đang xử lý {len(json_files)} mẫu dữ liệu mới từ khách hàng...")
    
    for json_path in json_files:
        # Lấy cái ID ngẫu nhiên (VD: a1b2c3d4)
        base_name = os.path.basename(json_path).replace("_label.json", "")
        
        # Tìm ảnh đi kèm (Dùng glob để quét đuôi jpg, png, jpeg...)
        img_files = glob.glob(os.path.join(RETRAIN_DIR, f"{base_name}_image.*"))
        if not img_files:
            continue
            
        img_src = img_files[0]
        img_filename = os.path.basename(img_src)
        
        # Đọc dữ liệu JSON: [{"cau": 1, "dap_an": "A"}, {"cau": 2, "dap_an": "B"}...]
        with open(json_path, 'r', encoding='utf-8') as f:
            feedback_data = json.load(f)
            
        # Lắp ráp thành 1 dòng (row) chuẩn với file CSV gốc
        row_data = {"filename": img_filename}
        for item in feedback_data:
            row_data[f"Q{item['cau']}"] = item['dap_an']
            
        new_rows.append(row_data)
        
        # Di chuyển ảnh vào thư mục synthetic_images
        shutil.move(img_src, os.path.join(TARGET_IMG_DIR, img_filename))
        
        # Xóa file JSON để tránh xử lý trùng lần sau
        os.remove(json_path)

    if new_rows:
        # 1. Chuyển thành DataFrame và Nối vào cuối file labels.csv
        df_new = pd.DataFrame(new_rows)
        # Đảm bảo thứ tự cột chuẩn xác (filename, Q1 -> Q40)
        cols = ['filename'] + [f'Q{i}' for i in range(1, 41)]
        df_new = df_new.reindex(columns=cols) 
        
        df_new.to_csv(TARGET_CSV, mode='a', header=False, index=False)
        print(f"✅ Đã ghi thêm {len(new_rows)} dòng vào {TARGET_CSV}")
        
        # 2. BÓP CÒ DVC & GIT TỰ ĐỘNG
        print("🚀 Bắt đầu chạy DVC Pipeline...")
        try:
            # Theo dõi các ảnh mới
            subprocess.run(["dvc", "add", "data/synthetic_images/"], check=True)
            
            # Phép thuật nằm ở đây: dvc repro sẽ tự thấy labels.csv đổi và chạy lại split_data.py
            subprocess.run(["dvc", "repro"], check=True)
            
            # Đẩy lên MinIO
            subprocess.run(["dvc", "push"], check=True)
            
            # Commit lên Git để lưu vết thay đổi file .dvc và .lock
            subprocess.run(["git", "add", "data/synthetic_images.dvc", "data/labels.csv", "dvc.lock"], check=True)
            subprocess.run(["git", "commit", "-m", "🔄 Auto-update: Bổ sung dữ liệu Ground Truth từ khách hàng"], check=True)
            
            print("🎉 MLOps Pipeline hoàn tất! Dữ liệu đã sẵn sàng trên MinIO.")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Lỗi khi chạy DVC/Git Pipeline: {e}")
            return False

if __name__ == "__main__":
    process_and_trigger_pipeline()