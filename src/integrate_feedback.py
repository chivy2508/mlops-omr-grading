import os
import json
import glob
import shutil
import pandas as pd
import subprocess
import fcntl
from filelock import FileLock

# --- CẤU HÌNH ĐƯỜNG DẪN ---
RETRAIN_DIR = "./retrain_dataset"
TARGET_IMG_DIR = "data/synthetic_images"
TARGET_CSV = "data/feedback_labels.csv"

def process_and_trigger_pipeline():
    json_files = glob.glob(os.path.join(RETRAIN_DIR, "*_label.json"))
    
    if not json_files:
        print("Kho dữ liệu trống. Không có gì để cập nhật.")
        return False

    new_rows = []
    
    print(f"🔄 Đang xử lý {len(json_files)} mẫu dữ liệu mới từ khách hàng...")
    
    for json_path in json_files:
        base_name = os.path.basename(json_path).replace("_label.json", "")
        
        img_files = glob.glob(os.path.join(RETRAIN_DIR, f"{base_name}_image.*"))
        if not img_files:
            continue
            
        img_src = img_files[0]
        img_filename = os.path.basename(img_src)
        
        with open(json_path, 'r', encoding='utf-8') as f:
            feedback_data = json.load(f)
            
        row_data = {"filename": img_filename}
        for item in feedback_data:
            row_data[f"Q{item['cau']}"] = item['dap_an']
            
        new_rows.append(row_data)
        
        # Di chuyển ảnh vào thư mục synthetic_images
        shutil.move(img_src, os.path.join(TARGET_IMG_DIR, img_filename))
        
        # Xóa file JSON để tránh xử lý trùng lần sau
        os.remove(json_path)

    if new_rows:
        df_new = pd.DataFrame(new_rows)
        cols = ['filename'] + [f'Q{i}' for i in range(1, 41)]
        df_new = df_new.reindex(columns=cols) 
        
        lock = FileLock(f"{TARGET_CSV}.lock")
        
        with lock:
            df_new.to_csv(TARGET_CSV, mode='a', header=False, index=False)

        print(f"✅ Đã ghi thêm {len(new_rows)} dòng vào {TARGET_CSV}")
        
        print("🚀 Bắt đầu chạy DVC Pipeline...")
        try:
            subprocess.run(["dvc", "add", "data/synthetic_images/"], check=True)
            
            subprocess.run(["dvc", "repro"], check=True)
            
            subprocess.run(["dvc", "push"], check=True)
            
            subprocess.run(["git", "config", "--global", "user.name", "OMR_Auto_Worker"], check=True)
            subprocess.run(["git", "config", "--global", "user.email", "worker@omr.local"], check=True)
            
            subprocess.run(["git", "add", "data/synthetic_images.dvc", "data/feedback_labels.csv", "dvc.lock"], check=True)
            
            status = subprocess.run(["git", "diff", "--cached", "--quiet"])
            if status.returncode != 0:
                subprocess.run(["git", "commit", "-m", f"🔄 Auto-update: Bổ sung {len(new_rows)} mẫu Ground Truth"], check=True)
                
                git_token = os.getenv("GIT_TOKEN")
                git_repo = os.getenv("GIT_REPO_URL") 
                
                if git_token and git_repo:
                    auth_url = f"https://{git_token}@{git_repo}"
                    subprocess.run(["git", "push", auth_url, "main"], check=True)
                    print("📦 Đã đẩy dvc.lock và code mới lên kho chứa an toàn!")
                else:
                    print("⚠️ Bỏ qua git push: Chưa cấu hình GIT_TOKEN hoặc GIT_REPO_URL.")
            else:
                print("ℹ️ Không có sự thay đổi nào về model/data để commit.")

            print("🎉 MLOps Pipeline hoàn tất! Dữ liệu đã sẵn sàng trên MinIO.")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Lỗi khi chạy DVC/Git Pipeline: {e}")
            return False

if __name__ == "__main__":
    process_and_trigger_pipeline()