import cv2
import os
import pandas as pd
import numpy as np
import json  # Đã thêm thư viện này

try:
    from src.align_document import get_aligned_paper
except ImportError:
    from align_document import get_aligned_paper

# Đọc file tọa độ chuẩn
with open("data/template_config.json", "r") as f:
    template_config = json.load(f)

def clean_and_binarize(aligned_image: np.ndarray) -> np.ndarray:
    """Đồng bộ hàm tiền xử lý này từ main.py để Train và Inference ăn khớp nhau"""
    blur = cv2.GaussianBlur(aligned_image, (3, 3), 0)
    cleaned = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    return cleaned

def create_bubble_dataset(csv_path, img_dir, output_dir):
    """
    Cắt tờ giấy 800x1200 thành 160 ô 32x32 và phân loại vào 0_empty / 1_filled.
    """
    df = pd.read_csv(csv_path)
    
    os.makedirs(os.path.join(output_dir, "0_empty"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "1_filled"), exist_ok=True)
    
    # Đã xóa cụm START_X thừa ở đây
    
    bubble_counter = 0
    # Đã xóa label_map thừa vì dùng trực tiếp ord() ở dưới
    
    for idx, row in df.iterrows():
        img_name = row.iloc[0]
        img_path = os.path.join(img_dir, img_name)
        
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        
        img = get_aligned_paper(img)
        img = clean_and_binarize(img)
        
        raw_labels = row.iloc[1:41].values
        
        for bubble in template_config["bubbles"]:
            q_idx = bubble["question"] - 1 # Câu hỏi từ 0-39
            ans_char = bubble["option"]    # 'A', 'B', 'C', 'D'
            center_x = bubble["x"]
            center_y = bubble["y"]
            
            # Tính Bounding Box 32x32
            x1, y1 = int(center_x - 16), int(center_y - 16)
            x2, y2 = int(center_x + 16), int(center_y + 16)
            
            bubble_patch = img[y1:y2, x1:x2]
            if bubble_patch.shape != (32, 32): continue 
            
            # So sánh đáp án đúng trong CSV để phân loại 0_empty hay 1_filled
            correct_ans_str = str(raw_labels[q_idx]).strip().upper()
            is_filled = 1 if ans_char == correct_ans_str else 0
            
            folder = "1_filled" if is_filled == 1 else "0_empty"
            lua_chon_idx = ord(ans_char) - 65 
            save_path = os.path.join(output_dir, folder, f"{img_name.split('.')[0]}_q{q_idx+1}_{lua_chon_idx}.jpg")
            cv2.imwrite(save_path, bubble_patch)
                
            bubble_counter += 1
                
    print(f"✅ Đã trích xuất thành công {bubble_counter} ô tròn (32x32) vào {output_dir}!")

if __name__ == "__main__":
    print("✂️ Đang tiến hành cắt dữ liệu Train...")
    create_bubble_dataset('data/processed/train.csv', 'data/synthetic_images/', 'data/bubbles/train')
    
    print("✂️ Đang tiến hành cắt dữ liệu Val...")
    create_bubble_dataset('data/processed/val.csv', 'data/synthetic_images/', 'data/bubbles/val')
    
    print("✂️ Đang tiến hành cắt dữ liệu Test...")
    create_bubble_dataset('data/processed/test.csv', 'data/synthetic_images/', 'data/bubbles/test')