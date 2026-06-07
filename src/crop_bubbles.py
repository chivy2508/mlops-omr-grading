import cv2
import os
import pandas as pd
import numpy as np

def create_bubble_dataset(csv_path, img_dir, output_dir):
    """
    Cắt tờ giấy 800x1200 thành 160 ô 32x32 và phân loại vào 0_empty / 1_filled.
    """
    df = pd.read_csv(csv_path)
    
    os.makedirs(os.path.join(output_dir, "0_empty"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "1_filled"), exist_ok=True)
    
    # --- BỘ TỌA ĐỘ CHUẨN TỪ MAIN.PY ---
    START_X = 135.0  
    START_Y = 665.0
    X_STRIDE = 30.5 
    Y_STRIDE = 35  
    COL_STRIDE = 150 
    BLOCK_GAP = 0    
    
    bubble_counter = 0
    label_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    
    for idx, row in df.iterrows():
        img_name = row.iloc[0]
        img_path = os.path.join(img_dir, img_name)
        
        # 1. Đọc ảnh và ép về đúng size chuẩn của Production
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        img = cv2.resize(img, (800, 1200)) # Rất quan trọng!
        
        # Đọc mảng 40 đáp án đúng từ CSV
        raw_labels = row.iloc[1:41].values
        
        # 2. Duyệt qua 40 câu hỏi
        for q in range(40):
            # Lấy đáp án đúng của câu này (Ví dụ: 'B' -> 1)
            ans_str = str(raw_labels[q]).strip().upper()
            correct_idx = label_map.get(ans_str, -1)
            
            # Tính toán Cột và Hàng y hệt API
            col_index = q // 10  
            row_index = q % 10   
            
            center_y = START_Y + (row_index * Y_STRIDE)
            if row_index >= 5:
                center_y += BLOCK_GAP
                
            # Duyệt qua 4 lựa chọn A, B, C, D của câu hỏi q
            for lua_chon_idx in range(4):
                center_x = START_X + (lua_chon_idx * X_STRIDE) + (col_index * COL_STRIDE)
                
                # Tính Bounding Box 32x32 quanh tâm
                x1, y1 = int(center_x - 16), int(center_y - 16)
                x2, y2 = int(center_x + 16), int(center_y + 16)
                
                # Cắt ra ô tròn
                bubble_patch = img[y1:y2, x1:x2]
                
                # Bỏ qua nếu thuật toán cắt lẹm ra ngoài viền ảnh (shape bất thường)
                if bubble_patch.shape != (32, 32): 
                    continue 
                
                # Xác định Label: Ô này có phải là đáp án đúng trong CSV không?
                is_filled = 1 if lua_chon_idx == correct_idx else 0
                
                # Lưu file vào đúng rổ
                folder = "1_filled" if is_filled == 1 else "0_empty"
                save_path = os.path.join(output_dir, folder, f"{img_name.split('.')[0]}_q{q+1}_{lua_chon_idx}.jpg")
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