import cv2
import os
import pandas as pd
import numpy as np

def create_bubble_dataset(csv_path, img_dir, output_dir):
    """
    Cắt tờ giấy to thành các ô 32x32 và lưu vào thư mục 0_empty hoặc 1_filled.
    """
    df = pd.read_csv(csv_path)
    
    # Tạo cấu trúc thư mục chuẩn cho PyTorch ImageFolder
    os.makedirs(os.path.join(output_dir, "0_empty"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "1_filled"), exist_ok=True)
    
    # Giả lập tọa độ bounding box cho 160 ô (40 câu x 4 đáp án)
    # TRONG THỰC TẾ: Bạn sẽ load tọa độ x, y từ file template_config.json
    # Ở đây mình ví dụ dùng grid cơ bản để mô phỏng
    
    bubble_counter = 0
    
    for idx, row in df.iterrows():
        img_name = row.iloc[0]
        img_path = os.path.join(img_dir, img_name)
        
        # Đọc ảnh xám
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        
        raw_labels = row.iloc[1:41].values
        label_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        
        for cau_hoi_idx, answer in enumerate(raw_labels):
            # Lấy đáp án đúng của câu này (ví dụ: 'A' -> 0)
            ans_str = str(answer).strip().upper()
            correct_idx = label_map.get(ans_str, -1)
            
            for lua_chon_idx in range(4): # A, B, C, D
                # --- TOÁN HỌC OPENCV ---
                # Tính tọa độ [START_Y:END_Y, START_X:END_X] từ JSON
                # Ví dụ (giả lập):
                start_x = 100 + lua_chon_idx * 50
                start_y = 200 + cau_hoi_idx * 40
                
                # Cắt ra đúng cái ô 32x32 pixel
                # LƯU Ý: Phải đảm bảo ảnh trước đó đã được Nắn Thẳng (Perspective Warp)
                bubble_patch = img[start_y:start_y+32, start_x:start_x+32]
                
                # Bỏ qua nếu cắt lố ra ngoài viền
                if bubble_patch.shape != (32, 32): continue 
                
                # Xác định nhãn (Label): Ô này có phải là ô đáp án đúng không?
                is_filled = 1 if lua_chon_idx == correct_idx else 0
                
                # Lưu file
                folder = "1_filled" if is_filled == 1 else "0_empty"
                save_path = os.path.join(output_dir, folder, f"bubble_{bubble_counter}.jpg")
                cv2.imwrite(save_path, bubble_patch)
                
                bubble_counter += 1
                
    print(f"✅ Đã trích xuất thành công {bubble_counter} ô tròn (32x32)!")

if __name__ == "__main__":
    # Chạy kịch bản này một lần duy nhất trước khi train
    create_bubble_dataset('data/processed/train.csv', 'data/synthetic_images/', 'data/bubbles/train')
    create_bubble_dataset('data/processed/val.csv', 'data/synthetic_images/', 'data/bubbles/val')