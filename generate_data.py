import cv2
import numpy as np
import random
import os
import csv

# ================= CẤU HÌNH =================
TEMPLATE_PATH = "image.png"  # Đường dẫn tới ảnh phiếu trắng của bạn
OUTPUT_DIR = "data/synthetic_images"
LABEL_FILE = "data/labels.csv"
NUM_SAMPLES = 5000  # Số lượng ảnh muốn sinh ra (để test trước)

# Tọa độ giả định (BẠN CẦN ĐO VÀ THAY ĐỔI CÁC SỐ NÀY)
# Giả sử câu 1 bắt đầu ở (x=100, y=200), các ô cách nhau 30px ngang, 40px dọc
START_X = 141.0 
START_Y = 343.0
X_STRIDE = 30.5  # Khoảng cách giữa A, B, C, D
Y_STRIDE = 23.7  # Khoảng cách giữa Câu 1, Câu 2...
COL_STRIDE = 179
BLOCK_GAP = 0

OPTIONS = ['A', 'B', 'C', 'D']
NUM_QUESTIONS = 40

# Bút chì tô đen (màu xám đậm, không đen đặc hoàn toàn)
PENCIL_COLOR = (50, 50, 50) 
BUBBLE_RADIUS = 10  # Bán kính ô tròn cần tô

# ============================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

import random

def draw_realistic_bubble(img, center_x, center_y, base_radius):
    """
    Vết tô được ép nằm lọt bên trong viền tròn, 
    vẫn giữ độ chân thực nhưng gọn gàng hơn.
    """
    # Ép bán kính tô tối đa phải nhỏ hơn viền tròn (trừ đi 1 pixel an toàn)
    max_fill_radius = base_radius - 1
    
    # 1. Lệch tâm cực nhỏ (chỉ 0.5 đến 1 pixel) để không trôi ra mép
    cx = int(center_x + random.uniform(-1, 1))
    cy = int(center_y + random.uniform(-1, 1))
    
    # 2. Hình dáng Elip: Bán kính ngẫu nhiên nhưng LUÔN NHỎ HƠN HOẶC BẰNG max_fill_radius
    axis_x = max_fill_radius + random.randint(-2, 0) # Chỉ trừ đi, không được cộng thêm
    axis_y = max_fill_radius + random.randint(-2, 0)
    
    angle = random.randint(0, 180)
    
    # 3. Vẽ 3 lớp (Lớp ngoài cùng cũng không bao giờ vượt qua axis_x, axis_y)
    
    # Lớp 1 (Ngoài cùng): Xám mờ (vẫn nằm gọn trong vòng)
    outer_color = (150, 150, 150)
    cv2.ellipse(img, (cx, cy), (axis_x, axis_y), angle, 0, 360, outer_color, -1)
    
    # Lớp 2 (Giữa): Xám đậm hơn, diện tích thu nhỏ lại 1 chút
    mid_shade = random.randint(80, 110)
    mid_color = (mid_shade, mid_shade, mid_shade)
    cv2.ellipse(img, (cx, cy), (max(1, axis_x - 1), max(1, axis_y - 1)), angle, 0, 360, mid_color, -1)
    
    # Lớp 3 (Lõi): Đậm nhất, diện tích bằng khoảng một nửa lõi
    core_shade = random.randint(40, 70)
    core_color = (core_shade, core_shade, core_shade)
    cv2.ellipse(img, (cx, cy), (max(1, axis_x - 3), max(1, axis_y - 3)), angle, 0, 360, core_color, -1)

def apply_augmentations(img):
    """Làm 'xấu' ảnh để giống thực tế"""
    # 1. Thỉnh thoảng làm mờ (Blur)
    if random.random() > 0.5:
        k_size = random.choice([(3,3), (5,5)])
        img = cv2.GaussianBlur(img, k_size, 0)
        
    # 2. Thay đổi độ sáng ngẫu nhiên
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    value = random.randint(-30, 30)
    if value > 0:
        cv2.add(v, value, v)
    else:
        cv2.subtract(v, abs(value), v)
    hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return img

def generate_dataset():
    # Load ảnh gốc
    base_img = cv2.imread(TEMPLATE_PATH)
    if base_img is None:
        print(f"Lỗi: Không tìm thấy ảnh {TEMPLATE_PATH}")
        return

    labels_data = []

    for i in range(NUM_SAMPLES):
        img_copy = base_img.copy()
        answers = {}

        # Duyệt qua 40 câu hỏi
        for q in range(NUM_QUESTIONS):
            ans_idx = random.randint(0, 3) 
            ans_letter = OPTIONS[ans_idx]
            answers[f"Q{q+1}"] = ans_letter

            col_index = q // 10  
            row_index = q % 10   

            # Tính tọa độ bằng float
            center_x = START_X + (ans_idx * X_STRIDE) + (col_index * COL_STRIDE)
            center_y = START_Y + (row_index * Y_STRIDE)

            # Bù trừ đường kẻ ngang
            if row_index >= 5:
                center_y += BLOCK_GAP

            # Gọi hàm vẽ vết bút chì chân thực (hàm tự động ép kiểu int bên trong)
            draw_realistic_bubble(img_copy, center_x, center_y, BUBBLE_RADIUS)

        # Áp dụng Augmentation
        img_copy = apply_augmentations(img_copy)

        # Lưu ảnh
        filename = f"sample_{i:04d}.jpg"
        filepath = os.path.join(OUTPUT_DIR, filename)
        cv2.imwrite(filepath, img_copy)

        # Lưu nhãn
        row = {'filename': filename}
        row.update(answers)
        labels_data.append(row)
        
        if (i+1) % 10 == 0:
            print(f"Đã sinh {i+1}/{NUM_SAMPLES} ảnh...")

    # Ghi file CSV
    fieldnames = ['filename'] + [f"Q{q+1}" for q in range(NUM_QUESTIONS)]
    
    # 2. Mở file và ghi dữ liệu
    with open(LABEL_FILE, 'w', newline='') as csvfile:
        # Truyền đúng danh sách cột đã tạo ở trên vào DictWriter
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Ghi dòng tiêu đề (Header)
        writer.writeheader()
        
        # Ghi TOÀN BỘ mảng dữ liệu (labels_data) đã được gom từ vòng lặp sinh ảnh
        # Lưu ý dùng writerows (có chữ s) thay vì writerow
        writer.writerows(labels_data) 

    print(f"Hoàn tất! Dữ liệu được lưu tại {OUTPUT_DIR} và {LABEL_FILE}")

if __name__ == "__main__":
    generate_dataset()