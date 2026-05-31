import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import cv2 # Dùng OpenCV để đọc và tiền xử lý ảnh
import numpy as np

class OMRDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        """
        csv_file: Đường dẫn tới file train.csv, val.csv hoặc test.csv
        img_dir: Đường dẫn tới thư mục gốc data/synthetic_images/
        """
        self.data_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        # 1. Đọc tên ảnh
        img_name = os.path.join(self.img_dir, self.data_frame.iloc[idx, 0])
        image = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (800, 1200))
        image = image.astype(np.float32) / 255.0
        image = torch.tensor(image).unsqueeze(0) 

        # --- ĐOẠN CODE ĐÃ ĐƯỢC SỬA ---
        # 3. Đọc 40 đáp án và ánh xạ chữ thành số
        raw_labels = self.data_frame.iloc[idx, 1:41].values
        
        # Từ điển ánh xạ (Nếu file csv của bạn dùng a,b,c,d thường thì hàm .upper() sẽ tự động bao quát luôn)
        label_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        
        # Dịch từng chữ cái trong 40 đáp án thành số tương ứng
        try:
            numeric_labels = [label_map[str(label).strip().upper()] for label in raw_labels]
        except KeyError as e:
            # Bắt lỗi nếu có ký tự lạ không nằm trong A, B, C, D
            print(f"Lỗi dữ liệu ở file {img_name}: Chứa ký tự lạ {e}")
            numeric_labels = [0] * 40 # Tạm gán 0 để không chết chương trình, hoặc bạn có thể raise lỗi
            
        labels = torch.tensor(numeric_labels, dtype=torch.float32)

        return image, labels

# --- TEST THỬ DATALOADER ---
if __name__ == "__main__":
    train_dataset = OMRDataset(
        csv_file='data/processed/train.csv', 
        img_dir='data/synthetic_images/'
    )
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    # In thử batch đầu tiên ra xem
    imgs, lbls = next(iter(train_loader))
    print(f"Kích thước 1 batch ảnh: {imgs.shape}") # Sẽ là [16, 1, 1200, 800]
    print(f"Kích thước 1 batch nhãn: {lbls.shape}") # Sẽ là [16, 40]