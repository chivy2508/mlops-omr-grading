import pandas as pd
from sklearn.model_selection import train_test_split
import os

print("🔄 Đang tiến hành chia tập dữ liệu...")

# Đọc file gốc
df = pd.read_csv('data/labels.csv')

# Chia tập Train (80%) và Temp (20%)
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
# Chia Temp thành Val (10%) và Test (10%)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# Tạo thư mục chứa file đã xử lý
os.makedirs('data/processed', exist_ok=True)

# Lưu 3 file CSV
train_df.to_csv('data/processed/train.csv', index=False)
val_df.to_csv('data/processed/val.csv', index=False)
test_df.to_csv('data/processed/test.csv', index=False)

print(f"✅ Đã chia xong tổng cộng {len(df)} mẫu dữ liệu:")
print(f"   📊 Train: {len(train_df)} mẫu (80%)")
print(f"   📊 Val:   {len(val_df)} mẫu (10%)")
print(f"   📊 Test:  {len(test_df)} mẫu (10%)")
print("📁 Đã lưu file thành công vào 'data/processed/'")