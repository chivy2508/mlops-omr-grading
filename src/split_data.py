import pandas as pd
from sklearn.model_selection import train_test_split
import os

print("🔄 Đang tiến hành chia tập dữ liệu...")

df_base = pd.read_csv('data/labels.csv')

train_df, temp_df = train_test_split(df_base, test_size=0.2, random_state=42)

val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

feedback_file = 'data/feedback_labels.csv'
if os.path.exists(feedback_file):
    df_feedback = pd.read_csv(feedback_file)
    
    train_df = pd.concat([train_df, df_feedback], ignore_index=True)
    print(f"➕ Đã bơm thêm {len(df_feedback)} mẫu ảnh thực tế vào tập Huấn luyện.")

os.makedirs('data/processed', exist_ok=True)

train_df.to_csv('data/processed/train.csv', index=False)
val_df.to_csv('data/processed/val.csv', index=False)
test_df.to_csv('data/processed/test.csv', index=False)

# Báo cáo kết quả
print(f"✅ Đã chia xong tổng cộng {len(train_df) + len(val_df) + len(test_df)} mẫu dữ liệu:")
print(f"   📊 Train: {len(train_df)} mẫu (Bao gồm gốc + thực tế)")
print(f"   📊 Val:   {len(val_df)} mẫu (Giữ nguyên)")
print(f"   📊 Test:  {len(test_df)} mẫu (Tinh khiết 100%)")
print("📁 Đã lưu file thành công vào 'data/processed/'")