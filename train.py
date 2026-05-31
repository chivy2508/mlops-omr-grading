import os
import subprocess
import mlflow
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.dataset import OMRDataset # Gọi file dataset.py bạn đã tạo lúc nãy

# --- 1. CẤU HÌNH KẾT NỐI MLFLOW & MINIO ---
os.environ["AWS_ACCESS_KEY_ID"] = "admin"
os.environ["AWS_SECRET_ACCESS_KEY"] = "password123"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("OMR_Grading_System")

# Hàm lấy mã phiên bản dữ liệu
def get_git_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except:
        return "unknown"

# --- 2. ĐỊNH NGHĨA MÔ HÌNH CNN CHO 40 CÂU HỎI ---
class OMRClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # Mạng CNN trích xuất đặc trưng đơn giản
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1), # Ảnh xám có 1 kênh màu
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )
        # Đầu ra: 40 câu hỏi * 4 đáp án (A,B,C,D) = 160 nơ-ron
        # (Lưu ý: Bạn cần căn chỉnh lại input_features tùy thuộc vào kích thước ảnh thật)
        self.classifier = nn.Linear(16 * 300 * 200, 160) 

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        # Reshape về dạng [Batch_size, 40 câu, 4 đáp án]
        return x.view(-1, 40, 4)

def evaluate_model(model, dataloader, criterion):
    model.eval() # Bật chế độ đánh giá (tắt Dropout, Batch Norm...)
    total_loss = 0.0
    correct_answers = 0
    total_answers = 0
    
    with torch.no_grad(): # Đóng băng gradient, không học thêm
        for images, labels in dataloader:
            outputs = model(images)
            loss = criterion(outputs.permute(0, 2, 1), labels.long())
            total_loss += loss.item()
            
            # Tính độ chính xác (Accuracy): đếm số câu trả lời khớp với nhãn
            predictions = torch.argmax(outputs, dim=2)
            correct_answers += (predictions == labels).sum().item()
            total_answers += labels.numel() # Tổng số câu hỏi (batch_size * 40)
            
    avg_loss = total_loss / len(dataloader)
    accuracy = correct_answers / total_answers
    return avg_loss, accuracy

# --- QUY TRÌNH HUẤN LUYỆN ĐẦY ĐỦ ---
if __name__ == "__main__":
    with mlflow.start_run() as run:
        print("🚀 Khởi động Dataloader với đủ Train, Val, Test...")
        
        # 1. LOAD 3 TẬP DỮ LIỆU TỪ DVC PIPELINE
        train_data = OMRDataset('data/processed/train.csv', 'data/synthetic_images/')
        val_data   = OMRDataset('data/processed/val.csv', 'data/synthetic_images/')
        test_data  = OMRDataset('data/processed/test.csv', 'data/synthetic_images/')
        
        train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
        val_loader   = DataLoader(val_data, batch_size=8, shuffle=False)
        test_loader  = DataLoader(test_data, batch_size=8, shuffle=False)

        model = OMRClassifier()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        epochs = 3

        mlflow.log_params({
            "dvc_git_commit": get_git_hash(),
            "train_samples": len(train_data),
            "val_samples": len(val_data),
            "test_samples": len(test_data)
        })

        print("🧠 Bắt đầu Training Loop...")
        for epoch in range(epochs):
            # --- PHA 1: HỌC TẬP (TRAIN) ---
            model.train()
            train_loss = 0.0
            for images, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs.permute(0, 2, 1), labels.long())
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            
            # --- PHA 2: THI THỬ (VALIDATION) ---
            val_loss, val_acc = evaluate_model(model, val_loader, criterion)
            
            # --- GHI LOG REAL-TIME LÊN MLFLOW ---
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)
            
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # --- PHA 3: THI THẬT (TESTING) ---
        print("🎯 Đang thực hiện bài test cuối cùng trên tập Test chưa từng nhìn thấy...")
        test_loss, test_acc = evaluate_model(model, test_loader, criterion)
        
        # Ghi log chỉ số test (Không có step, vì nó là điểm số tổng kết)
        mlflow.log_metric("final_test_loss", test_loss)
        mlflow.log_metric("final_test_accuracy", test_acc)
        print(f"🏆 Kết quả Test Cuối cùng: Accuracy = {test_acc:.4f}")

        # Đăng ký model nếu kết quả Test đủ tốt
        if test_acc > 0.8: # Ví dụ: Độ chính xác > 80% mới cho lên Registry
            mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path="model_weights",
                registered_model_name="OMR_Grading_Engine"
            )
            print("📦 Model đạt chuẩn! Đã lưu lên MinIO và Registry.")
        else:
            print("⚠️ Điểm Test quá thấp, từ chối lưu model vào Registry.")