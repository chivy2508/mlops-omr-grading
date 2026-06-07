import os
import subprocess
import mlflow
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import random
from torchvision import datasets, transforms

import yaml
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

batch_size = config['model']['batch_size']
epochs = config['model']['epochs']

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Tắt tính năng tối ưu non-deterministic của cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)

os.environ["AWS_ACCESS_KEY_ID"] = "admin"
os.environ["AWS_SECRET_ACCESS_KEY"] = "password123"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("OMR_Grading_System")


def get_git_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except:
        return "unknown"

class MobileNetBubble(nn.Module):
    def __init__(self):
        super(MobileNetBubble, self).__init__()
        self.backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        
        # Sửa lớp đầu tiên thành 1 kênh màu
        original_conv = self.backbone.features[0][0]
        self.backbone.features[0][0] = nn.Conv2d(
            1, original_conv.out_channels, kernel_size=original_conv.kernel_size,
            stride=original_conv.stride, padding=original_conv.padding, bias=False
        )
        
        # --- LỚP PHÒNG NGỰ 2: ĐÓNG BĂNG BACKBONE ---
        for param in self.backbone.features.parameters():
            param.requires_grad = False  # KHÔNG cập nhật trọng số phần này
        
        # --- LỚP PHÒNG NGỰ 3: DROPOUT MẠNH TAY ---
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.5), # Tắt ngẫu nhiên 50% nơ-ron để chống học vẹt
            nn.Linear(in_features, 2) # Output 2 class: 0 (Trống) và 1 (Đã tô)
        )

    def forward(self, x):
        return self.backbone(x)



def evaluate_model(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    correct_answers = 0
    total_answers = 0
    with torch.no_grad(): 
        for images, labels in dataloader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1)
            correct_answers += (predictions == labels).sum().item()
            total_answers += labels.numel() 

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_answers / total_answers
    return avg_loss, accuracy


if __name__ == "__main__":

    with mlflow.start_run() as run:

        print("🚀 Khởi động Dataloader với đủ Train, Val, Test...")
        train_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor()
        ])

        val_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])

        train_data = datasets.ImageFolder(root='data/bubbles/train', transform=train_transform)
        val_data   = datasets.ImageFolder(root='data/bubbles/val', transform=val_transform)
        test_data   = datasets.ImageFolder(root='data/bubbles/test', transform=val_transform)

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        test_loader  = DataLoader(test_data, batch_size=batch_size, shuffle=False)

        model = MobileNetBubble()

        # 2. Bạn có thể lấy luôn learning rate và patience từ config cho xịn!
        lr = config['model'].get('learning_rate', 0.001)
        patience = config['model'].get('patience', 2)

        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=lr, 
            weight_decay=1e-4  
        )

        criterion = nn.CrossEntropyLoss()
        
        # 3. Xóa dòng `epochs = 10` đi! Dùng biến epochs đã lấy từ YAML ở đầu file
        best_val_loss = float('inf')
        trigger_times = 0

        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for images, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            avg_train_loss = train_loss / len(train_loader)
            val_loss, val_acc = evaluate_model(model, val_loader, criterion)
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)
            print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                trigger_times = 0
                torch.save(model.state_dict(), "best_model.pth")
            else:
                trigger_times += 1
                if trigger_times >= patience:
                    print(f"🛑 Early Stopping tại Epoch {epoch+1}!")
                    break

        model.load_state_dict(torch.load("best_model.pth"))
        test_loss, test_acc = evaluate_model(model, test_loader, criterion)
        mlflow.log_metric("final_test_loss", test_loss)
        mlflow.log_metric("final_test_accuracy", test_acc)
        print(f"🏆 Kết quả Test Cuối cùng: Accuracy = {test_acc:.4f}")
        if test_acc > 0.8: 
            mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path="model_weights",
                registered_model_name="OMR_Grading_Engine"
            )
            print("📦 Model đạt chuẩn! Đã lưu lên MinIO và Registry.")
        else:
            print("⚠️ Điểm Test quá thấp, từ chối lưu model vào Registry.") 

