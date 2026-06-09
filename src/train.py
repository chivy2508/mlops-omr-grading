import os
import subprocess
import mlflow
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
import numpy as np
import random
from torchvision import datasets, transforms
from dotenv import load_dotenv

import yaml
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

load_dotenv()

batch_size = config['model']['batch_size']
epochs = config['model']['epochs']

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)

mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(mlflow_tracking_uri)
mlflow.set_experiment("OMR_Grading_System")


def get_git_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except:
        return "unknown"

class MobileNetBubble(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super(MobileNetBubble, self).__init__()
        self.backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        
        # Sửa lớp đầu tiên thành 1 kênh màu
        original_conv = self.backbone.features[0][0]
        self.backbone.features[0][0] = nn.Conv2d(
            1, original_conv.out_channels, kernel_size=original_conv.kernel_size,
            stride=original_conv.stride, padding=original_conv.padding, bias=False
        )
        
        for param in self.backbone.features.parameters():
            param.requires_grad = False  
        
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate), 
            nn.Linear(in_features, 2) 
        )

    def forward(self, x):
        return self.backbone(x)



def evaluate_model(model, dataloader, criterion):
    device = next(model.parameters()).device
    model.eval()
    total_loss = 0.0
    correct_answers = 0
    total_answers = 0
    with torch.no_grad(): 
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
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

        num_workers = config['model'].get('num_workers', 4)

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader   = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_loader  = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        lr = config['model'].get('learning_rate', 0.001)
        patience = config['model'].get('patience', 2)
        weight_decay = config['model'].get('weight_decay', 1e-5) 
        dropout_rate = config['model'].get('dropout', 0.2)
        class_weight_0 = config['model'].get('class_weight_0', 1.0)
        class_weight_1 = config['model'].get('class_weight_1', 3.0)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MobileNetBubble(dropout_rate=dropout_rate).to(device)

        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("learning_rate", lr)
        mlflow.log_param("patience", patience)
        mlflow.log_param("weight_decay", weight_decay)
        mlflow.log_param("dropout", dropout_rate)
        mlflow.log_param("class_weight_0", class_weight_0)
        mlflow.log_param("class_weight_1", class_weight_1)
        mlflow.log_param("num_workers", num_workers)
        mlflow.log_param("git_commit", get_git_hash()) 

        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=lr, 
            weight_decay=weight_decay  
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=3
        )

        class_weights = torch.tensor([class_weight_0, class_weight_1], dtype=torch.float).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        best_val_loss = float('inf')
        trigger_times = 0

        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            avg_train_loss = train_loss / len(train_loader)
            val_loss, val_acc = evaluate_model(model, val_loader, criterion)
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)
            mlflow.log_metric("learning_rate_step", current_lr, step=epoch)
            print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | LR: {current_lr}")

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
            import sys
            sys.exit(1) 
