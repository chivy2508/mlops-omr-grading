import os
import subprocess
import mlflow
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn as nn
from torch.utils.data import DataLoader
from src.dataset import OMRDataset # Gọi file dataset.py bạn đã tạo lúc nãy


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

class MobileNetOMR(nn.Module):
    def __init__(self, num_questions=40, num_choices=4):
        super(MobileNetOMR, self).__init__()

        self.backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)


        original_conv = self.backbone.features[0][0]
        self.backbone.features[0][0] = nn.Conv2d(
            in_channels=1, # Ảnh xám
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False
        )

        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, num_questions * num_choices)
        self.num_questions = num_questions
        self.num_choices = num_choices



    def forward(self, x):

        x = self.backbone(x)

        # Reshape về dạng [Batch_size, 40 câu, 4 đáp án]

        return x.view(-1, self.num_questions, self.num_choices)



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



        model = MobileNetOMR()

       

        # Chế độ học: Chúng ta sẽ dùng learning rate nhỏ hơn (0.0001)

        # vì chúng ta đang Fine-tuning trên một model đã mạnh sẵn

        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

        criterion = nn.CrossEntropyLoss()

       

        epochs = 10 # Tăng số Epoch lên nhưng sẽ dùng Early Stopping

        best_val_loss = float('inf')

        patience = 2 # Nếu 2 lần val_loss không giảm thì dừng

        trigger_times = 0



        for epoch in range(epochs):

            # PHA 1: TRAIN

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

           

            # PHA 2: VALIDATION

            val_loss, val_acc = evaluate_model(model, val_loader, criterion)

           

            # LOGGING

            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)

            mlflow.log_metric("val_loss", val_loss, step=epoch)

            mlflow.log_metric("val_accuracy", val_acc, step=epoch)

           

            print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")



            # --- EARLY STOPPING LOGIC ---

            if val_loss < best_val_loss:

                best_val_loss = val_loss

                trigger_times = 0

                # Lưu model tốt nhất tạm thời

                torch.save(model.state_dict(), "best_model.pth")

            else:

                trigger_times += 1

                if trigger_times >= patience:

                    print(f"🛑 Early Stopping tại Epoch {epoch+1}!")

                    break



        # Tải lại trọng số tốt nhất trước khi test

        model.load_state_dict(torch.load("best_model.pth"))

       

        # --- PHA 3: TEST CUỐI CÙNG ---

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

