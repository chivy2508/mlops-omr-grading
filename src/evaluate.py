import mlflow
from mlflow.tracking import MlflowClient
import mlflow.exceptions 
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import json
import os

os.environ["AWS_ACCESS_KEY_ID"] = "admin"
os.environ["AWS_SECRET_ACCESS_KEY"] = "password123"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
mlflow.set_tracking_uri("http://localhost:5000")

client = MlflowClient()
model_name = "OMR_Grading_Engine"
alias_name = "production"

# Lấy model version mới nhất trong registry
latest_versions = client.search_model_versions(f"name='{model_name}'")
if not latest_versions:
    print("❌ Không tìm thấy model nào trong registry.")
    import sys
    sys.exit(1)

latest_version_obj = sorted(latest_versions, key=lambda v: int(v.version))[-1]
new_model_version = latest_version_obj.version

print(f"🔄 Đang đánh giá model version {new_model_version} trên tập test...")

model_uri = f"models:/{model_name}/{new_model_version}"
try:
    model = mlflow.pytorch.load_model(model_uri)
    model.eval()
except Exception as e:
    print(f"❌ Lỗi load model: {e}")
    import sys
    sys.exit(1)

# Chạy model trên tập test để lấy true_labels và predictions
test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])
test_data = datasets.ImageFolder(root='data/bubbles/test', transform=test_transform)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

true_labels = []
predictions = []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)
        true_labels.extend(labels.cpu().numpy())
        predictions.extend(preds.cpu().numpy())

report = classification_report(true_labels, predictions, output_dict=True, zero_division=0)

metrics = {
    "test_accuracy": report.get("accuracy", 0.0),
    "precision_filled": report.get("1", {}).get("precision", 0.0), # Độ chuẩn xác khi đoán ô có tô
    "recall_filled": report.get("1", {}).get("recall", 0.0),       # Độ bao phủ khi tìm ô có tô
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

new_accuracy = metrics["test_accuracy"]

client.log_metric(latest_version_obj.run_id, "test_accuracy", new_accuracy)
client.log_metric(latest_version_obj.run_id, "precision_filled", metrics["precision_filled"])
client.log_metric(latest_version_obj.run_id, "recall_filled", metrics["recall_filled"])

try:
    prod_model = client.get_model_version_by_alias(name=model_name, alias=alias_name)
    
    prod_accuracy = client.get_run(prod_model.run_id).data.metrics.get("test_accuracy", 0.0)
except mlflow.exceptions.RestException:
    
    prod_accuracy = 0.0 
    print("⚠️ Chưa có mô hình mang nhãn @production nào. Bỏ qua bước so sánh cũ.")

threshold = prod_accuracy * 0.98

if new_accuracy >= threshold:
    print(f"✅ Pass! Mô hình mới ({new_accuracy:.4f}) đạt chuẩn so với mô hình cũ ({prod_accuracy:.4f})")
    
    print(f"🚀 Đang gắn nhãn @{alias_name} cho model version {new_model_version}...")
    
    client.set_registered_model_alias(
        name=model_name,
        alias=alias_name,
        version=new_model_version
    )
    print("🎉 Triển khai thành công!")
    
else:
    print(f"🚨 Failed! Mô hình mới ({new_accuracy:.4f}) quá tệ so với ngưỡng chấp nhận ({threshold:.4f}). Hủy deploy.")
    import sys
    sys.exit(1)