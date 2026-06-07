import mlflow
from mlflow.tracking import MlflowClient
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import json

client = MlflowClient()
model_name = "OMR_Grading_Engine"

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

try:
    prod_model = client.get_latest_versions(model_name, stages=["Production"])[0]
    prod_accuracy = client.get_run(prod_model.run_id).data.metrics.get("final_test_accuracy", 0.0)
except IndexError:
    prod_accuracy = 0.0 

if new_accuracy >= prod_accuracy * 0.98:
    print(f"✅ Pass! Mô hình mới ({new_accuracy}) tốt hơn/ngang mô hình cũ ({prod_accuracy})")
    
    # ĐÂY LÀ ĐOẠN CODE "THAY TRIỀU ĐỔI ĐẠI" CỦA MLFLOW
    print(f"🚀 Đang đưa mô hình phiên bản {new_model_version} lên Production...")
    client.transition_model_version_stage(
        name=model_name,
        version=new_model_version,
        stage="Production",
        archive_existing_versions=True  # Phép thuật nằm ở đây: Tự động gỡ tag model cũ!
    )
    print("🎉 Triển khai thành công!")
    
else:
    print(f"🚨 Failed! Mô hình mới quá tệ. Hủy deploy.")
    import sys
    sys.exit(1)