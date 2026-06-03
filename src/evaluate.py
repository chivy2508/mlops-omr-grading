import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()
model_name = "OMR_Grading_Engine"

# (Giả định bạn đã code xong phần tính toán new_accuracy)
new_accuracy = 0.96 
new_model_version = "3" # Phiên bản của mô hình bạn vừa train xong

try:
    prod_model = client.get_latest_versions(model_name, stages=["Production"])[0]
    prod_accuracy = client.get_run(prod_model.run_id).data.metrics.get("accuracy", 0.0)
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