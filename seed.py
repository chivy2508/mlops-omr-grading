import os
import sys
import time
from dotenv import load_dotenv

load_dotenv()

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MLFLOW_S3_ENDPOINT_URL = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "admin")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "password123")

print("Seed script: preparing demo environment (MinIO buckets + MLflow model)...")

try:
    import boto3
    from botocore.client import Config
except Exception:
    print("Please install boto3 (pip install boto3) before running seed.py")
    sys.exit(1)

try:
    import mlflow
    from mlflow.tracking import MlflowClient
    import mlflow.pytorch
except Exception:
    print("Please install mlflow and mlflow[pytorch] before running seed.py")
    sys.exit(1)

try:
    import torch
except Exception:
    print("Please install torch before running seed.py")
    sys.exit(1)

PRETRAINED_PATH = os.path.join(os.path.dirname(__file__), "pretrained", "best_model.pth")

def ensure_buckets():
    print("- Ensuring MinIO buckets exist...")
    s3 = boto3.client(
        "s3",
        endpoint_url=MLFLOW_S3_ENDPOINT_URL,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )
    for b in ("omr-models", "omr-data"):
        try:
            s3.head_bucket(Bucket=b)
            print(f"  - bucket '{b}' exists")
        except Exception:
            try:
                s3.create_bucket(Bucket=b)
                print(f"  - created bucket '{b}'")
            except Exception as e:
                print(f"  - failed to create bucket {b}: {e}")

def register_model():
    if not os.path.exists(PRETRAINED_PATH):
        print(f"No pretrained model found at {PRETRAINED_PATH}. Put your best_model.pth there and re-run.")
        sys.exit(1)

    # import model architecture from training module
    try:
        from src.train import MobileNetBubble
    except Exception as e:
        print("Cannot import MobileNetBubble from src.train:", e)
        sys.exit(1)

    device = torch.device("cpu")
    model = MobileNetBubble()
    state = torch.load(PRETRAINED_PATH, map_location=device)
    # support both state_dict and full model saved
    try:
        if isinstance(state, dict) and any(k.startswith("module.") or k in state for k in state.keys()):
            model.load_state_dict(state)
        else:
            model.load_state_dict(state)
    except Exception:
        try:
            # if saved as whole model
            model = state
        except Exception as e:
            print("Failed to load model state:", e)
            sys.exit(1)

    model.eval()

    print("- Logging model to MLflow and registering as 'OMR_Grading_Engine'...")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    with mlflow.start_run() as run:
        try:
            mlflow.pytorch.log_model(pytorch_model=model, artifact_path="model_weights", registered_model_name="OMR_Grading_Engine")
            print("  - model logged and registered (if Registry available)")
        except Exception as e:
            print("  - failed to log/register model:", e)
            sys.exit(1)

    # Wait and transition latest to Production
    time.sleep(2)
    try:
        versions = client.get_latest_versions("OMR_Grading_Engine")
        if versions:
            latest = sorted(versions, key=lambda v: int(v.version))[-1]
            client.transition_model_version_stage(name="OMR_Grading_Engine", version=latest.version, stage="Production", archive_existing_versions=True)
            print(f"  - set model version {latest.version} to Production")
        else:
            print("  - no registered versions found after logging")
    except Exception as e:
        print("  - warning: could not transition model to Production:", e)

def main():
    ensure_buckets()
    register_model()
    print("Seed complete. You can now restart API to load Production model.")

if __name__ == "__main__":
    main()
