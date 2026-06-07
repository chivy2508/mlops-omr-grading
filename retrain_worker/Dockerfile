FROM python:3.9-slim

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
WORKDIR /app

# 1. Cập nhật pip thông minh nhất
RUN pip install --upgrade pip setuptools wheel

# 2. Cài DVC S3 (Nó sẽ tự cài boto3 và botocore phiên bản cũ phù hợp)
RUN pip install --no-cache-dir "dvc[s3]"

# 3. Cài MLflow (Nó sẽ tự dùng lại boto3 mà DVC vừa cài)
RUN pip install --no-cache-dir mlflow

# 4. Cài các thư viện phổ thông còn lại (ĐÃ XÓA BOTO3 Ở ĐÂY)
RUN pip install --no-cache-dir prometheus_client requests numpy opencv-python-headless psycopg2-binary pandas filelock

COPY worker.py .
CMD ["python", "worker.py"]