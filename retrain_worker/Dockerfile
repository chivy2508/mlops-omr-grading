FROM python:3.9-slim

RUN apt-get update && apt-get install -y git gcc libpq-dev python3-dev && rm -rf /var/lib/apt/lists/*
WORKDIR /app

RUN pip install --upgrade pip setuptools wheel

RUN pip install --no-cache-dir \
    dvc==3.45.0 \
    dvc-s3==3.1.0 \
    mlflow==2.10.2 \
    boto3==1.34.69 \
    aiobotocore==2.12.3 \
    s3fs==2024.3.1 \
    prometheus_client==0.17.1 \
    requests \
    numpy \
    opencv-python-headless \
    psycopg2-binary==2.9.9 \
    pandas \
    filelock==3.13.1 \
    schedule==1.2.1

COPY worker.py .
CMD ["python", "worker.py"]