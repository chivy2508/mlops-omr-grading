FROM python:3.9-slim

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
WORKDIR /app

RUN pip install --upgrade pip setuptools wheel

RUN pip install --no-cache-dir "dvc[s3]"

RUN pip install --no-cache-dir mlflow

RUN pip install --no-cache-dir prometheus_client requests numpy opencv-python-headless psycopg2-binary pandas filelock

COPY worker.py .
CMD ["python", "worker.py"]