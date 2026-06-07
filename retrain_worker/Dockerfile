FROM python:3.9-slim

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
WORKDIR /app

RUN pip install prometheus_client requests numpy opencv-python-headless psycopg2-binary boto3 mlflow pandas filelock dvc dvc-s3 pandas filelock dvc dvc-s3
COPY worker.py .
CMD ["python", "worker.py"]