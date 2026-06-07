FROM python:3.9-slim
WORKDIR /app
RUN pip install prometheus_client requests numpy opencv-python-headless psycopg2-binary boto3 mlflow
COPY worker.py .
CMD ["python", "worker.py"]