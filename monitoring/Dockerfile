FROM python:3.9-slim
WORKDIR /app
RUN pip install prometheus_client requests numpy opencv-python-headless
COPY monitor.py .
CMD ["python", "monitor.py"]
