import sys
import os
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main import app  

client = TestClient(app)

def test_metrics_endpoint_is_alive():
    """Kiểm tra trạm phát sóng Prometheus có hoạt động không"""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "omr_total_requests" in response.text

def test_predict_endpoint_no_file():
    """Kiểm tra API có chặn request gửi thiếu file ảnh không"""
    response = client.post("/predict")
    assert response.status_code == 422  

def test_predict_endpoint_wrong_file_type():
    """Kiểm tra API có từ chối file không phải ảnh không (VD: file text)"""
    files = {"file": ("test.txt", b"Day la file text, khong phai anh", "text/plain")}
    response = client.post("/predict", files=files)
    assert response.status_code == 400
    assert response.json()["detail"] == "Chỉ chấp nhận định dạng JPG/PNG"