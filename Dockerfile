# Sử dụng base image Python 3.10 siêu nhẹ
FROM python:3.10-slim

# Khởi tạo thư mục làm việc bên trong bộ chứa
WORKDIR /app

# Ưu tiên copy requirements và cài đặt trước để tận dụng cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Mở cổng giao tiếp
EXPOSE 8000

# Kích hoạt máy chủ uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]