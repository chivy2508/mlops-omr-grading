import streamlit as st
import requests

# Cấu hình giao diện
st.set_page_config(page_title="Demo OMR API", page_icon="⚙️", layout="centered")

st.title("⚙️ Giao diện API Chấm Thi OMR")
st.info("Lưu ý: Đây là web nội bộ dùng để test độ chính xác của API & Model, không phải Web chính thức của dự án.")

# Khung upload ảnh
uploaded_file = st.file_uploader("Tải ảnh bài thi lên đây (JPG, JPEG, PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Ảnh bài thi đầu vào", use_container_width=True)
    
    if st.button("🚀 Gửi qua API Chấm Điểm"):
        with st.spinner("Đang gọi qua cổng 8000 của API..."):
            try:
                # Gói file lại để gửi đi
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                
                # Gọi thẳng tên service 'omr_api' và cổng 8000 trong mạng LAN của Docker Compose
                FASTAPI_URL = "http://omr_api:8000/predict" 
                
                response = requests.post(FASTAPI_URL, files=files)
                
                if response.status_code == 200:
                    data = response.json()
                    st.success("🎉 API trả kết quả thành công!")
                    
                    st.subheader("📋 Phản hồi từ API (JSON):")
                    st.json(data) #
                else:
                    st.error(f"API báo lỗi: Code {response.status_code}")
                    
            except Exception as e:
                st.error(f"Không kết nối được với API: {e}")