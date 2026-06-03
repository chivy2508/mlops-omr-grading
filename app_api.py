import streamlit as st
import requests
import base64
import cv2
import pandas as pd
import numpy as np
import torch

st.set_page_config(page_title="Demo OMR API", page_icon="⚡", layout="wide")

def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception as e:
        return ""

logo_base64 = get_base64_image("logo_fastapi.png")

st.markdown("""
<style>
[data-testid="stHeader"] {
    background-color: transparent !important;
}
[data-testid="stToolbar"] button {
    background-color: transparent !important; 
}
[data-testid="stToolbar"] button svg {
    fill: #000000 !important; 
    stroke: #000000 !important;
    stroke-width: 0.8px !important; 
}
.stApp {
    background-image: url("https://cdn2.fptshop.com.vn/unsafe/Uploads/images/tin-tuc/174931/Originals/background%20gradient%20(28).jpg"); 
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}
[data-testid="stAppViewContainer"] {
    background-color: transparent !important;
}
.block-container {
    background: rgba(255, 255, 255, 0.08) !important;
    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
    border: 1px solid rgba(255, 255, 255, 0.3) !important;
    border-radius: 25px !important;
    padding: 2rem 3rem !important;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4) !important;
}
h1, h2, h3, p, label, .stMarkdown, div[data-testid="stText"] {
    color: #ffffff !important;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.5); 
}
[data-testid="stFileUploadDropzone"] {
    background-color: rgba(0, 0, 0, 0.3) !important;
    border: 2px dashed rgba(255, 255, 255, 0.6) !important;
    border-radius: 15px !important;
}
[data-testid="stJson"], pre, code {
    background-color: rgba(14, 17, 23, 0.85) !important; 
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    border-radius: 10px !important;
    padding: 15px !important;
}
.stButton > button {
    background: linear-gradient(135deg, #ff7e5f, #feb47b) !important; 
    border: none !important;
    border-radius: 10px !important;
    font-weight: bold !important;
    color: white !important;
    box-shadow: 0 4px 15px rgba(255, 126, 95, 0.4) !important;
    transition: all 0.3s ease !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(255, 126, 95, 0.6) !important;
}
.stButton > button * {
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
    <div style='display: flex; align-items: center; gap: 15px; margin-bottom: 25px; border-bottom: 1px solid rgba(255,255,255,0.3); padding-bottom: 15px;'>
        <img src='data:image/png;base64,{logo_base64}' style='width: 60px; height: 60px; filter: drop-shadow(0px 4px 6px rgba(0,0,0,0.4));'>
        <h1 style='margin: 0; padding: 0; font-size: 2.5rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.5);'>API Chấm Thi OMR</h1>
    </div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

# ================= CỘT TRÁI: NHẬP LIỆU =================
with col1:
    st.subheader("📤 1. Tải ảnh & Gửi")
    uploaded_file = st.file_uploader("Tải ảnh bài thi lên đây (JPG, JPEG, PNG)...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # 👇 BỎ DẤU # Ở DÒNG NÀY ĐỂ HIỆN LẠI ẢNH NHÉ
        st.image(uploaded_file, caption="Ảnh bài thi đầu vào", use_container_width=True)
        
        submit_button = st.button("🚀 Gửi qua API Chấm Điểm", use_container_width=True, type="primary")
    else:
        submit_button = False

# ================= CỘT PHẢI: KẾT QUẢ =================
with col2:
    st.subheader("📥 2. Kết quả chấm & Phản hồi")
    
    if "api_data" not in st.session_state:
        st.session_state.api_data = None

    if uploaded_file is not None and submit_button:
        with st.spinner("Đang đợi Backend kéo Model và xử lý ảnh..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                FASTAPI_URL = "http://omr_api:8000/predict" 
                
                response = requests.post(FASTAPI_URL, files=files)
                
                if response.status_code == 200:
                    st.session_state.api_data = response.json()
                    st.success("🎉 API trả kết quả thành công!")
                else:
                    st.error(f"API báo lỗi: Code {response.status_code}")
            except Exception as e:
                st.error(f"Không kết nối được với API: {e}")
                
    elif uploaded_file is None:
        st.write("👈 *Hãy tải ảnh bài thi lên ở cột bên trái để xem kết quả tại đây.*")

    if st.session_state.api_data is not None:
        data = st.session_state.api_data
        
        st.markdown("### ✍️ Kiểm tra và Dạy lại mô hình")
        drift_score = data.get("drift_score", 0.0)
        
        if drift_score > 0.05:
            st.warning(f"⚠️ Cảnh báo: Độ lệch dữ liệu hơi cao (Drift: {drift_score}). Vui lòng dò kỹ các chấm đỏ và sửa lại bảng dưới đây nếu máy đoán sai!")
        else:
            st.info("💡 Trạng thái hệ thống tốt. Nếu tình cờ thấy lỗi nhỏ, bạn có thể sửa lại để giúp mô hình thông minh hơn.")
        
        predictions = data.get("predictions", [])
        if predictions:
            df = pd.DataFrame(predictions)
            display_df = df[["cau", "dap_an"]] 
        else:
            display_df = pd.DataFrame({"cau": [1, 2, 3], "dap_an": ["A", "B", "C"]})
            
        with st.form("feedback_form"):
            edited_df = st.data_editor(display_df, use_container_width=True)
            submit_feedback = st.form_submit_button("Lưu đáp án chuẩn & Cập nhật kho Data 🚀", type="primary", use_container_width=True)

            if submit_feedback:
                with st.spinner("Đang lưu dữ liệu chuẩn..."):
                    try:
                        feedback_files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                        feedback_data = {"correct_labels": edited_df.to_json(orient="records")}
                        
                        FEEDBACK_URL = "http://omr_api:8000/feedback"
                        feedback_response = requests.post(FEEDBACK_URL, files=feedback_files, data=feedback_data)
                        
                        if feedback_response.status_code == 200:
                            st.success("✅ Cảm ơn bạn! Dữ liệu Ground Truth đã được ghi nhận an toàn.")
                            st.balloons()
                            
                            st.session_state.api_data = None
                            st.rerun()
                        else:
                            st.error("❌ Có lỗi xảy ra khi lưu phản hồi.")
                    except Exception as e:
                        st.error(f"Không thể kết nối đến server lưu trữ: {e}")