import streamlit as st
import pandas as pd
import xgboost as xgb
import joblib
import numpy as np
import plotly.graph_objects as go

# --- 1. SETTING & CUSTOM CSS ---
st.set_page_config(page_title="TB Prediction", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;700&display=swap');
    html, body, [class*="css"] { font-family: 'Sarabun', sans-serif; }
    .input-card {
        background-color: #FFF9E1;
        border-radius: 15px;
        padding: 20px;
        border: 1px solid #FFE58F;
        margin-bottom: 10px;
        min-height: 550px;
    }
    .card-header { font-weight: bold; font-size: 1.2rem; color: #856404; margin-bottom: 15px; }
    .stButton>button { 
        background-color: #FFD600 !important; 
        color: black !important; 
        font-weight: bold !important;
        border-radius: 10px !important; width: 100% !important; height: 3.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. LOAD ASSETS ---
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('xgb_tb_model.pkl')
        features = joblib.load('model_features.pkl')
        return model, features
    except:
        return None, None

model, model_features = load_assets()

# --- 3. ENCODING ---
def encode_inputs(input_dict):
    mappings = {
        'Gender': {'ชาย': 0, 'หญิง': 1},
        'HIV': {'ไม่ติดเชื้อ (Negative)': 0, 'ติดเชื้อ (Positive)': 1, 'ไม่ทราบ/ไม่ตรวจ': 2},
        'Diabetes Mellitus': {'ไม่เป็น': 0, 'เป็น': 1},
        'Chronic Kidney Disease': {'ไม่เป็น': 0, 'เป็น': 1},
        ' Chronic Obstructive Pulmonary Disease': {'ไม่เป็น': 0, 'เป็น': 1},
        'Liver Disease': {'ไม่เป็น': 0, 'เป็น': 1},
        ' AFB resulf of first month': {'Negative': 0, '1+': 1, '2+': 2, '3+': 3, 'Scanty': 4},
        'position of TB': {'ในปอด': 0, 'นอกปอด': 1, 'ในและนอกปอด': 2},
        'Treatment of ARV': {'ไม่ได้รับ': 0, 'ได้รับ': 1}
    }
    encoded = input_dict.copy()
    for col, mapping in mappings.items():
        if col in encoded: encoded[col] = mapping.get(encoded[col], 0)
    return encoded

# --- 4. UI ---
st.title("🏥 ระบบพยากรณ์ความเสี่ยงผลการรักษาวัณโรค (TB)")

# บรรทัดนี้สำคัญ ถ้าโหลดโมเดลไม่ได้ ให้ลองรัน Form ก่อนเพื่อเช็คหน้าตา
if True: 
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="input-card"><div class="card-header">👤 ข้อมูลพื้นฐาน</div>', unsafe_allow_html=True)
            age = st.number_input("อายุ (Age)", 0, 120, 45)
            gender = st.selectbox("เพศ", ["ชาย", "หญิง"])
            bmi = st.number_input("BMI", 10.0, 50.0, 20.0)
            duration = st.number_input("ระยะเวลา (วัน)", 0, 1000, 180)
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="input-card"><div class="card-header">🩺 ประวัติสุขภาพ</div>', unsafe_allow_html=True)
            hiv = st.selectbox("สถานะ HIV", ["ไม่ติดเชื้อ (Negative)", "ติดเชื้อ (Positive)", "ไม่ทราบ"])
            dm = st.selectbox("เบาหวาน", ["ไม่เป็น", "เป็น"])
            ckd = st.selectbox("โรคไตเรื้อรัง", ["ไม่เป็น", "เป็น"])
            copd = st.selectbox("โรคปอดอุดกั้น", ["ไม่เป็น", "เป็น"])
            liver = st.selectbox("โรคตับ", ["ไม่เป็น", "เป็น"])
            st.markdown('</div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="input-card"><div class="card-header">🧪 ผลการตรวจ</div>', unsafe_allow_html=True)
            afb = st.selectbox("ผล AFB เดือนแรก", ["Negative", "1+", "2+", "3+", "Scanty"])
            pos_tb = st.selectbox("ตำแหน่งของโรค", ["ในปอด", "นอกปอด", "ในและนอกปอด"])
            follow_up = st.number_input("จำนวนครั้ง Follow up", 0, 100, 1)
            arv = st.selectbox("ได้รับยา ARV", ["ไม่ได้รับ", "ได้รับ"])
            st.markdown('</div>', unsafe_allow_html=True)
        
        submit_btn = st.form_submit_button("วิเคราะห์ความเสี่ยง")

    if submit_btn:
        if model is None:
            st.error("⚠️ ไม่สามารถทำนายได้ เนื่องจากโหลดไฟล์โมเดล (.pkl) ไม่สำเร็จ ตรวจสอบชื่อไฟล์ใน GitHub")
        else:
            raw_inputs = {
                'Age': age, 'Gender': gender, 'BMI': bmi, 'Duration': duration,
                'HIV': hiv, 'Diabetes Mellitus': dm, 'Chronic Kidney Disease': ckd,
                ' Chronic Obstructive Pulmonary Disease': copd, 'Liver Disease': liver,
                ' AFB resulf of first month': afb, 'position of TB': pos_tb,
                '(TB F/U) follow up': follow_up, 'Treatment of ARV': arv
            }
            encoded_data = encode_inputs(raw_inputs)
            input_df = pd.DataFrame(0, index=[0], columns=model_features)
            for col in encoded_data:
                if col in input_df.columns: input_df[col] = encoded_data[col]
            
            prob = model.predict_proba(input_df)[0][1]
            st.divider()
            st.subheader("📊 ผลการวิเคราะห์")
            fig = go.Figure(go.Indicator(
                mode = "gauge+number", value = prob * 100,
                number = {'suffix': "%"},
                gauge = {'
