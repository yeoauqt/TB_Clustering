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
        m = joblib.load('xgb_tb_model.pkl')
        f = joblib.load('model_features.pkl')
        return m, f
    except:
        return None, None

model, model_features = load_assets()

# --- 3. ENCODING ---
def encode_inputs(d):
    m = {
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
    for k, v in m.items():
        if k in d: d[k] = v.get(d[k], 0)
    return d

# --- 4. UI ---
st.title("🏥 ระบบพยากรณ์ความเสี่ยงผลการรักษาวัณโรค (TB)")

with st.form("main_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="input-card"><div class="card-header">👤 ข้อมูลพื้นฐาน</div>', unsafe_allow_html=True)
        age = st.number_input("อายุ (Age)", 0, 120, 45)
        gen = st.selectbox("เพศ (Gender)", ["ชาย", "หญิง"])
        bmi = st.number_input("ดัชนีมวลกาย (BMI)", 10.0, 50.0, 20.0)
        dur = st.number_input("ระยะเวลาการรักษา (วัน)", 0, 1000, 180)
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="input-card"><div class="card-header">🩺 ประวัติสุขภาพ</div>', unsafe_allow_html=True)
        hiv = st.selectbox("สถานะ HIV", ["ไม่ติดเชื้อ (Negative)", "ติดเชื้อ (Positive)", "ไม่ทราบ"])
        dm = st.selectbox("เบาหวาน", ["ไม่เป็น", "เป็น"])
        ckd = st.selectbox("โรคไตเรื้อรัง", ["ไม่เป็น", "เป็น"])
        copd = st.selectbox("โรคปอดอุดกั้น", ["ไม่เป็น", "เป็น"])
        liv = st.selectbox("โรคตับ", ["ไม่เป็น", "เป็น"])
        st.markdown('</div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="input-card"><div class="card-header">🧪 ผลตรวจ</div>', unsafe_allow_html=True)
        afb = st.selectbox("ผล AFB เดือนแรก", ["Negative", "1+", "2+", "3+", "Scanty"])
        pos = st.selectbox("ตำแหน่งของโรค", ["ในปอด", "นอกปอด", "ในและนอกปอด"])
        f_u = st.number_input("จำนวนครั้งที่ Follow up", 0, 100, 1)
        arv = st.selectbox("ได้รับยา ARV หรือไม่", ["ไม่ได้รับ", "ได้รับ"])
        st.markdown('</div>', unsafe_allow_html=True)
    
    sub = st.form_submit_button("วิเคราะห์ความเสี่ยงและพยากรณ์ผล")

if sub:
    if model is None:
        st.error("⚠️ ไม่สามารถโหลดโมเดลได้ ตรวจสอบไฟล์ .pkl")
    else:
        raw = {
            'Age': age, 'Gender': gen, 'BMI': bmi, 'Duration': dur,
            'HIV': hiv, 'Diabetes Mellitus': dm, 'Chronic Kidney Disease': ckd,
            ' Chronic Obstructive Pulmonary Disease': copd, 'Liver Disease': liv,
            ' AFB resulf of first month': afb, 'position of TB': pos,
            '(TB F/U) follow up': f_u, 'Treatment of ARV': arv
        }
        enc = encode_inputs(raw)
        df = pd.DataFrame(0, index=[0], columns=model_features)
        for k in enc:
            if k in df.columns: df[k] = enc[k]
        
        prob = model.predict_proba(df)[0][1]
        st.divider()
        st.subheader("📊 ผลการวิเคราะห์")
        
        # กราฟ Gauge ฉบับปลอดภัย Syntax ไม่หลุดแน่นอน
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prob * 100,
            number = {'suffix': "%"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "red"}
                ]
            }
        ))
        st.plotly_chart(fig)
