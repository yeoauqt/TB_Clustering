import streamlit as st
import pandas as pd
import xgboost as xgb
import joblib
import numpy as np
import plotly.graph_objects as go

# --- 1. SETTING & CUSTOM CSS (เป๊ะตาม Figma) ---
st.set_page_config(page_title="TB Treatment Prediction", page_icon="🏥", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;700&display=swap');
    
    html, body, [class*="css"] { font-family: 'Sarabun', sans-serif; }
    .main { background-color: #ffffff; }

    /* ปรับแต่งช่อง Input ให้ดู Minimal ตาม Figma */
    .stNumberInput, .stSelectbox {
        margin-bottom: -15px !important;
    }
    
    div[data-baseweb="input"], div[data-baseweb="select"] {
        border-radius: 10px !important;
    }

    /* กรอบการ์ดสีเหลือง */
    .input-card {
        background-color: #FFF9E1;
        border-radius: 25px;
        padding: 30px;
        border: 1px solid #FFE58F;
        min-height: 520px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    
    .card-title {
        font-weight: bold;
        font-size: 1.3rem;
        color: #856404;
        margin-bottom: 25px;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    /* ปุ่มวิเคราะห์สีเหลืองส้ม */
    .stButton>button {
        background-color: #FFD600 !important;
        color: black !important;
        font-weight: bold !important;
        border-radius: 15px !important;
        width: 300px !important;
        height: 3.5rem;
        border: none !important;
        font-size: 1.2rem !important;
        margin-top: 20px;
        box-shadow: 0 4px 15px rgba(255, 214, 0, 0.3);
    }

    /* ส่วนผลลัพธ์ */
    .result-section {
        background-color: #f8f9fa;
        border-radius: 25px;
        padding: 30px;
        margin-top: 40px;
    }

    .warning-box {
        background-color: #fce8e6;
        color: #b71c1c;
        padding: 20px;
        border-radius: 15px;
        border-left: 6px solid #d32f2f;
        margin-bottom: 15px;
    }

    .info-box {
        background-color: #f1f0ef;
        color: #4a4a4a;
        padding: 20px;
        border-radius: 15px;
        border-left: 6px solid #9e9e9e;
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

# --- 3. UI HEADER ---
st.markdown("<h1 style='text-align: center; color: #333;'>🏥 ระบบพยากรณ์ความเสี่ยงผลการรักษาวัณโรค (TB)</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666;'>ประเมินความเสี่ยงและวิเคราะห์ความเสี่ยงในการรักษาผู้ป่วยวัณโรคด้วยเทคโนโลยี AI</p>", unsafe_allow_html=True)
st.write("---")

# --- 4. PREDICTION FORM ---
with st.form("main_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="input-card"><div class="card-title">👤 ข้อมูลพื้นฐาน</div>', unsafe_allow_html=True)
        age = st.number_input("อายุ (ปี)", 0, 120, 45)
        gen = st.selectbox("เพศ", ["ชาย", "หญิง"])
        bmi = st.number_input("ดัชนีมวลกาย (BMI)", 10.0, 50.0, 20.0)
        dur = st.number_input("ระยะเวลาการรักษา (วัน)", 0, 1000, 180)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="input-card"><div class="card-title">❤️ ประวัติสุขภาพ</div>', unsafe_allow_html=True)
        hiv = st.selectbox("สถานะ HIV", ["ไม่ติดเชื้อ (Negative)", "ติดเชื้อ (Positive)", "ไม่ทราบ"])
        ckd = st.selectbox("โรคไตเรื้อรัง (CKD)", ["ไม่เป็น", "เป็น"])
        copd = st.selectbox("โรคปอดอุดกั้นเรื้อรัง (COPD)", ["ไม่เป็น", "เป็น"])
        liv = st.selectbox("โรคตับ", ["ไม่เป็น", "เป็น"])
        dm = st.selectbox("โรคเบาหวาน", ["ไม่เป็น", "เป็น"])
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="input-card"><div class="card-title">🧪 ผลการตรวจ</div>', unsafe_allow_html=True)
        afb = st.selectbox("ผล AFB เดือนที่ 1", ["Negative", "1+", "2+", "3+", "Scanty"])
        pos = st.selectbox("ตำแหน่งการติดเชื้อ", ["ในปอด", "นอกปอด", "ในและนอกปอด"])
        f_u = st.number_input("จำนวนครั้งที่ติดตาม", 0, 50, 1)
        arv = st.selectbox("สถานะ ARV", ["ไม่ได้รับ", "ได้รับ"])
        st.markdown('</div>', unsafe_allow_html=True)

    # ปุ่มวิเคราะห์อยู่ตรงกลาง
    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    sub = st.form_submit_button("วิเคราะห์ผลการรักษา")
    st.markdown("</div>", unsafe_allow_html=True)

# --- 5. PREDICTION LOGIC & RESULT ---
if sub:
    if model is not None:
        # (Encoding Logic เดียวกับที่คุณใช้อยู่)
        def encode(d):
            m = {
                'Gender': {'ชาย': 0, 'หญิง': 1},
                'HIV': {'ไม่ติดเชื้อ (Negative)': 0, 'ติดเชื้อ (Positive)': 1, 'ไม่ทราบ': 2},
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

        raw = {'Age': age, 'Gender': gen, 'BMI': bmi, 'Duration': dur, 'HIV': hiv, 'Diabetes Mellitus': dm, 'Chronic Kidney Disease': ckd, ' Chronic Obstructive Pulmonary Disease': copd, 'Liver Disease': liv, ' AFB resulf of first month': afb, 'position of TB': pos, '(TB F/U) follow up': f_u, 'Treatment of ARV': arv}
        enc = encode(raw)
        df = pd.DataFrame(0, index=[0], columns=model_features)
        for k in enc:
            if k in df.columns: df[k] = enc[k]
        
        prob = model.predict_proba(df)[0][1]
        risk_text = "สูง" if prob > 0.6 else "ปานกลาง" if prob > 0.3 else "ต่ำ"

        # --- ส่วนแสดงผลเป๊ะตาม Figma ---
        st.markdown('<div class="result-section">', unsafe_allow_html=True)
        st.subheader("📊 ผลการวิเคราะห์")
        
        res_col1, res_col2 = st.columns([1, 1.2])
        
        with res_col1:
            fig = go.Figure(go.Indicator(
                mode = "gauge+number", value = prob * 100,
                number = {'suffix': "%", 'font': {'size': 60}},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#333"},
                    'steps': [
                        {'range': [0, 30], 'color': "#A3E4D7"},
                        {'range': [30, 60], 'color': "#F9E79F"},
                        {'range': [60, 100], 'color': "#F1948A"}
                    ]
                }
            ))
            fig.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(f"<center><b>ระดับความเสี่ยง: {risk_text}</b></center>", unsafe_allow_html=True)

        with res_col2:
            st.markdown(f"""
                <div class="warning-box">
                    <b>⚠️ คำเตือน:</b> ผู้ป่วยอยู่ในกลุ่มเสี่ยงต่อผลการรักษาที่ไม่สำเร็จ (BMI < 18.5 หรืออายุมาก) ควรได้รับการติดตามอย่างใกล้ชิด
                </div>
                <div class="info-box">
                    <b>💡 หมายเหตุ:</b> ผลลัพธ์เป็นเพียงการพยากรณ์เบื้องต้นจาก AI ไม่สามารถใช้แทนการวินิจฉัยของแพทย์ได้
                </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.error("โหลดโมเดลไม่สำเร็จ")
