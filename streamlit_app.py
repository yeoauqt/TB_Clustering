import streamlit as st
import pandas as pd
import xgboost as xgb
import joblib
import numpy as np
import plotly.graph_objects as go

# --- 1. SETTING & CUSTOM CSS ---
st.set_page_config(page_title="TB Treatment Prediction", page_icon="🏥", layout="wide")

# CSS เพื่อจัดหน้าตาให้เป๊ะตาม Figma
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;700&display=swap');
    html, body, [class*="css"] { font-family: 'Sarabun', sans-serif; }
    
    /* สไตล์การ์ดข้อมูล */
    .input-card-container {
        display: flex;
        gap: 20px;
        margin-bottom: 20px;
    }

    .input-card {
        background-color: #FFF9E1;
        border-radius: 20px;
        padding: 30px 20px 20px 20px;
        border: 1px solid #FFE58F;
        flex: 1;
        min-height: 550px;
    }
    
    .card-header {
        font-weight: bold;
        font-size: 1.2rem;
        color: #856404;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    /* สไตล์ปุ่มวิเคราะห์ */
    .stButton>button {
        background-color: #FFD600 !important;
        color: black !important;
        font-weight: bold !important;
        border-radius: 10px !important;
        width: 250px !important;
        height: 3.5rem;
        border: none !important;
        display: block;
        margin: 0 auto; /* จัดกึ่งกลาง */
    }

    /* สไตล์ส่วนผลลัพธ์ */
    .result-container {
        background-color: #F0F2F6;
        border-radius: 20px;
        padding: 20px;
        margin-top: 20px;
    }

    .info-box {
        background-color: #D1ECF1;
        color: #0C5460;
        border-radius: 10px;
        padding: 15px;
        margin-top: 10px;
        border: 1px solid #BEE5EB;
    }

    .warning-box {
        background-color: #F8D7DA;
        color: #721C24;
        border-radius: 10px;
        padding: 15px;
        margin-top: 10px;
        border: 1px solid #F5C6CB;
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
    except FileNotFoundError:
        st.error("❌ ไม่พบไฟล์โมเดล! กรุณาตรวจสอบไฟล์ 'xgb_tb_model.pkl' และ 'model_features.pkl' ใน GitHub")
        return None, None

model, model_features = load_assets()

# --- 3. ENCODING FUNCTION ---
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

# --- 4. USER INTERFACE (UI) ---
st.title("🏥 ระบบพยากรณ์ความเสี่ยงผลการรักษาวัณโรค (TB)")
st.caption("ประเมินความเสี่ยงและวิเคราะห์ความเสี่ยงในการรักษาผู้ป่วยวัณโรคด้วยเทคโนโลยี AI")

if model:
    # เริ่มต้น Form เพื่อให้หน้าตาเหมือน Figma
    with st.form("prediction_form"):
        # ใช้ container ของ CSS เพื่อจัดวางการ์ด
        st.markdown('<div class="input-card-container">', unsafe_allow_html=True)

        # คอลัมน์ที่ 1: ข้อมูลพื้นฐาน
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">👤 ข้อมูลพื้นฐาน</div>', unsafe_allow_html=True)
        age = st.number_input("อายุ (Age)", 0, 120, 45)
        gender = st.selectbox("เพศ (Gender)", ["ชาย", "หญิง"])
        bmi = st.number_input("ดัชนีมวลกาย (BMI)", 10.0, 50.0, 20.0, step=0.1)
        duration = st.number_input("ระยะเวลาการรักษา (วัน)", 0, 1000, 180)
        st.markdown('</div>', unsafe_allow_html=True) # ปิดการ์ดที่ 1

        # คอลัมน์ที่ 2: ประวัติสุขภาพ
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">❤️ ประวัติสุขภาพ</div>', unsafe_allow_html=True)
        hiv = st.selectbox("สถานะ HIV", ["ไม่ติดเชื้อ (Negative)", "ติดเชื้อ (Positive)", "ไม่ทราบ/ไม่ตรวจ"])
        dm = st.selectbox("เบาหวาน", ["ไม่เป็น", "เป็น"])
        ckd = st.selectbox("โรคไตเรื้อรัง (CKD)", ["ไม่เป็น", "เป็น"])
        copd = st.selectbox("โรคปอดอุดกั้นเรื้อรัง (COPD)", ["ไม่เป็น", "เป็น"])
        liver = st.selectbox("โรคตับ (Liver Disease)", ["ไม่เป็น", "เป็น"])
        st.markdown('</div>', unsafe_allow_html=True) # ปิดการ์ดที่ 2

        # คอลัมน์ที่ 3: ผลการตรวจ
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">🧪 ผลการตรวจ</div>', unsafe_allow_html=True)
        afb = st.selectbox("ผล AFB เดือนแรก", ["Negative", "1+", "2+", "3+", "Scanty"])
        pos_tb = st.selectbox("ตำแหน่งของโรค", ["ในปอด", "นอกปอด", "ในและนอกปอด"])
        follow_up = st.number_input("จำนวนครั้งที่ Follow up", 0, 100, 1)
        arv = st.selectbox("ได้รับยา ARV หรือไม่", ["ไม่ได้รับ", "ได้รับ"])
        st.markdown('</div>', unsafe_allow_html=True) # ปิดการ์ดที่ 3

        st.markdown('</div>', unsafe_allow_html=True) # ปิด input-card-container

        # ปุ่มวิเคราะห์อยู่ตรงกลาง
        st.markdown("<br>", unsafe_allow_html=True)
        submit_btn = st.form_submit_button("วิเคราะห์ผลการรักษา")

    # --- 5. RESULT ---
    if submit_btn:
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
        
        st.markdown('<div class="result-container">', unsafe_allow_html=True)
        st.subheader("📊 ผลการวิเคราะห์")
        
        res_c1, res_c2 = st.columns([1, 1.2])
        
        with res_c1:
            # กราฟ Gauge ครึ่งวงกลมตาม Figma
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prob * 100,
                number = {'suffix': "%", 'font': {'size': 60}},
                title = {'text': "ความน่าจะเป็นกลุ่มเสี่ยง", 'font': {'size': 20}},
                gauge = {
                    'axis': {'range': [0, 100], 'tickwidth': 1},
                    'bar': {'color': "#444"},
                    'steps': [
                        {'range': [0, 30], 'color': "#A3E4D7"},
                        {'range': [30, 70], 'color': "#F9E79F"},
                        {'range': [70, 100], 'color': "#F1948A"}
                    ],
                }
            ))
            fig.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)
            
            # แสดงระดับความเสี่ยงเป็นตัวหนังสือ
            risk_level = "สูง" if prob > 0.7 else "ปานกลาง" if prob > 0.3 else "ต่ำ"
            st.markdown(f"<center><h4>ระดับความเสี่ยง: {risk_level}</h4></center>", unsafe_allow_html=True)

        with res_col2:
            st.markdown("<br>", unsafe_allow_html=True)
            # กล่องข้อความแจ้งเตือนตาม Figma
            st.markdown("""
                <div class="warning-box">
                    <b>⚠️ คำเตือน!</b><br>
                    ผู้ป่วยที่มีค่า BMI น้อยกว่า 18.5 ควรได้รับการดูแลเป็นพิเศษเกี่ยวกับโภชนาการ
                </div>
                <div class="info-box">
                    <b>💡 หมายเหตุ:</b><br>
                    ผลการประเมินนี้ใช้เพื่อประกอบการตัดสินใจเบื้องต้นเท่านั้น โปรดปรึกษาแพทย์ผู้เชี่ยวชาญเพื่อการวินิจฉัยที่ถูกต้อง
                </div>
            """, unsafe_allow_html=True)
            
        st.markdown('</div>', unsafe_allow_html=True) # ปิด result-container

else:
    st.warning("⚠️ กรุณาติดตั้งไฟล์โมเดลก่อนเริ่มใช้งาน")
