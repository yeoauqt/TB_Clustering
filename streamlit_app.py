import streamlit as st
import pandas as pd
import xgboost as xgb
import joblib
import numpy as np
import plotly.graph_objects as go

# --- 1. SETTING & CUSTOM CSS ---
st.set_page_config(
    page_title="TB Treatment Prediction",
    page_icon="🏥",
    layout="wide"
)

# ปรับแต่งหน้าตาด้วย CSS (จัดระเบียบใหม่เพื่อไม่ให้หลุดกรอบ)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Sarabun', sans-serif;
    }
    
    /* สไตล์ของการ์ดสีเหลือง */
    .input-card {
        background-color: #FFF9E1;
        border-radius: 15px;
        padding: 20px;
        border: 1px solid #FFE58F;
        margin-bottom: 20px;
        min-height: 520px; /* ความสูงคงที่เพื่อให้กรอบดูสวยงามเท่ากัน */
    }
    
    .card-header {
        font-weight: bold;
        font-size: 1.2rem;
        margin-bottom: 15px;
        color: #856404;
        border-bottom: 1px solid #FFE58F;
        padding-bottom: 10px;
    }

    /* ปรับแต่งปุ่มกดสีเหลืองเข้ม */
    .stButton>button {
        background-color: #FFD600 !important;
        color: black !important;
        font-weight: bold !important;
        border-radius: 10px !important;
        width: 100% !important;
        border: none !important;
        height: 3.5rem;
        font-size: 1.1rem !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. LOAD MODEL & ASSETS ---
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
        if col in encoded:
            encoded[col] = mapping.get(encoded[col], 0)
    return encoded

# --- 4. USER INTERFACE (UI) ---
st.title("🏥 ระบบพยากรณ์ความเสี่ยงผลการรักษาวัณโรค (TB)")
st.caption("AI-Powered Tuberculosis Treatment Outcome Prediction System")

if model:
    # เริ่มต้นการสร้าง Form
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown('<div class="input-card"><div class="card-header">👤 ข้อมูลพื้นฐาน</div>', unsafe_allow_html=True)
            age = st.number_input("อายุ (Age)", 0, 120, 45)
            gender = st.selectbox("เพศ (Gender)", ["ชาย", "หญิง"])
            bmi = st.number_input("ดัชนีมวลกาย (BMI)", 10.0, 50.0, 20.0, step=0.1)
            duration = st.number_input("ระยะเวลาการรักษา (วัน)", 0, 1000, 180)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="input-card"><div class="card-header">🩺 ประวัติสุขภาพ</div>', unsafe_allow_html=True)
            hiv = st.selectbox("สถานะ HIV", ["ไม่ติดเชื้อ (Negative)", "ติดเชื้อ (Positive)", "ไม่ทราบ/ไม่ตรวจ"])
            dm = st.selectbox("เบาหวาน (Diabetes Mellitus)", ["ไม่เป็น", "เป็น"])
            ckd = st.selectbox("โรคไตเรื้อรัง (CKD)", ["ไม่เป็น", "เป็น"])
            copd = st.selectbox("โรคปอดอุดกั้นเรื้อรัง (COPD)", ["ไม่เป็น", "เป็น"])
            liver = st.selectbox("โรคตับ (Liver Disease)", ["ไม่เป็น", "เป็น"])
            st.markdown('</div>', unsafe_allow_html=True)

        with col3:
            st.markdown('<div class="input-card"><div class="card-header">🧪 ผลการตรวจ</div>', unsafe_allow_html=True)
            afb = st.selectbox("ผล AFB เดือนแรก", ["Negative", "1+", "2+", "3+", "Scanty"])
            pos_tb = st.selectbox("ตำแหน่งของโรค", ["ในปอด", "นอกปอด", "ในและนอกปอด"])
            follow_up = st.number_input("จำนวนครั้งที่ Follow up", 0, 100, 1)
            arv = st.selectbox("ได้รับยา ARV หรือไม่", ["ไม่ได้รับ", "ได้รับ"])
            st.markdown('</div>', unsafe_allow_html=True)

        # ปุ่มกดต้องเยื้องเข้ามาให้ตรงกับพวก col1 เพื่อให้อยู่ใน Form
        st.markdown("<br>", unsafe_allow_html=True)
        submit_btn = st.form_submit_button("วิเคราะห์ความเสี่ยงและพยากรณ์ผล")

    # --- 5. PREDICTION LOGIC ---
    if submit_btn:
        # เก็บข้อมูลดิบ
        raw_inputs = {
            'Age': age, 'Gender': gender, 'BMI': bmi, 'Duration': duration,
            'HIV': hiv, 'Diabetes Mellitus': dm, 'Chronic Kidney Disease': ckd,
            ' Chronic Obstructive Pulmonary Disease': copd, 'Liver Disease': liver,
            ' AFB resulf of first month': afb, 'position of TB': pos_tb,
            '(TB F/U) follow up': follow_up, 'Treatment of ARV': arv
        }

        # แปลงเป็นตัวเลข
        encoded_data = encode_inputs(raw_inputs)

        # เตรียม DataFrame (ลำดับคอลัมน์ต้องตรงกับโมเดล)
        input_df = pd.DataFrame(0, index=[0], columns=model_features)
        for col in encoded_data:
            if col in input_df.columns:
                input_df[col] = encoded_data[col]

        # ทำนายผล
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        st.divider()
        st.subheader("📊 ผลการวิเคราะห์")
        
        res_col1, res_col2 = st.columns([1, 1.2])
        
        with res_col1:
            # หน้าปัดแสดงความเสี่ยง
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = probability * 100,
                number = {'suffix': "%", 'font': {'size': 50}},
                title = {'text': "ความน่าจะเป็นกลุ่มเสี่ยง", 'font': {'size': 20}},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#444"},
                    'steps': [
                        {'range': [0, 30], 'color': "#A3E4D7"},
                        {'range': [30, 70], 'color': "#F9E79F"},
                        {'range': [70, 100], 'color': "#F1948A"}
                    ],
                }
            ))
            fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)

        with res_col2:
            st.markdown("<br>", unsafe_allow_html=True)
            if prediction == 1:
                st.error("### ผลวินิจฉัย: กลุ่มเสี่ยง (At Risk)")
                st.write("ผู้ป่วยมีความเสี่ยงสูงที่จะเกิดผลการรักษาไม่พึงประสงค์ (เสียชีวิต/ล้มเหลว/ขาดยา)")
            else:
                st.success("### ผลวินิจฉัย: กลุ่มปกติ (Normal)")
                st.write("ผู้ป่วยมีแนวโน้มที่จะรักษาหายหรือครบตามกำหนด (รักษาหาย/รักษาครบ)")
            
            st.info("💡 **คำแนะนำ:** ผลลัพธ์นี้เป็นเพียงการประเมินเบื้องต้นจาก AI ควรปรึกษาแพทย์ผู้เชี่ยวชาญร่วมด้วย")

else:
    st.warning("⚠️ กรุณาตรวจสอบไฟล์โมเดลในโฟลเดอร์ให้เรียบร้อย")
