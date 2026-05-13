import streamlit as st
import pandas as pd
import xgboost as xgb
import joblib
import numpy as np

# --- 1. SETTING & CONFIG ---
st.set_page_config(
    page_title="TB Treatment Prediction",
    page_icon="🏥",
    layout="wide"
)

# --- 2. LOAD MODEL & ASSETS ---
@st.cache_resource
def load_assets():
    try:
        # ตรวจสอบชื่อไฟล์ให้ตรงกับที่โหลดจาก Colab
        model = joblib.load('xgb_tb_model.pkl')
        features = joblib.load('model_features.pkl')
        return model, features
    except FileNotFoundError:
        st.error("❌ ไม่พบไฟล์โมเดล! กรุณาตรวจสอบว่ามีไฟล์ 'xgb_tb_model.pkl' และ 'model_features.pkl' อยู่ในโฟลเดอร์เดียวกัน")
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
        'position of TB': {'ในปอด': 0, 'นอกปอด': 1},
        'Treatment of ARV': {'ไม่ได้รับ': 0, 'ได้รับ': 1}
    }
    
    encoded = input_dict.copy()
    for col, mapping in mappings.items():
        if col in encoded:
            encoded[col] = mapping.get(encoded[col], 0)
    return encoded

# --- 4. USER INTERFACE (UI) ---
st.title("🏥 ระบบพยากรณ์ความเสี่ยงผลการรักษาวัณโรค (TB)")
st.markdown("กรุณากรอกข้อมูลผู้ป่วยด้านล่าง เพื่อให้ AI ช่วยประเมินความเสี่ยงในการรักษา")

if model:
    # --- เริ่มต้น FORM ---
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("📌 ข้อมูลพื้นฐาน")
            age = st.number_input("อายุ (Age)", min_value=0, max_value=120, value=45)
            gender = st.selectbox("เพศ (Gender)", ["ชาย", "หญิง"])
            bmi = st.number_input("ดัชนีมวลกาย (BMI)", min_value=10.0, max_value=50.0, value=20.0, step=0.1)
            duration = st.number_input("ระยะเวลาการรักษา (Duration - วัน)", min_value=0, value=180)

        with col2:
            st.subheader("🩺 ประวัติสุขภาพ")
            hiv = st.selectbox("สถานะ HIV", ["ไม่ติดเชื้อ (Negative)", "ติดเชื้อ (Positive)", "ไม่ทราบ/ไม่ตรวจ"])
            dm = st.selectbox("เบาหวาน (Diabetes Mellitus)", ["ไม่เป็น", "เป็น"])
            ckd = st.selectbox("โรคไตเรื้อรัง (CKD)", ["ไม่เป็น", "เป็น"])
            copd = st.selectbox("โรคปอดอุดกั้นเรื้อรัง (COPD)", ["ไม่เป็น", "เป็น"])
            liver = st.selectbox("โรคตับ (Liver Disease)", ["ไม่เป็น", "เป็น"])

        with col3:
            st.subheader("🧪 ผลการตรวจ")
            afb = st.selectbox("ผล AFB เดือนแรก", ["Negative", "1+", "2+", "3+", "Scanty"])
            pos_tb = st.selectbox("ตำแหน่งของโรค (Position)", ["ในปอด", "นอกปอด"])
            follow_up = st.number_input("จำนวนครั้งที่ Follow up", min_value=0, value=1)
            arv = st.selectbox("ได้รับยา ARV หรือไม่", ["ไม่ได้รับ", "ได้รับ"])

        # ✅ บรรทัดนี้สำคัญ: ต้องย่อหน้าให้ตรงกับพวก col1, col2 เพื่อให้อยู่ข้างใน WITH ST.FORM
        submit_btn = st.form_submit_button("วิเคราะห์ผลการรักษา")

    # --- 5. PREDICTION LOGIC (อยู่นอก Form ได้) ---
    if submit_btn:
        raw_inputs = {
            'Age': age,
            'Gender': gender,
            'BMI': bmi,
            'Duration': duration,
            'HIV': hiv,
            'Diabetes Mellitus': dm,
            'Chronic Kidney Disease': ckd,
            ' Chronic Obstructive Pulmonary Disease': copd,
            'Liver Disease': liver,
            ' AFB resulf of first month': afb,
            'position of TB': pos_tb,
            '(TB F/U) follow up': follow_up,
            'Treatment of ARV': arv
        }

        # แปลงเป็นตัวเลขผ่านฟังก์ชัน Encoding
        encoded_data = encode_inputs(raw_inputs)

        # จัดเตรียมข้อมูลให้ตรงกับลำดับ Features ที่เทรนมา
        input_df = pd.DataFrame(0, index=[0], columns=model_features)
        for col in encoded_data:
            if col in input_df.columns:
                input_df[col] = encoded_data[col]

        # ทำนายผล
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        # แสดงผลลัพธ์
        st.divider()
        st.subheader("📊 ผลการวิเคราะห์")
        
        if prediction == 1:
            st.error(f"⚠️ **กลุ่มเสี่ยง:** มีโอกาสเกิดผลการรักษาที่ไม่พึงประสงค์ (เสียชีวิต/ล้มเหลว/ขาดยา)")
            st.write(f"ระดับความเสี่ยง: {probability:.2%}")
            st.progress(float(probability))
        else:
            st.success(f"✅ **กลุ่มปกติ:** มีแนวโน้มรักษาหายหรือครบตามกำหนด")
            st.write(f"โอกาสที่จะเป็นกลุ่มเสี่ยง: {probability:.2%}")
            st.progress(float(probability))

        st.info("หมายเหตุ: ผลลัพธ์นี้เป็นเพียงการพยากรณ์จาก AI โปรดปรึกษาแพทย์ผู้เชี่ยวชาญ")

else:
    st.warning("กรุณาตรวจสอบว่ามีไฟล์โมเดลในระบบเรียบร้อยแล้ว")
