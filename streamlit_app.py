import streamlit as st
import pandas as pd
import xgboost as xgb
import joblib
import numpy as np
import plotly.graph_objects as go

# --- 1. PAGE CONFIG ---
st.set_page_config(
    page_title="ระบบพยากรณ์ความเสี่ยงวัณโรค TB",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. CUSTOM CSS ตาม Figma ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;500;600;700&display=swap');

/* ===== GLOBAL ===== */
html, body, [class*="css"] {
    font-family: 'Sarabun', sans-serif !important;
}

.main .block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* ===== NAVBAR ===== */
.navbar {
    background-color: #7D5A3C;
    padding: 14px 48px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: sticky;
    top: 0;
    z-index: 999;
}
.navbar-brand {
    display: flex;
    align-items: center;
    gap: 10px;
    color: white;
    font-size: 1.1rem;
    font-weight: 600;
}
.navbar-links {
    display: flex;
    gap: 32px;
    color: white;
    font-size: 0.95rem;
}
.navbar-links span { cursor: pointer; opacity: 0.9; }
.navbar-links span:hover { opacity: 1; text-decoration: underline; }
.navbar-login {
    background: white;
    color: #7D5A3C;
    border-radius: 20px;
    padding: 6px 22px;
    font-weight: 600;
    font-size: 0.9rem;
    cursor: pointer;
}

/* ===== HERO SECTION ===== */
.hero-section {
    background: linear-gradient(135deg, #8B6343 0%, #A0784F 40%, #C4956A 100%);
    padding: 52px 64px 64px;
    color: white;
}
.hero-title {
    font-size: 2.2rem;
    font-weight: 700;
    line-height: 1.4;
    margin-bottom: 6px;
}
.hero-subtitle {
    font-size: 1.1rem;
    opacity: 0.88;
    font-weight: 400;
}

/* ===== MAIN CONTENT AREA ===== */
.content-area {
    background-color: #f5f0e8;
    padding: 40px 64px;
    min-height: 100vh;
}

/* ===== INPUT CARDS ===== */
.card-wrapper {
    background-color: #FEFCE8;
    border-radius: 20px;
    padding: 28px 26px;
    border: 1px solid #F5E08A;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    min-height: 460px;
}
.card-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 22px;
}
.card-icon {
    font-size: 1.5rem;
}
.card-title-text {
    font-size: 1.1rem;
    font-weight: 700;
    color: #5D4037;
}

/* ===== FORM INPUTS ===== */
.stSelectbox > label,
.stNumberInput > label {
    font-size: 0.85rem !important;
    color: #6B5744 !important;
    font-weight: 500 !important;
    margin-bottom: 2px !important;
}
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    background-color: white !important;
    border-radius: 10px !important;
    border: 1px solid #E8D9C0 !important;
}
div[data-baseweb="select"] > div:focus-within,
div[data-baseweb="input"] > div:focus-within {
    border-color: #C4956A !important;
    box-shadow: 0 0 0 2px rgba(196, 149, 106, 0.2) !important;
}

/* ===== SUBMIT BUTTON ===== */
div[data-testid="stFormSubmitButton"] > button {
    background-color: #F5C842 !important;
    color: #3D2B00 !important;
    font-family: 'Sarabun', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    border-radius: 14px !important;
    border: none !important;
    padding: 14px 40px !important;
    box-shadow: 0 4px 14px rgba(245, 200, 66, 0.4) !important;
    transition: all 0.2s ease !important;
    width: auto !important;
    min-width: 220px !important;
}
div[data-testid="stFormSubmitButton"] > button:hover {
    background-color: #E6B800 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 18px rgba(245, 200, 66, 0.5) !important;
}

/* ===== RESULT SECTION ===== */
.result-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 28px;
}
.result-title {
    font-size: 1.4rem;
    font-weight: 700;
    color: #3D2B00;
}
.result-icon {
    font-size: 1.8rem;
}

.gauge-card {
    background: white;
    border-radius: 20px;
    padding: 30px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.07);
    text-align: center;
}
.risk-badge {
    display: inline-block;
    padding: 6px 22px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 1rem;
    margin-top: 8px;
}
.risk-low    { background: #D4EDDA; color: #155724; }
.risk-medium { background: #FFF3CD; color: #856404; }
.risk-high   { background: #F8D7DA; color: #721C24; }

.legend-row {
    display: flex;
    justify-content: center;
    gap: 20px;
    margin-top: 12px;
    font-size: 0.85rem;
    color: #666;
}
.legend-dot {
    display: inline-block;
    width: 10px; height: 10px;
    border-radius: 50%;
    margin-right: 5px;
}

.warning-card {
    background: #FDEEEE;
    border-left: 5px solid #E53935;
    border-radius: 14px;
    padding: 18px 20px;
    margin-bottom: 14px;
    display: flex;
    gap: 12px;
    align-items: flex-start;
}
.warning-icon { font-size: 1.3rem; flex-shrink: 0; margin-top: 2px; }
.warning-text { font-size: 0.9rem; color: #5c1010; line-height: 1.6; }
.warning-text b { display: block; margin-bottom: 4px; font-size: 0.95rem; }

.note-card {
    background: #F1EEEA;
    border-left: 5px solid #A0907A;
    border-radius: 14px;
    padding: 18px 20px;
    display: flex;
    gap: 12px;
    align-items: flex-start;
}
.note-text { font-size: 0.9rem; color: #4a3728; line-height: 1.6; }
.note-text b { display: block; margin-bottom: 4px; font-size: 0.95rem; }

/* ===== HIDE STREAMLIT DEFAULTS ===== */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
</style>
""", unsafe_allow_html=True)

# --- 3. LOAD MODEL ---
@st.cache_resource
def load_assets():
    try:
        m = joblib.load('xgb_tb_model.pkl')
        f = joblib.load('model_features.pkl')
        return m, f
    except:
        return None, None

model, model_features = load_assets()

# --- 4. NAVBAR ---
st.markdown("""
<div class="navbar">
    <div class="navbar-brand">
        <span>🫁</span>
        <span>cmu</span>
    </div>
    <div class="navbar-links">
        <span>Home</span>
        <span>About us</span>
        <span>AI-powered disease analysis system</span>
    </div>
    <div class="navbar-login">Login</div>
</div>
""", unsafe_allow_html=True)

# --- 5. HERO ---
st.markdown("""
<div class="hero-section">
    <div class="hero-title">ระบบพยากรณ์ความเสี่ยงผลการรักษาวัณโรค (TB)</div>
    <div class="hero-subtitle">ระบบวิเคราะห์ความเสี่ยงในการรักษาผู้ป่วยวัณโรคด้วยเทคโนโลยี AI</div>
</div>
""", unsafe_allow_html=True)

# --- 6. CONTENT AREA ---
st.markdown('<div class="content-area">', unsafe_allow_html=True)

with st.form("main_form"):
    col1, col2, col3 = st.columns(3, gap="large")

    # ---- CARD 1: ข้อมูลพื้นฐาน ----
    with col1:
        st.markdown("""
        <div class="card-wrapper">
            <div class="card-header">
                <span class="card-icon">👤</span>
                <span class="card-title-text">ข้อมูลพื้นฐาน</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        age = st.number_input("อายุ (ปี)", 0, 120, 45, key="age")
        gen = st.selectbox("เพศ", ["ชาย", "หญิง"], key="gen")
        bmi = st.number_input("ดัชนีมวลกาย (BMI)", 10.0, 50.0, 20.0, key="bmi")
        dur = st.number_input("ระยะเวลาการรักษา (วัน)", 0, 1000, 180, key="dur")

    # ---- CARD 2: ประวัติสุขภาพ ----
    with col2:
        st.markdown("""
        <div class="card-wrapper">
            <div class="card-header">
                <span class="card-icon">❤️</span>
                <span class="card-title-text">ประวัติสุขภาพ</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        hiv  = st.selectbox("สถานะ HIV",                        ["ไม่ติดเชื้อ (Negative)", "ติดเชื้อ (Positive)", "ไม่ทราบ"])
        ckd  = st.selectbox("โรคไตเรื้อรัง (CKD)",              ["ไม่เป็น", "เป็น"])
        copd = st.selectbox("โรคปอดอุดกั้นเรื้อรัง (COPD)",    ["ไม่เป็น", "เป็น"])
        liv  = st.selectbox("โรคตับ",                           ["ไม่เป็น", "เป็น"])
        dm   = st.selectbox("โรคเบาหวาน",                       ["ไม่เป็น", "เป็น"])

    # ---- CARD 3: ผลการตรวจ ----
    with col3:
        st.markdown("""
        <div class="card-wrapper">
            <div class="card-header">
                <span class="card-icon">🧪</span>
                <span class="card-title-text">ผลการตรวจ</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        afb = st.selectbox("AFB เดือนที่ 1",        ["Negative", "1+", "2+", "3+", "Scanty"])
        pos = st.selectbox("ตำแหน่งการติดเชื้อ",    ["ในปอด", "นอกปอด", "ในและนอกปอด"])
        f_u = st.number_input("จำนวนครั้งติดตาม",   0, 50, 1)
        arv = st.selectbox("สถานะ ARV",              ["ไม่ได้รับ", "ได้รับ"])

    # ---- SUBMIT BUTTON ----
    st.markdown("<br>", unsafe_allow_html=True)
    btn_col = st.columns([1, 2, 1])
    with btn_col[1]:
        sub = st.form_submit_button("🔍 วิเคราะห์ผลการรักษา", use_container_width=True)

# --- 7. RESULT ---
if sub:
    if model is not None:
        def encode(d):
            mapping = {
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
            for k, v in mapping.items():
                if k in d:
                    d[k] = v.get(d[k], 0)
            return d

        raw = {
            'Age': age, 'Gender': gen, 'BMI': bmi, 'Duration': dur,
            'HIV': hiv, 'Diabetes Mellitus': dm, 'Chronic Kidney Disease': ckd,
            ' Chronic Obstructive Pulmonary Disease': copd, 'Liver Disease': liv,
            ' AFB resulf of first month': afb, 'position of TB': pos,
            '(TB F/U) follow up': f_u, 'Treatment of ARV': arv
        }
        enc = encode(raw)
        df  = pd.DataFrame(0, index=[0], columns=model_features)
        for k in enc:
            if k in df.columns:
                df[k] = enc[k]

        prob      = model.predict_proba(df)[0][1]
        risk_pct  = prob * 100
        risk_text = "สูง" if prob > 0.6 else "ปานกลาง" if prob > 0.3 else "ต่ำ"
        badge_cls = "risk-high" if prob > 0.6 else "risk-medium" if prob > 0.3 else "risk-low"

        st.markdown("<br>", unsafe_allow_html=True)

        # Result header
        st.markdown("""
        <div class="result-header">
            <span class="result-icon">📊</span>
            <span class="result-title">ผลการวิเคราะห์</span>
        </div>
        """, unsafe_allow_html=True)

        r1, r2 = st.columns([1, 1.3], gap="large")

        with r1:
            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_pct,
                number={
                    'suffix': "%",
                    'font': {'size': 52, 'family': 'Sarabun', 'color': '#3D2B00'},
                    'valueformat': '.2f'
                },
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': '#aaa'},
                    'bar':  {'color': "#5D4037", 'thickness': 0.25},
                    'bgcolor': "white",
                    'borderwidth': 0,
                    'steps': [
                        {'range': [0,   30],  'color': '#81C784'},
                        {'range': [30,  60],  'color': '#FFD54F'},
                        {'range': [60,  100], 'color': '#E57373'}
                    ],
                    'threshold': {
                        'line': {'color': "#3D2B00", 'width': 4},
                        'thickness': 0.8,
                        'value': risk_pct
                    }
                }
            ))
            fig.update_layout(
                height=280,
                margin=dict(l=30, r=30, t=30, b=10),
                paper_bgcolor='white',
                plot_bgcolor='white',
                font={'family': 'Sarabun'}
            )

            st.markdown('<div class="gauge-card">', unsafe_allow_html=True)
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            st.markdown(f"""
                <div style="text-align:center; margin-top:-10px;">
                    <b style="font-size:1rem; color:#5D4037;">ระดับความเสี่ยง:</b>
                    <span class="risk-badge {badge_cls}">{risk_text}</span>
                </div>
                <div class="legend-row" style="margin-top:16px;">
                    <span><span class="legend-dot" style="background:#81C784;"></span>ต่ำ (0–30%)</span>
                    <span><span class="legend-dot" style="background:#FFD54F;"></span>ปานกลาง (30–60%)</span>
                    <span><span class="legend-dot" style="background:#E57373;"></span>สูง (60–100%)</span>
                </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with r2:
            # Warning box
            warning_text = ""
            if bmi < 18.5:
                warning_text += "BMI ต่ำกว่าเกณฑ์ปกติ (< 18.5) "
            if age > 60:
                warning_text += "อายุมากกว่า 60 ปี "
            if hiv == "ติดเชื้อ (Positive)":
                warning_text += "ติดเชื้อ HIV "
            if not warning_text:
                warning_text = "พบปัจจัยเสี่ยงบางประการ"

            st.markdown(f"""
            <div class="warning-card">
                <span class="warning-icon">⚠️</span>
                <div class="warning-text">
                    <b>คำเตือน:</b>
                    ผู้ป่วยอยู่ในกลุ่มเสี่ยงต่อผลการรักษาที่ไม่สำเร็จ
                    ผู้ป่วยที่มีดัชนีมวลกายต่ำ (BMI &lt; 18.5) อายุมาก
                    และมีโรคประจำตัวหลายโรค มีโอกาสสูงที่จะเกิดผลการรักษาที่ไม่สำเร็จ
                    ควรได้รับการติดตามอย่างใกล้ชิด และพิจารณาแผนการรักษาเสริมเพิ่มเติม
                </div>
            </div>
            <div class="note-card">
                <span class="warning-icon">💡</span>
                <div class="note-text">
                    <b>หมายเหตุ:</b>
                    ผลลัพธ์นี้เป็นเพียงการพยากรณ์เบื้องต้นจากระบบปัญญาประดิษฐ์
                    ไม่สามารถใช้แทนการวินิจฉัยและการตัดสินใจทางคลินิกของแพทย์ผู้เชี่ยวชาญได้
                    กรุณาปรึกษาแพทย์เพื่อคำแนะนำที่เหมาะสมกับสภาวะของผู้ป่วย
                </div>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.error("⚠️ ไม่พบไฟล์โมเดล กรุณาตรวจสอบว่ามีไฟล์ `xgb_tb_model.pkl` และ `model_features.pkl` ในโฟลเดอร์เดียวกัน")

st.markdown('</div>', unsafe_allow_html=True)  # close content-area
