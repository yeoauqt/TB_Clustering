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

/* พื้นหลังทั้ง form */
div[data-testid="stForm"] {
    background-color: #f5f0e8 !important;
    border: none !important;
    padding: 40px 64px !important;
    box-shadow: none !important;
    border-radius: 0 !important;
}

/* gap ระหว่าง columns */
div[data-testid="stForm"] div[data-testid="stHorizontalBlock"] {
    gap: 24px !important;
    align-items: stretch !important;
}

/* ===== INPUT CARDS — ใช้ CSS ย้อม column แทน HTML div ===== */

/* ย้อม column block ให้เป็นการ์ด */
div[data-testid="stForm"] div[data-testid="stHorizontalBlock"] > div[data-testid="stVerticalBlock"] {
    background-color: #FEFCE8 !important;
    border-radius: 20px !important;
    border: 1px solid #F0DC82 !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06) !important;
    padding: 28px 22px 32px 22px !important;
    min-height: 480px !important;
}

.card-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 18px;
    padding-bottom: 12px;
    border-bottom: 1px solid #F0DC82;
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
    padding: 14px 32px !important;
    box-shadow: 0 4px 14px rgba(245, 200, 66, 0.4) !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
    white-space: nowrap !important;
}
div[data-testid="stFormSubmitButton"] > button:hover {
    background-color: #E6B800 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 18px rgba(245, 200, 66, 0.5) !important;
}

/* ===== RESULT AREA ===== */
div[data-testid="stVerticalBlock"]:has(div.result-header) {
    background-color: #f5f0e8 !important;
    padding: 0 64px 40px 64px !important;
    margin-top: -20px !important;
}
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
.warning-card-medium {
    background: #FFFBEA;
    border-left: 5px solid #F5C842;
    border-radius: 14px;
    padding: 18px 20px;
    margin-bottom: 14px;
    display: flex;
    gap: 12px;
    align-items: flex-start;
}
.warning-card-low {
    background: #EAFAF1;
    border-left: 5px solid #43A047;
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
.warning-text-medium { font-size: 0.9rem; color: #7B5800; line-height: 1.6; }
.warning-text-medium b { display: block; margin-bottom: 4px; font-size: 0.95rem; }
.warning-text-low { font-size: 0.9rem; color: #1B5E20; line-height: 1.6; }
.warning-text-low b { display: block; margin-bottom: 4px; font-size: 0.95rem; }

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

/* ===== RESPONSIVE — TABLET (max 1024px) ===== */
@media (max-width: 1024px) {
    .hero-section { padding: 40px 32px 48px !important; }
    .hero-title   { font-size: 1.8rem !important; }
    div[data-testid="stForm"] { padding: 28px 32px !important; }

    div[data-testid="stForm"] div[data-testid="stHorizontalBlock"] > div[data-testid="stVerticalBlock"] {
        padding: 22px 16px 26px !important;
        min-height: unset !important;
    }
    .navbar { padding: 12px 24px !important; }
    .navbar-links { gap: 18px !important; font-size: 0.85rem !important; }
}

/* ===== RESPONSIVE — MOBILE (max 768px) ===== */
@media (max-width: 768px) {
    /* Navbar */
    .navbar { padding: 12px 16px !important; flex-wrap: wrap; gap: 6px; }
    .navbar-links { display: none !important; }
    .navbar-brand { font-size: 1rem !important; }

    /* Hero */
    .hero-section  { padding: 28px 16px 36px !important; }
    .hero-title    { font-size: 1.35rem !important; line-height: 1.5 !important; }
    .hero-subtitle { font-size: 0.9rem !important; }

    /* Form padding */
    div[data-testid="stForm"] { padding: 16px 12px !important; }

    /* Stack 3 columns → 1 column */
    div[data-testid="stForm"] div[data-testid="stHorizontalBlock"] {
        flex-direction: column !important;
        gap: 16px !important;
    }
    div[data-testid="stForm"] div[data-testid="stHorizontalBlock"] > div[data-testid="stVerticalBlock"] {
        width: 100% !important;
        min-width: 100% !important;
        min-height: unset !important;
        padding: 20px 16px 24px !important;
    }

    /* Result columns stack */
    div[data-testid="stHorizontalBlock"]:has(div[data-testid="stVerticalBlock"]) {
        flex-direction: column !important;
    }

    /* Button full width */
    div[data-testid="stFormSubmitButton"] > button {
        font-size: 0.95rem !important;
        padding: 12px 20px !important;
    }

    /* Warning/note cards */
    .warning-card, .warning-card-medium, .warning-card-low, .note-card {
        padding: 14px 14px !important;
        font-size: 0.85rem !important;
    }

    /* Result area */
    div[data-testid="stVerticalBlock"]:has(div.result-header) {
        padding: 0 12px 40px !important;
        margin-top: -10px !important;
    }
    .result-title { font-size: 1.1rem !important; }
}

/* ===== RESPONSIVE — SMALL MOBILE (max 480px) ===== */
@media (max-width: 480px) {
    .hero-title    { font-size: 1.15rem !important; }
    .hero-subtitle { font-size: 0.82rem !important; }
    .card-title-text { font-size: 0.95rem !important; }

    div[data-testid="stForm"] { padding: 12px 8px !important; }
    div[data-testid="stForm"] div[data-testid="stVerticalBlock"] {
        padding: 16px 12px 20px !important;
    }

    .legend-row { flex-direction: column; align-items: center; gap: 6px !important; }
}
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

# --- 6. FORM ---
with st.form("main_form"):
    col1, col2, col3 = st.columns(3, gap="large")

    # ---- CARD 1: ข้อมูลพื้นฐาน ----
    with col1:
        st.markdown("""
        <div class="card-header">
            <span class="card-icon">👤</span>
            <span class="card-title-text">ข้อมูลพื้นฐาน</span>
        </div>
        """, unsafe_allow_html=True)
        age = st.number_input("อายุ (ปี)", 0, 120, 45, key="age")
        gen = st.selectbox("เพศ", ["ชาย", "หญิง"], key="gen")
        bmi = st.number_input("ดัชนีมวลกาย (BMI)", 10.0, 50.0, 20.0, key="bmi")
        dur = st.number_input("ระยะเวลาการรักษา (เดือน)", 0, 1000, 180, key="dur")

    # ---- CARD 2: ประวัติสุขภาพ ----
    with col2:
        st.markdown("""
        <div class="card-header">
            <span class="card-icon">❤️</span>
            <span class="card-title-text">ประวัติสุขภาพ</span>
        </div>
        """, unsafe_allow_html=True)
        hiv  = st.selectbox("สถานะ HIV",                        ["ไม่ติดเชื้อ (Negative)", "ติดเชื้อ (Positive)", "ไม่ทราบ"])
        ckd  = st.selectbox("โรคไตเรื้อรัง (CKD)",              ["ไม่เป็น", "เป็น"])
        copd = st.selectbox("โรคปอดอุดกั้นเรื้อรัง (COPD)",    ["ไม่เป็น", "เป็น"])
        liv  = st.selectbox("โรคตับ (Liver Disease)",                           ["ไม่เป็น", "เป็น"])
        dm   = st.selectbox("โรคเบาหวาน (Diabetes Mellitus)",                       ["ไม่เป็น", "เป็น"])

    # ---- CARD 3: ผลการตรวจ ----
    with col3:
        st.markdown("""
        <div class="card-header">
            <span class="card-icon">🧪</span>
            <span class="card-title-text">ผลการตรวจ</span>
        </div>
        """, unsafe_allow_html=True)
        afb = st.selectbox("AFB เดือนที่ 1",        ["Negative", "1+", "2+", "3+", "Scanty"])
        pos = st.selectbox("ตำแหน่งการติดเชื้อ",    ["ในปอด", "นอกปอด", "ในและนอกปอด"])
        f_u = st.number_input("จำนวนครั้งติดตาม",   0, 50, 1)
        arv = st.selectbox("สถานะ ARV",              ["ไม่ได้รับ", "ได้รับ"])

    # ---- SUBMIT BUTTON (กึ่งกลาง, single-line) ----
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div style="display:flex; justify-content:center;">', unsafe_allow_html=True)
    _, btn_mid, _ = st.columns([3, 2, 3])
    with btn_mid:
        sub = st.form_submit_button("🔍 วิเคราะห์ผลการรักษา", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

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
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'family': 'Sarabun'}
            )

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

        with r2:
            # คำเตือนแยกตามระดับความเสี่ยง
            if prob > 0.6:
                # สูง — แดง
                warning_card_class = "warning-card"
                warning_text_class = "warning-text"
                warning_icon = "🚨"
                warning_title = "ความเสี่ยงสูง — ต้องการการดูแลเร่งด่วน"
                warning_body = (
                    "ผู้ป่วยรายนี้มีความเสี่ยงสูงมากที่ผลการรักษาวัณโรคจะไม่สำเร็จ "
                    "ปัจจัยที่พบอาจได้แก่ ภาวะทุพโภชนาการ (BMI ต่ำกว่าเกณฑ์) การติดเชื้อร่วม เช่น HIV "
                    "หรือการมีโรคประจำตัวหลายโรคซึ่งส่งผลต่อภูมิคุ้มกันและการตอบสนองต่อยา<br><br>"
                    "ขอแนะนำให้ทีมแพทย์พิจารณาปรับแผนการรักษาโดยด่วน ตรวจสอบความเป็นไปได้ของเชื้อดื้อยา "
                    "และจัดให้มีการติดตามผู้ป่วยอย่างใกล้ชิดอย่างน้อยทุก 2 สัปดาห์ "
                    "รวมถึงประเมินการสนับสนุนด้านโภชนาการและสังคมเพื่อเพิ่มโอกาสความสำเร็จในการรักษา"
                )
                note_icon = "💡"
                note_body = (
                    "ผลลัพธ์นี้เป็นเพียงการพยากรณ์เบื้องต้นจากระบบปัญญาประดิษฐ์ ไม่สามารถใช้แทนการวินิจฉัยทางคลินิกได้ "
                    "การตัดสินใจในการรักษาควรอยู่บนพื้นฐานของการประเมินทางคลินิกโดยแพทย์ผู้เชี่ยวชาญ "
                    "กรุณาส่งต่อผู้ป่วยพบแพทย์เฉพาะทางโรควัณโรคโดยเร็วที่สุด"
                )
            elif prob > 0.3:
                # ปานกลาง — เหลือง
                warning_card_class = "warning-card-medium"
                warning_text_class = "warning-text-medium"
                warning_icon = "⚠️"
                warning_title = "ความเสี่ยงปานกลาง — ควรติดตามอย่างสม่ำเสมอ"
                warning_body = (
                    "ผู้ป่วยรายนี้มีปัจจัยเสี่ยงบางประการที่อาจส่งผลต่อความสำเร็จในการรักษา "
                    "แม้จะยังไม่อยู่ในระดับวิกฤต แต่ควรให้ความสำคัญกับการติดตามอาการและการรับยาอย่างต่อเนื่อง "
                    "ปัจจัยที่ควรเฝ้าระวัง ได้แก่ ความสม่ำเสมอในการรับประทานยา ภาวะโภชนาการ และโรคประจำตัวที่อาจกำเริบ<br><br>"
                    "แนะนำให้นัดติดตามผู้ป่วยตามกำหนดอย่างเคร่งครัด พร้อมประเมินผลข้างเคียงของยา "
                    "และให้ความรู้แก่ผู้ป่วยเกี่ยวกับความสำคัญของการรักษาอย่างครบถ้วนเพื่อป้องกันการดื้อยาในอนาคต"
                )
                note_icon = "💡"
                note_body = (
                    "ผลลัพธ์นี้เป็นเพียงการพยากรณ์เบื้องต้นจากระบบปัญญาประดิษฐ์ ไม่สามารถใช้แทนการวินิจฉัยทางคลินิกได้ "
                    "กรุณาใช้ข้อมูลนี้ประกอบการพิจารณาของแพทย์ผู้ดูแลร่วมกับข้อมูลทางคลินิกอื่น ๆ "
                    "และปรึกษาแพทย์ผู้เชี่ยวชาญหากพบความผิดปกติใด ๆ ระหว่างการรักษา"
                )
            else:
                # ต่ำ — เขียว
                warning_card_class = "warning-card-low"
                warning_text_class = "warning-text-low"
                warning_icon = "✅"
                warning_title = "ความเสี่ยงต่ำ — แนวโน้มการรักษาดี"
                warning_body = (
                    "ผู้ป่วยรายนี้มีแนวโน้มผลการรักษาวัณโรคที่ดี ปัจจัยเสี่ยงโดยรวมอยู่ในระดับต่ำ "
                    "ซึ่งบ่งชี้ว่าร่างกายมีศักยภาพในการตอบสนองต่อการรักษาได้อย่างมีประสิทธิภาพ "
                    "หากผู้ป่วยรับประทานยาครบตามกำหนดและดูแลสุขภาพโดยรวมอย่างเหมาะสม<br><br>"
                    "แนะนำให้รับประทานยาต่อเนื่องอย่างสม่ำเสมอโดยไม่หยุดยาเอง ดูแลโภชนาการให้เพียงพอ "
                    "พักผ่อนให้เต็มที่ และมาตรวจตามนัดทุกครั้งเพื่อติดตามผลการรักษาและป้องกันการกลับมาของโรค"
                )
                note_icon = "💡"
                note_body = (
                    "แม้ความเสี่ยงจะอยู่ในระดับต่ำ ผลลัพธ์นี้ยังคงเป็นเพียงการพยากรณ์เบื้องต้นจากระบบ AI "
                    "ไม่ควรใช้เป็นเหตุผลในการหยุดหรือปรับเปลี่ยนการรักษาด้วยตนเอง "
                    "ควรปรึกษาแพทย์และติดตามอาการอย่างสม่ำเสมอตลอดระยะเวลาการรักษา"
                )

            st.markdown(f"""
            <div class="{warning_card_class}">
                <span class="warning-icon">{warning_icon}</span>
                <div class="{warning_text_class}">
                    <b>{warning_title}</b>
                    {warning_body}
                </div>
            </div>
            <div class="note-card">
                <span class="warning-icon">{note_icon}</span>
                <div class="note-text">
                    <b>หมายเหตุ:</b>
                    {note_body}
                </div>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.error("⚠️ ไม่พบไฟล์โมเดล กรุณาตรวจสอบว่ามีไฟล์ `xgb_tb_model.pkl` และ `model_features.pkl` ในโฟลเดอร์เดียวกัน")
