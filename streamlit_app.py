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

# --- 2. CUSTOM CSS — Dark Navy Theme ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;500;600;700&display=swap');

/* ===== CSS VARIABLES ===== */
:root {
    --bg-base:       #0A0F1E;
    --bg-surface:    #111827;
    --bg-card:       #1A2235;
    --bg-card-hover: #1F2A40;
    --border:        #2A3A55;
    --border-light:  #243044;
    --accent:        #3B82F6;
    --accent-glow:   rgba(59,130,246,0.25);
    --accent2:       #60A5FA;
    --text-primary:  #E8F0FE;
    --text-secondary:#94A3C0;
    --text-muted:    #5A7090;
    --input-bg:      #0F172A;
}

/* ===== GLOBAL — ครอบคลุมทุกพื้นที่ ===== */
html, body {
    background-color: var(--bg-base) !important;
    font-family: 'Sarabun', sans-serif !important;
}

/* Streamlit root wrappers */
[class*="css"], .stApp, section[data-testid="stSidebar"],
div[data-testid="stAppViewContainer"], div[data-testid="stMain"],
div[data-testid="block-container"] {
    background-color: var(--bg-base) !important;
    font-family: 'Sarabun', sans-serif !important;
}

.main .block-container {
    padding: 0 !important;
    max-width: 100% !important;
    background-color: var(--bg-base) !important;
}

/* ===== HEADER ===== */
.page-header {
    background: linear-gradient(180deg, #0D1526 0%, #111827 100%);
    padding: 26px 64px 22px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 18px;
}
.header-logo {
    width: 54px;
    height: 54px;
    background: linear-gradient(135deg, #1D4ED8 0%, #3B82F6 100%);
    border-radius: 16px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.7rem;
    box-shadow: 0 0 20px rgba(59,130,246,0.4);
    flex-shrink: 0;
}
.header-text-title {
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--text-primary);
    line-height: 1.3;
    letter-spacing: -0.01em;
}
.header-text-sub {
    font-size: 0.87rem;
    color: var(--text-secondary);
    font-weight: 400;
    margin-top: 3px;
}

/* ===== SPACER ===== */
.top-spacer {
    height: 32px;
    background: var(--bg-base);
}

/* ===== FORM BACKGROUND ===== */
div[data-testid="stForm"] {
    background-color: var(--bg-base) !important;
    border: none !important;
    padding: 0 64px 32px !important;
    box-shadow: none !important;
    border-radius: 0 !important;
}

/* gap ระหว่าง columns */
div[data-testid="stForm"] div[data-testid="stHorizontalBlock"] {
    gap: 20px !important;
    align-items: stretch !important;
}

/* ===== INPUT CARDS ===== */
div[data-testid="stForm"] div[data-testid="stHorizontalBlock"] > div[data-testid="stVerticalBlock"] {
    background-color: var(--bg-card) !important;
    border-radius: 18px !important;
    border: 1px solid var(--border) !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4), 0 0 0 0 transparent !important;
    padding: 26px 22px 30px 22px !important;
    min-height: 460px !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}
div[data-testid="stForm"] div[data-testid="stHorizontalBlock"] > div[data-testid="stVerticalBlock"]:hover {
    border-color: #2E4470 !important;
    box-shadow: 0 6px 32px rgba(0,0,0,0.5), 0 0 0 1px rgba(59,130,246,0.1) !important;
}

/* ===== CARD HEADERS ===== */
.card-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 20px;
    padding-bottom: 14px;
    border-bottom: 1px solid var(--border-light);
}
.card-icon-circle {
    width: 38px;
    height: 38px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.05rem;
    flex-shrink: 0;
}
.icon-blue  { background: linear-gradient(135deg, #1D4ED8, #3B82F6); box-shadow: 0 0 12px rgba(59,130,246,0.4); }
.icon-teal  { background: linear-gradient(135deg, #0E7490, #22D3EE); box-shadow: 0 0 12px rgba(34,211,238,0.35); }
.icon-indigo{ background: linear-gradient(135deg, #4338CA, #818CF8); box-shadow: 0 0 12px rgba(129,140,248,0.35); }

.card-title-text {
    font-size: 1.03rem;
    font-weight: 700;
    color: var(--text-primary);
}

/* ===== FORM LABELS ===== */
.stSelectbox > label,
.stNumberInput > label {
    font-size: 0.83rem !important;
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
    margin-bottom: 3px !important;
}

/* ===== FORM INPUTS ===== */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    background-color: var(--input-bg) !important;
    border-radius: 10px !important;
    border: 1.5px solid var(--border) !important;
    font-family: 'Sarabun', sans-serif !important;
    color: var(--text-primary) !important;
}
div[data-baseweb="select"] > div:focus-within,
div[data-baseweb="input"] > div:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-glow) !important;
}
/* dropdown text color */
div[data-baseweb="select"] span,
div[data-baseweb="input"] input {
    color: var(--text-primary) !important;
}
/* dropdown menu */
ul[data-baseweb="menu"] {
    background-color: #1A2235 !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
}
ul[data-baseweb="menu"] li {
    color: var(--text-primary) !important;
}
ul[data-baseweb="menu"] li:hover {
    background-color: #243044 !important;
}
/* number input arrows */
div[data-baseweb="input"] button {
    background-color: var(--input-bg) !important;
    color: var(--text-secondary) !important;
    border-color: var(--border) !important;
}

/* ===== SUBMIT BUTTON ===== */
div[data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(135deg, #1D4ED8 0%, #3B82F6 100%) !important;
    color: white !important;
    font-family: 'Sarabun', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    border-radius: 50px !important;
    border: none !important;
    padding: 14px 52px !important;
    box-shadow: 0 6px 24px rgba(59,130,246,0.45) !important;
    transition: all 0.2s ease !important;
    white-space: nowrap !important;
    letter-spacing: 0.01em !important;
}
div[data-testid="stFormSubmitButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 32px rgba(59,130,246,0.6) !important;
    filter: brightness(1.08) !important;
}

/* ===== RESULT SECTION ===== */
.result-section-bg {
    background: var(--bg-base);
    padding: 8px 64px 52px;
}
.result-header {
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 24px;
    padding-top: 8px;
}
.result-icon-circle {
    width: 44px;
    height: 44px;
    border-radius: 14px;
    background: linear-gradient(135deg, #1D4ED8, #3B82F6);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.3rem;
    box-shadow: 0 0 16px rgba(59,130,246,0.4);
}
.result-title {
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--text-primary);
}

/* ===== WARNING CARDS ===== */
.warning-card {
    background: rgba(239,68,68,0.08);
    border: 1px solid rgba(239,68,68,0.3);
    border-left: 4px solid #EF4444;
    border-radius: 14px;
    padding: 18px 20px;
    margin-bottom: 14px;
    display: flex;
    gap: 12px;
    align-items: flex-start;
}
.warning-card-medium {
    background: rgba(245,158,11,0.08);
    border: 1px solid rgba(245,158,11,0.25);
    border-left: 4px solid #F59E0B;
    border-radius: 14px;
    padding: 18px 20px;
    margin-bottom: 14px;
    display: flex;
    gap: 12px;
    align-items: flex-start;
}
.warning-card-low {
    background: rgba(34,197,94,0.08);
    border: 1px solid rgba(34,197,94,0.25);
    border-left: 4px solid #22C55E;
    border-radius: 14px;
    padding: 18px 20px;
    margin-bottom: 14px;
    display: flex;
    gap: 12px;
    align-items: flex-start;
}
.warning-icon { font-size: 1.2rem; flex-shrink: 0; margin-top: 2px; }
.warning-text { font-size: 0.9rem; color: #FCA5A5; line-height: 1.7; }
.warning-text b { display: block; margin-bottom: 4px; font-size: 0.95rem; color: #FEE2E2; }
.warning-text-medium { font-size: 0.9rem; color: #FCD34D; line-height: 1.7; }
.warning-text-medium b { display: block; margin-bottom: 4px; font-size: 0.95rem; color: #FEF3C7; }
.warning-text-low { font-size: 0.9rem; color: #86EFAC; line-height: 1.7; }
.warning-text-low b { display: block; margin-bottom: 4px; font-size: 0.95rem; color: #DCFCE7; }

.note-card {
    background: rgba(59,130,246,0.08);
    border: 1px solid rgba(59,130,246,0.25);
    border-left: 4px solid #3B82F6;
    border-radius: 14px;
    padding: 18px 20px;
    display: flex;
    gap: 12px;
    align-items: flex-start;
}
.note-text { font-size: 0.9rem; color: #93C5FD; line-height: 1.7; }
.note-text b { display: block; margin-bottom: 4px; font-size: 0.95rem; color: #BFDBFE; }

/* ===== GAUGE CARD ===== */
.gauge-card {
    background: var(--bg-card);
    border-radius: 18px;
    padding: 28px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
    border: 1px solid var(--border);
    text-align: center;
    margin-bottom: 14px;
}
.risk-badge {
    display: inline-block;
    padding: 5px 20px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 0.93rem;
    margin-top: 6px;
}
.risk-low    { background: rgba(34,197,94,0.15);  color: #86EFAC; border: 1px solid rgba(34,197,94,0.3); }
.risk-medium { background: rgba(245,158,11,0.15); color: #FCD34D; border: 1px solid rgba(245,158,11,0.3); }
.risk-high   { background: rgba(239,68,68,0.15);  color: #FCA5A5; border: 1px solid rgba(239,68,68,0.3); }

.legend-row {
    display: flex;
    justify-content: center;
    gap: 20px;
    margin-top: 14px;
    font-size: 0.83rem;
    color: var(--text-secondary);
}
.legend-dot {
    display: inline-block;
    width: 9px; height: 9px;
    border-radius: 50%;
    margin-right: 5px;
    vertical-align: middle;
}

/* ===== HIDE STREAMLIT DEFAULTS ===== */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* Fix stray white background from Streamlit */
.stApp > div:first-child { background: var(--bg-base) !important; }
section.main > div { background: var(--bg-base) !important; }

/* ===== RESPONSIVE — TABLET ===== */
@media (max-width: 1024px) {
    .page-header { padding: 20px 32px !important; }
    div[data-testid="stForm"] { padding: 0 32px 28px !important; }
    .result-section-bg { padding: 8px 32px 40px !important; }
}

/* ===== RESPONSIVE — MOBILE ===== */
@media (max-width: 768px) {
    .page-header { padding: 16px 16px !important; gap: 12px !important; }
    .header-text-title { font-size: 1.05rem !important; }
    .header-text-sub { font-size: 0.78rem !important; }
    .header-logo { width: 44px !important; height: 44px !important; }

    div[data-testid="stForm"] { padding: 0 12px 20px !important; }
    div[data-testid="stForm"] div[data-testid="stHorizontalBlock"] {
        flex-direction: column !important; gap: 14px !important;
    }
    div[data-testid="stForm"] div[data-testid="stHorizontalBlock"] > div[data-testid="stVerticalBlock"] {
        width: 100% !important; min-width: 100% !important;
        min-height: unset !important; padding: 18px 14px 22px !important;
    }
    .result-section-bg { padding: 8px 12px 40px !important; }
    .result-title { font-size: 1.15rem !important; }
    div[data-testid="stFormSubmitButton"] > button {
        font-size: 0.95rem !important; padding: 12px 28px !important;
    }
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

# --- 4. HEADER ---
st.markdown("""
<div class="page-header">
    <div class="header-logo">🏥</div>
    <div>
        <div class="header-text-title">ระบบพยากรณ์ความเสี่ยงผลการรักษาวัณโรค (TB)</div>
        <div class="header-text-sub">ระบบวิเคราะห์ความเสี่ยงในการรักษาผู้ป่วยวัณโรคด้วยเทคโนโลยี AI</div>
    </div>
</div>
<div class="top-spacer"></div>
""", unsafe_allow_html=True)

# --- 5. FORM ---
with st.form("main_form"):
    col1, col2, col3 = st.columns(3, gap="large")

    # ---- CARD 1 ----
    with col1:
        st.markdown("""
        <div class="card-header">
            <div class="card-icon-circle icon-blue">📍</div>
            <span class="card-title-text">ข้อมูลพื้นฐาน</span>
        </div>
        """, unsafe_allow_html=True)
        age = st.number_input("อายุ (ปี)", 0, 120, 45, key="age")
        gen = st.selectbox("เพศ", ["ชาย", "หญิง"], key="gen")
        bmi = st.number_input("ดัชนีมวลกาย (BMI)", 10.0, 50.0, 20.0, key="bmi")
        dur = st.number_input("ระยะเวลาการรักษา (เดือน)", 0, 1000, 180, key="dur")

    # ---- CARD 2 ----
    with col2:
        st.markdown("""
        <div class="card-header">
            <div class="card-icon-circle icon-teal">💙</div>
            <span class="card-title-text">ประวัติสุขภาพ</span>
        </div>
        """, unsafe_allow_html=True)
        hiv  = st.selectbox("สถานะ HIV",                        ["ไม่ติดเชื้อ (Negative)", "ติดเชื้อ (Positive)", "ไม่ทราบ"])
        ckd  = st.selectbox("โรคไตเรื้อรัง (CKD)",              ["ไม่เป็น", "เป็น"])
        copd = st.selectbox("โรคปอดอุดกั้นเรื้อรัง (COPD)",    ["ไม่เป็น", "เป็น"])
        liv  = st.selectbox("โรคตับ (Liver Disease)",           ["ไม่เป็น", "เป็น"])
        dm   = st.selectbox("โรคเบาหวาน (Diabetes Mellitus)",   ["ไม่เป็น", "เป็น"])

    # ---- CARD 3 ----
    with col3:
        st.markdown("""
        <div class="card-header">
            <div class="card-icon-circle icon-indigo">🧪</div>
            <span class="card-title-text">ผลการตรวจ</span>
        </div>
        """, unsafe_allow_html=True)
        afb = st.selectbox("AFB เดือนที่ 1",        ["Negative", "1+", "2+", "3+", "Scanty"])
        pos = st.selectbox("ตำแหน่งการติดเชื้อ",    ["ในปอด", "นอกปอด", "ในและนอกปอด"])
        f_u = st.number_input("จำนวนครั้งติดตาม",   0, 50, 1)
        arv = st.selectbox("สถานะ ARV",              ["ไม่ได้รับ", "ได้รับ"])

    st.markdown("<br>", unsafe_allow_html=True)
    _, btn_mid, _ = st.columns([3, 2, 3])
    with btn_mid:
        sub = st.form_submit_button("🔍 วิเคราะห์ผลการรักษา", use_container_width=True)

# --- 6. RESULT ---
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

        st.markdown('<div class="result-section-bg">', unsafe_allow_html=True)

        st.markdown("""
        <div class="result-header">
            <div class="result-icon-circle">📊</div>
            <span class="result-title">ผลการวิเคราะห์</span>
        </div>
        """, unsafe_allow_html=True)

        # Warning logic
        if prob > 0.6:
            wc, wt, wi = "warning-card", "warning-text", "⚠️"
            wtitle = "คำเตือน: ผู้ป่วยอยู่ในกลุ่มเสี่ยงต่อผลการรักษาที่ไม่สำเร็จ"
            wbody  = ("ผู้ป่วยที่มีดัชนีมวลกายต่ำ (BMI &lt; 18.5) อายุมาก และมีโรคประจำตัวหลายโรค "
                      "มีโอกาสสูงที่จะเกิดผลการรักษาที่ไม่สำเร็จ ควรได้รับการติดตามอย่างใกล้ชิด "
                      "และพิจารณาแผนการรักษาเสริมเพิ่มเติม")
            ni = "ℹ️"
            nb = ("ผลลัพธ์นี้เป็นเพียงการพยากรณ์เบื้องต้นจากระบบปัญญาประดิษฐ์ ไม่สามารถใช้แทนการวินิจฉัยทางคลินิกได้ "
                  "กรุณาปรึกษาแพทย์เพื่อรับคำแนะนำที่เหมาะสมกับสภาวะของผู้ป่วย")
        elif prob > 0.3:
            wc, wt, wi = "warning-card-medium", "warning-text-medium", "⚠️"
            wtitle = "ควรติดตามอาการ: ผู้ป่วยมีปัจจัยเสี่ยงบางประการ"
            wbody  = ("พบปัจจัยเสี่ยงบางส่วนที่อาจส่งผลต่อผลการรักษา ควรนัดติดตามผู้ป่วยอย่างสม่ำเสมอ "
                      "และให้ความรู้เกี่ยวกับการรับประทานยาอย่างต่อเนื่องเพื่อป้องกันการดื้อยา")
            ni = "ℹ️"
            nb = ("ผลลัพธ์นี้เป็นเพียงการพยากรณ์เบื้องต้นจากระบบปัญญาประดิษฐ์ ไม่สามารถใช้แทนการวินิจฉัยทางคลินิกได้ "
                  "กรุณาใช้ข้อมูลนี้ประกอบการพิจารณาของแพทย์ผู้ดูแล")
        else:
            wc, wt, wi = "warning-card-low", "warning-text-low", "✅"
            wtitle = "แนวโน้มดี: ความเสี่ยงต่ำ"
            wbody  = ("ผู้ป่วยรายนี้มีแนวโน้มการรักษาที่ดี หากรับประทานยาครบตามกำหนดและดูแลสุขภาพอย่างเหมาะสม "
                      "โอกาสหายขาดจากวัณโรคอยู่ในระดับสูง ควรมาตรวจตามนัดทุกครั้ง")
            ni = "ℹ️"
            nb = ("แม้ความเสี่ยงจะอยู่ในระดับต่ำ ผลลัพธ์นี้ยังคงเป็นเพียงการพยากรณ์เบื้องต้นจากระบบ AI "
                  "ไม่ควรใช้เป็นเหตุผลในการหยุดหรือปรับเปลี่ยนการรักษาด้วยตนเอง")

        st.markdown(f"""
        <div class="{wc}">
            <span class="warning-icon">{wi}</span>
            <div class="{wt}"><b>{wtitle}</b>{wbody}</div>
        </div>
        """, unsafe_allow_html=True)

        # Gauge
        st.markdown('<div class="gauge-card">', unsafe_allow_html=True)

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_pct,
            number={
                'suffix': "%",
                'font': {'size': 56, 'family': 'Sarabun', 'color': '#60A5FA'},
                'valueformat': '.2f'
            },
            gauge={
                'axis': {
                    'range': [0, 100],
                    'tickwidth': 1,
                    'tickcolor': '#2A3A55',
                    'tickfont': {'family': 'Sarabun', 'color': '#5A7090'}
                },
                'bar':  {'color': "rgba(0,0,0,0)", 'thickness': 0},
                'bgcolor': "rgba(0,0,0,0)",
                'borderwidth': 0,
                'steps': [
                    {'range': [0,   30],  'color': '#166534'},
                    {'range': [30,  60],  'color': '#92400E'},
                    {'range': [60,  100], 'color': '#7F1D1D'}
                ],
                'threshold': {
                    'line': {'color': "#60A5FA", 'width': 5},
                    'thickness': 0.85,
                    'value': risk_pct
                }
            }
        ))
        fig.update_layout(
            height=260,
            margin=dict(l=40, r=40, t=30, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'family': 'Sarabun'}
        )

        col_g1, col_g2, col_g3 = st.columns([1, 2, 1])
        with col_g2:
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            st.markdown(f"""
                <div style="text-align:center; margin-top:-14px; margin-bottom:8px;">
                    <span style="font-size:0.93rem; color:#94A3C0; font-weight:500;">ระดับความเสี่ยง: </span>
                    <span class="risk-badge {badge_cls}">{risk_text}</span>
                </div>
                <div class="legend-row">
                    <span><span class="legend-dot" style="background:#22C55E;"></span>ต่ำ (0–30%)</span>
                    <span><span class="legend-dot" style="background:#F59E0B;"></span>ปานกลาง (30–60%)</span>
                    <span><span class="legend-dot" style="background:#EF4444;"></span>สูง (60–100%)</span>
                </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="note-card" style="margin-top:4px;">
            <span class="warning-icon">{ni}</span>
            <div class="note-text"><b>หมายเหตุ:</b>{nb}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.error("⚠️ ไม่พบไฟล์โมเดล กรุณาตรวจสอบว่ามีไฟล์ `xgb_tb_model.pkl` และ `model_features.pkl` ในโฟลเดอร์เดียวกัน")
