import streamlit as st
import pandas as pd
import xgboost as xgb
import joblib
import numpy as np
import plotly.graph_objects as go

st.set_page_config(
    page_title="ระบบพยากรณ์ผลการรักษาวัณโรค TB",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;500;600;700&display=swap');

:root {
    --bg-base:    #F0F4FA;
    --bg-card:    #FFFFFF;
    --border:     #DDE5F0;
    --blue-main:  #1D4ED8;
    --blue-mid:   #3B82F6;
    --blue-light: #EFF6FF;
    --blue-glow:  rgba(59,130,246,0.18);
    --text-dark:  #0F172A;
    --text-mid:   #475569;
    --text-soft:  #94A3B8;
}

html, body, [class*="css"], .stApp,
div[data-testid="stAppViewContainer"],
div[data-testid="stMain"],
div[data-testid="block-container"],
section.main > div {
    background-color: var(--bg-base) !important;
    font-family: 'Sarabun', sans-serif !important;
}

.main .block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* ===== HEADER ===== */
.page-header {
    background: #FFFFFF;
    padding: 24px 64px 20px;
    border-bottom: 2px solid var(--border);
    display: flex;
    align-items: center;
    gap: 16px;
}
.header-logo {
    width: 52px; height: 52px;
    background: linear-gradient(135deg, var(--blue-main), var(--blue-mid));
    border-radius: 14px;
    display: flex; align-items: center; justify-content: center;
    box-shadow: 0 4px 16px var(--blue-glow);
    flex-shrink: 0;
}
.header-text-title {
    font-size: 1.45rem; font-weight: 700;
    color: var(--blue-main); line-height: 1.3;
}
.header-text-sub {
    font-size: 0.86rem; color: var(--text-soft);
    font-weight: 400; margin-top: 3px;
}
.top-spacer { height: 28px; background: var(--bg-base); }

/* ===== FORM ===== */
div[data-testid="stForm"] {
    background-color: var(--bg-base) !important;
    border: none !important;
    padding: 24px 64px 28px !important;
    box-shadow: none !important;
    border-radius: 0 !important;
}
div[data-testid="stForm"] div[data-testid="stHorizontalBlock"] {
    gap: 20px !important; align-items: stretch !important;
}

/* ===== CARDS ===== */
div[data-testid="stForm"] div[data-testid="stHorizontalBlock"] > div[data-testid="stVerticalBlock"] {
    background-color: var(--bg-card) !important;
    border-radius: 18px !important;
    border: 1.5px solid var(--border) !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.05) !important;
    padding: 26px 22px 30px !important;
    min-height: 460px !important;
    transition: box-shadow 0.2s, border-color 0.2s !important;
}
div[data-testid="stForm"] div[data-testid="stHorizontalBlock"] > div[data-testid="stVerticalBlock"]:hover {
    border-color: #BFDBFE !important;
    box-shadow: 0 4px 20px rgba(59,130,246,0.10) !important;
}

/* ===== CARD HEADER ===== */
.card-header {
    display: flex; align-items: center; gap: 12px;
    margin-bottom: 20px; padding-bottom: 14px;
    border-bottom: 1.5px solid var(--blue-light);
}
.card-icon-circle {
    width: 38px; height: 38px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
    background: linear-gradient(135deg, var(--blue-main), var(--blue-mid));
    box-shadow: 0 2px 10px var(--blue-glow);
}
.card-title-text {
    font-size: 1.02rem; font-weight: 700; color: var(--blue-main);
}

/* ===== BMI DISPLAY ===== */
.bmi-display {
    background: var(--blue-light);
    border: 1.5px solid #BFDBFE;
    border-radius: 10px;
    padding: 10px 14px;
    margin-top: 2px;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.bmi-value {
    font-size: 1.3rem;
    font-weight: 700;
    color: var(--blue-main);
}
.bmi-label {
    font-size: 0.82rem;
    color: var(--text-mid);
    line-height: 1.4;
}
.bmi-category {
    font-size: 0.8rem;
    font-weight: 600;
    padding: 2px 10px;
    border-radius: 20px;
    margin-left: auto;
}
.bmi-under   { background: #FEE2E2; color: #991B1B; }
.bmi-normal  { background: #DCFCE7; color: #166534; }
.bmi-over    { background: #FEF9C3; color: #854D0E; }
.bmi-obese   { background: #FEE2E2; color: #991B1B; }

/* ===== RADIO PILLS ===== */
div[data-testid="stElementContainer"] { width: 100% !important; }

div[data-testid="stRadio"] > label {
    font-size: 0.83rem !important;
    color: var(--text-mid) !important;
    font-weight: 500 !important;
    margin-bottom: 4px !important;
}
div[data-testid="stRadio"] [data-testid="stWidgetLabel"] p {
    font-size: 0.83rem !important;
    color: var(--text-mid) !important;
    font-weight: 500 !important;
}
div[data-testid="stRadio"] { width: 100% !important; }
div[data-testid="stRadio"] [role="radiogroup"] {
    background: #F8FAFF !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 3px !important;
    display: flex !important;
    width: 100% !important;
    gap: 2px !important;
    margin-top: 2px !important;
}
div[data-testid="stRadio"] [role="radiogroup"] label {
    flex: 1 1 0 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    padding: 10px 12px !important;
    margin: 0 !important;
    border-radius: 10px !important;
    background: transparent !important;
    border: none !important;
    font-size: 0.88rem !important;
    color: var(--text-mid) !important;
    min-height: 44px !important;
    white-space: nowrap !important;
}
div[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) {
    background: white !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.10) !important;
    color: var(--blue-main) !important;
    font-weight: 700 !important;
}
div[data-testid="stRadio"] [role="radiogroup"] label input[type="radio"] { display: none !important; }
div[data-testid="stRadio"] [role="radiogroup"] label > div:first-child { display: none !important; }

/* ===== LABELS & INPUTS ===== */
.stSelectbox > label, .stNumberInput > label {
    font-size: 0.83rem !important;
    color: var(--text-mid) !important;
    font-weight: 500 !important;
    margin-bottom: 3px !important;
}
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    background-color: #F8FAFF !important;
    border-radius: 10px !important;
    border: 1.5px solid var(--border) !important;
    font-family: 'Sarabun', sans-serif !important;
}
div[data-baseweb="select"] > div:focus-within,
div[data-baseweb="input"] > div:focus-within {
    border-color: var(--blue-mid) !important;
    box-shadow: 0 0 0 3px var(--blue-glow) !important;
}

/* ===== BUTTON ===== */
div[data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(135deg, var(--blue-main), var(--blue-mid)) !important;
    color: white !important;
    font-family: 'Sarabun', sans-serif !important;
    font-weight: 700 !important; font-size: 1.05rem !important;
    border-radius: 50px !important; border: none !important;
    padding: 14px 52px !important;
    box-shadow: 0 6px 20px rgba(59,130,246,0.35) !important;
    transition: all 0.2s ease !important;
}
div[data-testid="stFormSubmitButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 28px rgba(59,130,246,0.5) !important;
    filter: brightness(1.07) !important;
}

/* ===== RESULT ===== */
.result-section-bg { background: var(--bg-base); padding: 8px 64px 52px; }
.result-header { display: flex; align-items: center; gap: 14px; margin-bottom: 22px; padding-top: 8px; }
.result-icon-circle {
    width: 42px; height: 42px; border-radius: 13px;
    background: linear-gradient(135deg, var(--blue-main), var(--blue-mid));
    display: flex; align-items: center; justify-content: center;
    box-shadow: 0 4px 14px var(--blue-glow);
}
.result-title { font-size: 1.4rem; font-weight: 700; color: var(--blue-main); }

/* ===== WARNING CARDS ===== */
.warning-card {
    background: #FFF5F5; border-left: 4px solid #EF4444;
    border-radius: 14px; padding: 18px 20px; margin-bottom: 14px;
    display: flex; gap: 12px; align-items: flex-start;
}
.warning-card-low {
    background: #F0FDF4; border-left: 4px solid #22C55E;
    border-radius: 14px; padding: 18px 20px; margin-bottom: 14px;
    display: flex; gap: 12px; align-items: flex-start;
}
.warning-icon { font-size: 1.2rem; flex-shrink: 0; margin-top: 2px; }
.warning-text { font-size: 0.9rem; color: #7F1D1D; line-height: 1.7; }
.warning-text b { display: block; margin-bottom: 4px; font-size: 0.95rem; color: #991B1B; }
.warning-text-low { font-size: 0.9rem; color: #14532D; line-height: 1.7; }
.warning-text-low b { display: block; margin-bottom: 4px; font-size: 0.95rem; color: #166534; }
.note-card {
    background: var(--blue-light); border-left: 4px solid var(--blue-mid);
    border-radius: 14px; padding: 18px 20px;
    display: flex; gap: 12px; align-items: flex-start;
}
.note-text { font-size: 0.9rem; color: #1E3A5F; line-height: 1.7; }
.note-text b { display: block; margin-bottom: 4px; font-size: 0.95rem; color: var(--blue-main); }

/* ===== GAUGE ===== */
.gauge-card {
    background: white; border-radius: 18px; padding: 28px;
    box-shadow: 0 2px 14px rgba(0,0,0,0.06);
    border: 1.5px solid var(--border);
    text-align: center; margin-bottom: 14px;
}
.risk-badge { display: inline-block; padding: 5px 20px; border-radius: 20px; font-weight: 700; font-size: 0.93rem; margin-top: 6px; }
.risk-low    { background: #DCFCE7; color: #166534; }
.risk-high   { background: #FEE2E2; color: #991B1B; }
.legend-row {
    display: flex; justify-content: center; gap: 20px;
    margin-top: 14px; font-size: 0.83rem; color: var(--text-soft);
}
.legend-dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 5px; vertical-align: middle; }

/* ===== ACTION BOX ===== */
.action-box {
    background: white; border-radius: 14px; padding: 20px 22px;
    border: 1.5px solid var(--border);
    box-shadow: 0 2px 10px rgba(0,0,0,0.04);
    margin-bottom: 14px;
}
.action-box-title {
    font-size: 0.92rem; font-weight: 700; color: var(--text-dark);
    margin-bottom: 12px;
}
.action-item {
    display: flex; gap: 10px; align-items: flex-start;
    font-size: 0.88rem; color: var(--text-mid);
    line-height: 1.6; margin-bottom: 8px;
}
.action-item:last-child { margin-bottom: 0; }
.action-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: var(--blue-mid); flex-shrink: 0; margin-top: 7px;
}
.action-dot-red {
    width: 6px; height: 6px; border-radius: 50%;
    background: #EF4444; flex-shrink: 0; margin-top: 7px;
}

#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

@media (max-width: 1024px) {
    .page-header { padding: 20px 32px !important; }
    div[data-testid="stForm"] { padding: 16px 32px 24px !important; }
    .result-section-bg { padding: 8px 32px 40px !important; }
}
@media (max-width: 768px) {
    .page-header { padding: 16px 16px !important; gap: 12px !important; }
    .header-text-title { font-size: 1.05rem !important; }
    div[data-testid="stForm"] { padding: 12px 12px 20px !important; }
    div[data-testid="stForm"] div[data-testid="stHorizontalBlock"] { flex-direction: column !important; gap: 14px !important; }
    div[data-testid="stForm"] div[data-testid="stHorizontalBlock"] > div[data-testid="stVerticalBlock"] {
        width: 100% !important; min-height: unset !important; padding: 18px 14px 22px !important;
    }
    .result-section-bg { padding: 8px 12px 40px !important; }
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    try:
        m = joblib.load('xgb_tb_model.pkl')
        f = joblib.load('model_features.pkl')
        return m, f
    except:
        return None, None

model, model_features = load_assets()

def calc_bmi(weight_kg, height_cm):
    if height_cm <= 0: return 0.0
    return round(weight_kg / (height_cm / 100) ** 2, 1)

def bmi_category(bmi):
    if bmi < 18.5: return ("น้ำหนักน้อยกว่าเกณฑ์", "bmi-under")
    if bmi < 23.0: return ("น้ำหนักปกติ", "bmi-normal")
    if bmi < 25.0: return ("น้ำหนักเกิน", "bmi-over")
    return ("อ้วน", "bmi-obese")

# ─── Header ────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
    <div class="header-logo">
        <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2"
             stroke-linecap="round" stroke-linejoin="round">
            <path d="M12 5v14M5 12h14"/><rect x="3" y="3" width="18" height="18" rx="3"/>
        </svg>
    </div>
    <div>
        <div class="header-text-title">ระบบพยากรณ์ผลการรักษาวัณโรค (TB)</div>
        <div class="header-text-sub">กรอกข้อมูลสุขภาพของคุณเพื่อประเมินแนวโน้มการรักษา · ผลนี้ใช้ประกอบการพิจารณาของแพทย์เท่านั้น</div>
    </div>
</div>
<div class="top-spacer"></div>
""", unsafe_allow_html=True)

# ─── Form ──────────────────────────────────────────────────
with st.form("main_form"):
    col1, col2, col3 = st.columns(3, gap="large")

    # ── Card 1: ข้อมูลพื้นฐาน ──────────────────────────────
    with col1:
        st.markdown('''
        <div class="card-header">
            <div class="card-icon-circle">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2"
                     stroke-linecap="round" stroke-linejoin="round">
                    <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/>
                    <circle cx="12" cy="7" r="4"/>
                </svg>
            </div>
            <span class="card-title-text">ข้อมูลพื้นฐาน</span>
        </div>''', unsafe_allow_html=True)

        age = st.number_input("อายุ (ปี)", 0, 120, 45, key="age")
        gen = st.selectbox("เพศ", ["ชาย", "หญิง"], key="gen")

        w_col, h_col = st.columns(2)
        with w_col:
            weight = st.number_input("น้ำหนัก (กิโลกรัม)", 20.0, 200.0, 55.0, step=0.5, key="weight")
        with h_col:
            height = st.number_input("ส่วนสูง (เซนติเมตร)", 100.0, 220.0, 165.0, step=0.5, key="height")

        bmi = calc_bmi(weight, height)
        cat_text, cat_cls = bmi_category(bmi)
        st.markdown(f"""
        <div class="bmi-display">
            <div>
                <div class="bmi-value">{bmi}</div>
                <div class="bmi-label">ดัชนีมวลกาย (BMI)</div>
            </div>
            <span class="bmi-category {cat_cls}">{cat_text}</span>
        </div>""", unsafe_allow_html=True)

        dur = st.number_input("ระยะเวลาที่เข้ารับการรักษามาแล้ว (เดือน)", 0, 200, 6, key="dur")

    # ── Card 2: ประวัติสุขภาพ ──────────────────────────────
    with col2:
        st.markdown('''
        <div class="card-header">
            <div class="card-icon-circle">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2"
                     stroke-linecap="round" stroke-linejoin="round">
                    <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
                </svg>
            </div>
            <span class="card-title-text">ประวัติสุขภาพ</span>
        </div>''', unsafe_allow_html=True)

        hiv = st.selectbox(
            "สถานะการติดเชื้อ HIV",
            ["ไม่ติดเชื้อ (ผลลบ)", "ติดเชื้อ (ผลบวก)", "ไม่ทราบ / ไม่เคยตรวจ"],
            key="hiv"
        )
        dm_r   = st.radio("โรคเบาหวาน",        ["ไม่เป็น", "เป็น"], horizontal=True, key="dm")
        ckd_r  = st.radio("โรคไตเรื้อรัง",      ["ไม่เป็น", "เป็น"], horizontal=True, key="ckd")
        copd_r = st.radio("โรคถุงลมโป่งพอง",    ["ไม่เป็น", "เป็น"], horizontal=True, key="copd")
        liv_r  = st.radio("โรคตับ",             ["ไม่เป็น", "เป็น"], horizontal=True, key="liv")

        dm   = dm_r   == "เป็น"
        ckd  = ckd_r  == "เป็น"
        copd = copd_r == "เป็น"
        liv  = liv_r  == "เป็น"

    # ── Card 3: ผลการตรวจ ──────────────────────────────────
    with col3:
        st.markdown('''
        <div class="card-header">
            <div class="card-icon-circle">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2"
                     stroke-linecap="round" stroke-linejoin="round">
                    <path d="M9 3h6m-6 0v6l-4 9a1 1 0 0 0 .9 1.45h12.2A1 1 0 0 0 19 18L15 9V3m-6 0h6"/>
                </svg>
            </div>
            <span class="card-title-text">ผลการตรวจ</span>
        </div>''', unsafe_allow_html=True)

        pos = st.selectbox(
            "ตำแหน่งที่ติดวัณโรค",
            ["ในปอด (มีอาการไอ หรือเอกซเรย์ปอด)", "นอกปอด (ต่อมน้ำเหลือง กระดูก ฯลฯ)", "ทั้งในและนอกปอด"],
            key="pos"
        )

        afb = st.selectbox(
            "ผลตรวจเสมหะ เดือนที่ 1 (แพทย์จะแจ้งผล)",
            [
                "ไม่ทราบ / ไม่มีผล",
                "ไม่พบเชื้อ",
                "พบเชื้อน้อยมาก (Scanty)",
                "พบเชื้อน้อย (1+)",
                "พบเชื้อปานกลาง (2+)",
                "พบเชื้อมาก (3+)",
            ],
            key="afb"
        )

        arv_r = st.radio(
            "ได้รับยาต้านไวรัส HIV (ARV) หรือไม่",
            ["ไม่ได้รับ", "ได้รับ"],
            horizontal=True, key="arv"
        )
        arv = arv_r == "ได้รับ"

        f_u = st.number_input("จำนวนครั้งที่มาติดตามการรักษา (ครั้ง)", 0, 50, 1, key="f_u")

    st.markdown("<br>", unsafe_allow_html=True)
    _, btn_mid, _ = st.columns([3, 2, 3])
    with btn_mid:
        sub = st.form_submit_button("ประเมินผลการรักษา", use_container_width=True)

# ─── Result ────────────────────────────────────────────────
if sub:
    if model is not None:

        hiv_map = {"ไม่ติดเชื้อ (ผลลบ)": 0, "ติดเชื้อ (ผลบวก)": 1, "ไม่ทราบ / ไม่เคยตรวจ": 2}
        pos_map = {
            "ในปอด (มีอาการไอ หรือเอกซเรย์ปอด)": 0,
            "นอกปอด (ต่อมน้ำเหลือง กระดูก ฯลฯ)": 1,
            "ทั้งในและนอกปอด": 2
        }
        # "ไม่ทราบ / ไม่มีผล" → fallback เป็น 0 (Negative = safe default)
        afb_map = {
            "ไม่ทราบ / ไม่มีผล": 0,
            "ไม่พบเชื้อ": 0,
            "พบเชื้อน้อยมาก (Scanty)": 1,
            "พบเชื้อน้อย (1+)": 2,
            "พบเชื้อปานกลาง (2+)": 3,
            "พบเชื้อมาก (3+)": 4,
        }
        gen_map = {"ชาย": 0, "หญิง": 1}

        enc = {
            'Age':                                    age,
            'Gender':                                 gen_map.get(gen, 0),
            'BMI':                                    bmi,
            'Duration':                               dur,
            'HIV':                                    hiv_map.get(hiv, 0),
            'Diabetes Mellitus':                      int(dm),
            'Chronic Kidney Disease':                 int(ckd),
            ' Chronic Obstructive Pulmonary Disease': int(copd),
            'Liver Disease':                          int(liv),
            ' AFB resulf of first month':             afb_map.get(afb, 0),
            'position of TB':                         pos_map.get(pos, 0),
            '(TB F/U) follow up':                     f_u,
            'Treatment of ARV':                       int(arv),
        }

        df = pd.DataFrame(0, index=[0], columns=model_features)
        for k, v in enc.items():
            if k in df.columns:
                df[k] = v

        prob     = model.predict_proba(df)[0][1]
        risk_pct = prob * 100

        OPTIMAL_THRESHOLD = 0.5
        is_high = prob > OPTIMAL_THRESHOLD

        booster = model.get_booster()

        importance = booster.get_score(importance_type='gain')

        imp_df = pd.DataFrame(
            importance.items(),
            columns=['Feature', 'Importance']
        ).sort_values('Importance', ascending=False)

        st.dataframe(imp_df)

        # ── Result layout ────────────────────────────────────
        st.markdown('<div class="result-section-bg">', unsafe_allow_html=True)
        st.markdown("""
        <div class="result-header">
            <div class="result-icon-circle">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2"
                     stroke-linecap="round" stroke-linejoin="round">
                    <line x1="18" y1="20" x2="18" y2="10"/>
                    <line x1="12" y1="20" x2="12" y2="4"/>
                    <line x1="6" y1="20" x2="6" y2="14"/>
                </svg>
            </div>
            <span class="result-title">ผลการประเมิน</span>
        </div>""", unsafe_allow_html=True)

        # ── Summary card ─────────────────────────────────────
        if is_high:
            wc   = "warning-card"
            wt   = "warning-text"
            wi   = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#EF4444" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>'
            wtitle = "ผลการประเมิน: มีปัจจัยที่ควรติดตามอย่างใกล้ชิด"
            wbody  = ("จากข้อมูลที่กรอก ระบบพบว่ามีปัจจัยบางอย่างที่อาจทำให้การรักษาไม่เป็นไปตามแผน "
                      "ซึ่งอาจหมายถึงการดูแลรักษาที่ต้องปรับเปลี่ยน ไม่ได้หมายความว่าการรักษาจะล้มเหลวเสมอไป "
                      "ขอแนะนำให้แจ้งแพทย์ผู้ดูแลเพื่อประเมินซ้ำโดยเร็ว")
        else:
            wc   = "warning-card-low"
            wt   = "warning-text-low"
            wi   = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#22C55E" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>'
            wtitle = "ผลการประเมิน: แนวโน้มการรักษาอยู่ในเกณฑ์ดี"
            wbody  = ("จากข้อมูลที่กรอก ระบบประเมินว่าการรักษาของคุณมีแนวโน้มที่ดี "
                      "การรักษาวัณโรคต้องใช้เวลาและความสม่ำเสมอ ผลอาจเปลี่ยนแปลงได้ "
                      "ขอให้รับประทานยาต่อเนื่องและมาพบแพทย์ตามนัดทุกครั้ง")

        st.markdown(f"""
        <div class="{wc}">
            <span class="warning-icon">{wi}</span>
            <div class="{wt}"><b>{wtitle}</b>{wbody}</div>
        </div>""", unsafe_allow_html=True)

        # ── Gauge ────────────────────────────────────────────
        st.markdown('<div class="gauge-card">', unsafe_allow_html=True)

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_pct,
            number={
                'suffix': "%",
                'font': {'size': 56, 'family': 'Sarabun', 'color': '#1D4ED8'},
                'valueformat': '.1f'
            },
            gauge={
                'axis': {
                    'range': [0, 100],
                    'tickwidth': 1,
                    'tickcolor': '#CBD5E1',
                    'tickfont': {'family': 'Sarabun', 'color': '#94A3B8'}
                },
                'bar':  {'color': "rgba(0,0,0,0)", 'thickness': 0},
                'bgcolor': "rgba(0,0,0,0)", 'borderwidth': 0,
                'steps': [
                    {'range': [0,  65], 'color': '#86EFAC'},
                    {'range': [65, 100], 'color': '#FCA5A5'}
                ],
                'threshold': {
                    'line': {'color': "#1D4ED8", 'width': 5},
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

        badge_cls  = "risk-high" if is_high else "risk-low"
        risk_label = "มีปัจจัยเสี่ยง" if is_high else "แนวโน้มดี"

        col_g1, col_g2, col_g3 = st.columns([1, 2, 1])
        with col_g2:
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            st.markdown(f"""
            <div style="text-align:center; margin-top:-14px; margin-bottom:4px;">
                <span style="font-size:0.93rem; color:#64748B; font-weight:500;">ผลประเมิน: </span>
                <span class="risk-badge {badge_cls}">{risk_label}</span>
            </div>
            <div style="text-align:center; margin-bottom:12px;">
                <span style="font-size:0.78rem; color:#94A3B8;">
                    ค่านี้แสดงโอกาสที่การรักษาอาจไม่เป็นไปตามแผน ประเมิน ณ วันที่กรอกข้อมูล
                </span>
            </div>
            <div class="legend-row">
                <span><span class="legend-dot" style="background:#22C55E;"></span>แนวโน้มดี (0–65%)</span>
                <span><span class="legend-dot" style="background:#EF4444;"></span>มีปัจจัยเสี่ยง (65–100%)</span>
            </div>""", unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # ── Action items ─────────────────────────────────────
        if is_high:
            action_html = """
            <div class="action-box">
                <div class="action-box-title">สิ่งที่ควรทำต่อ</div>
                <div class="action-item">
                    <div class="action-dot-red"></div>
                    <span>แจ้งแพทย์หรือพยาบาลผู้ดูแลโดยเร็ว ไม่ต้องรอนัดถัดไป</span>
                </div>
                <div class="action-item">
                    <div class="action-dot-red"></div>
                    <span>รับประทานยาให้ครบทุกวัน ห้ามหยุดยาเองเด็ดขาด เพราะอาจทำให้เชื้อดื้อยา</span>
                </div>
                <div class="action-item">
                    <div class="action-dot-red"></div>
                    <span>แจ้งแพทย์ทันทีหากมีอาการเปลี่ยนแปลง เช่น ไอมากขึ้น มีไข้ น้ำหนักลดเร็ว หรือเหนื่อยผิดปกติ</span>
                </div>
                <div class="action-item">
                    <div class="action-dot-red"></div>
                    <span>กลับมาประเมินอีกครั้งหลังพบแพทย์หรือเมื่อข้อมูลสุขภาพเปลี่ยนแปลง</span>
                </div>
            </div>"""
        else:
            action_html = """
            <div class="action-box">
                <div class="action-box-title">สิ่งที่ควรทำต่อ</div>
                <div class="action-item">
                    <div class="action-dot"></div>
                    <span>รับประทานยาให้ครบทุกวันตามที่แพทย์สั่ง ไม่หยุดยาเองแม้อาการจะดีขึ้นแล้ว</span>
                </div>
                <div class="action-item">
                    <div class="action-dot"></div>
                    <span>มาพบแพทย์ตามนัดทุกครั้ง และแจ้งอาการเปลี่ยนแปลงที่พบ</span>
                </div>
                <div class="action-item">
                    <div class="action-dot"></div>
                    <span>ดูแลโภชนาการและพักผ่อนให้เพียงพอ ช่วยให้ร่างกายตอบสนองต่อยาได้ดีขึ้น</span>
                </div>
                <div class="action-item">
                    <div class="action-dot"></div>
                    <span>กลับมาประเมินอีกครั้งในเดือนถัดไปหรือเมื่อข้อมูลสุขภาพเปลี่ยนแปลง</span>
                </div>
            </div>"""

        st.markdown(action_html, unsafe_allow_html=True)

        # ── Disclaimer ───────────────────────────────────────
        st.markdown(f"""
        <div class="note-card" style="margin-top:4px;">
            <span class="warning-icon">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#3B82F6" stroke-width="2"
                     stroke-linecap="round" stroke-linejoin="round">
                    <circle cx="12" cy="12" r="10"/>
                    <line x1="12" y1="8" x2="12" y2="12"/>
                    <line x1="12" y1="16" x2="12.01" y2="16"/>
                </svg>
            </span>
            <div class="note-text">
                <b>หมายเหตุสำคัญ</b>
                ผลนี้ประเมินจากข้อมูล ณ วันที่กรอก สามารถกลับมาประเมินใหม่ได้ทุกครั้งที่ข้อมูลเปลี่ยนแปลง
                ระบบนี้ใช้ AI วิเคราะห์เบื้องต้นเท่านั้น ไม่สามารถแทนการวินิจฉัยหรือคำแนะนำของแพทย์ได้
                หากมีข้อสงสัย ควรปรึกษาบุคลากรทางการแพทย์โดยตรง
            </div>
        </div>
        </div>""", unsafe_allow_html=True)

    else:
        st.error("ไม่พบไฟล์โมเดล กรุณาตรวจสอบว่ามีไฟล์ `xgb_tb_model.pkl` และ `model_features.pkl` อยู่ในโฟลเดอร์เดียวกัน")
