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
    background-color: #F5F5FA !important;
}

.main .block-container {
    padding: 0 !important;
    max-width: 100% !important;
    background-color: #F5F5FA !important;
}

/* ===== HEADER ===== */
.page-header {
    background: #FFFFFF;
    padding: 28px 64px 24px;
    border-bottom: 1px solid #EBEBF0;
    display: flex;
    align-items: center;
    gap: 18px;
}
.header-logo {
    width: 56px;
    height: 56px;
    background: linear-gradient(135deg, #FF3D6B 0%, #C9186C 100%);
    border-radius: 16px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.8rem;
    box-shadow: 0 4px 14px rgba(201, 24, 108, 0.3);
    flex-shrink: 0;
}
.header-text-title {
    font-size: 1.55rem;
    font-weight: 700;
    color: #7B2FBE;
    line-height: 1.3;
}
.header-text-sub {
    font-size: 0.88rem;
    color: #8A8AAA;
    font-weight: 400;
    margin-top: 2px;
}

/* ===== MAIN CONTENT ===== */
.content-wrapper {
    padding: 36px 64px 48px;
    background-color: #F5F5FA;
}

/* พื้นหลังทั้ง form */
div[data-testid="stForm"] {
    background-color: #F5F5FA !important;
    border: none !important;
    padding: 0 64px !important;
    box-shadow: none !important;
    border-radius: 0 !important;
}

/* gap ระหว่าง columns */
div[data-testid="stForm"] div[data-testid="stHorizontalBlock"] {
    gap: 24px !important;
    align-items: stretch !important;
}

/* ===== INPUT CARDS ===== */
div[data-testid="stForm"] div[data-testid="stHorizontalBlock"] > div[data-testid="stVerticalBlock"] {
    background-color: #FFFFFF !important;
    border-radius: 20px !important;
    border: 1px solid #EBEBF0 !important;
    box-shadow: 0 2px 16px rgba(0,0,0,0.06) !important;
    padding: 28px 24px 32px 24px !important;
    min-height: 460px !important;
}

/* ===== CARD HEADERS ===== */
.card-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 20px;
    padding-bottom: 14px;
    border-bottom: 1px solid #F0F0F5;
}
.card-icon-circle {
    width: 38px;
    height: 38px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
    flex-shrink: 0;
}
.icon-pink   { background: linear-gradient(135deg, #FF6B9D, #FF3D6B); }
.icon-blue   { background: linear-gradient(135deg, #60B8FF, #3D7EFF); }
.icon-green  { background: linear-gradient(135deg, #56E0A0, #00B96B); }

.card-title-text {
    font-size: 1.05rem;
    font-weight: 700;
    color: #1A1A2E;
}

/* ===== FORM LABELS ===== */
.stSelectbox > label,
.stNumberInput > label {
    font-size: 0.84rem !important;
    color: #6B6B8A !important;
    font-weight: 500 !important;
    margin-bottom: 3px !important;
}

/* ===== FORM INPUTS ===== */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    background-color: #F9F9FC !important;
    border-radius: 10px !important;
    border: 1.5px solid #E8E8F0 !important;
    font-family: 'Sarabun', sans-serif !important;
}
div[data-baseweb="select"] > div:focus-within,
div[data-baseweb="input"] > div:focus-within {
    border-color: #A855F7 !important;
    box-shadow: 0 0 0 3px rgba(168, 85, 247, 0.12) !important;
}

/* ===== SUBMIT BUTTON ===== */
div[data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(135deg, #C026A8 0%, #7B2FBE 100%) !important;
    color: white !important;
    font-family: 'Sarabun', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    border-radius: 50px !important;
    border: none !important;
    padding: 14px 48px !important;
    box-shadow: 0 6px 20px rgba(168, 56, 180, 0.4) !important;
    transition: all 0.2s ease !important;
    white-space: nowrap !important;
}
div[data-testid="stFormSubmitButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(168, 56, 180, 0.55) !important;
    filter: brightness(1.05) !important;
}

/* ===== RESULT AREA ===== */
.result-section {
    padding: 0 64px 48px;
    background-color: #F5F5FA;
}
.result-header {
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 28px;
}
.result-icon-circle {
    width: 44px;
    height: 44px;
    border-radius: 14px;
    background: linear-gradient(135deg, #7B2FBE, #C026A8);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.3rem;
}
.result-title {
    font-size: 1.45rem;
    font-weight: 700;
    color: #1A1A2E;
}

/* ===== WARNING CARDS ===== */
.warning-card {
    background: #FFF5F5;
    border-left: 4px solid #E53935;
    border-radius: 14px;
    padding: 18px 20px;
    margin-bottom: 14px;
    display: flex;
    gap: 12px;
    align-items: flex-start;
}
.warning-card-medium {
    background: #FFFBEA;
    border-left: 4px solid #F59E0B;
    border-radius: 14px;
    padding: 18px 20px;
    margin-bottom: 14px;
    display: flex;
    gap: 12px;
    align-items: flex-start;
}
.warning-card-low {
    background: #F0FDF4;
    border-left: 4px solid #22C55E;
    border-radius: 14px;
    padding: 18px 20px;
    margin-bottom: 14px;
    display: flex;
    gap: 12px;
    align-items: flex-start;
}
.warning-icon { font-size: 1.2rem; flex-shrink: 0; margin-top: 2px; }
.warning-text { font-size: 0.9rem; color: #7F1D1D; line-height: 1.7; }
.warning-text b { display: block; margin-bottom: 4px; font-size: 0.95rem; color: #991B1B; }
.warning-text-medium { font-size: 0.9rem; color: #78350F; line-height: 1.7; }
.warning-text-medium b { display: block; margin-bottom: 4px; font-size: 0.95rem; color: #92400E; }
.warning-text-low { font-size: 0.9rem; color: #14532D; line-height: 1.7; }
.warning-text-low b { display: block; margin-bottom: 4px; font-size: 0.95rem; color: #166534; }

.note-card {
    background: #EFF6FF;
    border-left: 4px solid #60A5FA;
    border-radius: 14px;
    padding: 18px 20px;
    display: flex;
    gap: 12px;
    align-items: flex-start;
}
.note-text { font-size: 0.9rem; color: #1E3A5F; line-height: 1.7; }
.note-text b { display: block; margin-bottom: 4px; font-size: 0.95rem; color: #1D4ED8; }

/* ===== GAUGE CARD ===== */
.gauge-card {
    background: white;
    border-radius: 20px;
    padding: 28px;
    box-shadow: 0 2px 16px rgba(0,0,0,0.07);
    text-align: center;
    border: 1px solid #EBEBF0;
}
.risk-badge {
    display: inline-block;
    padding: 6px 22px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 0.95rem;
    margin-top: 6px;
}
.risk-low    { background: #DCFCE7; color: #166534; }
.risk-medium { background: #FEF9C3; color: #854D0E; }
.risk-high   { background: #FEE2E2; color: #991B1B; }

.legend-row {
    display: flex;
    justify-content: center;
    gap: 20px;
    margin-top: 14px;
    font-size: 0.84rem;
    color: #6B6B8A;
}
.legend-dot {
    display: inline-block;
    width: 10px; height: 10px;
    border-radius: 50%;
    margin-right: 5px;
}

/* ===== HIDE STREAMLIT DEFAULTS ===== */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ===== RESPONSIVE — TABLET ===== */
@media (max-width: 1024px) {
    .page-header { padding: 22px 32px 20px !important; }
    div[data-testid="stForm"] { padding: 0 32px !important; }
    .result-section { padding: 0 32px 40px !important; }
    .header-text-title { font-size: 1.25rem !important; }
}

/* ===== RESPONSIVE — MOBILE ===== */
@media (max-width: 768px) {
    .page-header { padding: 18px 16px !important; gap: 12px !important; }
    .header-text-title { font-size: 1.05rem !important; }
    .header-text-sub { font-size: 0.78rem !important; }
    .header-logo { width: 44px !important; height: 44px !important; border-radius: 12px !important; }

    div[data-testid="stForm"] { padding: 0 12px !important; }
    div[data-testid="stForm"] div[data-testid="stHorizontalBlock"] {
        flex-direction: column !important; gap: 14px !important;
    }
    div[data-testid="stForm"] div[data-testid="stHorizontalBlock"] > div[data-testid="stVerticalBlock"] {
        width: 100% !important; min-width: 100% !important;
        min-height: unset !important; padding: 20px 16px 24px !important;
    }
    .result-section { padding: 0 12px 40px !important; }
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

# --- 4. HEADER (แทน navbar) ---
st.markdown("""
<div class="page-header">
    <div class="header-logo">🏥</div>
    <div>
        <div class="header-text-title">ระบบพยากรณ์ความเสี่ยงผลการรักษาวัณโรค (TB)</div>
        <div class="header-text-sub">ระบบวิเคราะห์ความเสี่ยงในการรักษาผู้ป่วยวัณโรคด้วยเทคโนโลยี AI</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)

# --- 5. FORM ---
with st.form("main_form"):
    col1, col2, col3 = st.columns(3, gap="large")

    # ---- CARD 1: ข้อมูลพื้นฐาน ----
    with col1:
        st.markdown("""
        <div class="card-header">
            <div class="card-icon-circle icon-pink">📍</div>
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
            <div class="card-icon-circle icon-blue">💙</div>
            <span class="card-title-text">ประวัติสุขภาพ</span>
        </div>
        """, unsafe_allow_html=True)
        hiv  = st.selectbox("สถานะ HIV",                        ["ไม่ติดเชื้อ (Negative)", "ติดเชื้อ (Positive)", "ไม่ทราบ"])
        ckd  = st.selectbox("โรคไตเรื้อรัง (CKD)",              ["ไม่เป็น", "เป็น"])
        copd = st.selectbox("โรคปอดอุดกั้นเรื้อรัง (COPD)",    ["ไม่เป็น", "เป็น"])
        liv  = st.selectbox("โรคตับ (Liver Disease)",           ["ไม่เป็น", "เป็น"])
        dm   = st.selectbox("โรคเบาหวาน (Diabetes Mellitus)",   ["ไม่เป็น", "เป็น"])

    # ---- CARD 3: ผลการตรวจ ----
    with col3:
        st.markdown("""
        <div class="card-header">
            <div class="card-icon-circle icon-green">🧪</div>
            <span class="card-title-text">ผลการตรวจ</span>
        </div>
        """, unsafe_allow_html=True)
        afb = st.selectbox("AFB เดือนที่ 1",        ["Negative", "1+", "2+", "3+", "Scanty"])
        pos = st.selectbox("ตำแหน่งการติดเชื้อ",    ["ในปอด", "นอกปอด", "ในและนอกปอด"])
        f_u = st.number_input("จำนวนครั้งติดตาม",   0, 50, 1)
        arv = st.selectbox("สถานะ ARV",              ["ไม่ได้รับ", "ได้รับ"])

    # ---- SUBMIT BUTTON ----
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

        st.markdown("<div class='result-section'>", unsafe_allow_html=True)

        # Result header
        st.markdown("""
        <div class="result-header">
            <div class="result-icon-circle">📊</div>
            <span class="result-title">ผลการวิเคราะห์</span>
        </div>
        """, unsafe_allow_html=True)

        # ---- Warning card (แสดงก่อน gauge ตาม Figma) ----
        if prob > 0.6:
            warning_card_class = "warning-card"
            warning_text_class = "warning-text"
            warning_icon = "⚠️"
            warning_title = "คำเตือน: ผู้ป่วยอยู่ในกลุ่มเสี่ยงต่อผลการรักษาที่ไม่สำเร็จ"
            warning_body = (
                "ผู้ป่วยที่มีดัชนีมวลกายต่ำ (BMI &lt; 18.5) อายุมาก และมีโรคประจำตัวหลายโรค "
                "มีโอกาสสูงที่จะเกิดผลการรักษาที่ไม่สำเร็จ ควรได้รับการติดตามอย่างใกล้ชิด "
                "และพิจารณาแผนการรักษาเสริมเพิ่มเติม"
            )
            note_icon = "ℹ️"
            note_body = (
                "ผลลัพธ์นี้เป็นเพียงการพยากรณ์เบื้องต้นจากระบบปัญญาประดิษฐ์ ไม่สามารถใช้แทนการวินิจฉัยทางคลินิกได้ "
                "กรุณาปรึกษาแพทย์เพื่อรับคำแนะนำที่เหมาะสมกับสภาวะของผู้ป่วย"
            )
        elif prob > 0.3:
            warning_card_class = "warning-card-medium"
            warning_text_class = "warning-text-medium"
            warning_icon = "⚠️"
            warning_title = "ควรติดตามอาการ: ผู้ป่วยมีปัจจัยเสี่ยงบางประการ"
            warning_body = (
                "พบปัจจัยเสี่ยงบางส่วนที่อาจส่งผลต่อผลการรักษา ควรนัดติดตามผู้ป่วยอย่างสม่ำเสมอ "
                "และให้ความรู้เกี่ยวกับการรับประทานยาอย่างต่อเนื่องเพื่อป้องกันการดื้อยา"
            )
            note_icon = "ℹ️"
            note_body = (
                "ผลลัพธ์นี้เป็นเพียงการพยากรณ์เบื้องต้นจากระบบปัญญาประดิษฐ์ ไม่สามารถใช้แทนการวินิจฉัยทางคลินิกได้ "
                "กรุณาใช้ข้อมูลนี้ประกอบการพิจารณาของแพทย์ผู้ดูแล"
            )
        else:
            warning_card_class = "warning-card-low"
            warning_text_class = "warning-text-low"
            warning_icon = "✅"
            warning_title = "แนวโน้มดี: ความเสี่ยงต่ำ"
            warning_body = (
                "ผู้ป่วยรายนี้มีแนวโน้มการรักษาที่ดี หากรับประทานยาครบตามกำหนดและดูแลสุขภาพอย่างเหมาะสม "
                "โอกาสหายขาดจากวัณโรคอยู่ในระดับสูง ควรมาตรวจตามนัดทุกครั้ง"
            )
            note_icon = "ℹ️"
            note_body = (
                "แม้ความเสี่ยงจะอยู่ในระดับต่ำ ผลลัพธ์นี้ยังคงเป็นเพียงการพยากรณ์เบื้องต้นจากระบบ AI "
                "ไม่ควรใช้เป็นเหตุผลในการหยุดหรือปรับเปลี่ยนการรักษาด้วยตนเอง"
            )

        # Warning card แสดงก่อน (ตาม Figma)
        st.markdown(f"""
        <div class="{warning_card_class}">
            <span class="warning-icon">{warning_icon}</span>
            <div class="{warning_text_class}">
                <b>{warning_title}</b>
                {warning_body}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Gauge chart (กึ่งกลาง ตาม Figma)
        st.markdown('<div class="gauge-card" style="margin-bottom:14px;">', unsafe_allow_html=True)

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_pct,
            number={
                'suffix': "%",
                'font': {'size': 56, 'family': 'Sarabun', 'color': '#C026A8'},
                'valueformat': '.2f'
            },
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': '#ccc',
                         'tickfont': {'family': 'Sarabun', 'color': '#999'}},
                'bar':  {'color': "rgba(0,0,0,0)", 'thickness': 0},
                'bgcolor': "rgba(0,0,0,0)",
                'borderwidth': 0,
                'steps': [
                    {'range': [0,   30],  'color': '#4ADE80'},
                    {'range': [30,  60],  'color': '#FBBF24'},
                    {'range': [60,  100], 'color': '#F87171'}
                ],
                'threshold': {
                    'line': {'color': "#7B2FBE", 'width': 5},
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
                    <span style="font-size:0.95rem; color:#6B6B8A; font-weight:500;">ระดับความเสี่ยง: </span>
                    <span class="risk-badge {badge_cls}">{risk_text}</span>
                </div>
                <div class="legend-row">
                    <span><span class="legend-dot" style="background:#4ADE80;"></span>ต่ำ (0–30%)</span>
                    <span><span class="legend-dot" style="background:#FBBF24;"></span>ปานกลาง (30–60%)</span>
                    <span><span class="legend-dot" style="background:#F87171;"></span>สูง (60–100%)</span>
                </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # Note card (ล่างสุด)
        st.markdown(f"""
        <div class="note-card" style="margin-top:4px;">
            <span class="warning-icon">{note_icon}</span>
            <div class="note-text">
                <b>หมายเหตุ:</b>
                {note_body}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.error("⚠️ ไม่พบไฟล์โมเดล กรุณาตรวจสอบว่ามีไฟล์ `xgb_tb_model.pkl` และ `model_features.pkl` ในโฟลเดอร์เดียวกัน")
