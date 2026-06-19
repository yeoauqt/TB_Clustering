import streamlit as st
import pandas as pd
import xgboost as xgb
import joblib
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score

st.set_page_config(
    page_title="ระบบพยากรณ์ความเสี่ยงวัณโรค TB",
    page_icon="TB",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# เกณฑ์ความเสี่ยงที่แท้จริงจากการตรวจสอบความถูกต้องของแบบจำลอง (Validation Set)
OPTIMAL_THRESHOLD = 0.6499847769737244

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

html, body,
[class*="css"],
.stApp,
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

/* ===== FORM & SPACING ADJUSTMENT ===== */
div[data-testid="stForm"] {
    background-color: var(--bg-base) !important;
    border: none !important;
    padding: 24px 64px 28px !important; /* แก้ไขระยะห่างด้านบนไม่ให้ชิดกรอบเกินไป */
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

/* ===== LABELS ===== */
.stSelectbox > label, .stNumberInput > label, .stCheckbox > label {
    font-size: 0.83rem !important;
    color: var(--text-mid) !important;
    font-weight: 500 !important;
    margin-bottom: 3px !important;
}

/* ===== INPUTS ===== */
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

#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

@media (max-width: 1024px) {
    .page-header { padding: 20px 32px !important; }
    div[data-testid="stForm"] { padding: 24px 32px 24px !important; }
    .result-section-bg { padding: 8px 32px 40px !important; }
}
@media (max-width: 768px) {
    .page-header { padding: 16px 16px !important; gap: 12px !important; }
    .header-text-title { font-size: 1.05rem !important; }
    div[data-testid="stForm"] { padding: 24px 12px 20px !important; }
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

# ตัวแปรฟอร์มหลัก
with st.form("main_form"):
    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        st.markdown('<div class="card-header"><div class="card-icon-circle"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg></div><span class="card-title-text">ข้อมูลพื้นฐาน</span></div>', unsafe_allow_html=True)
        age = st.number_input("อายุ (ปี)", 0, 120, 45, key="age")
        gen = st.selectbox("เพศ", ["ชาย", "หญิง"], key="gen")
        
        # ปรับเปลี่ยนช่องกรอก BMI เป็น น้ำหนัก และ ส่วนสูง เพื่อเพิ่มประสบการณ์ผู้ใช้งาน (UX)
        weight = st.number_input("น้ำหนัก (กิโลกรัม)", 5.0, 250.0, 60.0, key="weight")
        height = st.number_input("ส่วนสูง (เซนติเมตร)", 50.0, 250.0, 170.0, key="height")
        # สูตรการคำนวณดัชนีมวลกายอัตโนมัติ
        bmi = weight / ((height / 100) ** 2)
        
        dur = st.number_input("ระยะเวลาการรักษา (เดือน)", 0, 1000, 6, key="dur")

    with col2:
        st.markdown('<div class="card-header"><div class="card-icon-circle"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg></div><span class="card-title-text">ประวัติสุขภาพ</span></div>', unsafe_allow_html=True)
        hiv = st.selectbox("สถานะ HIV", ["ไม่ติดเชื้อ (Negative)", "ติดเชื้อ (Positive)", "ไม่ทราบ"])
        
        # ปรับโรคร่วมต่างๆ จาก Selectbox เป็น Checkbox ตามที่ระบุ
        st.markdown("<br>", unsafe_allow_html=True)
        dm = st.checkbox("โรคเบาหวาน", key="dm")
        ckd = st.checkbox("โรคไตเรื้อรัง", key="ckd")
        copd = st.checkbox("โรคถุงลมโป่งพอง", key="copd")
        liv = st.checkbox("โรคตับ", key="liv")

    with col3:
        st.markdown('<div class="card-header"><div class="card-icon-circle"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 3h6m-6 0v6l-4 9a1 1 0 0 0 .9 1.45h12.2A1 1 0 0 0 19 18L15 9V3m-6 0h6"/></svg></div><span class="card-title-text">ผลการตรวจ</span></div>', unsafe_allow_html=True)
        # ปรับปรุงข้อความรายงานเสมหะ
        afb = st.selectbox("AFB เดือนที่ 1", ["ไม่พบเชื้อ", "พบเชื้อน้อยมาก (Scanty)", "พบเชื้อน้อย (1+)", "พบเชื้อปานกลาง (2+)", "พบเชื้อมาก (3+)"])
        pos = st.selectbox("ตำแหน่งการติดเชื้อ", ["ในปอด", "นอกปอด", "ทั้งในและนอกปอด"])
        f_u = st.number_input("จำนวนครั้งติดตาม", 0, 50, 1)
        
        # ปรับปรุงโครงสร้างช่องตรวจสอบสถานะการรับยาต้านไวรัส
        arv = st.checkbox("ได้รับยาต้านไวรัส HIV (ARV) หรือไม่")

    st.markdown("<br>", unsafe_allow_html=True)
    _, btn_mid, _ = st.columns([3, 2, 3])
    with btn_mid:
        sub = st.form_submit_button("วิเคราะห์ผลการรักษา", use_container_width=True)

if sub:
    if model is not None:
        def encode(d):
            mapping = {
                'Gender': {'ชาย': 0, 'หญิง': 1},
                'HIV': {'ไม่ติดเชื้อ (Negative)': 0, 'ติดเชื้อ (Positive)': 1, 'ไม่ทราบ': 2},
                'AFB resulf of first month': {
                    "ไม่พบเชื้อ": 0,
                    "พบเชื้อน้อยมาก (Scanty)": 4,
                    "พบเชื้อน้อย (1+)": 1,
                    "พบเชื้อปานกลาง (2+)": 2,
                    "พบเชื้อมาก (3+)": 3
                },
                'position of TB': {
                    "ในปอด": 0,
                    "นอกปอด": 1,
                    "ทั้งในและนอกปอด": 2
                }
            }
            for k, v in mapping.items():
                if k in d: d[k] = v.get(d[k], 0)
            return d

        raw = {
            'Age': age, 
            'Gender': gen, 
            'BMI': bmi, 
            'Duration': dur,
            'HIV': hiv, 
            'Diabetes Mellitus': int(dm), 
            'Chronic Kidney Disease': int(ckd),
            ' Chronic Obstructive Pulmonary Disease': int(copd), 
            'Liver Disease': int(liv),
            ' AFB resulf of first month': afb, 
            'position of TB': pos,
            '(TB F/U) follow up': f_u, 
            'Treatment of ARV': int(arv)
        }
        
        enc = encode(raw)
        df = pd.DataFrame(0, index=[0], columns=model_features)
        for k in enc:
            if k in df.columns: df[k] = enc[k]

        prob = model.predict_proba(df)[0][1]
        risk_pct = prob * 100
        
        # ปรับเปลี่ยนโครงสร้างการวัดผลให้เหลือ 2 ระดับ (เสี่ยงต่ำ / เสี่ยงสูง) ตามเกณฑ์ Optimal Threshold ของโมเดล
        if prob >= OPTIMAL_THRESHOLD:
            risk_text = "เสี่ยงสูง"
            badge_cls = "risk-high"
            wc, wt, wi = "warning-card", "warning-text", '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#EF4444" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>'
            wtitle = "ผลการประเมิน: กลุ่มเสี่ยงสูง"[cite: 1]
            wbody = "จากข้อมูลที่กรอก ระบบประเมินว่าผู้ป่วยรายนี้มีโอกาสเกิดผลการรักษาไม่สำเร็จสูงกว่าเกณฑ์ที่กำหนด ควรได้รับการติดตามและประเมินโดยบุคลากรทางการแพทย์อย่างต่อเนื่อง"[cite: 1]
        else:
            risk_text = "เสี่ยงต่ำ"
            badge_cls = "risk-low"
            wc, wt, wi = "warning-card-low", "warning-text-low", '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#22C55E" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>'
            wtitle = "ผลการประเมิน: กลุ่มเสี่ยงต่ำ"[cite: 1]
            wbody = "จากข้อมูลที่กรอก ระบบประเมินว่าผู้ป่วยรายนี้มีโอกาสเกิดผลการรักษาไม่สำเร็จต่ำกว่าเกณฑ์ที่กำหนด อย่างไรก็ตาม ควรปฏิบัติตามแผนการรักษาและคำแนะนำของแพทย์อย่างต่อเนื่อง"[cite: 1]

        st.markdown('<div class="result-section-bg">', unsafe_allow_html=True)
        st.markdown("""
        <div class="result-header">
            <div class="result-icon-circle"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg></div>
            <span class="result-title">ผลการวิเคราะห์</span>
        </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="{wc}">
            <span class="warning-icon">{wi}</span>
            <div class="{wt}"><b>{wtitle}</b>{wbody}</div>
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="gauge-card">', unsafe_allow_html=True)
        
        # ปรับแต่งเกจวัดค่าความน่าจะเป็นให้สอดรับกับการแบ่งโซนเสี่ยงต่ำ-สูง (0-65% และ 65-100%)
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_pct,
            number={'suffix': "%", 'font': {'size': 56, 'family': 'Sarabun', 'color': '#1D4ED8'}, 'valueformat': '.2f'},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': '#CBD5E1',
                         'tickfont': {'family': 'Sarabun', 'color': '#94A3B8'}},
                'bar':  {'color': "rgba(0,0,0,0)", 'thickness': 0},
                'bgcolor': "rgba(0,0,0,0)", 'borderwidth': 0,
                'steps': [
                    {'range': [0, 65],  'color': '#86EFAC'},
                    {'range': [65, 100], 'color': '#FCA5A5'}
                ],
                'threshold': {'line': {'color': "#1D4ED8", 'width': 5}, 'thickness': 0.85, 'value': risk_pct}
            }
        ))
        fig.update_layout(
            height=260, margin=dict(l=40, r=40, t=30, b=0),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font={'family': 'Sarabun'}
        )
        col_g1, col_g2, col_g3 = st.columns([1, 2, 1])
        with col_g2:
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            st.markdown(f"""
                <div style="text-align:center; margin-top:-14px; margin-bottom:8px;">
                    <span style="font-size:0.93rem; color:#64748B; font-weight:500;">ระดับความเสี่ยง: </span>
                    <span class="risk-badge {badge_cls}">{risk_text}</span>
                </div>
                <div class="legend-row">
                    <span><span class="legend-dot" style="background:#22C55E;"></span>เสี่ยงต่ำ (ต่ำกว่า 65%)</span>
                    <span><span class="legend-dot" style="background:#EF4444;"></span>เสี่ยงสูง (ตั้งแต่ 65% ขึ้นไป)</span>
                </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="note-card" style="margin-top:4px;">
            <span class="warning-icon"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#3B82F6" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg></span>
            <div class="note-text"><b>คำอธิบายเพิ่มเติม:</b> * ค่านี้แสดงความน่าจะเป็นที่โมเดลประเมินว่าผลการรักษาอาจไม่สำเร็จ โดยใช้ข้อมูลทางคลินิกของผู้ป่วยเป็นปัจจัยในการคำนวณ</div>
        </div>
        </div>""", unsafe_allow_html=True)[cite: 1]

    else:
        st.error("ไม่พบไฟล์โมเดล กรุณาตรวจสอบว่ามีไฟล์ `xgb_tb_model.pkl` และ `model_features.pkl` ในโฟลเดอร์เดียวกัน")
