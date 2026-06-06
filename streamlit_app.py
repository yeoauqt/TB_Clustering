import streamlit as st
import pandas as pd
import xgboost as xgb
import joblib
import numpy as np
import plotly.graph_objects as go

st.set_page_config(
    page_title="ระบบพยากรณ์ความเสี่ยงวัณโรค TB",
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
    font-size: 1.6rem;
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
    padding: 0 64px 28px !important;
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
    width: 36px; height: 36px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem; flex-shrink: 0;
    background: linear-gradient(135deg, var(--blue-main), var(--blue-mid));
    box-shadow: 0 2px 10px var(--blue-glow);
}
.card-title-text {
    font-size: 1.02rem; font-weight: 700; color: var(--blue-main);
}

/* ===== LABELS ===== */
.stSelectbox > label, .stNumberInput > label {
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
    font-size: 1.2rem; box-shadow: 0 4px 14px var(--blue-glow);
}
.result-title { font-size: 1.4rem; font-weight: 700; color: var(--blue-main); }

/* ===== WARNING CARDS ===== */
.warning-card {
    background: #FFF5F5; border-left: 4px solid #EF4444;
    border-radius: 14px; padding: 18px 20px; margin-bottom: 14px;
    display: flex; gap: 12px; align-items: flex-start;
}
.warning-card-medium {
    background: #FFFBEA; border-left: 4px solid #F59E0B;
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
.warning-text-medium { font-size: 0.9rem; color: #78350F; line-height: 1.7; }
.warning-text-medium b { display: block; margin-bottom: 4px; font-size: 0.95rem; color: #92400E; }
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
.risk-medium { background: #FEF9C3; color: #854D0E; }
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
    div[data-testid="stForm"] { padding: 0 32px 24px !important; }
    .result-section-bg { padding: 8px 32px 40px !important; }
}
@media (max-width: 768px) {
    .page-header { padding: 16px 16px !important; gap: 12px !important; }
    .header-text-title { font-size: 1.05rem !important; }
    div[data-testid="stForm"] { padding: 0 12px 20px !important; }
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

with st.form("main_form"):
    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        st.markdown('<div class="card-header"><div class="card-icon-circle">📍</div><span class="card-title-text">ข้อมูลพื้นฐาน</span></div>', unsafe_allow_html=True)
        age = st.number_input("อายุ (ปี)", 0, 120, 45, key="age")
        gen = st.selectbox("เพศ", ["ชาย", "หญิง"], key="gen")
        bmi = st.number_input("ดัชนีมวลกาย (BMI)", 10.0, 50.0, 20.0, key="bmi")
        dur = st.number_input("ระยะเวลาการรักษา (เดือน)", 0, 1000, 180, key="dur")

    with col2:
        st.markdown('<div class="card-header"><div class="card-icon-circle">💙</div><span class="card-title-text">ประวัติสุขภาพ</span></div>', unsafe_allow_html=True)
        hiv  = st.selectbox("สถานะ HIV",                      ["ไม่ติดเชื้อ (Negative)", "ติดเชื้อ (Positive)", "ไม่ทราบ"])
        ckd  = st.selectbox("โรคไตเรื้อรัง (CKD)",            ["ไม่เป็น", "เป็น"])
        copd = st.selectbox("โรคปอดอุดกั้นเรื้อรัง (COPD)",  ["ไม่เป็น", "เป็น"])
        liv  = st.selectbox("โรคตับ (Liver Disease)",         ["ไม่เป็น", "เป็น"])
        dm   = st.selectbox("โรคเบาหวาน (Diabetes Mellitus)", ["ไม่เป็น", "เป็น"])

    with col3:
        st.markdown('<div class="card-header"><div class="card-icon-circle">🧪</div><span class="card-title-text">ผลการตรวจ</span></div>', unsafe_allow_html=True)
        afb = st.selectbox("AFB เดือนที่ 1",      ["Negative", "1+", "2+", "3+", "Scanty"])
        pos = st.selectbox("ตำแหน่งการติดเชื้อ",  ["ในปอด", "นอกปอด", "ในและนอกปอด"])
        f_u = st.number_input("จำนวนครั้งติดตาม", 0, 50, 1)
        arv = st.selectbox("สถานะ ARV",            ["ไม่ได้รับ", "ได้รับ"])

    st.markdown("<br>", unsafe_allow_html=True)
    _, btn_mid, _ = st.columns([3, 2, 3])
    with btn_mid:
        sub = st.form_submit_button("🔍 วิเคราะห์ผลการรักษา", use_container_width=True)

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
                if k in d: d[k] = v.get(d[k], 0)
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
            if k in df.columns: df[k] = enc[k]

        prob      = model.predict_proba(df)[0][1]
        risk_pct  = prob * 100
        risk_text = "สูง" if prob > 0.6 else "ปานกลาง" if prob > 0.3 else "ต่ำ"
        badge_cls = "risk-high" if prob > 0.6 else "risk-medium" if prob > 0.3 else "risk-low"

        st.markdown('<div class="result-section-bg">', unsafe_allow_html=True)
        st.markdown("""
        <div class="result-header">
            <div class="result-icon-circle">📊</div>
            <span class="result-title">ผลการวิเคราะห์</span>
        </div>""", unsafe_allow_html=True)

        if prob > 0.6:
            wc, wt, wi = "warning-card", "warning-text", "⚠️"
            wtitle = "คำเตือน: ผู้ป่วยอยู่ในกลุ่มเสี่ยงต่อผลการรักษาที่ไม่สำเร็จ"
            wbody  = ("ผู้ป่วยที่มีดัชนีมวลกายต่ำ (BMI &lt; 18.5) อายุมาก และมีโรคประจำตัวหลายโรค "
                      "มีโอกาสสูงที่จะเกิดผลการรักษาที่ไม่สำเร็จ ควรได้รับการติดตามอย่างใกล้ชิด "
                      "และพิจารณาแผนการรักษาเสริมเพิ่มเติม")
            nb = ("ผลลัพธ์นี้เป็นเพียงการพยากรณ์เบื้องต้นจากระบบปัญญาประดิษฐ์ ไม่สามารถใช้แทนการวินิจฉัยทางคลินิกได้ "
                  "กรุณาปรึกษาแพทย์เพื่อรับคำแนะนำที่เหมาะสมกับสภาวะของผู้ป่วย")
        elif prob > 0.3:
            wc, wt, wi = "warning-card-medium", "warning-text-medium", "⚠️"
            wtitle = "ควรติดตามอาการ: ผู้ป่วยมีปัจจัยเสี่ยงบางประการ"
            wbody  = ("พบปัจจัยเสี่ยงบางส่วนที่อาจส่งผลต่อผลการรักษา ควรนัดติดตามผู้ป่วยอย่างสม่ำเสมอ "
                      "และให้ความรู้เกี่ยวกับการรับประทานยาอย่างต่อเนื่องเพื่อป้องกันการดื้อยา")
            nb = ("ผลลัพธ์นี้เป็นเพียงการพยากรณ์เบื้องต้นจากระบบปัญญาประดิษฐ์ ไม่สามารถใช้แทนการวินิจฉัยทางคลินิกได้ "
                  "กรุณาใช้ข้อมูลนี้ประกอบการพิจารณาของแพทย์ผู้ดูแล")
        else:
            wc, wt, wi = "warning-card-low", "warning-text-low", "✅"
            wtitle = "แนวโน้มดี: ความเสี่ยงต่ำ"
            wbody  = ("ผู้ป่วยรายนี้มีแนวโน้มการรักษาที่ดี หากรับประทานยาครบตามกำหนดและดูแลสุขภาพอย่างเหมาะสม "
                      "โอกาสหายขาดจากวัณโรคอยู่ในระดับสูง ควรมาตรวจตามนัดทุกครั้ง")
            nb = ("แม้ความเสี่ยงจะอยู่ในระดับต่ำ ผลลัพธ์นี้ยังคงเป็นเพียงการพยากรณ์เบื้องต้นจากระบบ AI "
                  "ไม่ควรใช้เป็นเหตุผลในการหยุดหรือปรับเปลี่ยนการรักษาด้วยตนเอง")

        st.markdown(f"""
        <div class="{wc}">
            <span class="warning-icon">{wi}</span>
            <div class="{wt}"><b>{wtitle}</b>{wbody}</div>
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="gauge-card">', unsafe_allow_html=True)
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
                    {'range': [0,   30],  'color': '#86EFAC'},
                    {'range': [30,  60],  'color': '#FDE68A'},
                    {'range': [60,  100], 'color': '#FCA5A5'}
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
                    <span><span class="legend-dot" style="background:#22C55E;"></span>ต่ำ (0–30%)</span>
                    <span><span class="legend-dot" style="background:#F59E0B;"></span>ปานกลาง (30–60%)</span>
                    <span><span class="legend-dot" style="background:#EF4444;"></span>สูง (60–100%)</span>
                </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="note-card" style="margin-top:4px;">
            <span class="warning-icon">ℹ️</span>
            <div class="note-text"><b>หมายเหตุ:</b>{nb}</div>
        </div>
        </div>""", unsafe_allow_html=True)

    else:
        st.error("⚠️ ไม่พบไฟล์โมเดล กรุณาตรวจสอบว่ามีไฟล์ `xgb_tb_model.pkl` และ `model_features.pkl` ในโฟลเดอร์เดียวกัน")
