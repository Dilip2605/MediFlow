import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import hashlib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ─── Internal imports ────────────────────────────────────────
from database.db import get_db
from utils.helpers import (
    get_risk_color, get_risk_level, get_risk_emoji,
    get_today, validate_phone
)
from services.diabetes_service import (
    predict_diabetes, get_diabetes_recommendations, load_diabetes_models
)
from services.heart_service import (
    predict_heart_disease, get_heart_recommendations, load_heart_models
)
from services.nlp_service import (
    analyze_symptoms, get_disease_info,
    get_supported_diseases, load_nlp_models
)
from services.xray_service import (
    analyze_xray, get_xray_recommendations,
    get_cnn_model_status, load_cnn_model
)
from utils.pdf_generator import generate_medical_report, generate_inventory_report

# ═══════════════════════════════════════════════════════════════
# PAGE CONFIG — Must be FIRST Streamlit command
# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="MediFlow AI v2.0",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════════
# PROFESSIONAL CSS
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
* { font-family: 'Plus Jakarta Sans', sans-serif !important; }
.stApp { background: linear-gradient(135deg, #0a0f1e 0%, #0d1a2e 50%, #071422 100%); color: #e2e8f0; }
section[data-testid="stSidebar"] { background: linear-gradient(180deg, #0d1a2e 0%, #071422 100%) !important; border-right: 1px solid rgba(0,200,255,0.15); }
.main-title { font-size: 3rem; font-weight: 800; background: linear-gradient(135deg, #00c8ff 0%, #0080ff 50%, #00ffb3 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; text-align: center; margin-bottom: 0.5rem; }
.metric-card { background: linear-gradient(135deg, rgba(13,26,46,0.8), rgba(7,20,34,0.9)); border: 1px solid rgba(0,200,255,0.2); border-radius: 16px; padding: 1.5rem; text-align: center; transition: all 0.3s ease; }
.metric-card:hover { border-color: rgba(0,200,255,0.5); transform: translateY(-2px); box-shadow: 0 8px 32px rgba(0,200,255,0.1); }
.metric-value { font-size: 2.5rem; font-weight: 800; background: linear-gradient(135deg, #00c8ff, #00ffb3); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
.metric-label { color: #64748b; font-size: 0.85rem; font-weight: 500; text-transform: uppercase; letter-spacing: 1px; }
.metric-icon { font-size: 2rem; margin-bottom: 0.5rem; }
.section-header { font-size: 1.5rem; font-weight: 700; color: #e2e8f0; margin-bottom: 1rem; padding-bottom: 0.5rem; border-bottom: 2px solid rgba(0,200,255,0.3); }
.result-positive { background: linear-gradient(135deg, rgba(220,38,38,0.15), rgba(153,27,27,0.1)); border: 1px solid rgba(220,38,38,0.4); border-radius: 16px; padding: 1.5rem; text-align: center; }
.result-negative { background: linear-gradient(135deg, rgba(5,150,105,0.15), rgba(4,120,87,0.1)); border: 1px solid rgba(5,150,105,0.4); border-radius: 16px; padding: 1.5rem; text-align: center; }
.result-title { font-size: 1.8rem; font-weight: 800; margin-bottom: 0.5rem; }
.text-red { color: #f87171; } .text-green { color: #34d399; }
.info-box { background: rgba(0,200,255,0.05); border: 1px solid rgba(0,200,255,0.2); border-left: 4px solid #00c8ff; border-radius: 8px; padding: 1rem 1.25rem; margin: 0.75rem 0; }
.warn-box { background: rgba(251,191,36,0.05); border: 1px solid rgba(251,191,36,0.2); border-left: 4px solid #fbbf24; border-radius: 8px; padding: 1rem 1.25rem; margin: 0.75rem 0; }
.danger-box { background: rgba(220,38,38,0.05); border: 1px solid rgba(220,38,38,0.2); border-left: 4px solid #dc2626; border-radius: 8px; padding: 1rem 1.25rem; margin: 0.75rem 0; }
.ok-box { background: rgba(5,150,105,0.05); border: 1px solid rgba(5,150,105,0.2); border-left: 4px solid #059669; border-radius: 8px; padding: 1rem 1.25rem; margin: 0.75rem 0; }
.stButton>button { background: linear-gradient(135deg,#0080ff,#00c8ff); color:white; border:none; border-radius:10px; padding:0.6rem 2rem; font-weight:600; font-size:1rem; transition:all 0.3s ease; width:100%; }
.stButton>button:hover { background: linear-gradient(135deg,#00c8ff,#0080ff); transform:translateY(-1px); box-shadow:0 4px 20px rgba(0,200,255,0.3); }
.stTextInput>div>div>input, .stNumberInput>div>div>input { background:rgba(13,26,46,0.8)!important; border:1px solid rgba(0,200,255,0.2)!important; border-radius:8px!important; color:#e2e8f0!important; }
.stTextArea>div>div>textarea { background:rgba(13,26,46,0.8)!important; border:1px solid rgba(0,200,255,0.2)!important; border-radius:8px!important; color:#e2e8f0!important; }
.stTabs [aria-selected="true"] { background:linear-gradient(135deg,#0080ff,#00c8ff)!important; color:white!important; }
.logo-text { font-size:1.8rem; font-weight:800; background:linear-gradient(135deg,#00c8ff,#00ffb3); -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; }
.custom-divider { border:none; height:1px; background:linear-gradient(90deg,transparent,rgba(0,200,255,0.3),transparent); margin:1.5rem 0; }
#MainMenu{visibility:hidden;} footer{visibility:hidden;} header{visibility:hidden;}
</style>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# DATABASE
# ═══════════════════════════════════════════════════════════════
conn = get_db()

# ═══════════════════════════════════════════════════════════════
# UI COMPONENT HELPERS
# ═══════════════════════════════════════════════════════════════
def mc(icon, value, label):
    """Render metric card"""
    st.markdown(f'<div class="metric-card"><div class="metric-icon">{icon}</div><div class="metric-value">{value}</div><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)

def cbar(prob, label):
    """Render confidence bar"""
    pct = int(prob * 100)
    c = get_risk_color(prob)
    st.markdown(f'<div style="margin:0.5rem 0;"><div style="display:flex;justify-content:space-between;margin-bottom:4px;"><span style="color:#94a3b8;font-size:0.85rem;">{label}</span><span style="color:{c};font-weight:700;font-size:0.85rem;">{pct}%</span></div><div style="background:rgba(255,255,255,0.1);border-radius:50px;height:12px;overflow:hidden;"><div style="width:{pct}%;height:100%;border-radius:50px;background:linear-gradient(90deg,#0080ff,{c});"></div></div></div>', unsafe_allow_html=True)

def info(text): st.markdown(f'<div class="info-box">{text}</div>', unsafe_allow_html=True)
def warn(text): st.markdown(f'<div class="warn-box">{text}</div>', unsafe_allow_html=True)
def danger(text): st.markdown(f'<div class="danger-box">{text}</div>', unsafe_allow_html=True)
def ok(text): st.markdown(f'<div class="ok-box">{text}</div>', unsafe_allow_html=True)
def divider(): st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

def dark_fig(w=6, h=4):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor('#0d1a2e')
    ax.set_facecolor('#0d1a2e')
    ax.tick_params(colors='#64748b')
    for s in ax.spines.values(): s.set_color('#1e3a5f')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.3, color='#1e3a5f', linestyle='--')
    return fig, ax

# ═══════════════════════════════════════════════════════════════
# AUTH HELPERS
# ═══════════════════════════════════════════════════════════════
def hash_pw(p): return hashlib.sha256(p.encode()).hexdigest()

def verify_login(username, password):
    cur = conn.cursor()
    cur.execute("SELECT password_hash, role FROM users WHERE username=?", (username,))
    r = cur.fetchone()
    return (True, r[1]) if r and r[0] == hash_pw(password) else (False, None)

# ═══════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════
for k, v in [('logged_in', False), ('username', ''), ('role', ''), ('show_reg', False)]:
    if k not in st.session_state:
        st.session_state[k] = v

# ═══════════════════════════════════════════════════════════════
# LOGIN PAGE
# ═══════════════════════════════════════════════════════════════
if not st.session_state.logged_in:
    st.markdown('<h1 class="main-title">🏥 MediFlow AI</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align:center;color:#64748b;font-size:1.1rem;">AI-Powered Hospital Management System v2.0</p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        st.markdown("""
        <div style="background:linear-gradient(135deg,rgba(13,26,46,0.9),rgba(7,20,34,0.95));
                    border:1px solid rgba(0,200,255,0.3);border-radius:20px;padding:2rem;
                    box-shadow:0 20px 60px rgba(0,0,0,0.5);">
            <h2 style="text-align:center;color:#00c8ff;margin-bottom:0.5rem;">Welcome Back</h2>
            <p style="text-align:center;color:#64748b;margin-bottom:1.5rem;">Sign in to MediFlow</p>
        </div>""", unsafe_allow_html=True)

        username = st.text_input("👤 Username", placeholder="Enter username")
        password = st.text_input("🔒 Password", type="password", placeholder="Enter password")

        c1, c2 = st.columns(2)
        with c1: login_btn = st.button("🔑 Login", use_container_width=True)
        with c2:
            if st.button("📝 Register", use_container_width=True):
                st.session_state.show_reg = True

        if login_btn:
            if username and password:
                ok_login, role = verify_login(username, password)
                if ok_login:
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.session_state.role = role
                    st.toast(f"Welcome {username}! 🎉", icon="✅")
                    st.rerun()
                else:
                    danger("❌ Invalid username or password!")
            else:
                warn("⚠️ Please enter username and password!")

        info("🔑 Default login: <b>admin</b> / <b>admin123</b>")

        if st.session_state.show_reg:
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown("**Create New Account**")
            nu = st.text_input("New Username")
            np_ = st.text_input("New Password", type="password")
            nr = st.selectbox("Role", ["doctor", "nurse", "admin"])
            if st.button("✅ Create Account"):
                if nu and np_:
                    try:
                        conn.execute(
                            "INSERT INTO users (username,password_hash,role,created_at) VALUES (?,?,?,?)",
                            (nu, hash_pw(np_), nr, get_today()))
                        conn.commit()
                        ok(f"✅ Account created: {nu}")
                        st.session_state.show_reg = False
                    except Exception:
                        danger("❌ Username already exists!")
                else:
                    warn("⚠️ Fill all fields!")
    st.stop()

# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center;padding:1.5rem 0;border-bottom:1px solid rgba(0,200,255,0.1);margin-bottom:1rem;">
        <div class="logo-text">🏥 MediFlow</div>
        <div style="font-size:0.75rem;color:#475569;letter-spacing:2px;text-transform:uppercase;">AI Hospital v2.0</div>
        <div style="margin-top:0.5rem;color:#00c8ff;font-size:0.85rem;">
            👤 {st.session_state.username} ({st.session_state.role})
        </div>
    </div>""", unsafe_allow_html=True)

    menu = st.selectbox("Navigation", [
        "🏠 Home Dashboard",
        "👥 Patient Management",
        "📋 Patient History",
        "🩸 Diabetes Prediction",
        "❤️ Heart Disease",
        "📝 Symptom Analyzer",
        "🫁 X-ray Analysis",
        "📊 Analytics",
        "💊 Inventory",
        "⚙️ Settings"
    ], label_visibility="collapsed")

    st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

    # Model status
    st.markdown("**MODEL STATUS**")
    _, _, d_ok = load_diabetes_models()
    _, _, _, h_ok = load_heart_models()
    _, _, n_ok = load_nlp_models()
    cnn_status = get_cnn_model_status()

    for name, loaded in [("🩸 Diabetes", d_ok), ("❤️ Heart", h_ok), ("📝 NLP", n_ok), ("🫁 CNN", cnn_status['loaded'])]:
        c = "#34d399" if loaded else "#f87171"
        st.markdown(f'<div style="display:flex;justify-content:space-between;padding:0.3rem 0;font-size:0.82rem;"><span style="color:#94a3b8;">{name}</span><span style="color:{c};font-weight:700;">{"✅ Ready" if loaded else "❌ Not loaded"}</span></div>', unsafe_allow_html=True)

    st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)
    if st.button("🚪 Logout", use_container_width=True):
        for k in ['logged_in', 'username', 'role']:
            st.session_state[k] = '' if k != 'logged_in' else False
        st.rerun()

    st.markdown(f'<div style="text-align:center;color:#475569;font-size:0.75rem;margin-top:1rem;">🕐 {datetime.now().strftime("%d %b %Y | %H:%M")}</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# PAGE: HOME DASHBOARD
# ═══════════════════════════════════════════════════════════════
if menu == "🏠 Home Dashboard":
    st.markdown('<h1 class="main-title">MediFlow AI v2.0</h1>', unsafe_allow_html=True)
    st.markdown(f'<p style="text-align:center;color:#64748b;">Welcome, <b style="color:#00c8ff">{st.session_state.username}</b> | {datetime.now().strftime("%A, %d %B %Y")}</p>', unsafe_allow_html=True)

    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM patients"); tp = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM xray_results"); tx = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM symptoms"); ts = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM inventory WHERE quantity < 50"); ls = cur.fetchone()[0]

    c1, c2, c3, c4 = st.columns(4)
    with c1: mc("👥", tp, "Total Patients")
    with c2: mc("🫁", tx, "X-ray Records")
    with c3: mc("📝", ts, "Symptoms Records")
    with c4: mc("⚠️", ls, "Low Stock Items")

    if ls > 0:
        st.toast(f"⚠️ {ls} medicines below minimum stock!", icon="⚠️")
        warn(f"⚠️ <b>{ls} medicines</b> running low. Check Inventory page.")

    divider()

    col1, col2 = st.columns([1.5, 1])
    with col1:
        st.markdown('<div class="section-header">📊 Disease Distribution</div>', unsafe_allow_html=True)
        cur.execute("SELECT disease, COUNT(*) FROM xray_results GROUP BY disease")
        dd = cur.fetchall()
        if dd:
            fig, ax = dark_fig(7, 4)
            colors = ['#00c8ff', '#f87171', '#fbbf24', '#34d399', '#a78bfa']
            bars = ax.bar([r[0] for r in dd], [r[1] for r in dd],
                         color=colors[:len(dd)], width=0.6, edgecolor='none')
            ax.set_title('X-ray Disease Records', color='#94a3b8', fontsize=11)
            ax.grid(axis='y', alpha=0.4, color='#1e3a5f', linestyle='--')
            for bar, count in zip(bars, [r[1] for r in dd]):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                       str(count), ha='center', color='#94a3b8', fontweight='bold')
            plt.tight_layout(); st.pyplot(fig); plt.close()
        else:
            info("No disease records yet. Run database/init_db.py and add patients.")

    with col2:
        st.markdown('<div class="section-header">⚡ AI Models</div>', unsafe_allow_html=True)
        models_info = [
            ("🩸", "Diabetes", "XGBoost", "~74%", d_ok),
            ("❤️", "Heart Disease", "XGBoost", "~84%", h_ok),
            ("📝", "Symptom NLP", "TF-IDF+LR", "~90%", n_ok),
            ("🫁", "X-ray CNN", "DenseNet121", "~90%", cnn_status['loaded']),
        ]
        for icon, name, algo, acc, loaded in models_info:
            color = "#34d399" if loaded else "#f87171"
            st.markdown(f"""
            <div style="background:rgba(13,26,46,0.6);border:1px solid rgba(0,200,255,0.15);
                        border-radius:12px;padding:0.75rem 1rem;margin-bottom:0.5rem;
                        display:flex;align-items:center;gap:0.75rem;">
                <span style="font-size:1.4rem;">{icon}</span>
                <div style="flex:1;">
                    <div style="color:#e2e8f0;font-weight:600;font-size:0.9rem;">{name}</div>
                    <div style="color:#64748b;font-size:0.75rem;">{algo} | CV: {acc}</div>
                </div>
                <div style="width:10px;height:10px;border-radius:50%;
                            background:{color};box-shadow:0 0 8px {color};"></div>
            </div>""", unsafe_allow_html=True)

    divider()
    st.markdown('<div class="section-header">👥 Recent Patients</div>', unsafe_allow_html=True)
    cur.execute("SELECT * FROM patients ORDER BY id DESC LIMIT 5")
    recent = cur.fetchall()
    if recent:
        df_r = pd.DataFrame(recent, columns=["ID", "Name", "Age", "Gender", "Phone"])
        st.dataframe(df_r, use_container_width=True, hide_index=True)
    else:
        info("No patients yet. Add patients in Patient Management.")

# ═══════════════════════════════════════════════════════════════
# PAGE: PATIENT MANAGEMENT
# ═══════════════════════════════════════════════════════════════
elif menu == "👥 Patient Management":
    st.markdown('<h2 class="section-header">👥 Patient Management</h2>', unsafe_allow_html=True)
    cur = conn.cursor()

    tab1, tab2, tab3, tab4 = st.tabs(["➕ Add Patient", "📋 View All", "🔍 Search", "🗑️ Delete"])

    with tab1:
        st.markdown("### Register New Patient")
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("👤 Full Name", placeholder="Patient full name")
            age = st.number_input("🎂 Age", 1, 120, 30)
        with col2:
            gender = st.selectbox("⚧ Gender", ["Male", "Female", "Other"])
            phone = st.text_input("📱 Phone", placeholder="10-digit number")

        if st.button("✅ Register Patient", use_container_width=True):
            if name and phone:
                if not validate_phone(phone):
                    warn("⚠️ Enter valid 10-digit phone number!")
                else:
                    cur.execute(
                        "INSERT INTO patients (name,age,gender,phone) VALUES (?,?,?,?)",
                        (name, age, gender, phone))
                    conn.commit()
                    new_id = cur.lastrowid
                    st.toast(f"✅ {name} registered!", icon="✅")
                    ok(f"✅ Patient <b>{name}</b> registered! Patient ID: <b>#{new_id}</b>")
                    st.balloons()
            else:
                warn("⚠️ Please fill Name and Phone!")

    with tab2:
        cur.execute("SELECT COUNT(*) FROM patients")
        count = cur.fetchone()[0]
        info(f"📊 Total registered patients: <b>{count}</b>")

        cur.execute("SELECT * FROM patients ORDER BY id DESC")
        pts = cur.fetchall()
        if pts:
            df_p = pd.DataFrame(pts, columns=["ID", "Name", "Age", "Gender", "Phone"])
            st.download_button("📥 Export to CSV", df_p.to_csv(index=False),
                             "patients.csv", "text/csv", use_container_width=True)
            st.dataframe(df_p, use_container_width=True, hide_index=True)

            col1, col2 = st.columns(2)
            with col1:
                fig, ax = dark_fig()
                ax.hist(df_p['Age'], bins=8, color='#00c8ff', edgecolor='#0d1a2e', alpha=0.85)
                ax.set_title('Age Distribution', color='#94a3b8')
                plt.tight_layout(); st.pyplot(fig); plt.close()
            with col2:
                gc = df_p['Gender'].value_counts()
                fig, ax = dark_fig()
                ax.pie(gc.values, labels=gc.index,
                      colors=['#00c8ff','#f87171','#fbbf24'][:len(gc)],
                      autopct='%1.1f%%', textprops={'color':'#94a3b8'})
                ax.set_title('Gender Distribution', color='#94a3b8')
                st.pyplot(fig); plt.close()

    with tab3:
        search = st.text_input("🔍 Search by name or phone")
        if search:
            cur.execute(
                "SELECT * FROM patients WHERE name LIKE ? OR phone LIKE ?",
                (f"%{search}%", f"%{search}%"))
            results = cur.fetchall()
            if results:
                ok(f"Found <b>{len(results)}</b> patient(s)")
                df_s = pd.DataFrame(results, columns=["ID","Name","Age","Gender","Phone"])
                st.dataframe(df_s, use_container_width=True, hide_index=True)
            else:
                warn("No patients found with that name or phone.")

    with tab4:
        if st.session_state.role == "admin":
            del_id = st.number_input("Patient ID to delete", min_value=1, value=1)
            cur.execute("SELECT name FROM patients WHERE id=?", (del_id,))
            found = cur.fetchone()
            if found:
                warn(f"⚠️ This will permanently delete patient: <b>{found[0]}</b>")
                if st.button("🗑️ Confirm Delete", use_container_width=True):
                    cur.execute("DELETE FROM patients WHERE id=?", (del_id,))
                    conn.commit()
                    st.toast("Patient deleted", icon="🗑️")
                    ok("✅ Patient deleted successfully.")
            else:
                info("No patient found with that ID.")
        else:
            danger("❌ Admin access required for patient deletion.")
            
            
            
elif menu == "📅 Appointments":
    st.markdown(
        '<h2 class="section-header">📅 Appointments</h2>',
        unsafe_allow_html=True)
    cur = conn.cursor()

    tab1, tab2, tab3 = st.tabs([
        "📋 Today's Schedule",
        "➕ Book Appointment",
        "📊 All Appointments"
    ])

    with tab1:
        today = get_today()
        cur.execute("""
            SELECT a.id, p.name, p.age, a.doctor_name,
                   a.department, a.time_slot, a.status
            FROM appointments a
            JOIN patients p ON a.patient_id = p.patient_id
            WHERE a.date = ?
            ORDER BY a.time_slot
        """, (today,))
        todays = cur.fetchall()

        st.markdown(
            f"### 📅 {datetime.now().strftime('%d %B %Y')}")

        if todays:
            for apt in todays:
                status_color = (
                    "#34d399" if apt[6] == "Completed"
                    else "#fbbf24" if apt[6] == "Scheduled"
                    else "#f87171"
                )
                st.markdown(f"""
                <div style="background:rgba(13,26,46,0.6);
                            border:1px solid rgba(0,200,255,0.1);
                            border-radius:12px;
                            padding:1rem;
                            margin-bottom:0.5rem;">
                    <div style="display:flex;
                                justify-content:space-between;">
                        <div>
                            <b style="color:#e2e8f0;">
                                🕐 {apt[5]} — {apt[1]}
                            </b>
                            <div style="color:#64748b;
                                        font-size:0.85rem;">
                                Dr. {apt[3]} | {apt[4]}
                                | Age: {apt[2]}
                            </div>
                        </div>
                        <span style="color:{status_color};
                                     font-weight:700;
                                     font-size:0.85rem;">
                            {apt[6]}
                        </span>
                    </div>
                </div>""", unsafe_allow_html=True)
        else:
            st.info("No appointments today.")

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            cur.execute(
                "SELECT id, name FROM patients ORDER BY name")
            pts = cur.fetchall()
            pt_opts = {
                f"#{p[0]} — {p[1]}": p[0] for p in pts
            }
            sel = st.selectbox(
                "👤 Patient", list(pt_opts.keys()))
            pid = pt_opts[sel]

            doctor = st.selectbox("👨‍⚕️ Doctor", [
                "Dr. Kumar — Cardiology",
                "Dr. Priya — Neurology",
                "Dr. Rajan — Pulmonology",
                "Dr. Meena — Endocrinology",
                "Dr. Suresh — General Medicine"
            ])
            dept = doctor.split("—")[1].strip()

        with col2:
            apt_date = st.date_input("📅 Date")
            time_slot = st.selectbox("🕐 Time Slot", [
                "09:00 AM", "09:30 AM",
                "10:00 AM", "10:30 AM",
                "11:00 AM", "11:30 AM",
                "02:00 PM", "02:30 PM",
                "03:00 PM", "03:30 PM",
                "04:00 PM", "04:30 PM"
            ])
            notes = st.text_area("📝 Notes", height=80)

        if st.button(
                "✅ Book Appointment",
                use_container_width=True):
            cur.execute("""
                INSERT INTO appointments
                    (patient_id, doctor_name, department,
                     date, time_slot, status, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (pid, doctor.split("—")[0].strip(),
                  dept, str(apt_date), time_slot,
                  "Scheduled", notes))
            conn.commit()
            st.toast("✅ Appointment booked!", icon="📅")
            ok(f"✅ Appointment booked for "
               f"<b>{sel}</b> on "
               f"<b>{apt_date}</b> at <b>{time_slot}</b>")

# ═══════════════════════════════════════════════════════════════
# PAGE: PATIENT HISTORY
# ═══════════════════════════════════════════════════════════════
elif menu == "📋 Patient History":
    st.markdown('<h2 class="section-header">📋 Complete Patient History</h2>', unsafe_allow_html=True)
    cur = conn.cursor()

    info("🔍 Complete medical history using SQL JOIN queries — X-rays + symptoms linked by patient ID.")

    cur.execute("SELECT id, name FROM patients ORDER BY name")
    pts = cur.fetchall()

    if pts:
        pt_opts = {f"#{p[0]} — {p[1]}": p[0] for p in pts}
        selected = st.selectbox("Select Patient", list(pt_opts.keys()))
        pid = pt_opts[selected]

        cur.execute("SELECT * FROM patients WHERE id=?", (pid,))
        patient = cur.fetchone()

        if patient:
            c1, c2, c3 = st.columns(3)
            with c1: mc("👤", patient[1][:12], "Name")
            with c2: mc("🎂", patient[2], "Age")
            with c3: mc("⚧", patient[3], "Gender")

            divider()
            tab1, tab2, tab3 = st.tabs(["🫁 X-ray Records", "📝 Symptom Records", "📊 Full Summary"])

            with tab1:
                cur.execute("""
                    SELECT x.id, x.disease, x.confidence, x.date
                    FROM xray_results x
                    WHERE x.patient_id = ?
                    ORDER BY x.date DESC
                """, (pid,))
                xrays = cur.fetchall()
                if xrays:
                    ok(f"Found <b>{len(xrays)}</b> X-ray record(s)")
                    df_x = pd.DataFrame(xrays, columns=["ID", "Disease", "Confidence", "Date"])
                    st.dataframe(df_x, use_container_width=True, hide_index=True)

                    if len(xrays) > 1:
                        fig, ax = dark_fig(8, 3)
                        ax.bar(range(len(xrays)), [x[2] for x in xrays], color='#00c8ff', edgecolor='none')
                        ax.set_xticks(range(len(xrays)))
                        ax.set_xticklabels([x[3] for x in xrays], rotation=45, color='#64748b', fontsize=8)
                        ax.set_title('Confidence Over Time', color='#94a3b8', fontsize=10)
                        ax.set_ylabel('Confidence', color='#64748b')
                        plt.tight_layout(); st.pyplot(fig); plt.close()
                else:
                    info("No X-ray records for this patient.")

            with tab2:
                cur.execute("""
                    SELECT s.id, s.symptoms_text, s.diagnosis, s.date
                    FROM symptoms s
                    WHERE s.patient_id = ?
                    ORDER BY s.date DESC
                """, (pid,))
                syms = cur.fetchall()
                if syms:
                    ok(f"Found <b>{len(syms)}</b> symptom record(s)")
                    df_s = pd.DataFrame(syms, columns=["ID","Symptoms","Diagnosis","Date"])
                    st.dataframe(df_s, use_container_width=True, hide_index=True)
                else:
                    info("No symptom records for this patient.")

            with tab3:
                # Complex JOIN combining 3 tables
                cur.execute("""
                    SELECT p.name, p.age, p.gender, p.phone,
                           COUNT(DISTINCT x.id) as xray_count,
                           COUNT(DISTINCT s.id) as symptom_count
                    FROM patients p
                    LEFT JOIN xray_results x ON p.id = x.patient_id
                    LEFT JOIN symptoms s ON p.id = s.patient_id
                    WHERE p.id = ?
                    GROUP BY p.id
                """, (pid,))
                summary = cur.fetchone()
                if summary:
                    st.markdown(f"""
                    <div style="background:rgba(13,26,46,0.8);border:1px solid rgba(0,200,255,0.2);
                                border-radius:16px;padding:2rem;">
                        <h3 style="color:#00c8ff;">Patient Summary</h3>
                        <p style="color:#94a3b8;">Name: <b style="color:#e2e8f0;">{summary[0]}</b></p>
                        <p style="color:#94a3b8;">Age: <b style="color:#e2e8f0;">{summary[1]} years</b></p>
                        <p style="color:#94a3b8;">Gender: <b style="color:#e2e8f0;">{summary[2]}</b></p>
                        <p style="color:#94a3b8;">Phone: <b style="color:#e2e8f0;">{summary[3]}</b></p>
                        <p style="color:#94a3b8;">X-ray Records: <b style="color:#00c8ff;">{summary[4]}</b></p>
                        <p style="color:#94a3b8;">Symptom Records: <b style="color:#00c8ff;">{summary[5]}</b></p>
                    </div>""", unsafe_allow_html=True)
    else:
        info("No patients registered. Add patients first.")
        
        
elif menu == "💊 Prescriptions":
    st.markdown(
        '<h2 class="section-header">💊 Prescriptions</h2>',
        unsafe_allow_html=True)
    cur = conn.cursor()

    tab1, tab2 = st.tabs([
        "✍️ Write Prescription",
        "📋 View Prescriptions"
    ])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            cur.execute(
                "SELECT id, name FROM patients ORDER BY name")
            pts = cur.fetchall()
            pt_opts = {
                f"#{p[0]} — {p[1]}": p[0] for p in pts
            }
            sel = st.selectbox(
                "👤 Patient", list(pt_opts.keys()))
            pid = pt_opts[sel]

            doctor = st.text_input(
                "👨‍⚕️ Doctor Name",
                value=f"Dr. {st.session_state.username}")
            diagnosis = st.text_input(
                "🏥 Diagnosis",
                placeholder="e.g. Type 2 Diabetes")

        with col2:
            # Get medicines from inventory
            cur.execute(
                "SELECT medicine_name FROM inventory "
                "ORDER BY medicine_name")
            meds = [m[0] for m in cur.fetchall()]
            selected_meds = st.multiselect(
                "💊 Select Medicines", meds)

            dosage = st.text_area(
                "📋 Dosage Instructions",
                placeholder="Paracetamol: 1 tablet "
                            "twice daily after food\n"
                            "Amoxicillin: 1 capsule "
                            "3 times daily",
                height=100)
            duration = st.number_input(
                "📅 Duration (days)", 1, 365, 7)

        if st.button(
                "✅ Save Prescription",
                use_container_width=True):
            if pid and diagnosis and selected_meds:
                meds_str = ", ".join(selected_meds)
                cur.execute("""
                    INSERT INTO prescriptions
                        (patient_id, doctor_name,
                         diagnosis, medicines,
                         dosage, duration_days, date)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (pid, doctor, diagnosis,
                      meds_str, dosage,
                      duration, get_today()))
                conn.commit()

                # Reduce inventory automatically!
                for med in selected_meds:
                    cur.execute("""
                        UPDATE inventory
                        SET quantity = quantity - ?
                        WHERE medicine_name = ?
                    """, (duration, med))
                conn.commit()

                st.toast(
                    "✅ Prescription saved!", icon="💊")
                ok(f"✅ Prescription saved! "
                   f"Inventory updated for "
                   f"{len(selected_meds)} medicines.")
            else:
                warn("⚠️ Fill all required fields!")

# ═══════════════════════════════════════════════════════════════
# PAGE: DIABETES PREDICTION
# ═══════════════════════════════════════════════════════════════
elif menu == "🩸 Diabetes Prediction":
    st.markdown('<h2 class="section-header">🩸 Diabetes Prediction</h2>', unsafe_allow_html=True)
    cur = conn.cursor()

    if not d_ok:
        danger("❌ Diabetes model not loaded! Run: <code>python scripts/diabetes_model.py</code>")
        st.stop()

    info("🤖 <b>XGBoost</b> with Feature Engineering | CV Accuracy: <b>~74%</b> | Pima Indians Dataset")

    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.markdown("### 🧪 Blood Test Values")

        # Patient selector
        cur.execute("SELECT id, name FROM patients")
        pts = cur.fetchall()
        patient_name = "Unknown"; patient_age = 30; patient_gender = "N/A"; pid = None
        if pts:
            pt_opts = {"Manual Entry (no patient)": None}
            pt_opts.update({f"#{p[0]} — {p[1]}": p[0] for p in pts})
            sel = st.selectbox("🔗 Link to Patient (optional)", list(pt_opts.keys()))
            pid = pt_opts[sel]
            if pid:
                cur.execute("SELECT name, age, gender FROM patients WHERE id=?", (pid,))
                pd_ = cur.fetchone()
                if pd_: patient_name, patient_age, patient_gender = pd_[0], pd_[1], pd_[2]

        c1, c2 = st.columns(2)
        with c1:
            pregnancies = st.number_input("🤰 Pregnancies", 0, 20, 1)
            glucose = st.number_input("🍬 Glucose (mg/dL)", 0, 300, 120)
            blood_pressure = st.number_input("💉 Blood Pressure (mm Hg)", 0, 200, 70)
            skin_thickness = st.number_input("📏 Skin Thickness (mm)", 0, 100, 20)
        with c2:
            insulin = st.number_input("💉 Insulin (mu U/ml)", 0, 900, 80)
            bmi = st.number_input("⚖️ BMI", 0.0, 70.0, 25.0, 0.1)
            dpf = st.number_input("🧬 Diabetes Pedigree", 0.0, 3.0, 0.5, 0.01)
            age_inp = st.number_input("👴 Age", 1, 120, patient_age)

        predict_btn = st.button("🔬 Predict Diabetes Risk", use_container_width=True)

    with col2:
        st.markdown("### 📊 Reference Ranges")
        for rn, val, low, high, unit in [
            ("Glucose", glucose, 70, 99, "mg/dL"),
            ("Blood Pressure", blood_pressure, 60, 80, "mm Hg"),
            ("BMI", bmi, 18.5, 24.9, "kg/m²"),
            ("Insulin", insulin, 16, 166, "mu U/ml")
        ]:
            if val < low: status, color = "🔵 Low", "#60a5fa"
            elif val > high: status, color = "🔴 High", "#f87171"
            else: status, color = "🟢 Normal", "#34d399"
            st.markdown(f'<div style="background:rgba(13,26,46,0.6);border:1px solid rgba(0,200,255,0.1);border-radius:10px;padding:0.6rem 1rem;margin-bottom:0.5rem;"><div style="display:flex;justify-content:space-between;"><span style="color:#94a3b8;font-size:0.85rem;">{rn}</span><span style="color:{color};font-size:0.85rem;font-weight:600;">{status}</span></div><div style="color:#e2e8f0;font-weight:700;">{val} {unit}</div></div>', unsafe_allow_html=True)

    if predict_btn:
        result = predict_diabetes(
            pregnancies, glucose, blood_pressure, skin_thickness,
            insulin, bmi, dpf, age_inp
        )

        if result.get("error"):
            danger(f"❌ {result['error']}")
        else:
            divider()
            st.markdown("### 🎯 Prediction Results")

            c1, c2, c3 = st.columns(3)
            with c1:
                if result["prediction"] == 1:
                    st.markdown('<div class="result-positive"><div class="result-title text-red">⚠️ DIABETIC</div><div style="color:#94a3b8;">Diabetes Detected</div></div>', unsafe_allow_html=True)
                    st.toast("⚠️ Diabetes detected!", icon="⚠️")
                else:
                    st.markdown('<div class="result-negative"><div class="result-title text-green">✅ HEALTHY</div><div style="color:#94a3b8;">No Diabetes</div></div>', unsafe_allow_html=True)
                    st.toast("✅ Patient healthy!", icon="✅")

            with c2:
                rc = get_risk_color(result["probability"])
                st.markdown(f'<div style="background:rgba(13,26,46,0.8);border:1px solid {rc}40;border-radius:16px;padding:1.5rem;text-align:center;"><div style="color:#64748b;font-size:0.8rem;">Risk Level</div><div style="font-size:2rem;font-weight:800;color:{rc};">{int(result["probability"]*100)}%</div><div style="color:{rc};font-weight:600;">{result["risk_level"]}</div></div>', unsafe_allow_html=True)

            with c3:
                recs = get_diabetes_recommendations(result["prediction"], glucose, bmi)
                pdf_bytes = generate_medical_report(
                    patient_name, age_inp, patient_gender,
                    "Diabetes Screening", result["result_text"],
                    result["probability"], result["risk_level"],
                    result.get("features_used", {}), recs
                )
                if pdf_bytes:
                    st.download_button("📄 Download PDF Report", pdf_bytes,
                                     f"diabetes_{patient_name}.pdf",
                                     "application/pdf", use_container_width=True)
                else:
                    info("Install fpdf for PDF: <code>pip install fpdf</code>")

            cbar(result["probability"], "Diabetes Risk")
            cbar(result["probability_healthy"], "Healthy Probability")

            if result["prediction"] == 1:
                danger("<b>⚕️ Recommendation:</b> Consult endocrinologist. Monitor blood glucose daily. HbA1c test recommended.")
            else:
                ok("<b>✅ Health Note:</b> No diabetes detected. Annual check-up recommended.")

            # Save to patient record if linked
            if pid:
                cur.execute(
                    "INSERT INTO symptoms (patient_id,symptoms_text,diagnosis,date) VALUES (?,?,?,?)",
                    (pid, f"Diabetes screening: Glucose={glucose}, BMI={bmi}",
                     result["result_text"], get_today()))
                conn.commit()
                st.toast("Saved to patient record!", icon="💾")

# ═══════════════════════════════════════════════════════════════
# PAGE: HEART DISEASE
# ═══════════════════════════════════════════════════════════════
elif menu == "❤️ Heart Disease":
    st.markdown('<h2 class="section-header">❤️ Heart Disease Prediction</h2>', unsafe_allow_html=True)

    if not h_ok:
        danger("❌ Heart model not loaded! Run: <code>python scripts/heart_model.py</code>")
        st.stop()

    info("🤖 <b>XGBoost</b> | CV Accuracy: <b>~84%</b> | CV Recall: <b>~87%</b> | 302 unique patients")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🫀 Clinical Parameters")
        age = st.number_input("👴 Age", 20, 100, 55)
        sex = st.selectbox("⚧ Sex", ["Male (1)", "Female (0)"])
        sex_val = 1 if "Male" in sex else 0
        cp = st.selectbox("💔 Chest Pain", ["0 - Typical Angina", "1 - Atypical Angina", "2 - Non-Anginal", "3 - Asymptomatic"])
        cp_val = int(cp[0])
        trestbps = st.number_input("🩸 Resting BP (mm Hg)", 80, 250, 130)
        chol = st.number_input("🧪 Cholesterol (mg/dl)", 100, 600, 240)
        fbs = st.selectbox("🍬 Fasting Blood Sugar > 120", ["No (0)", "Yes (1)"])
        fbs_val = int(fbs[-2])

    with col2:
        st.markdown("### 📊 Test Results")
        restecg = st.selectbox("📈 Resting ECG", ["0 - Normal", "1 - ST-T Abnormality", "2 - LV Hypertrophy"])
        restecg_val = int(restecg[0])
        thalach = st.number_input("💓 Max Heart Rate", 60, 220, 150)
        exang = st.selectbox("🏃 Exercise Angina", ["No (0)", "Yes (1)"])
        exang_val = int(exang[-2])
        oldpeak = st.number_input("📉 ST Depression", 0.0, 7.0, 1.0, 0.1)
        slope = st.selectbox("📈 ST Slope", ["0 - Upsloping", "1 - Flat", "2 - Downsloping"])
        slope_val = int(slope[0])
        ca = st.number_input("🔵 Major Vessels (0-4)", 0, 4, 0)
        thal = st.selectbox("🧬 Thalassemia", ["0 - Normal", "1 - Fixed Defect", "2 - Reversible"])
        thal_val = int(thal[0])

    if st.button("❤️ Predict Heart Disease", use_container_width=True):
        result = predict_heart_disease(
            age, sex_val, cp_val, trestbps, chol, fbs_val,
            restecg_val, thalach, exang_val, oldpeak,
            slope_val, ca, thal_val
        )

        if result.get("error"):
            danger(f"❌ {result['error']}")
        else:
            divider()
            c1, c2, c3 = st.columns(3)
            with c1:
                if result["prediction"] == 1:
                    st.markdown('<div class="result-positive"><div class="result-title text-red">❤️‍🩹 DISEASE</div><div style="color:#94a3b8;">Heart Disease Detected</div></div>', unsafe_allow_html=True)
                    st.toast("🚨 Heart disease detected!", icon="🚨")
                else:
                    st.markdown('<div class="result-negative"><div class="result-title text-green">❤️ HEALTHY</div><div style="color:#94a3b8;">No Heart Disease</div></div>', unsafe_allow_html=True)
                    st.toast("✅ Heart healthy!", icon="✅")

            with c2:
                rc = get_risk_color(result["probability"])
                st.markdown(f'<div style="background:rgba(13,26,46,0.8);border:1px solid {rc}40;border-radius:16px;padding:1.5rem;text-align:center;"><div style="color:#64748b;font-size:0.8rem;">Risk</div><div style="font-size:2rem;font-weight:800;color:{rc};">{int(result["probability"]*100)}%</div><div style="color:{rc};font-weight:600;">{result["risk_level"]}</div></div>', unsafe_allow_html=True)

            with c3:
                recs = get_heart_recommendations(result["prediction"], cp_val, chol)
                pdf_bytes = generate_medical_report(
                    "Patient", age, sex, "Heart Disease Screening",
                    result["result_text"], result["probability"],
                    result["risk_level"], result.get("features_used", {}), recs
                )
                if pdf_bytes:
                    st.download_button("📄 PDF Report", pdf_bytes, "heart_report.pdf",
                                     "application/pdf", use_container_width=True)

            cbar(result["probability"], "Heart Disease Risk")

            if result.get("clinical_flags"):
                st.markdown("### 🚩 Clinical Flags")
                for flag, status in result["clinical_flags"].items():
                    st.markdown(f'<div style="padding:0.4rem 0;border-bottom:1px solid rgba(0,200,255,0.1);color:#94a3b8;font-size:0.85rem;"><b>{flag}:</b> {status}</div>', unsafe_allow_html=True)

            if result["prediction"] == 1:
                danger("<b>⚕️ Recommendation:</b> Urgent cardiology consultation. ECG and Echo recommended. Avoid strenuous activity.")



# Show BP trend over time for a patient
cur.execute("""
    SELECT recorded_at, pulse_rate, spo2
    FROM vitals
    WHERE patient_id = ?
    ORDER BY recorded_at
""", (patient_id,))
vitals_data = cur.fetchall()

if len(vitals_data) > 1:
    df_v = pd.DataFrame(
        vitals_data,
        columns=['Date', 'Pulse', 'SpO2'])

    fig, ax = dark_fig(8, 3)
    ax.plot(df_v['Date'], df_v['Pulse'],
            color='#f87171', marker='o',
            label='Pulse Rate', linewidth=2)
    ax.plot(df_v['Date'], df_v['SpO2'],
            color='#34d399', marker='s',
            label='SpO2 %', linewidth=2)
    ax.set_title(
        'Vitals Trend', color='#94a3b8')
    ax.legend(
        labelcolor='#94a3b8',
        facecolor='#0d1a2e',
        edgecolor='#1e3a5f')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    
    
# Monthly patient registration trend
cur.execute("""
    SELECT
        strftime('%Y-%m', rowid) as month,
        COUNT(*) as count
    FROM patients
    GROUP BY month
    ORDER BY month
""")
monthly = cur.fetchall()

if monthly:
    fig, ax = dark_fig(8, 4)
    months = [m[0] for m in monthly]
    counts = [m[1] for m in monthly]
    ax.plot(months, counts,
            color='#00c8ff',
            marker='o',
            linewidth=2,
            markersize=8)
    ax.fill_between(
        months, counts,
        color='#00c8ff',
        alpha=0.1)
    ax.set_title(
        'Monthly Patient Registrations',
        color='#94a3b8')
    plt.xticks(rotation=45, color='#64748b')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ═══════════════════════════════════════════════════════════════
# PAGE: SYMPTOM ANALYZER
# ═══════════════════════════════════════════════════════════════
elif menu == "📝 Symptom Analyzer":
    st.markdown('<h2 class="section-header">📝 AI Symptom Analyzer</h2>', unsafe_allow_html=True)
    cur = conn.cursor()

    if not n_ok:
        danger("❌ NLP model not loaded! Run: <code>python scripts/nlp_model.py</code>")
        st.stop()

    diseases = get_supported_diseases()
    info(f"🤖 <b>TF-IDF + Logistic Regression</b> | <b>{len(diseases)}</b> diseases | 1200 symptom descriptions trained")

    col1, col2 = st.columns([1.5, 1])
    with col1:
        st.markdown("### 💬 Describe Patient Symptoms")
        symptom_text = st.text_area(
            "Symptoms:", height=150, label_visibility="collapsed",
            placeholder="Patient has severe chest pain, shortness of breath, and dizziness for 3 days..."
        )

        # Link to patient
        cur.execute("SELECT id, name FROM patients")
        pts = cur.fetchall()
        save_pid = 0
        if pts:
            pt_opts = {"Don't save to record": 0}
            pt_opts.update({f"#{p[0]} — {p[1]}": p[0] for p in pts})
            sel = st.selectbox("💾 Save to patient record", list(pt_opts.keys()))
            save_pid = pt_opts[sel]

        analyze_btn = st.button("🔍 Analyze Symptoms", use_container_width=True)

    with col2:
        st.markdown(f"### 📋 Supported Diseases ({len(diseases)})")
        for d in diseases[:14]:
            st.markdown(f'<div style="display:flex;align-items:center;gap:0.5rem;padding:0.2rem 0;font-size:0.8rem;color:#64748b;"><div style="width:6px;height:6px;border-radius:50%;background:#00c8ff;flex-shrink:0;"></div>{d}</div>', unsafe_allow_html=True)
        if len(diseases) > 14:
            st.markdown(f'<div style="color:#475569;font-size:0.78rem;margin-top:0.3rem;">+{len(diseases)-14} more...</div>', unsafe_allow_html=True)

    if analyze_btn:
        if not symptom_text:
            warn("⚠️ Please enter symptom description!")
        else:
            result = analyze_symptoms(symptom_text)

            if result.get("error"):
                danger(f"❌ {result['error']}")
            else:
                divider()
                c1, c2 = st.columns([1, 1.5])
                with c1:
                    conf = result["primary_confidence"]
                    st.markdown(f"""
                    <div style="background:linear-gradient(135deg,rgba(0,128,255,0.15),rgba(0,200,255,0.1));
                                border:1px solid rgba(0,200,255,0.3);border-radius:16px;padding:2rem;text-align:center;">
                        <div style="color:#64748b;font-size:0.8rem;text-transform:uppercase;letter-spacing:1px;">Primary Diagnosis</div>
                        <div style="font-size:1.8rem;font-weight:800;color:#00c8ff;margin:1rem 0;">{result["primary_disease"]}</div>
                        <div style="color:#94a3b8;">Confidence: {conf*100:.1f}%</div>
                    </div>""", unsafe_allow_html=True)

                    # Save to patient
                    if save_pid > 0:
                        cur.execute(
                            "INSERT INTO symptoms (patient_id,symptoms_text,diagnosis,date) VALUES (?,?,?,?)",
                            (save_pid, symptom_text[:200], result["primary_disease"], get_today()))
                        conn.commit()
                        st.toast("Saved to patient record!", icon="💾")
                        ok("✅ Saved to patient record!")

                    # Disease info
                    dinfo = get_disease_info(result["primary_disease"])
                    if dinfo:
                        st.markdown(f'<div class="info-box" style="margin-top:1rem;font-size:0.85rem;"><b>Urgency:</b> {dinfo.get("urgency","N/A")}<br><b>Specialist:</b> {dinfo.get("specialist","N/A")}</div>', unsafe_allow_html=True)

                with c2:
                    st.markdown("**Top 5 Possible Conditions:**")
                    for disease, prob in result["top_diseases"]:
                        pct = int(prob * 100)
                        color = "#00c8ff" if disease == result["primary_disease"] else "#64748b"
                        fw = "700" if disease == result["primary_disease"] else "400"
                        prefix = "🏆 " if disease == result["primary_disease"] else "   "
                        st.markdown(f'<div style="margin-bottom:0.6rem;"><div style="display:flex;justify-content:space-between;margin-bottom:4px;"><span style="color:{color};font-size:0.9rem;font-weight:{fw};">{prefix}{disease}</span><span style="color:{color};font-size:0.85rem;">{pct}%</span></div><div style="background:rgba(255,255,255,0.05);border-radius:4px;height:6px;"><div style="width:{pct}%;height:100%;border-radius:4px;background:{"linear-gradient(90deg,#0080ff,#00c8ff)" if disease==result["primary_disease"] else "#1e3a5f"};"></div></div></div>', unsafe_allow_html=True)

                warn("⚠️ AI analysis is for assistance only. Always consult a qualified physician.")

# ═══════════════════════════════════════════════════════════════
# PAGE: X-RAY ANALYSIS
# ═══════════════════════════════════════════════════════════════
elif menu == "🫁 X-ray Analysis":
    st.markdown(
        '<h2 class="section-header">🫁 X-ray Analysis</h2>',
        unsafe_allow_html=True)
    cur = conn.cursor()

    if not cnn_status['loaded']:
        st.markdown("""
        <div class="warn-box">
            <b>⚠️ CNN Model not loaded.</b><br>
            Run locally: <code>python scripts/cnn_model.py</code><br>
            Or train on Google Colab (free GPU).
        </div>""", unsafe_allow_html=True)
        st.stop()

    info("🤖 <b>DenseNet121</b> Transfer Learning | "
         "Classes: NORMAL, PNEUMONIA")

    # ─── STEP 1: SELECT PATIENT ──────────────────────────
    st.markdown("### Step 1 — Select Patient")

    cur.execute("SELECT id, name, age, gender FROM patients ORDER BY name")
    pts = cur.fetchall()

    if not pts:
        warn("⚠️ No patients registered yet! "
             "Add patients in Patient Management first.")
        st.stop()

    # Build patient dropdown
    pt_opts = {f"#{p[0]} — {p[1]} (Age: {p[2]}, {p[3]})": p[0]
               for p in pts}
    selected_pt = st.selectbox(
        "🔍 Select Patient for X-ray",
        list(pt_opts.keys()),
        label_visibility="collapsed"
    )
    patient_id = pt_opts[selected_pt]

    # Show selected patient info
    cur.execute("SELECT * FROM patients WHERE id=?", (patient_id,))
    patient = cur.fetchone()

    col1, col2, col3, col4 = st.columns(4)
    with col1: mc("👤", patient[1][:10], "Name")
    with col2: mc("🎂", patient[2], "Age")
    with col3: mc("⚧", patient[3], "Gender")
    with col4:
        # Count existing X-rays for this patient
        cur.execute(
            "SELECT COUNT(*) FROM xray_results WHERE patient_id=?",
            (patient_id,))
        xray_count = cur.fetchone()[0]
        mc("🫁", xray_count, "X-ray Records")

    divider()

    # ─── STEP 2: UPLOAD X-RAY ────────────────────────────
    st.markdown("### Step 2 — Upload X-ray Image")

    col1, col2 = st.columns([1, 1.2])

    with col1:
        uploaded = st.file_uploader(
            "Upload chest X-ray (JPG/PNG)",
            type=['jpg', 'jpeg', 'png'],
            label_visibility="collapsed"
        )

        if uploaded:
            from PIL import Image
            image = Image.open(uploaded)
            st.image(image,
                     caption=f"X-ray for {patient[1]}",
                     use_container_width=True)

            # X-ray type selector
            xray_type = st.selectbox(
                "📋 X-ray Type",
                ["Chest PA View",
                 "Chest AP View",
                 "Lateral View",
                 "Other"]
            )

            notes = st.text_area(
                "📝 Doctor Notes (optional)",
                placeholder="Any additional observations...",
                height=80
            )

            analyze_btn = st.button(
                "🔬 Analyze X-ray & Save to Patient",
                use_container_width=True
            )

    with col2:
        # Show previous X-rays for this patient
        st.markdown(
            f"### 📋 Previous X-rays for {patient[1]}")

        cur.execute("""
            SELECT disease, confidence, date
            FROM xray_results
            WHERE patient_id = ?
            ORDER BY date DESC
            LIMIT 5
        """, (patient_id,))
        prev_xrays = cur.fetchall()

        if prev_xrays:
            for xr in prev_xrays:
                color = (
                    "#f87171"
                    if xr[0] != "NORMAL"
                    else "#34d399"
                )
                icon = "⚠️" if xr[0] != "NORMAL" else "✅"
                st.markdown(f"""
                <div style="background:rgba(13,26,46,0.6);
                            border:1px solid rgba(0,200,255,0.1);
                            border-radius:10px;
                            padding:0.75rem 1rem;
                            margin-bottom:0.5rem;">
                    <div style="display:flex;
                                justify-content:space-between;">
                        <span style="color:{color};
                                     font-weight:700;">
                            {icon} {xr[0]}
                        </span>
                        <span style="color:#64748b;
                                     font-size:0.82rem;">
                            {xr[2]}
                        </span>
                    </div>
                    <div style="color:#94a3b8;font-size:0.82rem;">
                        Confidence: {int(xr[1]*100)}%
                    </div>
                </div>""", unsafe_allow_html=True)
        else:
            info(f"No previous X-ray records for {patient[1]}.")

    # ─── STEP 3: ANALYZE AND SAVE ────────────────────────
    if uploaded and analyze_btn:
        with st.spinner("🔬 Analyzing X-ray with DenseNet121..."):
            result = analyze_xray(uploaded)

        if result.get("error"):
            danger(f"❌ {result['error']}")

        else:
            divider()
            st.markdown("### Step 3 — Analysis Result")

            col1, col2, col3 = st.columns(3)

            with col1:
                if not result["is_normal"]:
                    st.markdown(f"""
                    <div class="result-positive">
                        <div class="result-title text-red">
                            ⚠️ {result['predicted_class']}
                        </div>
                        <div style="color:#94a3b8;">
                            Disease Detected
                        </div>
                    </div>""", unsafe_allow_html=True)
                    st.toast(
                        f"⚠️ {result['predicted_class']} detected!",
                        icon="⚠️")
                else:
                    st.markdown("""
                    <div class="result-negative">
                        <div class="result-title text-green">
                            ✅ NORMAL
                        </div>
                        <div style="color:#94a3b8;">
                            No Disease Found
                        </div>
                    </div>""", unsafe_allow_html=True)
                    st.toast("✅ X-ray normal!", icon="✅")

            with col2:
                rc = get_risk_color(result["confidence"])
                st.markdown(f"""
                <div style="background:rgba(13,26,46,0.8);
                            border:1px solid {rc}40;
                            border-radius:16px;
                            padding:1.5rem;
                            text-align:center;">
                    <div style="color:#64748b;font-size:0.8rem;">
                        Confidence
                    </div>
                    <div style="font-size:2rem;font-weight:800;
                                color:{rc};">
                        {int(result['confidence']*100)}%
                    </div>
                    <div style="color:{rc};font-weight:600;">
                        {result['predicted_class']}
                    </div>
                </div>""", unsafe_allow_html=True)

            with col3:
                # PDF Report
                from utils.pdf_generator import (
                    generate_medical_report)
                recs = get_xray_recommendations(
                    result["predicted_class"],
                    result["confidence"])
                pdf_bytes = generate_medical_report(
                    patient[1], patient[2], patient[3],
                    f"Chest X-ray ({xray_type})",
                    result["predicted_class"],
                    result["confidence"],
                    get_risk_level(result["confidence"]),
                    {"X-ray Type": xray_type,
                     "Predicted": result["predicted_class"],
                     "Confidence": f"{int(result['confidence']*100)}%"},
                    recs
                )
                if pdf_bytes:
                    st.download_button(
                        "📄 Download PDF Report",
                        pdf_bytes,
                        f"xray_{patient[1]}_{get_today()}.pdf",
                        "application/pdf",
                        use_container_width=True
                    )

            # Confidence bars
            for cls, prob in result["all_predictions"].items():
                cbar(prob, cls)

            # ── AUTO-SAVE TO DATABASE ─────────────────────
            # This is the KEY feature —
            # saves result linked to patient!
            cur.execute("""
                INSERT INTO xray_results
                    (patient_id, disease, confidence, date)
                VALUES (?, ?, ?, ?)
            """, (
                patient_id,
                result["predicted_class"],
                result["confidence"],
                get_today()
            ))
            conn.commit()

            st.toast(
                f"✅ X-ray result saved to {patient[1]}'s record!",
                icon="💾")
            ok(f"""
                ✅ <b>Saved to Patient Record!</b><br>
                Patient: <b>{patient[1]}</b><br>
                Result: <b>{result['predicted_class']}</b><br>
                Confidence: <b>{int(result['confidence']*100)}%</b><br>
                Date: <b>{get_today()}</b>
            """)

            # Medical recommendation
            if not result["is_normal"]:
                danger(f"""
                    <b>⚕️ Recommendation:</b>
                    {recs[1] if len(recs) > 1 else recs[0]}
                """)

            # ── SHOW ALL X-RAYS FOR THIS PATIENT ─────────
            divider()
            st.markdown(
                f"### 📋 Complete X-ray History — {patient[1]}")

            cur.execute("""
                SELECT id, disease, confidence, date
                FROM xray_results
                WHERE patient_id = ?
                ORDER BY date DESC
            """, (patient_id,))
            all_xrays = cur.fetchall()

            if all_xrays:
                df_x = pd.DataFrame(
                    all_xrays,
                    columns=["Record ID", "Disease",
                             "Confidence", "Date"])
                df_x["Confidence"] = df_x["Confidence"].apply(
                    lambda x: f"{int(x*100)}%")
                st.dataframe(
                    df_x,
                    use_container_width=True,
                    hide_index=True)

# ═══════════════════════════════════════════════════════════════
# PAGE: ANALYTICS
# ═══════════════════════════════════════════════════════════════
elif menu == "📊 Analytics":
    st.markdown('<h2 class="section-header">📊 Hospital Analytics</h2>', unsafe_allow_html=True)
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM patients"); tp = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM xray_results"); tx = cur.fetchone()[0]
    cur.execute("SELECT AVG(age) FROM patients"); aa = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM inventory WHERE quantity < 50"); ls = cur.fetchone()[0]

    c1,c2,c3,c4 = st.columns(4)
    with c1: mc("👥", tp, "Patients")
    with c2: mc("🫁", tx, "X-rays")
    with c3: mc("📅", f"{aa:.0f}" if aa else "N/A", "Avg Age")
    with c4: mc("⚠️", ls, "Low Stock")

    divider()
    tab1, tab2, tab3 = st.tabs(["👥 Patients","🫁 Diseases","💊 Inventory"])

    with tab1:
        cur.execute("SELECT * FROM patients")
        pts = cur.fetchall()
        if pts:
            df_p = pd.DataFrame(pts, columns=["ID","Name","Age","Gender","Phone"])
            col1, col2 = st.columns(2)
            with col1:
                fig, ax = dark_fig()
                ax.hist(df_p['Age'], bins=8, color='#00c8ff', edgecolor='#0d1a2e', alpha=0.85)
                ax.set_title('Age Distribution', color='#94a3b8')
                plt.tight_layout(); st.pyplot(fig); plt.close()
            with col2:
                gc = df_p['Gender'].value_counts()
                fig, ax = dark_fig()
                ax.pie(gc.values, labels=gc.index,
                      colors=['#00c8ff','#f87171','#fbbf24'][:len(gc)],
                      autopct='%1.1f%%', textprops={'color':'#94a3b8'})
                ax.set_title('Gender Distribution', color='#94a3b8')
                st.pyplot(fig); plt.close()
            st.download_button("📥 Export Patients CSV", df_p.to_csv(index=False), "patients.csv", "text/csv")
        else:
            info("No patient data.")

    with tab2:
        cur.execute("SELECT disease, COUNT(*), AVG(confidence) FROM xray_results GROUP BY disease")
        ds = cur.fetchall()
        if ds:
            df_d = pd.DataFrame(ds, columns=['Disease','Count','Avg Confidence'])
            fig, ax = dark_fig(8, 4)
            colors = ['#00c8ff','#f87171','#fbbf24','#34d399','#a78bfa']
            bars = ax.bar(df_d['Disease'], df_d['Count'],
                         color=colors[:len(df_d)], edgecolor='none')
            ax.set_title('Disease Count', color='#94a3b8')
            ax.grid(axis='y', alpha=0.3, color='#1e3a5f', linestyle='--')
            for bar, c in zip(bars, df_d['Count']):
                ax.text(bar.get_x()+bar.get_width()/2., bar.get_height()+0.05,
                       str(int(c)), ha='center', color='#94a3b8', fontweight='bold')
            plt.tight_layout(); st.pyplot(fig); plt.close()
            st.dataframe(df_d, use_container_width=True, hide_index=True)

    with tab3:
        cur.execute("SELECT * FROM inventory ORDER BY quantity ASC")
        inv = cur.fetchall()
        if inv:
            df_inv = pd.DataFrame(inv, columns=["ID","Medicine","Quantity","Last Updated"])
            for _, r in df_inv[df_inv['Quantity']<30].iterrows():
                danger(f"🚨 CRITICAL: <b>{r['Medicine']}</b> — Only {r['Quantity']} units!")
            for _, r in df_inv[(df_inv['Quantity']>=30)&(df_inv['Quantity']<50)].iterrows():
                warn(f"⚠️ LOW STOCK: <b>{r['Medicine']}</b> — {r['Quantity']} units")

            fig, ax = dark_fig(10, 4)
            ax.barh(df_inv['Medicine'], df_inv['Quantity'],
                   color=['#f87171' if q < 50 else '#00c8ff' for q in df_inv['Quantity']],
                   edgecolor='none')
            ax.axvline(50, color='#fbbf24', linestyle='--', alpha=0.7, label='Low Stock (50)')
            ax.set_title('Medicine Stock Levels', color='#94a3b8')
            ax.set_xlabel('Quantity', color='#64748b')
            ax.legend(labelcolor='#94a3b8', facecolor='#0d1a2e', edgecolor='#1e3a5f')
            plt.tight_layout(); st.pyplot(fig); plt.close()

            pdf_b = generate_inventory_report(inv)
            col1, col2 = st.columns(2)
            with col1:
                st.download_button("📥 Export CSV", df_inv.to_csv(index=False), "inventory.csv", "text/csv")
            with col2:
                if pdf_b:
                    st.download_button("📄 Inventory PDF", pdf_b, "inventory_report.pdf", "application/pdf")
                    
                    
                    
elif menu == "🔍 Search":
    st.markdown(
        '<h2 class="section-header">'
        '🔍 Global Search</h2>',
        unsafe_allow_html=True)

    search = st.text_input(
        "Search anything...",
        placeholder="Patient name, disease, medicine...")

    if search and len(search) >= 2:
        cur = conn.cursor()

        # Search patients
        cur.execute("""
            SELECT 'Patient' as type, name as result,
                   age as detail, id
            FROM patients
            WHERE name LIKE ? OR phone LIKE ?
        """, (f"%{search}%", f"%{search}%"))
        patient_results = cur.fetchall()

        # Search diagnoses
        cur.execute("""
            SELECT 'Diagnosis' as type,
                   s.diagnosis as result,
                   p.name as detail,
                   s.patient_id as id
            FROM symptoms s
            JOIN patients p ON s.patient_id = p.id
            WHERE s.diagnosis LIKE ? OR s.symptoms_text LIKE ?
        """, (f"%{search}%", f"%{search}%"))
        diagnosis_results = cur.fetchall()

        # Search inventory
        cur.execute("""
            SELECT 'Medicine' as type,
                   medicine_name as result,
                   quantity as detail,
                   id
            FROM inventory
            WHERE medicine_name LIKE ?
        """, (f"%{search}%",))
        med_results = cur.fetchall()

        total = (len(patient_results) +
                 len(diagnosis_results) +
                 len(med_results))

        if total > 0:
            ok(f"Found <b>{total}</b> results "
               f"for '<b>{search}</b>'")

            if patient_results:
                st.markdown("**👥 Patients:**")
                for r in patient_results:
                    st.markdown(
                        f"• Patient #{r[3]}: "
                        f"<b>{r[1]}</b> — Age {r[2]}",
                        unsafe_allow_html=True)

            if diagnosis_results:
                st.markdown("**🏥 Diagnoses:**")
                for r in diagnosis_results:
                    st.markdown(
                        f"• {r[2]}: <b>{r[1]}</b>",
                        unsafe_allow_html=True)

            if med_results:
                st.markdown("**💊 Medicines:**")
                for r in med_results:
                    st.markdown(
                        f"• <b>{r[1]}</b> — "
                        f"{r[2]} units",
                        unsafe_allow_html=True)
        else:
            warn(f"No results found for '{search}'")

# ═══════════════════════════════════════════════════════════════
# PAGE: INVENTORY
# ═══════════════════════════════════════════════════════════════
elif menu == "💊 Inventory":
    st.markdown('<h2 class="section-header">💊 Hospital Inventory</h2>', unsafe_allow_html=True)
    cur = conn.cursor()
    tab1, tab2 = st.tabs(["📋 View Stock","➕ Add/Update"])

    with tab1:
        cur.execute("SELECT * FROM inventory ORDER BY quantity ASC")
        inv = cur.fetchall()
        if inv:
            df_inv = pd.DataFrame(inv, columns=["ID","Medicine","Quantity","Last Updated"])
            for _, r in df_inv[df_inv['Quantity']<30].iterrows():
                danger(f"🚨 CRITICAL: <b>{r['Medicine']}</b> — {r['Quantity']} units only!")
            for _, r in df_inv[(df_inv['Quantity']>=30)&(df_inv['Quantity']<50)].iterrows():
                warn(f"⚠️ LOW STOCK: <b>{r['Medicine']}</b> — {r['Quantity']} units")
            st.dataframe(df_inv, use_container_width=True, hide_index=True)
            st.download_button("📥 Export CSV", df_inv.to_csv(index=False), "inventory.csv", "text/csv")
        else:
            info("No inventory data.")

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            med_name = st.text_input("💊 Medicine Name", placeholder="e.g. Paracetamol 500mg")
            qty = st.number_input("📦 Quantity", 0, 10000, 100)
        with col2:
            action = st.selectbox("Action", ["Add New Medicine","Update Existing"])

        if st.button("✅ Save Medicine", use_container_width=True):
            if med_name:
                today = get_today()
                if action == "Add New Medicine":
                    cur.execute(
                        "INSERT OR IGNORE INTO inventory (medicine_name,quantity,last_updated) VALUES (?,?,?)",
                        (med_name, qty, today))
                else:
                    cur.execute(
                        "UPDATE inventory SET quantity=?,last_updated=? WHERE medicine_name=?",
                        (qty, today, med_name))
                conn.commit()
                st.toast(f"💊 {med_name} saved!", icon="💊")
                ok(f"✅ <b>{med_name}</b> — {qty} units saved on {today}")
            else:
                warn("⚠️ Please enter medicine name!")
                
                




# ═══════════════════════════════════════════════════════════════
# PAGE: SETTINGS
# ═══════════════════════════════════════════════════════════════
elif menu == "⚙️ Settings":
    st.markdown('<h2 class="section-header">⚙️ Settings</h2>', unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["🔐 Change Password","👥 User Management"])

    with tab1:
        st.markdown("### Change Password")
        cur_p = st.text_input("Current Password", type="password")
        new_p = st.text_input("New Password", type="password")
        conf_p = st.text_input("Confirm New Password", type="password")
        if st.button("🔐 Update Password", use_container_width=True):
            ok_login, _ = verify_login(st.session_state.username, cur_p)
            if not ok_login:
                danger("❌ Current password incorrect!")
            elif new_p != conf_p:
                warn("⚠️ Passwords don't match!")
            elif len(new_p) < 6:
                warn("⚠️ Password must be at least 6 characters!")
            else:
                conn.execute(
                    "UPDATE users SET password_hash=? WHERE username=?",
                    (hash_pw(new_p), st.session_state.username))
                conn.commit()
                st.toast("Password updated! 🔐", icon="✅")
                ok("✅ Password updated successfully!")

    with tab2:
        if st.session_state.role == "admin":
            cur = conn.cursor()
            cur.execute("SELECT id, username, role, created_at FROM users")
            users = cur.fetchall()
            if users:
                df_u = pd.DataFrame(users, columns=["ID","Username","Role","Created"])
                st.dataframe(df_u, use_container_width=True, hide_index=True)
        else:
            info("Admin access required to view all users.")
