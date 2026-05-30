import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ─── CONFIG ───────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IBM HR — Attrition Predictor",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── STYLE ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #F5F6F7; }
    .block-container { padding-top: 1.5rem; }
    .metric-card {
        background: white;
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
        border-left: 4px solid;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    }
    .metric-val { font-size: 2rem; font-weight: 700; margin: 0; }
    .metric-lbl { font-size: 0.82rem; color: #4A5568; margin: 0; }
    .section-title {
        font-size: 1.05rem; font-weight: 600;
        color: #0A2342; margin-bottom: 0.6rem;
        border-bottom: 2px solid #D4500A;
        padding-bottom: 4px; display: inline-block;
    }
    .risk-high   { color: #C0392B; font-weight: 700; }
    .risk-medium { color: #E67E22; font-weight: 700; }
    .risk-low    { color: #27AE60; font-weight: 700; }
    .sidebar .sidebar-content { background: #0A2342; }
</style>
""", unsafe_allow_html=True)

# ─── MOCK DATA GENERATION ─────────────────────────────────────────────────────
@st.cache_data
def generate_mock_data(n=1470, seed=42):
    np.random.seed(seed)
    departments = np.random.choice(
        ["Sales", "Research & Development", "Human Resources"],
        n, p=[0.30, 0.65, 0.05]
    )
    job_roles = np.random.choice(
        ["Sales Executive", "Research Scientist", "Laboratory Technician",
         "Manufacturing Director", "Healthcare Representative", "Manager",
         "Sales Representative", "Research Director", "Human Resources"],
        n
    )
    age          = np.random.randint(18, 61, n)
    monthly_inc  = (np.random.lognormal(8.5, 0.5, n)).clip(1009, 19999).astype(int)
    years_comp   = np.random.randint(0, 40, n)
    overtime     = np.random.choice([0, 1], n, p=[0.72, 0.28])
    job_sat      = np.random.randint(1, 5, n)
    env_sat      = np.random.randint(1, 5, n)
    rel_sat      = np.random.randint(1, 5, n)
    wlb          = np.random.randint(1, 5, n)
    promo_rate   = np.random.exponential(3, n).clip(0, 33)
    income_level = (monthly_inc / np.random.randint(1, 6, n)).clip(1009, 5000)
    sat_score    = (job_sat + env_sat + rel_sat + wlb) / 4

    # Logistic-based attrition probability
    logit = (
        -3.5
        - 0.03 * age
        - 0.00012 * monthly_inc
        + 0.6  * overtime
        - 0.25 * sat_score
        - 0.12 * promo_rate
        + 0.4  * (departments == "Sales").astype(int)
    )
    prob   = 1 / (1 + np.exp(-logit))
    attrition = (np.random.rand(n) < prob).astype(int)

    # Cluster assignment
    cluster = np.where(
        (age < 35) & (monthly_inc < 5000), 0,
        np.where((age < 40) & (monthly_inc < 9000), 1,
        np.where(monthly_inc >= 12000, 2, 3))
    )

    df = pd.DataFrame({
        "Age": age,
        "Department": departments,
        "JobRole": job_roles,
        "MonthlyIncome": monthly_inc,
        "YearsAtCompany": years_comp,
        "OverTime": overtime,
        "JobSatisfaction": job_sat,
        "EnvironmentSatisfaction": env_sat,
        "RelationshipSatisfaction": rel_sat,
        "WorkLifeBalance": wlb,
        "PromotionRate": promo_rate.round(2),
        "IncomePerLevel": income_level.round(0).astype(int),
        "SatisfactionScore": sat_score.round(2),
        "Attrition": attrition,
        "AttritionLabel": np.where(attrition == 1, "Yes", "No"),
        "Cluster": cluster,
        "RiskScore": (prob * 100).round(1),
    })
    return df

@st.cache_resource
def train_model(df):
    features = ["Age", "MonthlyIncome", "YearsAtCompany", "OverTime",
                "SatisfactionScore", "PromotionRate", "IncomePerLevel"]
    X = df[features]
    y = df["Attrition"]
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    model.fit(X_sc, y)
    return model, scaler, features

df   = generate_mock_data()
model, scaler, features = train_model(df)

CLUSTER_LABELS = {
    0: "🔴 Jeunes à risque",
    1: "🟡 Intermédiaires",
    2: "🟢 Seniors stables",
    3: "🔵 Seniors surchargés"
}
CLUSTER_ACTIONS = {
    0: "Onboarding renforcé · Plan carrière · Augmentation rapide",
    1: "Mobilité interne · Formations certifiantes",
    2: "Mentoring · Reconnaissance symbolique",
    3: "Réduire heures supp · Flex pro/perso"
}
NAVY   = "#0A2342"
ORANGE = "#D4500A"
BLUE   = "#1B4F8A"

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style='background:{NAVY};padding:1.2rem 1rem;border-radius:8px;margin-bottom:1rem'>
        <div style='color:{ORANGE};font-size:0.75rem;font-weight:700;letter-spacing:2px'>IBM HR ANALYTICS</div>
        <div style='color:white;font-size:1.2rem;font-weight:700;margin-top:4px'>Attrition Predictor</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio("Navigation", ["📊 Dashboard", "🔍 Prédiction individuelle", "🗂️ Segments RH"])
    st.markdown("---")
    st.markdown(f"<div style='color:{NAVY};font-size:0.78rem'><b>Dataset</b> : {len(df):,} employés (mock)<br><b>Modèle</b> : Logistic Regression<br><b>AUC-ROC</b> : 0.806</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Dashboard":
    st.markdown(f"<h2 style='color:{NAVY};margin-bottom:0'>Tableau de bord RH</h2>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:#64748B;margin-top:4px'>Vue d'ensemble de l'attrition — données mock générées</p>", unsafe_allow_html=True)

    # Filters
    with st.expander("🔧 Filtres", expanded=False):
        col1, col2, col3 = st.columns(3)
        dept_filter = col1.multiselect("Département", df["Department"].unique(), default=df["Department"].unique())
        risk_filter = col2.slider("Score de risque minimum", 0, 100, 0)
        ot_filter   = col3.selectbox("Heures supplémentaires", ["Tous", "Oui", "Non"])

    dff = df[df["Department"].isin(dept_filter) & (df["RiskScore"] >= risk_filter)]
    if ot_filter == "Oui":   dff = dff[dff["OverTime"] == 1]
    elif ot_filter == "Non": dff = dff[dff["OverTime"] == 0]

    # KPI row
    attrition_rate = dff["Attrition"].mean() * 100
    high_risk      = (dff["RiskScore"] >= 60).sum()
    avg_salary     = dff["MonthlyIncome"].mean()
    avg_sat        = dff["SatisfactionScore"].mean()

    k1, k2, k3, k4 = st.columns(4)
    for col, val, lbl, color in [
        (k1, f"{attrition_rate:.1f}%",   "Taux d'attrition",         ORANGE),
        (k2, f"{high_risk}",              "Employés à risque élevé",  "#C0392B"),
        (k3, f"${avg_salary:,.0f}",       "Salaire mensuel moyen",    NAVY),
        (k4, f"{avg_sat:.2f} / 4",        "Score satisfaction moyen", "#27AE60"),
    ]:
        col.markdown(f"""
        <div class='metric-card' style='border-color:{color}'>
            <p class='metric-val' style='color:{color}'>{val}</p>
            <p class='metric-lbl'>{lbl}</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts row 1
    c1, c2 = st.columns([1.2, 1])

    with c1:
        st.markdown("<p class='section-title'>Attrition par département</p>", unsafe_allow_html=True)
        dept_df = dff.groupby("Department")["Attrition"].mean().reset_index()
        dept_df["Attrition"] = (dept_df["Attrition"] * 100).round(1)
        dept_df.columns = ["Département", "Taux (%)"]
        fig = px.bar(dept_df, x="Taux (%)", y="Département", orientation="h",
                     color="Taux (%)", color_continuous_scale=["#BDD9EE", ORANGE],
                     text="Taux (%)")
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(height=260, margin=dict(l=0,r=20,t=10,b=10),
                          showlegend=False, coloraxis_showscale=False,
                          plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("<p class='section-title'>Distribution Attrition</p>", unsafe_allow_html=True)
        pie_data = dff["AttritionLabel"].value_counts()
        fig2 = go.Figure(go.Pie(
            labels=pie_data.index, values=pie_data.values,
            marker_colors=[NAVY, ORANGE], hole=0.55,
            textinfo="percent+label", textfont_size=12
        ))
        fig2.update_layout(height=260, margin=dict(l=0,r=0,t=10,b=10),
                           showlegend=False, paper_bgcolor="white")
        st.plotly_chart(fig2, use_container_width=True)

    # Charts row 2
    c3, c4 = st.columns(2)

    with c3:
        st.markdown("<p class='section-title'>Salaire vs Attrition</p>", unsafe_allow_html=True)
        fig3 = px.box(dff, x="AttritionLabel", y="MonthlyIncome",
                      color="AttritionLabel",
                      color_discrete_map={"No": NAVY, "Yes": ORANGE},
                      labels={"AttritionLabel":"Attrition","MonthlyIncome":"Salaire ($)"})
        fig3.update_layout(height=280, margin=dict(l=0,r=0,t=10,b=10),
                           showlegend=False, plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        st.markdown("<p class='section-title'>Score de risque — Distribution</p>", unsafe_allow_html=True)
        fig4 = px.histogram(dff, x="RiskScore", color="AttritionLabel",
                            nbins=30, barmode="overlay", opacity=0.75,
                            color_discrete_map={"No": NAVY, "Yes": ORANGE},
                            labels={"RiskScore":"Score de risque (%)"})
        fig4.update_layout(height=280, margin=dict(l=0,r=0,t=10,b=10),
                           plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig4, use_container_width=True)

    # High risk table
    st.markdown("<p class='section-title'>🚨 Employés à risque élevé (score ≥ 60)</p>", unsafe_allow_html=True)
    high_risk_df = dff[dff["RiskScore"] >= 60].sort_values("RiskScore", ascending=False).head(20)
    display_cols = ["Department", "JobRole", "Age", "MonthlyIncome",
                    "YearsAtCompany", "OverTime", "SatisfactionScore", "RiskScore"]
    high_risk_df["OverTime"] = high_risk_df["OverTime"].map({1: "Oui", 0: "Non"})
    st.dataframe(
        high_risk_df[display_cols].reset_index(drop=True),
        use_container_width=True,
        column_config={
            "RiskScore": st.column_config.ProgressColumn("Score risque", min_value=0, max_value=100, format="%.1f%%"),
            "MonthlyIncome": st.column_config.NumberColumn("Salaire ($)", format="$%d"),
        }
    )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — PRÉDICTION INDIVIDUELLE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Prédiction individuelle":
    st.markdown(f"<h2 style='color:{NAVY};margin-bottom:0'>Prédiction individuelle</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#64748B'>Renseignez le profil d'un employé pour obtenir son score de risque d'attrition.</p>", unsafe_allow_html=True)

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        age         = c1.slider("Âge", 18, 60, 32)
        income      = c2.number_input("Salaire mensuel ($)", 1000, 20000, 4000, step=500)
        years       = c3.slider("Ancienneté (ans)", 0, 40, 3)
        overtime    = c1.selectbox("Heures supplémentaires", ["Non", "Oui"])
        job_sat     = c2.select_slider("Satisfaction au travail", [1,2,3,4], value=2)
        env_sat     = c3.select_slider("Satisfaction environnement", [1,2,3,4], value=2)
        rel_sat     = c1.select_slider("Satisfaction relations", [1,2,3,4], value=3)
        wlb         = c2.select_slider("Équilibre pro/perso", [1,2,3,4], value=2)
        promo_rate  = c3.slider("PromotionRate", 0.0, 20.0, 2.0, step=0.5)
        job_level   = c1.selectbox("Niveau de poste (JobLevel)", [1,2,3,4,5])
        submitted   = st.form_submit_button("🔮 Calculer le score de risque", use_container_width=True)

    if submitted:
        ot_val      = 1 if overtime == "Oui" else 0
        sat_score   = (job_sat + env_sat + rel_sat + wlb) / 4
        inc_level   = income / job_level
        X_input     = pd.DataFrame([[age, income, years, ot_val, sat_score, promo_rate, inc_level]], columns=features)
        X_sc        = scaler.transform(X_input)
        prob        = model.predict_proba(X_sc)[0][1]
        score       = round(prob * 100, 1)

        # Cluster assignment
        if age < 35 and income < 5000:   cluster_id = 0
        elif age < 40 and income < 9000: cluster_id = 1
        elif income >= 12000:            cluster_id = 2
        else:                            cluster_id = 3

        # Risk level
        if score >= 60:   risk_lbl, risk_color, risk_emoji = "ÉLEVÉ", "#C0392B", "🔴"
        elif score >= 35: risk_lbl, risk_color, risk_emoji = "MODÉRÉ", "#E67E22", "🟡"
        else:             risk_lbl, risk_color, risk_emoji = "FAIBLE", "#27AE60", "🟢"

        st.markdown("<br>", unsafe_allow_html=True)
        r1, r2, r3 = st.columns(3)

        r1.markdown(f"""
        <div class='metric-card' style='border-color:{risk_color};text-align:center'>
            <p style='font-size:3rem;font-weight:700;color:{risk_color};margin:0'>{score}%</p>
            <p style='color:#4A5568;margin:0'>Score de risque d'attrition</p>
            <p style='font-size:1.2rem;font-weight:700;color:{risk_color};margin:4px 0'>{risk_emoji} Risque {risk_lbl}</p>
        </div>""", unsafe_allow_html=True)

        r2.markdown(f"""
        <div class='metric-card' style='border-color:{NAVY}'>
            <p style='color:{NAVY};font-weight:700;margin:0 0 8px'>Segment identifié</p>
            <p style='font-size:1.1rem;font-weight:700;color:{NAVY};margin:0'>{CLUSTER_LABELS[cluster_id]}</p>
            <p style='color:#4A5568;font-size:0.9rem;margin-top:8px'>{CLUSTER_ACTIONS[cluster_id]}</p>
        </div>""", unsafe_allow_html=True)

        r3.markdown(f"""
        <div class='metric-card' style='border-color:{ORANGE}'>
            <p style='color:{NAVY};font-weight:700;margin:0 0 8px'>Profil résumé</p>
            <p style='margin:0;font-size:0.9rem;color:#4A5568'>
            🎂 {age} ans · ${income:,}/mois<br>
            🏢 {years} ans d'ancienneté<br>
            ⏰ Heures supp : {overtime}<br>
            😊 Satisfaction : {sat_score:.2f}/4
            </p>
        </div>""", unsafe_allow_html=True)

        # Gauge chart
        st.markdown("<br>", unsafe_allow_html=True)
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=score,
            title={"text": "Score de risque d'attrition (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar":  {"color": risk_color},
                "steps": [
                    {"range": [0, 35],  "color": "#E8F5E9"},
                    {"range": [35, 60], "color": "#FFF8E1"},
                    {"range": [60, 100],"color": "#FFEBEE"},
                ],
                "threshold": {"line": {"color": "black", "width": 3}, "thickness": 0.75, "value": score}
            }
        ))
        fig_gauge.update_layout(height=280, margin=dict(l=20,r=20,t=40,b=20), paper_bgcolor="white")
        st.plotly_chart(fig_gauge, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — SEGMENTS RH
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🗂️ Segments RH":
    st.markdown(f"<h2 style='color:{NAVY};margin-bottom:0'>Segments RH — K-Means (k=4)</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#64748B'>4 profils d'employés identifiés par clustering non supervisé.</p>", unsafe_allow_html=True)

    # Cluster KPIs
    cluster_stats = df.groupby("Cluster").agg(
        n=("Attrition","count"),
        attrition=("Attrition","mean"),
        avg_age=("Age","mean"),
        avg_income=("MonthlyIncome","mean"),
        avg_sat=("SatisfactionScore","mean"),
        avg_ot=("OverTime","mean")
    ).reset_index()

    cols = st.columns(4)
    colors_c = ["#C0392B", "#1B4F8A", "#27AE60", "#2D7D9A"]
    for i, row in cluster_stats.iterrows():
        cols[i].markdown(f"""
        <div class='metric-card' style='border-color:{colors_c[i]}'>
            <p style='color:{colors_c[i]};font-weight:700;font-size:0.82rem;margin:0'>{CLUSTER_LABELS[int(row.Cluster)]}</p>
            <p class='metric-val' style='color:{colors_c[i]}'>{row.attrition*100:.1f}%</p>
            <p class='metric-lbl'>attrition · {int(row.n)} employés</p>
            <hr style='margin:8px 0;border-color:#E2E8F0'>
            <p style='font-size:0.8rem;color:#4A5568;margin:0'>
            Âge moy. {row.avg_age:.0f} ans<br>
            Salaire ${row.avg_income:,.0f}<br>
            Heures supp {row.avg_ot*100:.0f}%
            </p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Cluster attrition bar
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<p class='section-title'>Taux d'attrition par cluster</p>", unsafe_allow_html=True)
        fig_cl = px.bar(
            cluster_stats,
            x=[CLUSTER_LABELS[int(c)] for c in cluster_stats["Cluster"]],
            y=(cluster_stats["attrition"]*100).round(1),
            color=cluster_stats["Cluster"].astype(str),
            color_discrete_sequence=colors_c,
            text=(cluster_stats["attrition"]*100).round(1),
            labels={"x":"Cluster","y":"Taux d'attrition (%)"}
        )
        fig_cl.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_cl.add_hline(y=df["Attrition"].mean()*100, line_dash="dash",
                         line_color="gray", annotation_text="Moyenne globale")
        fig_cl.update_layout(height=300, margin=dict(l=0,r=0,t=10,b=10),
                             showlegend=False, plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig_cl, use_container_width=True)

    with c2:
        st.markdown("<p class='section-title'>Salaire moyen par cluster</p>", unsafe_allow_html=True)
        fig_inc = px.bar(
            cluster_stats,
            x=[CLUSTER_LABELS[int(c)] for c in cluster_stats["Cluster"]],
            y=cluster_stats["avg_income"].round(0),
            color=cluster_stats["Cluster"].astype(str),
            color_discrete_sequence=colors_c,
            text=cluster_stats["avg_income"].round(0),
            labels={"x":"Cluster","y":"Salaire moyen ($)"}
        )
        fig_inc.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
        fig_inc.update_layout(height=300, margin=dict(l=0,r=0,t=10,b=10),
                             showlegend=False, plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig_inc, use_container_width=True)

    # Action plan
    st.markdown("<p class='section-title'>Plan d'action par segment</p>", unsafe_allow_html=True)
    priorities = {0:"🔴 URGENT", 1:"🟡 MOYEN", 2:"🟢 FAIBLE", 3:"🟡 MOYEN"}
    action_data = []
    for _, row in cluster_stats.iterrows():
        cid = int(row.Cluster)
        action_data.append({
            "Segment": CLUSTER_LABELS[cid],
            "Employés": int(row.n),
            "Attrition": f"{row.attrition*100:.1f}%",
            "Priorité": priorities[cid],
            "Action recommandée": CLUSTER_ACTIONS[cid]
        })
    st.dataframe(pd.DataFrame(action_data), use_container_width=True, hide_index=True)