import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from joblib import load
import json
import plotly.graph_objects as go
import plotly.express as px
import shap
import io
import base64


# ==========================================================
#   BACKGROUND + GLOBAL STYLING (Glassmorphism)
# ==========================================================
def add_bg_from_local(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
            color: white;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


st.markdown("""
<style>

/* --- Sidebar --- */
[data-testid="stSidebar"] {
    background: rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(15px);
    border-right: 1px solid rgba(255,255,255,0.15);
}
[data-testid="stSidebar"] * {
    color: #ffffff !important;
}

/* --- App Title and Text --- */
h1, h2, h3, h4, h5, h6, p, label, span {
    color: #f5f5f5 !important;
}

/* --- Main Form Container --- */
div[data-testid="stForm"] {
    background: rgba(255, 255, 255, 0.06);
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.2);
    backdrop-filter: blur(20px);
    box-shadow: 0 4px 40px rgba(0,0,0,0.4);
    padding: 25px;
    transition: all 0.3s ease;
}
div[data-testid="stForm"]:hover {
    background: rgba(255, 255, 255, 0.1);
}

/* --- Input Fields --- */
.stNumberInput > div, 
.stTextInput > div,
.stSelectbox > div,
.stTextArea > div,
.stDateInput > div {
    background: rgba(255,255,255,0.08) !important;
    backdrop-filter: blur(15px) !important;
    border-radius: 10px !important;
    border: 1px solid rgba(255,255,255,0.3) !important;
    box-shadow: inset 0 1px 3px rgba(255,255,255,0.1);
}
.stNumberInput input, 
.stTextInput input, 
.stSelectbox div[data-baseweb="select"] > div, 
textarea {
    background: transparent !important;
    color: #ffffff !important;
    font-weight: 500 !important;
}
.stNumberInput label, 
.stTextInput label, 
.stSelectbox label {
    color: #eeeeee !important;
    font-weight: 500 !important;
}

/* --- Sliders --- */
.stSlider > div > div > div > div {
    background: linear-gradient(90deg, #00ffff, #007bff);
    height: 6px;
    border-radius: 10px;
}

/* --- Buttons --- */
.stButton > button {
    background: rgba(255,255,255,0.12);
    color: #ffffff;
    border: 1px solid rgba(255,255,255,0.25);
    border-radius: 10px;
    backdrop-filter: blur(10px);
    transition: all 0.3s ease;
}
.stButton > button:hover {
    background: rgba(255,255,255,0.22);
    transform: scale(1.03);
    box-shadow: 0 0 15px rgba(0,255,255,0.3);
}

/* --- Metric and Result Boxes --- */
[data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
    color: #ffffff !important;
}
div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 10px;
    border: 1px solid rgba(255,255,255,0.25);
    backdrop-filter: blur(10px);
}

/* --- Chart Containers --- */
[data-testid="stPlotlyChart"] {
    background: rgba(255,255,255,0.05);
    border-radius: 15px;
    padding: 15px;
    border: 1px solid rgba(255,255,255,0.2);
    backdrop-filter: blur(15px);
    box-shadow: 0 4px 25px rgba(0,0,0,0.25);
}

/* --- Tables and DataFrames --- */
[data-testid="stDataFrame"] {
    background: rgba(255,255,255,0.06);
    border-radius: 12px;
    backdrop-filter: blur(10px);
}

/* --- Expander --- */
[data-testid="stExpander"] {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.2);
}

/* --- Scrollbars --- */
::-webkit-scrollbar {
    width: 8px;
}
::-webkit-scrollbar-thumb {
    background: rgba(255,255,255,0.25);
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)


# ==========================================================
#   BACKGROUND IMAGE
# ==========================================================
add_bg_from_local("assets/background.jpg")


# ==========================================================
#   APP CONFIGURATION
# ==========================================================
st.set_page_config(
    page_title="🇱🇰 Sri Lanka Bank Churn Predictor",
    layout="wide",
    page_icon="🇱🇰"
)
st.title("🇱🇰 Bank Customer Churn Predictor")
st.caption("Estimate churn probability, understand the drivers, and run what-if & batch analyses.")


# ==========================================================
#   MODEL LOADING
# ==========================================================
MODEL_PATH = Path("models/churn_model.joblib")
SPEC_PATH  = Path("models/feature_spec.json")
QT_PATH    = Path("models/quantile_transformers.pkl")

@st.cache_resource(show_spinner=False)
def load_artifacts():
    if not MODEL_PATH.exists() or not SPEC_PATH.exists() or not QT_PATH.exists():
        raise FileNotFoundError("Model/spec/transformers not found.")
    pipe = load(MODEL_PATH)
    qt   = load(QT_PATH)
    with open(SPEC_PATH, "r") as f:
        spec = json.load(f)
    return pipe, qt, spec

try:
    pipe, qt, spec = load_artifacts()
except Exception as e:
    st.error(f"❌ Failed to load model assets: {e}")
    st.stop()


threshold = float(spec.get("localization", {}).get("threshold", 0.5))
region_map = spec.get("localization", {}).get("geography_mapping", {})
sl_regions = list(region_map.keys()) if region_map else ["Colombo","Gampaha","Kandy","Jaffna","Galle","Matara"]


# ==========================================================
#   HELPERS
# ==========================================================
FEATURES_ORDER = [
    "CreditScore","Geography","Gender","Age","Tenure","Balance",
    "NumOfProducts","HasCrCard","IsActiveMember","EstimatedSalary"
]

NUMERIC_FEATURES = ["CreditScore","Age","Tenure","Balance","NumOfProducts","EstimatedSalary"]

def scale_quantiles(balance_val, salary_val):
    b = qt["Balance"].transform(pd.DataFrame([[balance_val]], columns=["Balance"]))[0,0] if "Balance" in qt else balance_val
    s = qt["EstimatedSalary"].transform(pd.DataFrame([[salary_val]], columns=["EstimatedSalary"]))[0,0] if "EstimatedSalary" in qt else salary_val
    return b, s

def build_single_row(CreditScore, Geography_SL, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary):
    Geography = region_map.get(Geography_SL, Geography_SL)
    Balance_scaled, Salary_scaled = scale_quantiles(Balance, EstimatedSalary)
    return pd.DataFrame({
        "CreditScore": [CreditScore],
        "Geography": [Geography],
        "Gender": [Gender],
        "Age": [Age],
        "Tenure": [Tenure],
        "Balance": [Balance_scaled],
        "NumOfProducts": [NumOfProducts],
        "HasCrCard": [HasCrCard],
        "IsActiveMember": [IsActiveMember],
        "EstimatedSalary": [Salary_scaled],
    })[FEATURES_ORDER]

def predict_one(df_one_row: pd.DataFrame):
    proba = float(pipe.predict_proba(df_one_row)[:,1][0])
    pred  = int(proba >= threshold)
    return proba, pred

def risk_bucket(proba):
    if proba < 0.30: return "Low", "green"
    if proba < 0.60: return "Medium", "orange"
    return "High", "red"


# ==========================================================
#   SIDEBAR NAVIGATION
# ==========================================================
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Single Prediction", "What-If Analysis", "Batch Prediction", "About"], index=0)


# ==========================================================
#   PAGE 1 — SINGLE PREDICTION
# ==========================================================
if page == "Single Prediction":
    st.subheader("Single Customer Prediction")

    c1, c2 = st.columns([1,1])
    with c1:
        with st.form("input_form"):
            colA, colB = st.columns(2)
            with colA:
                CreditScore = st.number_input("Credit Score", 0, 1000, 650)
                Age = st.number_input("Age", 18, 100, 40)
                Tenure = st.number_input("Tenure with Bank (years)", 0, 50, 5)
                Balance = st.number_input("Account Balance (LKR)", 0.0, 5_000_000.0, 600_000.0, step=10_000.0, format="%.2f")
            with colB:
                NumOfProducts = st.number_input("Bank Products Count", 1, 10, 2)
                HasCrCard = st.selectbox("Has Credit Card?", [0,1], index=1)
                IsActiveMember = st.selectbox("Active Customer?", [0,1], index=1)
                EstimatedSalary = st.number_input("Monthly Income (LKR)", 0.0, 1_000_000.0, 150_000.0, step=5_000.0, format="%.2f")

            Geography_SL = st.selectbox("Customer Region (Sri Lanka)", sl_regions, index=0)
            Gender = st.selectbox("Gender", ["Male","Female"], index=0)
            submitted = st.form_submit_button("Predict")

    with c2:
        st.markdown("### Decision Summary")
        threshold = st.slider("Decision Threshold", 0.0, 1.0, float(threshold), 0.05,
                            help="If probability ≥ threshold → predict Churn (1), else Stay (0).")
        st.metric("Current Cutoff", f"{threshold:.2f}")
        st.caption(
            "Adjust the cutoff to balance **false alarms** vs **missed churns**. "
            "Higher threshold = stricter churn detection."
        )



    if submitted:
        X = build_single_row(
            CreditScore, Geography_SL, Gender, Age, Tenure, Balance,
            NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary
        )
        proba, pred = predict_one(X)

        r1, r2 = st.columns([1.3, 1])
        with r1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba * 100,
                title={'text': "Churn Risk (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "red" if proba >= 0.6 else "orange" if proba >= 0.3 else "green"},
                    'steps': [
                        {'range': [0, 30], 'color': "#d7f7d9"},
                        {'range': [30, 60], 'color': "#fff4c2"},
                        {'range': [60, 100], 'color': "#ffd6d6"},
                    ]
                }
            ))
            st.plotly_chart(fig, use_container_width=True)

        with r2:
            st.markdown(
                """
                <div style='padding-right:25px;'>
                """,
                unsafe_allow_html=True
            )

            st.subheader("Result")
            st.metric("Churn Probability", f"{proba * 100:.1f}%")

            # Assign color-coded risk bucket
            if proba < 0.3:
                bucket = "Low Risk"
                color = "#00b300"   # green
            elif proba < 0.6:
                bucket = "Medium Risk"
                color = "#ff9900"   # orange
            else:
                bucket = "High Risk"
                color = "#e60000"   # red

            # Color-coded card with subtle background
            st.markdown(
                f"""
                <div style='
                    background-color:{color}20;
                    padding:12px 16px;
                    border-left:6px solid {color};
                    border-radius:8px;
                    margin-top:10px;
                    margin-bottom:15px;
                '>
                    <b style='color:{color}; font-size:18px;'>Risk Level: {bucket}</b>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                "**Prediction:** " +
                ("🚨 <b style='color:#e60000;'>Churn (1)</b>"
                if pred == 1 else "✅ <b style='color:#00b300;'>Stay (0)</b>"),
                unsafe_allow_html=True
            )

            st.markdown("</div>", unsafe_allow_html=True)



# ==========================================================
#   PAGE 2 — WHAT IF ANALYSIS
# ==========================================================
elif page == "What-If Analysis":
    st.subheader("Scenario Simulation (What-If)")
    st.caption("Tweak inputs to see how churn probability moves.")
    colA, colB, colC = st.columns(3)
    with colA:
        CreditScore = st.slider("Credit Score", 300, 900, 650, step=10)
        Age = st.slider("Age", 18, 85, 40)
        Geography_SL = st.selectbox("Region (Sri Lanka)", sl_regions)
        Gender = st.selectbox("Gender", ["Male","Female"])
    with colB:
        Tenure = st.slider("Tenure (years)", 0, 30, 5)
        NumOfProducts = st.slider("Products Count", 1, 6, 2)
        HasCrCard = st.selectbox("Has Credit Card?", [0,1], index=1)
        IsActiveMember = st.selectbox("Active Customer?", [0,1], index=1)
    with colC:
        Balance = st.slider("Account Balance (LKR)", 0, 5_000_000, 600_000, step=50_000)
        EstimatedSalary = st.slider("Monthly Income (LKR)", 0, 1_000_000, 150_000, step=10_000)

    X = build_single_row(CreditScore, Geography_SL, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary)
    proba, pred = predict_one(X)

    r1, r2 = st.columns([1.3,1])
    with r1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=proba*100,
            title={'text': "Churn Risk (%)"},
            gauge={'axis': {'range': [0,100]},
                   'bar': {'color': "red" if proba>=0.6 else "orange" if proba>=0.3 else "green"}}
        ))
        st.plotly_chart(fig, use_container_width=True)
    with r2:
        st.metric("Churn Probability", f"{proba*100:.1f}%")
        bucket, color = risk_bucket(proba)
        st.markdown(f"**Risk Level:** <span style='color:{color}'>{bucket}</span>", unsafe_allow_html=True)
        st.markdown("**Prediction:** " + ("🚨 **Churn (1)**" if pred==1 else "✅ **Stay (0)**"))



# -----------------------------
# Batch Prediction
# -----------------------------
elif page == "Batch Prediction":
    st.subheader("Batch Prediction (CSV Upload)")
    st.caption("Upload a CSV with columns: "
               "`CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary` "
               "OR with `Geography_SL` (we’ll map to dataset geography).")

    sample = pd.DataFrame([{
        "CreditScore": 650, "Geography_SL": "Colombo", "Gender":"Male", "Age":40, "Tenure":5,
        "Balance":600000, "NumOfProducts":2, "HasCrCard":1, "IsActiveMember":1, "EstimatedSalary":150000
    }])
    with st.expander("Download CSV Template"):
        st.dataframe(sample, use_container_width=True)
        csv_buf = io.StringIO()
        sample.to_csv(csv_buf, index=False)
        st.download_button("Download template CSV", csv_buf.getvalue(), "churn_batch_template.csv", "text/csv")

    up = st.file_uploader("Upload CSV", type=["csv"])
    if up is not None:
        try:
            df_in = pd.read_csv(up)
            # Allow Geography_SL as convenience; map to Geography
            if "Geography_SL" in df_in.columns:
                df_in["Geography"] = df_in["Geography_SL"].map(lambda r: region_map.get(r, r) if region_map else r)

            missing = [c for c in FEATURES_ORDER if c not in df_in.columns]
            if missing:
                st.error(f"❌ Missing columns: {missing}")
            else:
                # Scale Balance & Salary
                df_proc = df_in.copy()
                # Safely scale each row
                b_vals = []
                s_vals = []
                for _, row in df_proc.iterrows():
                    b, s = scale_quantiles(row["Balance"], row["EstimatedSalary"])
                    b_vals.append(b); s_vals.append(s)
                df_proc["Balance"] = b_vals
                df_proc["EstimatedSalary"] = s_vals

                Xb = df_proc[FEATURES_ORDER]
                probas = pipe.predict_proba(Xb)[:,1]
                preds  = (probas >= threshold).astype(int)

                out = df_in.copy()
                out["churn_probability"] = np.round(probas, 6)
                out["prediction"]        = preds
                out["risk_level"]        = pd.cut(probas, bins=[-1,0.3,0.6,1.01], labels=["Low","Medium","High"])

                st.success("✅ Batch predictions ready.")
                st.dataframe(out, use_container_width=True, height=400)

                # Summary chart
                st.markdown("### Risk Distribution")
                fig_hist = px.histogram(out, x="churn_probability", nbins=20, title="Churn Probability Distribution")
                st.plotly_chart(fig_hist, use_container_width=True)

                # Download results
                buf = io.StringIO()
                out.to_csv(buf, index=False)
                st.download_button("Download Results (CSV)", buf.getvalue(), "churn_predictions.csv", "text/csv")

        except Exception as e:
            st.error(f"❌ Could not process CSV: {e}")

# -----------------------------
# About
# -----------------------------
else:
    st.subheader("About this App")
    st.markdown("""
- **Purpose**: Practical, Sri Lanka-aware churn prediction with **explainability** and **scenario analysis**.
- **Model Inputs**:
  - Numeric: `CreditScore, Age, Tenure, Balance (quantile-scaled), NumOfProducts, EstimatedSalary (quantile-scaled)`
  - Binary: `HasCrCard, IsActiveMember`
  - Categorical: `Gender, Geography` (Sri Lankan region auto-mapped to dataset labels)
- **Explainability**: Tries Tree/Linear SHAP; falls back to Kernel (slow) or a quick sensitivity bar.
- **Batch**: Upload CSV → predictions + downloadable results.
- **Threshold**: From `feature_spec.json` → `localization.threshold`.
- **Note**: The gauge and risk buckets are for **decision support**. Always combine with domain expertise.
    """)

    st.markdown("**Tips for better decisions**")
    st.markdown("""
- Use **What-If** to see how policy levers (e.g., tenure offers, product bundles, engagement nudges)
  might reduce risk.
- Track **risk distribution** from Batch → prioritize outreach for the **High** bucket.
- Log real outcomes back to a data store to **re-calibrate the threshold** for Sri Lankan context.
    """)
