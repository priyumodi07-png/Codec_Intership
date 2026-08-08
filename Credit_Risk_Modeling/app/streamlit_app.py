"""
Interactive credit scoring demo.

Run locally:
    streamlit run app/streamlit_app.py

Deploy for free at https://streamlit.io/cloud (point it at this file in
your GitHub repo -- no server needed).
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from features import transform_woe_features  # noqa: E402
from scorecard import risk_band  # noqa: E402

st.set_page_config(page_title="Credit Risk Scoring Demo", page_icon="💳", layout="centered")

st.title("💳 Credit Risk Scoring Model")
st.caption(
    "Enter applicant details below to see a predicted default probability, "
    "credit score, and the top factors driving the decision. "
    "Trained on a synthetic dataset mirroring real credit-bureau data structure."
)


@st.cache_resource
def load_artifacts():
    with open(ROOT / "outputs" / "lr_model.pkl", "rb") as f:
        lr_bundle = pickle.load(f)
    return lr_bundle


bundle = load_artifacts()
lr, encoders, scaler = bundle["model"], bundle["encoders"], bundle["scaler"]

st.subheader("Applicant Information")

col1, col2 = st.columns(2)
with col1:
    age = st.slider("Age", 21, 70, 35)
    income = st.number_input("Annual Income ($)", 8000, 300000, 45000, step=1000)
    credit_amount = st.number_input("Requested Credit Amount ($)", 500, 80000, 15000, step=500)
    duration = st.selectbox("Duration (months)", [6, 12, 18, 24, 36, 48, 60], index=3)
    employment_years = st.slider("Years Employed", 0.0, 40.0, 5.0)
    delinquencies = st.slider("Delinquencies (last 2 years)", 0, 5, 0)

with col2:
    checking = st.selectbox("Checking Account Status",
                             ["no_account", "negative", "0_to_200", "200_plus"])
    savings = st.selectbox("Savings Account Status",
                            ["unknown", "low", "medium", "high"])
    purpose = st.selectbox("Loan Purpose",
                            ["car", "furniture", "electronics", "business", "education", "renovation"])
    housing = st.selectbox("Housing", ["own", "rent", "with_parents"])
    num_dependents = st.slider("Number of Dependents", 0, 5, 0)
    existing_credits = st.slider("Existing Credit Lines", 0, 5, 1)

debt_to_income = credit_amount / (income + 1)
credit_utilization = st.slider("Estimated Credit Utilization", 0.0, 1.0, 0.3)

if st.button("Score Applicant", type="primary"):
    row = pd.DataFrame([{
        "AGE": age, "AMT_INCOME_TOTAL": income, "AMT_CREDIT": credit_amount,
        "DURATION_MONTHS": duration, "EMPLOYMENT_YEARS": employment_years,
        "CHECKING_STATUS": checking, "SAVINGS_STATUS": savings, "PURPOSE": purpose,
        "HOUSING": housing, "NUM_DEPENDENTS": num_dependents,
        "EXISTING_CREDITS": existing_credits, "DELINQUENCIES_2YR": delinquencies,
        "DEBT_TO_INCOME": debt_to_income, "CREDIT_UTILIZATION": credit_utilization,
    }])

    woe_row = transform_woe_features(row, encoders)
    prob_default = lr.predict_proba(woe_row)[:, 1][0]
    score = int(scaler.prob_to_score(np.array([prob_default]))[0])
    band = risk_band(score)

    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("Credit Score", score)
    m2.metric("Risk Band", band)
    m3.metric("Est. Default Probability", f"{prob_default:.1%}")

    band_colors = {
        "Excellent": "🟢", "Good": "🟢", "Fair": "🟡", "Poor": "🟠", "Very Poor": "🔴"
    }
    st.write(f"{band_colors.get(band, '')} This applicant falls into the **{band}** risk band.")

    st.subheader("Top factors influencing this score")
    woe_contribs = (woe_row.iloc[0] * lr.coef_[0]).sort_values()
    top = pd.concat([woe_contribs.head(3), woe_contribs.tail(3)])
    contrib_df = pd.DataFrame({
        "Feature": [c.replace("WOE_", "") for c in top.index],
        "Impact on log-odds of default": top.values,
    }).sort_values("Impact on log-odds of default")
    st.bar_chart(contrib_df.set_index("Feature"))
    st.caption(
        "Negative bars push the score up (lower risk); positive bars push it down (higher risk)."
    )

st.divider()
st.caption(
    "⚠️ This is a portfolio demo trained on synthetic data — not a real credit decisioning tool."
)
