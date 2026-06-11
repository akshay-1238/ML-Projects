import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="centered",
)

# ── Styles ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.hero {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 16px;
    padding: 2.5rem 2rem;
    margin-bottom: 2rem;
    text-align: center;
}
.hero h1 {
    color: #e2e8f0;
    font-size: 2rem;
    font-weight: 700;
    margin: 0 0 0.5rem 0;
}
.hero p {
    color: #94a3b8;
    font-size: 1rem;
    margin: 0;
}

.model-card {
    border: 1.5px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 0.75rem;
    cursor: pointer;
    transition: border-color 0.2s;
}
.model-card.selected {
    border-color: #3b82f6;
    background: #eff6ff;
}

.result-approved {
    background: #f0fdf4;
    border: 2px solid #22c55e;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
}
.result-rejected {
    background: #fef2f2;
    border: 2px solid #ef4444;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
}
.result-approved h2 { color: #16a34a; margin: 0; }
.result-rejected h2 { color: #dc2626; margin: 0; }
.result-approved p, .result-rejected p { color: #64748b; margin: 0.5rem 0 0 0; font-size: 0.9rem; }

.section-label {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #94a3b8;
    margin: 1.5rem 0 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

# ── Load models ───────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
@st.cache_resource
def load_models():
    with open(os.path.join(BASE_DIR, 'loan_model_1.pkl'), 'rb') as f:
        m1 = pickle.load(f)
    with open(os.path.join(BASE_DIR, 'loan_model_2.pkl'), 'rb') as f:
        m2 = pickle.load(f)
    return m1, m2

try:
    model_1, model_2 = load_models()
except FileNotFoundError:
    st.error("Model files not found. Make sure `loan_model_1.pkl` and `loan_model_2.pkl` are in the same folder as this app.")
    st.stop()

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🏦 Loan Approval Predictor</h1>
    <p>Enter your details below to check your loan approval chances.</p>
</div>
""", unsafe_allow_html=True)

# ── Model selection ───────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Choose a model</div>', unsafe_allow_html=True)

model_choice = st.radio(
    label="Model",
    options=["Quick Check (3 inputs)", "Detailed Check (12 inputs)"],
    label_visibility="collapsed",
    horizontal=True,
)

if model_choice == "Quick Check (3 inputs)":
    st.caption("Uses the 3 most statistically significant features. Faster, fewer inputs.")
else:
    st.caption("Uses all available features. More inputs, more thorough prediction.")

st.divider()

# ── Inputs ────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Your details</div>', unsafe_allow_html=True)

if model_choice == "Quick Check (3 inputs)":
    col1, col2, col3 = st.columns(3)
    with col1:
        credit_history = st.selectbox("Credit History", options=[1, 0], format_func=lambda x: "Good (1)" if x == 1 else "Bad (0)")
    with col2:
        property_semiurban = st.selectbox("Property Area", options=[0, 1], format_func=lambda x: "Semiurban" if x == 1 else "Rural / Urban")
    with col3:
        is_married = st.selectbox("Marital Status", options=[1, 0], format_func=lambda x: "Married" if x == 1 else "Unmarried")

    input_data = pd.DataFrame([{
        'Credit_History': credit_history,
        'Property_Area_Semiurban': property_semiurban,
        'Is_Married': is_married,
    }])
    selected_model = model_1

else:
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Marital Status", ["Yes", "No"])
        dependents = st.selectbox("Dependents", [0, 1, 2, 3])
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self Employed", ["No", "Yes"])
        credit_history = st.selectbox("Credit History", [1, 0], format_func=lambda x: "Good (1)" if x == 1 else "Bad (0)")
    with col2:
        applicant_income = st.number_input("Applicant Income (₹/month)", min_value=0, value=5000, step=500)
        coapplicant_income = st.number_input("Co-applicant Income (₹/month)", min_value=0, value=0, step=500)
        loan_amount = st.number_input("Loan Amount (₹ thousands)", min_value=0, value=100, step=10)
        loan_term = st.selectbox("Loan Term (months)", [360, 180, 120, 84, 60, 36, 12])
        property_area = st.selectbox("Property Area", ["Rural", "Semiurban", "Urban"])

    input_data = pd.DataFrame([{
        'Dependents': dependents,
        'ApplicantIncome': applicant_income,
        'CoapplicantIncome': coapplicant_income,
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': loan_term,
        'Credit_History': credit_history,
        'Is_Male': 1 if gender == "Male" else 0,
        'Is_Married': 1 if married == "Yes" else 0,
        'Is_Education': 1 if education == "Graduate" else 0,
        'Is_Self_Employed': 1 if self_employed == "Yes" else 0,
        'Property_Area_Semiurban': 1 if property_area == "Semiurban" else 0,
        'Property_Area_Urban': 1 if property_area == "Urban" else 0,
    }])
    selected_model = model_2

# ── Predict ───────────────────────────────────────────────────────────────────
st.divider()

if st.button("Check Approval", type="primary", use_container_width=True):
    prediction = selected_model.predict(input_data)[0]
    probability = selected_model.predict_proba(input_data)[0]
    confidence = round(max(probability) * 100, 1)

    if prediction == 1:
        st.markdown(f"""
        <div class="result-approved">
            <h2>✅ Likely Approved</h2>
            <p>Model confidence: {confidence}%</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-rejected">
            <h2>❌ Likely Rejected</h2>
            <p>Model confidence: {confidence}%</p>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.caption("Built by Akshay Balani · Logistic Regression · GTU Diploma CS · [GitHub](https://github.com/akshay-1238/ML-Projects)")
