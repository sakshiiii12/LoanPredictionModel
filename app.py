import streamlit as st
import pandas as pd
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Loan Default Predictor",
    page_icon="🏦",
    layout="centered"
)

# ---------------- LOAD MODEL ----------------
model = joblib.load('model/loanData_model.pkl')

# ---------------- HEADER ----------------
st.title("🏦 Loan Default Prediction")
st.write("Enter applicant details to check loan risk.")

st.markdown("---")

# ---------------- INPUT FIELDS ----------------
credit_policy = st.selectbox("Credit Policy Met?", [0, 1])

purpose = st.selectbox(
    "Loan Purpose",
    [
        "debt_consolidation",
        "credit_card",
        "home_improvement",
        "small_business",
        "major_purchase",
        "all_other"
    ]
)

int_rate = st.number_input("Interest Rate", min_value=0.0)
installment = st.number_input("Installment Amount", min_value=0.0)
log_annual_inc = st.number_input("Log Annual Income", min_value=0.0)
dti = st.number_input("Debt-to-Income Ratio", min_value=0.0)
fico = st.number_input("FICO Score", min_value=300, max_value=850)
days_credit_line = st.number_input("Days with Credit Line", min_value=0.0)
revol_bal = st.number_input("Revolving Balance", min_value=0.0)
revol_util = st.number_input("Revolving Utilization (%)", min_value=0.0)
inq_6m = st.number_input("Inquiries in Last 6 Months", min_value=0)
delinq_2yrs = st.number_input("Delinquencies in 2 Years", min_value=0)
pub_rec = st.number_input("Public Records", min_value=0)

st.markdown("---")

# ---------------- PREDICTION ----------------
if st.button("Predict"):

    input_data = pd.DataFrame([{
        "credit.policy": credit_policy,
        "purpose": purpose,
        "int.rate": int_rate,
        "installment": installment,
        "log.annual.inc": log_annual_inc,
        "dti": dti,
        "fico": fico,
        "days.with.cr.line": days_credit_line,
        "revol.bal": revol_bal,
        "revol.util": revol_util,
        "inq.last.6mths": inq_6m,
        "delinq.2yrs": delinq_2yrs,
        "pub.rec": pub_rec
    }])

    prediction = model.predict(input_data)

    if prediction[0] == 0:
        st.success("✅ Loan Likely to be Fully Paid")
    else:
        st.error("⚠ Loan May Not Be Fully Paid")