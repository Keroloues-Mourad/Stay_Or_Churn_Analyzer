import streamlit as st
import joblib
import pandas as pd

model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

st.set_page_config(page_title="Churn Predictor", page_icon="📊", layout="centered")
st.title("📊 Customer Churn Prediction")
st.markdown("Answer the following questions to predict whether the customer is likely to churn or stay.")


def styled_radio(label):
    answer = st.radio(label, ["Yes", "No"], horizontal=True,
                      format_func=lambda x: f"✅ Yes" if x == "Yes" else "❌ No",
                      index=None)
    return 1 if answer == "Yes" else 0 if answer == "No" else st.stop()


tenure = st.slider("Customer Tenure (months)", 0, 72, step=1)
MonthlyCharges = st.slider("Monthly Charges ($)", 0.0, 500.0, step=1.0)
TotalCharges = st.slider("Total Charges ($)", 0.0, 10000.0, step=10.0)


SeniorCitizen = styled_radio("Is the customer a senior citizen?")
gender_Male = styled_radio("Is the customer male?")
Partner_Yes = styled_radio("Does the customer have a partner?")
Dependents_Yes = styled_radio("Does the customer have dependents?")
PhoneService_Yes = styled_radio("Does the customer have phone service?")
MultipleLines_NoPhone = styled_radio("Does the customer have NO phone service?")
MultipleLines_Yes = styled_radio("Does the customer have multiple lines?")
InternetService_Fiber = styled_radio("Is the internet service Fiber optic?")
InternetService_No = styled_radio("Does the customer have NO internet service?")
OnlineSecurity_NoInternet = styled_radio("No internet service for online security?")
OnlineSecurity_Yes = styled_radio("Does the customer have online security?")
OnlineBackup_NoInternet = styled_radio("No internet service for online backup?")
OnlineBackup_Yes = styled_radio("Does the customer have online backup?")
DeviceProtection_NoInternet = styled_radio("No internet service for device protection?")
DeviceProtection_Yes = styled_radio("Does the customer have device protection?")
TechSupport_NoInternet = styled_radio("No internet service for tech support?")
TechSupport_Yes = styled_radio("Does the customer have tech support?")
StreamingTV_NoInternet = styled_radio("No internet service for streaming TV?")
StreamingTV_Yes = styled_radio("Does the customer stream TV?")
StreamingMovies_NoInternet = styled_radio("No internet service for streaming movies?")
StreamingMovies_Yes = styled_radio("Does the customer stream movies?")
Contract_OneYear = styled_radio("Is the contract one year?")
Contract_TwoYear = styled_radio("Is the contract two years?")
PaperlessBilling_Yes = styled_radio("Is paperless billing enabled?")
PaymentMethod_CreditCard = styled_radio("Payment method: Credit card (automatic)?")
PaymentMethod_ElectronicCheck = styled_radio("Payment method: Electronic check?")
PaymentMethod_MailedCheck = styled_radio("Payment method: Mailed check?")

# Arrange input data
input_data = [[
    SeniorCitizen, tenure, MonthlyCharges, TotalCharges,
    gender_Male, Partner_Yes, Dependents_Yes, PhoneService_Yes,
    MultipleLines_NoPhone, MultipleLines_Yes,
    InternetService_Fiber, InternetService_No,
    OnlineSecurity_NoInternet, OnlineSecurity_Yes,
    OnlineBackup_NoInternet, OnlineBackup_Yes,
    DeviceProtection_NoInternet, DeviceProtection_Yes,
    TechSupport_NoInternet, TechSupport_Yes,
    StreamingTV_NoInternet, StreamingTV_Yes,
    StreamingMovies_NoInternet, StreamingMovies_Yes,
    Contract_OneYear, Contract_TwoYear,
    PaperlessBilling_Yes,
    PaymentMethod_CreditCard, PaymentMethod_ElectronicCheck, PaymentMethod_MailedCheck
]]


input_df = pd.DataFrame(input_data, columns=columns)
scaled_input = scaler.transform(input_df)
thresh = 0.37
proba = model.predict_proba(scaled_input)[0][1]
prediction = (proba >= thresh)


st.markdown("---")
if st.button("🔍 Predict"):
    percentage = round(proba * 100, 2)
    if prediction:
        st.error(f"❌ The customer is **likely to churn**.\n\n**Churn Probability: {percentage}%**")
    else:
        st.success(f"✅ The customer is **likely to stay**.\n\n**Stay Probability: {100 - percentage}%**")
