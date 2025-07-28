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

multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])


has_internet = styled_radio("Does the customer have internet service?")


if has_internet:
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber"])
else:
    internet_service = "No"


if has_internet:
    online_security = st.selectbox("Online Security", ["No", "Yes"])
    online_backup = st.selectbox("Online Backup", ["No", "Yes"])
    device_protection = st.selectbox("Device Protection", ["No", "Yes"])
    tech_support = st.selectbox("Tech Support", ["No", "Yes"])
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes"])
    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes"])
else:
    online_security = "No internet service"
    online_backup = "No internet service"
    device_protection = "No internet service"
    tech_support = "No internet service"
    streaming_tv = "No internet service"
    streaming_movies = "No internet service"

contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
paperless_billing = styled_radio("Is paperless billing enabled?")
payment_method = st.selectbox("Payment Method", [
    "Bank transfer (automatic)",
    "CreditCard",
    "ElectronicCheck",
    "MailedCheck"
])


data_dict = {
    "SeniorCitizen": SeniorCitizen,
    "tenure": tenure,
    "MonthlyCharges": MonthlyCharges,
    "TotalCharges": TotalCharges,
    "gender_Male": gender_Male,
    "Partner_Yes": Partner_Yes,
    "Dependents_Yes": Dependents_Yes,
    "PhoneService_Yes": PhoneService_Yes,

    "MultipleLines_NoPhone": 1 if multiple_lines == "No phone service" else 0,
    "MultipleLines_Yes": 1 if multiple_lines == "Yes" else 0,

    "InternetService_Fiber": 1 if has_internet and internet_service == "Fiber" else 0,
    "InternetService_No": 0 if has_internet else 1,

    "OnlineSecurity_NoInternet": 0 if has_internet else 1,
    "OnlineSecurity_Yes": 1 if online_security == "Yes" else 0,

    "OnlineBackup_NoInternet": 0 if has_internet else 1,
    "OnlineBackup_Yes": 1 if online_backup == "Yes" else 0,

    "DeviceProtection_NoInternet": 0 if has_internet else 1,
    "DeviceProtection_Yes": 1 if device_protection == "Yes" else 0,

    "TechSupport_NoInternet": 0 if has_internet else 1,
    "TechSupport_Yes": 1 if tech_support == "Yes" else 0,

    "StreamingTV_NoInternet": 0 if has_internet else 1,
    "StreamingTV_Yes": 1 if streaming_tv == "Yes" else 0,

    "StreamingMovies_NoInternet": 0 if has_internet else 1,
    "StreamingMovies_Yes": 1 if streaming_movies == "Yes" else 0,

    "Contract_OneYear": 1 if contract == "One year" else 0,
    "Contract_TwoYear": 1 if contract == "Two year" else 0,

    "PaperlessBilling_Yes": paperless_billing,

    "PaymentMethod_CreditCard": 1 if payment_method == "CreditCard" else 0,
    "PaymentMethod_ElectronicCheck": 1 if payment_method == "ElectronicCheck" else 0,
    "PaymentMethod_MailedCheck": 1 if payment_method == "MailedCheck" else 0,
}


input_df = pd.DataFrame([data_dict])
input_df = input_df.reindex(columns=columns, fill_value=0)
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
