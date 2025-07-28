# Stay_Or_Churn_Analyzer
Modern website that predicts whether a customer will churn or stay 

💡 Customer Churn Predictor

This is an interactive and modern website designed to predict customer churn for telecom services.
It leverages a logistic regression model trained on preprocessed Telco customer data.

The app allows users to simulate customer profiles by answering intuitive Yes/No questions using a clean visual interface and provides real time predictions in the form of percentage.

---

## 🚀 Features


- 📊 Predicts whether a customer is likely to **churn** or **stay**
- 🎨 Intuitive Yes/No radio buttons with visual cues (no 0/1)
- 🎯 Custom threshold logic for improved recall
- 📈 Displays the result as a **percentage probability**
- ⚙️ Backend model: Logistic Regression (balanced class weights)
- 🧠 Trained on preprocessed Telco customer churn dataset
- 🔒 Ready to deploy on [Streamlit Cloud]

---

## 🗂️ Project Structure

```bash
.
├── app.py                  # Streamlit UI App
├── churn_model.pkl         # Trained Logistic Regression model
├── scaler.pkl              # StandardScaler used during training
├── columns.pkl             # Column order for consistent predictions
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```


## 📬 Try It Online

👉 [Launch the app on Streamlit Cloud](([https://share.streamlit.io/yourusername/customer-churn-predictor/main/app.py](https://stay-or-churn-analyzer-by-keroloues-mourad.streamlit.app/))

---

## 📊 Dataset Info

This model is trained on the "Telco Customer Churn" dataset available via Kaggle.
It includes customer demographics, account info, internet/phone service details and churn labels.

---

## 🧠 Model Summary

- Algorithm: `LogisticRegression`
- Preprocessing: `StandardScaler` + `get_dummies`
- Class balancing: `class_weight='balanced'`
- Custom probability threshold: `0.37` to optimize recall

---

**Created by Keroloues Mourad**
