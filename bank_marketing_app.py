
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained model and encoders
model = joblib.load('random_forest_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')  # Dictionary of label encoders for categorical columns

st.set_page_config(page_title="Bank Term Deposit Subscription Predictor", layout="centered")

st.title("💰 Bank Marketing Prediction App")
st.markdown("Predict whether a customer will subscribe to a **term deposit** based on input features.")

st.sidebar.header("Customer Information")

# Input fields
age = st.sidebar.slider("Age", 18, 95, 30)
job = st.sidebar.selectbox("Job", ['admin.', 'blue-collar', 'entrepreneur', 'housemaid',
                                   'management', 'retired', 'self-employed', 'services',
                                   'student', 'technician', 'unemployed', 'unknown'])
marital = st.sidebar.selectbox("Marital Status", ['married', 'single', 'divorced'])
education = st.sidebar.selectbox("Education", ['primary', 'secondary', 'tertiary', 'unknown'])
default = st.sidebar.selectbox("Has Credit in Default?", ['no', 'yes'])
housing = st.sidebar.selectbox("Has Housing Loan?", ['no', 'yes'])
loan = st.sidebar.selectbox("Has Personal Loan?", ['no', 'yes'])
contact = st.sidebar.selectbox("Contact Communication Type", ['cellular', 'telephone'])
month = st.sidebar.selectbox("Last Contact Month", ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                                                    'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
day = st.sidebar.slider("Last Contact Day of Month", 1, 31, 15)
duration = st.sidebar.slider("Call Duration (in seconds)", 0, 3000, 300)
campaign = st.sidebar.slider("Number of Contacts During Campaign", 1, 50, 1)
pdays = st.sidebar.slider("Days Since Last Contact", -1, 999, -1)
previous = st.sidebar.slider("Previous Contacts", 0, 10, 0)
poutcome = st.sidebar.selectbox("Outcome of Previous Campaign", ['failure', 'other', 'success', 'unknown'])
balance = st.sidebar.number_input("Account Balance", min_value=-5000, max_value=100000, value=1000)

# Create input DataFrame
input_dict = {
    'age': age, 'job': job, 'marital': marital, 'education': education, 'default': default,
    'balance': balance, 'housing': housing, 'loan': loan, 'contact': contact, 'day': day,
    'month': month, 'duration': duration, 'campaign': campaign, 'pdays': pdays,
    'previous': previous, 'poutcome': poutcome
}

input_df = pd.DataFrame([input_dict])

# Apply label encoding to categorical variables
for col in input_df.select_dtypes(include='object').columns:
    if col in label_encoders:
        input_df[col] = label_encoders[col].transform(input_df[col])
    else:
        st.error(f"Missing label encoder for: {col}")

# Predict
if st.button("Predict Subscription"):
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]

    st.subheader("🔍 Prediction Result:")
    if prediction == 1:
        st.success(f"The customer is likely to **subscribe** to the term deposit. ✅ (Confidence: {proba:.2%})")
    else:
        st.warning(f"The customer is **not likely** to subscribe. ❌ (Confidence: {proba:.2%})")

# Optional: Feature Importance Plot
if st.checkbox("Show Feature Importances"):
    importances = model.feature_importances_
    features = input_df.columns
    feature_df = pd.DataFrame({'Feature': features, 'Importance': importances})
    feature_df = feature_df.sort_values(by='Importance', ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_df, ax=ax)
    ax.set_title("Feature Importance (Random Forest)")
    st.pyplot(fig)
