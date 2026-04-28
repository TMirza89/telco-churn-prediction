import os

import streamlit as st
import pandas as pd
import numpy as np
import shap
import pickle
import matplotlib.pyplot as plt

st.set_page_config(page_title="Telco Churn Prediction", layout="wide")
def get_feature_names(preprocessor):
    try:
        ohe = preprocessor.named_transformers_['cat']
        ohe_features = ohe.get_feature_names_out()
    except:
        ohe_features = []

    # Remainder columns (numeric + engineered)
    remainder = preprocessor.transformers_[1][2]

    return list(ohe_features) + list(remainder)


def engineer_features(df):
    df = df.copy()

    df['LTV'] = df['MonthlyCharges'] * df['tenure']
    df['PriceToTenureRatio'] = df['MonthlyCharges'] / (df['tenure'] + 1)

    payment_map = {
        'Electronic check': 3,
        'Mailed check': 2,
        'Bank transfer': 1,
        'Credit card': 1
    }
    df['PaymentRiskScore'] = df['PaymentMethod'].map(payment_map)

    service_cols = [
        'PhoneService', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    df['BundleDepth'] = df[service_cols].apply(lambda row: sum(row == 'Yes'), axis=1)

    df['ProtectionScore'] = df[['OnlineSecurity', 'DeviceProtection', 'TechSupport']].apply(
        lambda row: sum(row == 'Yes'), axis=1
    )

    df['EntertainmentScore'] = df[['StreamingTV', 'StreamingMovies']].apply(
        lambda row: sum(row == 'Yes'), axis=1
    )

    df['TenureBucket'] = pd.cut(
        df['tenure'],
        bins=[0, 6, 12, 24, 48, 100],
        labels=['0-6', '6-12', '12-24', '24-48', '48+']
    )

    contract_map = {'Month-to-month': 3, 'One year': 2, 'Two year': 1}
    df['ContractRiskScore'] = df['Contract'].map(contract_map)

    df['PaperlessFlag'] = (df['PaperlessBilling'] == 'Yes').astype(int)

    df['Fiber_NoTechSupport'] = (
        (df['InternetService'] == 'Fiber optic') &
        (df['TechSupport'] == 'No')
    ).astype(int)

    internet_map = {'DSL': 1, 'Fiber optic': 2, 'None': 0}
    df['InternetChargeInteraction'] = df['MonthlyCharges'] * df['InternetService'].map(internet_map)

    return df

# ---------------------------------------------------------
# Load Models + Preprocessor
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    with open("models/log_model.pkl", "rb") as f:
        log_model = pickle.load(f)
    with open("models/xgb_model.pkl", "rb") as f:
        xgb_model = pickle.load(f)
    with open("models/preprocessor.pkl", "rb") as f:
        preprocessor = pickle.load(f)
    return log_model, xgb_model, preprocessor

log_model, xgb_model, preprocessor = load_artifacts()

model_dict = {
    "Logistic Regression": log_model,
    "XGBoost": xgb_model
}

# ---------------------------------------------------------
# Sidebar Inputs
# ---------------------------------------------------------
st.sidebar.header("Customer Information")

model_choice = st.sidebar.selectbox("Select Model", list(model_dict.keys()))

# Basic features
tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.sidebar.slider("Monthly Charges", 10, 150, 70)
total_charges = st.sidebar.slider("Total Charges", 0, 10000, 2000)

contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "None"])
tech_support = st.sidebar.selectbox("Tech Support", ["Yes", "No"])
payment = st.sidebar.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.sidebar.selectbox("Multiple Lines", ["Yes", "No"])
online_security = st.sidebar.selectbox("Online Security", ["Yes", "No"])
online_backup = st.sidebar.selectbox("Online Backup", ["Yes", "No"])
device_protection = st.sidebar.selectbox("Device Protection", ["Yes", "No"])
streaming_tv = st.sidebar.selectbox("Streaming TV", ["Yes", "No"])
streaming_movies = st.sidebar.selectbox("Streaming Movies", ["Yes", "No"])
paperless_billing = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
senior_citizen = st.sidebar.selectbox("Senior Citizen", [0, 1])
partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])


# ---------------------------------------------------------
# Input DataFrame
# ---------------------------------------------------------
input_df = pd.DataFrame({
    "tenure": [tenure],
    "MonthlyCharges": [monthly_charges],
    "TotalCharges": [total_charges],
    "Contract": [contract],
    "InternetService": [internet],
    "TechSupport": [tech_support],
    "PaymentMethod": [payment],
    "PhoneService": [phone_service],
    "MultipleLines": [multiple_lines],
    "OnlineSecurity": [online_security],
    "OnlineBackup": [online_backup],
    "DeviceProtection": [device_protection],
    "StreamingTV": [streaming_tv],
    "StreamingMovies": [streaming_movies],
    "PaperlessBilling": [paperless_billing],
    "SeniorCitizen": [senior_citizen],
    "Partner": [partner],
    "Dependents": [dependents],
    "gender": [gender]
})


# ---------------------------------------------------------
# Tabs for clean UI
# ---------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["Prediction", "Explainability", "Business Impact", "High-Risk Customers"])

# ---------------------------------------------------------
# 1. Prediction Tab
# ---------------------------------------------------------
with tab1:
    st.subheader("📄 Input Data")
    st.write(input_df)

    if st.button("Predict Churn"):
        model = model_dict[model_choice]

        engineered_df = engineer_features(input_df)
        X_processed = preprocessor.transform(engineered_df)

        churn_prob = model.predict_proba(X_processed)[0][1]

        st.subheader("🔮 Prediction Result")
        st.metric("Churn Probability", f"{churn_prob:.2%}")

        if churn_prob > 0.6:
            st.error("⚠️ High Risk of Churn")
        elif churn_prob > 0.3:
            st.warning("🟠 Medium Risk of Churn")
        else:
            st.success("🟢 Low Risk of Churn")

# ---------------------------------------------------------
# 2. Explainability Tab
# ---------------------------------------------------------
with tab2:
    st.subheader("🧠 SHAP Explainability")

    model = model_dict[model_choice]

    # Engineer features
    engineered_df = engineer_features(input_df)
    X_processed = preprocessor.transform(engineered_df)

    # Choose correct SHAP explainer
    if model_choice == "XGBoost":
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_processed)
    else:
        # Logistic Regression → use LinearExplainer
        # Use a small background dataset (first 100 rows)
        try:
            background = preprocessor.transform(
                pd.read_csv("data/Telco-Customer-Churn.csv").head(100)
            )
        except:
            background = np.zeros((1, X_processed.shape[1]))

        explainer = shap.LinearExplainer(model, background)
        shap_values = explainer.shap_values(X_processed)

    # Summary Plot
    st.write("### Global Feature Importance")
    fig_summary = plt.figure(figsize=(10, 6))
    feature_names = get_feature_names(preprocessor)
    X_display = pd.DataFrame(X_processed, columns=feature_names)
    shap.summary_plot(shap_values, X_display, feature_names=feature_names, show=False)

    st.pyplot(fig_summary)

    # Waterfall Plot
    st.write("### Individual Prediction")
    feature_names = get_feature_names(preprocessor)

    fig_waterfall = plt.figure(figsize=(10, 6))
    shap.plots._waterfall.waterfall_legacy(
        explainer.expected_value,
        shap_values[0],
        feature_names=feature_names
    )
    st.pyplot(fig_waterfall)

    # Force Plot (HTML)
    st.write("### Force Plot (Interactive)")
    X_display = pd.DataFrame(X_processed, columns=feature_names)
    force_plot = shap.force_plot(
        explainer.expected_value,
        shap_values[0],
        X_display.iloc[0, :]
    )

    html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
    st.components.v1.html(html, height=300)

# ---------------------------------------------------------
# 3. Business Impact Tab
# ---------------------------------------------------------
with tab3:
    st.subheader("💰 Revenue Impact Calculator")

    retention_cost = st.number_input("Retention Cost per Customer ($)", 0, 200, 10)

    model = model_dict[model_choice]
    engineered_df = engineer_features(input_df)
    X_processed = preprocessor.transform(engineered_df)
    churn_prob = model.predict_proba(X_processed)[0][1]

    revenue_at_risk = monthly_charges
    net_revenue = revenue_at_risk - retention_cost if churn_prob > 0.5 else 0

    st.write(f"**Revenue at Risk:** ${revenue_at_risk:.2f}")
    st.write(f"**Net Revenue Saved (if retained):** ${net_revenue:.2f}")

# ---------------------------------------------------------
# 4. High-Risk Customers Tab
# ---------------------------------------------------------
with tab4:
    st.subheader("📁 High-Risk Customer Explorer")
    risk_file_map = {
        "Logistic Regression": "LogesticRegression_top10_20260423_123951.csv",
        "XGBoost": "XGBoost_top10_20260423_123958.csv"  # adjust if needed
    }
    filename = risk_file_map[model_choice]
    filepath = f"results/high_risk_customers/{filename}"

    if os.path.exists(filepath):
        df_risk = pd.read_csv(filepath)
        st.dataframe(df_risk)
    else:
        st.warning(f"No high-risk customer file found for {model_choice}.")

    # try:
    #     df_risk = pd.read_csv(f"results/high_risk_customers/{model_choice}_top_customers.csv")
    #     st.dataframe(df_risk)
    # except:
    #     st.info("No high-risk customer file found for this model.")