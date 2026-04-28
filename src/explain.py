import shap
import matplotlib.pyplot as plt
import os
import numpy as np

# Ensure results folder exists
os.makedirs("../results", exist_ok=True)

def explain_model(model, X_test, model_name="Model"):
    """
    Generate SHAP summary and bar plots for any model.
    Works for Logistic Regression, Random Forest, and XGBoost.
    Saves plots into ../results/
    """

    # -----------------------------
    # Choose correct SHAP explainer
    # -----------------------------
    if model_name.lower() in ["logisticregression", "logistic_regression"]:
        explainer = shap.LinearExplainer(model, X_test)
        shap_values = explainer(X_test)
        shap_vals = shap_values.values.astype(float)
        expected_value = explainer.expected_value

    else:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        # For tree models, shap_values is a list → pick class 1
        if isinstance(shap_values, list):
            shap_vals = shap_values[1]
            expected_value = explainer.expected_value[1]
        else:
            shap_vals = shap_values
            expected_value = explainer.expected_value

    # Normalize expected_value to a single float
    if isinstance(expected_value, (list, tuple, np.ndarray)):
        expected_value = expected_value[0]

    print(f"\nGenerating SHAP summary plot for {model_name}...")

    # Summary plot
    plt.figure(figsize=(12, 6))
    shap.summary_plot(shap_vals, X_test, show=False)
    plt.title(f"{model_name} — SHAP Summary Plot")
    plt.tight_layout()
    plt.savefig(f"../results/shap/{model_name}_shap_summary.png")
    plt.close()

    # Bar plot
    plt.figure(figsize=(12, 6))
    shap.summary_plot(shap_vals, X_test, plot_type="bar", show=False)
    plt.title(f"{model_name} — SHAP Feature Importance")
    plt.tight_layout()
    plt.savefig(f"../results/shap/{model_name}_shap_bar.png")
    plt.close()

    print(f"SHAP plots saved for {model_name}.")



if __name__ == "__main__":
    import pandas as pd
    from train_models import (
        train_logistic_regression,
        train_random_forest,
        train_xgboost
    )

    df = pd.read_csv("../data/Telco-Customer-Churn.csv")

    # Logistic Regression
    log_model, X_test, y_test, df_test_original = train_logistic_regression(df)
    explain_model(log_model, X_test, model_name="LogisticRegression")

    # Random Forest
    rf_model, X_test, y_test, df_test_original = train_random_forest(df)
    explain_model(rf_model, X_test, model_name="RandomForest")

    # XGBoost
    xgb_model, X_test, y_test, df_test_original = train_xgboost(df)
    explain_model(xgb_model, X_test, model_name="XGBoost")
