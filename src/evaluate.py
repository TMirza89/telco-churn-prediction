from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from src.utils import save_plot, save_json, save_csv, create_results_folders, timestamp
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, X_test, y_test, original_test_df, model_name="Model"):
    create_results_folders()

    preds = model.predict(X_test)

    # -----------------------------
    # 1. Save Metrics
    # -----------------------------
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "classification_report": classification_report(y_test, preds, output_dict=True)
    }
    save_json(metrics, f"{model_name}_metrics_{timestamp()}.json")

    print("Accuracy:", metrics["accuracy"])
    print("\nClassification Report:\n", classification_report(y_test, preds))

    # -----------------------------
    # 2. Confusion Matrix (save + show)
    # -----------------------------
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{model_name} — Confusion Matrix")
    fig = plt.gcf()
    save_plot(fig, f"{model_name}_confusion_matrix_{timestamp()}.png", "../results/confusion_matrices")

    print("\nConfusion Matrix:\n", cm)

    # -----------------------------
    # 3. Revenue at Risk
    # -----------------------------
    monthly_risk, ltv_risk = revenue_at_risk(model, X_test, original_test_df)
    print(f"\nMonthly Revenue at Risk: ${monthly_risk:,.2f}")
    print(f"Lifetime Value at Risk: ${ltv_risk:,.2f}")

    save_json(
        {"monthly_risk": monthly_risk, "ltv_risk": ltv_risk},
        f"{model_name}_revenue_risk_{timestamp()}.json"
    )

    # -----------------------------
    # 4. Top High-Risk Customers
    # -----------------------------
    top_customers = top_high_risk_customers(model, X_test, original_test_df)
    print("\nTop 10 High-Risk Customers:")
    print(top_customers)

    save_csv(
        top_customers,
        f"high_risk_customers/{model_name}_top10_{timestamp()}.csv",
        "../results"
    )

    # -----------------------------
    # 5. Threshold Optimization
    # -----------------------------
    print("\nGenerating Threshold Optimization Curve...")
    df_thresholds = threshold_optimization_curve(model, X_test, y_test, original_test_df, model_name=model_name)

    best_row = df_thresholds.loc[df_thresholds["net_revenue_saved"].idxmax()]
    print(f"\nBest Threshold for Net Revenue Saved: {best_row['threshold']:.2f}")
    print(f"Net Revenue Saved: ${best_row['net_revenue_saved']:,.2f}")

    save_csv(
        df_thresholds,
        f"threshold_optimization/{model_name}_threshold_table_{timestamp()}.csv",
        "../results"
    )


def revenue_at_risk(model, X_test, original_test_df, threshold=0.5):
    """
    Compute monthly revenue at risk and lifetime value at risk.
    Requires the original (non-encoded) test dataframe.
    """

    # Predict churn probabilities
    churn_prob = model.predict_proba(X_test)[:, 1]

    # Convert probabilities to churn predictions
    churn_pred = (churn_prob >= threshold).astype(int)

    # Attach predictions back to original test data
    df_eval = original_test_df.copy()
    df_eval["PredictedChurn"] = churn_pred
    df_eval["ChurnProb"] = churn_prob

    # Monthly revenue at risk
    monthly_risk = df_eval.loc[df_eval["PredictedChurn"] == 1, "MonthlyCharges"].sum()

    # Lifetime value at risk
    ltv_risk = df_eval.loc[df_eval["PredictedChurn"] == 1, "LTV"].sum()

    return monthly_risk, ltv_risk

def top_high_risk_customers(model, X_test, original_test_df, top_n=10):
    """
    Return a dataframe of the top N highest-risk customers based on predicted churn probability.
    Includes MonthlyCharges, LTV, and engineered risk features.
    """

    # Predict churn probabilities
    churn_prob = model.predict_proba(X_test)[:, 1]

    # Attach predictions to original test data
    df_eval = original_test_df.copy()
    df_eval["ChurnProb"] = churn_prob

    # Sort by highest churn probability
    df_sorted = df_eval.sort_values(by="ChurnProb", ascending=False)

    # Select top N
    top_customers = df_sorted.head(top_n)

    # Select useful columns for business insight
    cols_to_show = [
        "ChurnProb",
        "MonthlyCharges",
        "LTV",
        "tenure",
        "Contract",
        "InternetService",
        "TechSupport",
        "Fiber_NoTechSupport",
        "PaymentMethod",
        "ContractRiskScore",
        "BundleDepth",
        "ProtectionScore",
        "EntertainmentScore"
    ]

    # Only keep columns that exist (safe for future changes)
    cols_to_show = [c for c in cols_to_show if c in top_customers.columns]

    return top_customers[cols_to_show]

import numpy as np
import matplotlib.pyplot as plt

def threshold_optimization_curve(model, X_test, y_test, original_test_df, model_name="Model", retention_cost=10):
    """
    Computes precision, recall, F1, revenue saved, and net revenue saved
    for thresholds 0–1. Includes retention cost for false positives.
    """

    churn_prob = model.predict_proba(X_test)[:, 1]
    monthly_charges = original_test_df["MonthlyCharges"].values

    thresholds = np.linspace(0, 1, 101)
    precision_list = []
    recall_list = []
    f1_list = []
    revenue_saved_list = []
    net_revenue_saved_list = []

    for t in thresholds:
        preds = (churn_prob >= t).astype(int)

        TP = np.sum((preds == 1) & (y_test == 1))
        FP = np.sum((preds == 1) & (y_test == 0))
        FN = np.sum((preds == 0) & (y_test == 1))

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # Revenue saved = monthly charges of correctly predicted churners
        revenue_saved = monthly_charges[(preds == 1) & (y_test == 1)].sum()

        # Net revenue saved = revenue saved - cost of contacting false positives
        net_revenue_saved = revenue_saved - (FP * retention_cost)

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        revenue_saved_list.append(revenue_saved)
        net_revenue_saved_list.append(net_revenue_saved)

    # ---- PLOT 1: Precision, Recall, F1 ----
    fig1 = plt.figure(figsize=(12, 6))
    plt.plot(thresholds, precision_list, label="Precision", color="blue")
    plt.plot(thresholds, recall_list, label="Recall", color="orange")
    plt.plot(thresholds, f1_list, label="F1 Score", color="green")
    plt.xlabel("Threshold")
    plt.ylabel("Metric Value")
    plt.title(f"{model_name} — Precision, Recall, and F1 vs Threshold")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    save_plot(fig1, f"{model_name}_precision_recall_f1_{timestamp()}.png", "../results/plots")

    # ---- PLOT 2: Net Revenue Saved ----
    fig2 = plt.figure(figsize=(12, 6))
    plt.plot(thresholds, net_revenue_saved_list, label="Net Revenue Saved ($)", color="red")
    plt.xlabel("Threshold")
    plt.ylabel("Net Revenue Saved ($)")
    plt.title(f"{model_name} — Net Revenue Saved vs Threshold")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    save_plot(fig2, f"{model_name}_net_revenue_saved_{timestamp()}.png", "../results/plots")

    # Return results as a dataframe
    import pandas as pd
    df_results = pd.DataFrame({
        "threshold": thresholds,
        "precision": precision_list,
        "recall": recall_list,
        "f1": f1_list,
        "revenue_saved": revenue_saved_list,
        "net_revenue_saved": net_revenue_saved_list
    })

    return df_results




if __name__ == "__main__":
    import pandas as pd
    from preprocess import preprocess_telco_data
    from train_models import (
        train_logistic_regression,
        train_random_forest,
        train_xgboost
    )

    df = pd.read_csv("../data/Telco-Customer-Churn.csv")

    print("\n--- Logistic Regression ---")
    log_model, X_test, y_test, df_test_original = train_logistic_regression(df)
    evaluate_model(log_model, X_test, y_test, df_test_original, model_name="LogesticRegression")

    print("\n--- Random Forest ---")
    rf_model, X_test, y_test, df_test_original = train_random_forest(df)
    evaluate_model(rf_model, X_test, y_test, df_test_original, model_name="RandomForest")

    print("\n--- XGBoost ---")
    xgb_model, X_test, y_test, df_test_original = train_xgboost(df)
    evaluate_model(xgb_model, X_test, y_test, df_test_original, model_name="XGBoost")
