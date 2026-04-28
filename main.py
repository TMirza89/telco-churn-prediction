import pandas as pd
from src.preprocess import preprocess_telco_data
from src.train_models import (
    train_logistic_regression,
    train_random_forest,
    train_xgboost, save_models
)
from src.evaluate import evaluate_model
from src.explain import explain_model

def main():
    # Load dataset
    df = pd.read_csv("data/Telco-Customer-Churn.csv")

    print("\n--- Logistic Regression ---")
    log_model, X_test, y_test, df_test_original, preprocessor = train_logistic_regression(df)
    evaluate_model(log_model, X_test, y_test, df_test_original, model_name="LogisticRegression")
    explain_model(log_model, X_test, model_name="LogisticRegression")

    print("\n--- Random Forest ---")
    rf_model, X_test, y_test, df_test_original = train_random_forest(df)
    evaluate_model(rf_model, X_test, y_test, df_test_original, model_name="RandomForest")
    explain_model(rf_model, X_test, model_name="RandomForest")

    print("\n--- XGBoost ---")
    xgb_model, X_test, y_test, df_test_original = train_xgboost(df)
    evaluate_model(xgb_model, X_test, y_test, df_test_original, model_name="XGBoost")
    explain_model(xgb_model, X_test, model_name="XGBoost")

    # ------------------------------------------------
    # SAVE ALL MODELS + PREPROCESSOR
    # ------------------------------------------------
    save_models(preprocessor, log_model, rf_model, xgb_model)

if __name__ == "__main__":
    main()
