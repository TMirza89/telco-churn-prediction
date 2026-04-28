import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
from src.preprocess import preprocess_telco_data

def train_logistic_regression(df):
    X_train, X_test, y_train, y_test, df_test_original, preprocessor = preprocess_telco_data(df)

    model = LogisticRegression(max_iter=2000, solver="liblinear")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\nLogistic Regression Results:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_prob))

    return model, X_test, y_test, df_test_original, preprocessor


def train_random_forest(df):
    X_train, X_test, y_train, y_test, df_test_original, preprocessor = preprocess_telco_data(df)

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\nRandom Forest Results:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_prob))

    return model, X_test, y_test, df_test_original


def train_xgboost(df):
    X_train, X_test, y_train, y_test, df_test_original, preprocessor = preprocess_telco_data(df)

    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\nXGBoost Results:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_prob))

    return model, X_test, y_test, df_test_original

import os
import pickle

def save_models(preprocessor, log_model, rf_model, xgb_model):
    """
    Saves preprocessor and trained models into models/ folder.
    """

    os.makedirs("models", exist_ok=True)

    # Save preprocessor
    with open("models/preprocessor.pkl", "wb") as f:
        pickle.dump(preprocessor, f)

    # Save Logistic Regression
    with open("models/log_model.pkl", "wb") as f:
        pickle.dump(log_model, f)

    # Save Random Forest
    with open("models/rf_model.pkl", "wb") as f:
        pickle.dump(rf_model, f)

    # Save XGBoost
    with open("models/xgb_model.pkl", "wb") as f:
        pickle.dump(xgb_model, f)

    print("\nAll models and preprocessor saved successfully in models/ folder.")


# if __name__ == "__main__":
#     import pandas as pd
#     from preprocess import preprocess_telco_data
#
#     df = pd.read_csv("../data/Telco-Customer-Churn.csv")
#     X_train, X_test, y_train, y_test = preprocess_telco_data(df)
#
#     log_model = train_logistic_regression(df)
#     rf_model = train_random_forest(df)
#     xg_model = train_xgboost(df)
#
#     print(log_model)
#     print(rf_model)
#     print(xg_model)