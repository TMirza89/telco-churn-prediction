import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

def preprocess_telco_data(df, test_size=0.2, random_state=42):
    df = df.copy()

    # -----------------------------
    # Basic Cleaning
    # -----------------------------
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna(subset=['TotalCharges'])

    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])

    df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

    # -----------------------------
    # Feature Engineering
    # -----------------------------
    df['LTV'] = df['MonthlyCharges'] * df['tenure']
    df['PriceToTenureRatio'] = df['MonthlyCharges'] / (df['tenure'] + 1)

    payment_map = {
        'Electronic check': 3,
        'Mailed check': 2,
        'Bank transfer (automatic)': 1,
        'Credit card (automatic)': 1
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

    contract_map = {
        'Month-to-month': 3,
        'One year': 2,
        'Two year': 1
    }
    df['ContractRiskScore'] = df['Contract'].map(contract_map)

    df['PaperlessFlag'] = (df['PaperlessBilling'] == 'Yes').astype(int)

    df['Fiber_NoTechSupport'] = (
        (df['InternetService'] == 'Fiber optic') &
        (df['TechSupport'] == 'No')
    ).astype(int)

    internet_map = {'DSL': 1, 'Fiber optic': 2, 'No': 0}
    df['InternetChargeInteraction'] = df['MonthlyCharges'] * df['InternetService'].map(internet_map)

    # Save original engineered dataframe BEFORE encoding
    df_original = df.copy()

    # -----------------------------
    # Split Features and Target
    # -----------------------------
    X = df.drop(columns=['Churn'])
    y = df['Churn']

    # Identify categorical columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # -----------------------------
    # Build Preprocessor
    # -----------------------------
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
        ],
        remainder='passthrough'
    )

    # -----------------------------
    # Train-test split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # Fit on train, transform both
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Extract original test rows
    df_test_original = df_original.loc[X_test.index].copy()

    return X_train_processed, X_test_processed, y_train, y_test, df_test_original, preprocessor


# if __name__ == "__main__":
#     df = pd.read_csv("../data/Telco-Customer-Churn.csv")
#     X_train, X_test, y_train, y_test, df_test_original, preprocessor = preprocess_telco_data(df)
#     print("Train shape:", X_train.shape)
#     print("Test shape:", X_test.shape)
#     print("Original Test DF shape:", df_test_original.shape)




