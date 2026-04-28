import pandas as pd

def load_dataset(path):
    """
    Loads the churn dataset from a CSV file.
    """
    df = pd.read_csv(path)
    print("Dataset loaded successfully.")
    print("Shape:", df.shape)
    return df

# if __name__ == "__main__":
#     df = load_dataset("../data/Telco-Customer-Churn.csv")
#     df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
#     df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
#     df['TotalCharges'] = df['TotalCharges'].fillna(0)
#     print(df.head())
#     print(df.info())
#     print(df.describe())
#     print(df.isnull().sum())
