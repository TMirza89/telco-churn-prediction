import seaborn as sns
import matplotlib.pyplot as plt


def plot_churn_distribution(df):
    sns.countplot(data=df, x="Churn")
    plt.title("Churn Distribution")
    plt.show()
    df['Churn'].value_counts(normalize=True)


def plot_numeric_distribution(df):
    sns.countplot(data=df, x='gender', hue='Churn')
    plt.title("Gender vs Churn")
    plt.show()

    sns.barplot(data=df, x='SeniorCitizen', y='Churn')
    plt.title("Senior Citizen vs Churn")
    plt.show()

    sns.histplot(data=df, x='tenure', hue='Churn', bins=30)
    plt.title("Tenure Distribution by Churn")
    plt.show()

    sns.countplot(data=df, x='InternetService', hue='Churn')
    plt.title("Internet Service vs Churn")
    plt.show()

    for col in ['OnlineSecurity', 'TechSupport', 'OnlineBackup', 'DeviceProtection']:
        sns.barplot(data=df, x=col, y='Churn')
        plt.title(f"{col} vs Churn")
        plt.show()

    sns.countplot(data=df, x='Contract', hue='Churn')
    plt.title("Contract Type vs Churn")
    plt.show()

    sns.boxplot(data=df, x='Churn', y='MonthlyCharges')
    plt.title("Monthly Charges vs Churn")
    plt.show()

    sns.boxplot(data=df, x='Churn', y='TotalCharges')
    plt.title("Total Charges vs Churn")
    plt.show()

def correlation_heatmap(df):
    numeric_df = df.select_dtypes(include=['number'])
    plt.figure(figsize=(12, 6))
    sns.heatmap(numeric_df.corr(), annot=True,cmap='coolwarm')
    plt.title("Correlation Heatmap")
    plt.show()

# if __name__ == "__main__":
#     import pandas as pd
#     df = pd.read_csv("../data/Telco-Customer-Churn.csv")
#     df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
#     df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
#     plot_churn_distribution(df)
#     plot_numeric_distribution(df)
#     correlation_heatmap(df)