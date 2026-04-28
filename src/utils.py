import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# -----------------------------
# Folder Setup
# -----------------------------

def create_results_folders(base_path="../results"):
    folders = [
        base_path,
        f"{base_path}/plots",
        f"{base_path}/metrics",
        f"{base_path}/high_risk_customers",
        f"{base_path}/threshold_optimization",
        f"{base_path}/confusion_matrices",
        f"{base_path}/shap"
    ]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)


# -----------------------------
# Saving Helpers
# -----------------------------

def save_plot(fig, filename, base_path="../results/plots"):
    create_results_folders()
    fig.savefig(f"{base_path}/{filename}", bbox_inches="tight")
    plt.close(fig)


def save_json(data, filename, base_path="../results/metrics"):
    create_results_folders()
    with open(f"{base_path}/{filename}", "w") as f:
        json.dump(data, f, indent=4)


def save_csv(df, filename, base_path="../results"):
    create_results_folders()
    df.to_csv(f"{base_path}/{filename}", index=False)


# -----------------------------
# Timestamp Helper
# -----------------------------

def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")
