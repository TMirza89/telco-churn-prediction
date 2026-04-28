#  Telco Customer Churn Prediction  
## A Full ML Engineering Pipeline + Streamlit App with Business Insights, SHAP Explainability & Revenue‑Driven Threshold Optimization

This project builds a complete, production‑style machine learning pipeline to predict customer churn for a telecom company.  
It goes beyond accuracy — focusing on **business impact**, **explainability**, and **actionable retention strategy**.

Recruiters and hiring managers will find:

- Clean, modular, reproducible ML code  
- SHAP explainability for model transparency  
- Threshold optimization tied to **net revenue saved**  
- A structured `results/` folder with all artifacts  
- Business storytelling + technical depth  
Telco Customer Churn Prediction
A Full ML Engineering Pipeline + Streamlit App with Business Insights, SHAP Explainability & Revenue‑Driven Threshold Optimization
This project delivers a complete, production‑style machine learning system for predicting telecom customer churn.
It combines ML engineering, business analytics, and explainability to simulate how real retention teams use predictive models.

Recruiters and hiring managers will find:

- Clean, modular, reproducible ML code
- A professional Streamlit app for real‑time predictions
- SHAP explainability (waterfall, force, bar plots)
- Revenue‑driven threshold optimization
- High‑risk customer identification
- A structured results/ folder with all artifacts
- Strong business storytelling + technical depth

## 1. Business Problem
Telecom companies lose millions each year due to customer churn.
This project aims to:

- Predict which customers are likely to churn
- Understand why they churn using SHAP
- Quantify revenue at risk (monthly + LTV)
- Recommend optimal retention thresholds
- Identify top high‑risk customers for targeted outreach
- The pipeline mirrors a real-world churn analytics system used by retention teams.

## 2. Project Architecture
Code
project/
│── data/
│── src/
│   ├── load_data.py
│   ├── preprocess.py
│   ├── train_models.py
│   ├── evaluate.py
│   ├── explain.py
│   └── utils.py
│
│── streamlit_app.py
│── results/
│── main.py
│── README.md
│── SHAP_Explanation.md
The codebase is fully modular and follows ML engineering best practices.

## 3. Feature Engineering Summary
Custom engineered features significantly improve interpretability and business alignment:

- LTV (Lifetime Value)
- ContractRiskScore
- BundleDepth
- ProtectionScore
- EntertainmentScore
- Fiber_NoTechSupport (high‑risk interaction feature)
- Tenure‑based stability indicators
- Service‑usage depth features
- These features help the model capture real churn drivers beyond raw columns.

## 4. Models Used
Three models are trained and compared:

- Logistic Regression (baseline, interpretable)
- Random Forest (removed from app for stability)
- XGBoost (best performing)

Each model outputs:

- Metrics
- Confusion matrix
- SHAP explainability
- Threshold optimization curves
- High‑risk customer lists

## 5. Evaluation Metrics
Saved automatically in:

Code
results/metrics/
Includes:

- Accuracy
- Precision
- Recall
- F1 Score
- Full classification report
- Revenue at risk (monthly + LTV)

## 6. SHAP Explainability
SHAP is used to understand why the model predicts churn.

Generated plots:

- Local SHAP waterfall plot
- Local SHAP force plot
- Local SHAP bar plot
- Global SHAP summary plot (offline, saved in results)

Saved in:

Code
results/shap/
A full business‑focused interpretation is provided in:

SHAP_Explanation.md

## 7. Threshold Optimization (Revenue‑Driven)
Instead of using a default 0.5 threshold, the project computes:

- Precision
- Recall
- F1
- Revenue saved
- Net revenue saved (after retention cost)

This identifies the optimal threshold for maximum business value.

Outputs saved in:

Code
results/plots/
results/threshold_optimization/
## 8. High‑Risk Customer Identification
For each model, the top 10 highest‑risk customers are saved with:

- Churn probability
- MonthlyCharges
- LTV
- Contract type
- Internet service
- Engineered risk scores

Files are timestamped, e.g.:

Code
LogesticRegression_top10_20260423_123951.csv
Saved in:

Code
results/high_risk_customers/
This simulates real retention‑team workflows.

## 9. Results Folder Structure
All outputs are automatically saved.
See the section “Project Outputs — Results Folder Structure” for a full breakdown.

## 10. How to Run the Project
### 1. Install dependencies
Code
pip install -r requirements.txt
### 2. Run the full pipeline
Code
python main.py
This will:

1. Train all models

2. Evaluate them

3. Generate SHAP plots

4. Save all results into the results/ folder

## 11. Streamlit App (Interactive Dashboard)
A fully interactive Streamlit application is included to simulate how a real retention analyst would use the churn model in production.

The app provides:

- Real‑time churn prediction from user‑entered customer attributes
- Local SHAP explainability (waterfall, force, and bar plots)
- Business impact calculator (monthly + LTV revenue at risk)
- High‑risk customer viewer with timestamped outputs
- A clean, business‑friendly UI designed for decision makers

Run the app:

Code
streamlit run streamlit_app.py
This transforms the project from a static ML pipeline into a practical decision‑support tool that demonstrates real‑world usability.
---

## 12. Future Work
- Hyperparameter tuning (Optuna)
- Deploy Streamlit app
- Add model monitoring + drift detection
- Build REST API endpoint
- Add customer segmentation + uplift modeling

## 12. Why This Project Stands Out
This project demonstrates:

1. ML engineering maturity

2. Business‑aligned modeling

3. SHAP explainability

4. Revenue‑driven decision-making

5. Clean, modular architecture

6. A polished Streamlit app

7. Recruiter‑ready documentation

It showcases both technical depth and business storytelling — the combination hiring managers look for.




