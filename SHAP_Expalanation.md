#  SHAP Explainability Report  
### Understanding Why Customers Churn

This document provides a clear, business‑focused interpretation of SHAP values generated for the churn prediction models.

SHAP (SHapley Additive exPlanations) assigns each feature a contribution value for every prediction — making the model transparent and trustworthy.

---

#  1. What SHAP Tells Us

SHAP helps answer:

- **Why did this customer churn?**  
- **Which features increase churn risk?**  
- **Which features protect against churn?**  
- **How do features interact?**  

This is essential for retention teams who need **actionable insights**, not just predictions.

---

#  2. Global SHAP Summary Plot

The summary plot shows:

- Feature importance  
- Direction of impact  
- Distribution of effects  

**Top churn drivers identified:**

1. **Month‑to‑Month Contract**  
   - Strongest churn indicator  
   - Customers without commitment are more likely to leave  

2. **High MonthlyCharges**  
   - Expensive plans increase churn risk  

3. **Fiber + No Tech Support**  
   - High‑speed internet without support creates frustration  
   - This engineered interaction feature is highly predictive  

4. **Electronic Check Payment**  
   - Historically associated with higher churn  

5. **Low Tenure**  
   - New customers are more likely to churn  

---

#  3. Key Retention Stabilizers

SHAP also highlights features that **reduce** churn risk:

- **Long Tenure**  
- **AutoPay**  
- **Two‑Year Contracts**  
- **Multi‑Service Bundles**  
- **Tech Support**  

These insights help design targeted retention campaigns.

---

# 4. SHAP Bar Plot (Feature Importance)

The bar plot shows the **average absolute impact** of each feature.

This helps answer:

- Which features matter most overall?  
- Where should the business focus its retention strategy?  

---

#  5. Individual Customer Explanations

SHAP can also explain **single predictions**, answering:

> “Why is this specific customer likely to churn?”

This is useful for:

- Customer service agents  
- Retention specialists  
- Personalized outreach  

---

# 6. Why SHAP Matters for This Project

- Builds trust with stakeholders  
- Makes the model transparent  
- Connects ML outputs to business actions  
- Helps prioritize retention interventions  
- Supports regulatory compliance (explainability requirements)  

---

#  7. SHAP Artifacts Saved

All SHAP plots are saved automatically in:

results/shap/

Code

Files include:

- `<ModelName>_shap_summary.png`  
- `<ModelName>_shap_bar.png`  

---

#  Final Takeaway

SHAP transforms the churn model from a “black box” into a **business‑intelligent system**.  