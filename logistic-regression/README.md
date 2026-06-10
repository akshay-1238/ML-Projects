# Loan Approval Prediction — Logistic Regression
Predicting whether a loan application will be approved or rejected based on applicant details.

---

## Dataset
- **Source:** loan_data_1.csv
- **Rows:** 381
- **Target:** Loan_Status (Y/N) → encoded as `Is_Loan_Approved` (1/0)

**Features:**
- Applicant demographics: Gender, Married, Dependents, Education, Self_Employed
- Financial: ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History
- Location: Property_Area

---

## Work Done

### Data Cleaning
- Handled missing values — median for skewed numerical features, mode for categorical
- Verified `Credit_History` null strategy: 26/30 missing rows were approved → mode imputation (1.0) justified
- Binary encoded: Gender, Married, Education, Self_Employed, Loan_Status
- One-hot encoded: Property_Area with `drop_first=True` to avoid multicollinearity
- Converted Dependents — replaced `3+` with `3`, cast to int
- Dropped: Loan_ID (non-informative identifier)

### EDA
- Categorical distributions: dataset is male-heavy, graduate-heavy, and target is imbalanced (~70% approved)
- Continuous distributions: all four features are right-skewed with outliers
- Grouped boxplots: continuous features show heavy overlap between approved/rejected classes — weak individual predictors
- Pearson correlation heatmap: all continuous features have near-zero correlation with target (max 0.05)
- Chi-square test results:
  - Significant: `Credit_History` (p≈0.00), `Property_Area_Semiurban` (p=0.004)
  - Borderline: `Is_Married` (p=0.09)
  - Not significant: Dependents, Is_Male, Is_Education, Is_Self_Employed, Property_Area_Urban

### Model Training
- **Model 1** (3 selected features): 81.8% accuracy, but class 0 recall = 0.33 — misses 67% of rejections
- **Model 2** (all features + `class_weight='balanced'`): 75.3% accuracy, class 0 recall improves to 0.52
- Model 2 is more reliable given class imbalance — higher overall accuracy in Model 1 is misleading

---

## Project Status
| Step | Status |
|---|---|
| Data Cleaning | ✅ Done |
| EDA | ✅ Done |
| Feature Selection | ✅ Done |
| Model Training | ✅ Done |
| Evaluation | ✅ Done |
