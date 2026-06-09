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

### EDA & Preprocessing
- Handled missing values — median for numerical, mode for categorical
- Binary encoded: Gender, Married, Education, Self_Employed, Loan_Status
- One-hot encoded: Property_Area (drop_first=True)
- Converted Dependents — replaced `3+` with `3`, cast to int
- Dropped: Loan_ID (non-informative)

---

## Project Status

| Step | Status |
|---|---|
| Data Cleaning | ✅ Done |
| EDA | ✅ Done |
| Feature Engineering | ✅ Done |
| Model Training | 🔄 In Progress |
| Evaluation | ⏳ Pending |
