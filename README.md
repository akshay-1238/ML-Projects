# First Linear Regression ML Project

This repository documents my first end-to-end Machine Learning workflow using Linear Regression to predict medical insurance charges.

## Project Status ✅

- ✅ Data cleaning
- ✅ Exploratory Data Analysis (EDA)
- ✅ Feature engineering
- ✅ Feature selection
- ✅ Model training
- ✅ Model evaluation

## Model Results

| Metric | Score |
|---|---|
| R² Score | 0.865 |
| Adjusted R² | 0.857 |

The model explains **86.5%** of the variance in insurance charges.

## Repository Structure

```
├── notebooks/
│   ├── 01_eda_and_preprocessing.ipynb   # Data cleaning, EDA, feature engineering
│   └── Training.ipynb                   # Model training and evaluation
├── data/
│   ├── insurance.csv                    # Original dataset
│   └── Data.csv                         # Cleaned and processed dataset
└── README.md
```

## Dataset

The dataset contains medical insurance charges for **1338 patients** with the following features:

- `age` — Age of the patient
- `bmi` — Body Mass Index
- `children` — Number of children
- `sex` — Gender
- `smoker` — Whether the patient is a smoker
- `region` — US region (northeast, northwest, southeast, southwest)
- `charges` — Medical insurance charges (target variable)

## Approach

- Applied **log transformation** on the target variable (`charges`) to handle skewness and avoid negative predictions
- Used **one-hot encoding** for categorical variables (sex, smoker, region)
- Created **BMI categories** (Underweight, Normal, Overweight, Obese) as additional features
- Split data into **80% training / 20% testing**

## Tools Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
