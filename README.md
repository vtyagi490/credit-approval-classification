# Australian Credit Approval Classification

This project analyzes the Statlog (Australian Credit Approval) dataset to build and evaluate statistical and machine learning models for predicting credit approval decisions. The study focuses on data cleaning, exploratory analysis, and classification modeling to support data-driven credit risk assessment.

## Dataset Overview

### Source
- **Name**: Statlog (Australian Credit Approval)
- **Repository**: UCI Machine Learning Repository
- **Samples**: 690 credit applications
- **Features**: 14 predictors (6 numerical + 8 categorical) + 1 binary target
- **Target Variable (A15)**: 1 = Approved (+), 2 = Denied (−)
- **Missing Values**: None detected
- **Class Distribution**: 44% Approved, 56% Denied (modest imbalance)

### Feature Summary

#### Numerical Features (6)
| Feature | Code | Range | Mean | Description |
|---------|------|-------|------|-------------|
| A2 | Continuous | 13.75–80.25 | 31.6 | Age or similar metric |
| A3 | Continuous | 0–28 | 4.76 | Financial metric (right-skewed) |
| A7 | Continuous | 0–28.5 | 2.22 | Debt/loan amount (heavy tail) |
| A10 | Continuous | 0–67 | 2.4 | Months/tenure (sparse, zero-heavy) |
| A13 | Continuous | 0–2000 | 184 | Credit limit or balance |
| A14 | Continuous | 1–100001 | 1018 | Monthly payment/income (extreme outliers) |

#### Categorical Features (8)
| Feature | Code | Type | Cardinality | Description |
|---------|------|------|-------------|-------------|
| A1 | Binary | Nominal | 2 | Account status (a/b) |
| A4 | Ordinal | 1–3 | 3 | Credit history (p/g/gg) |
| A5 | Nominal | 1–14 | 14 | Purpose of credit |
| A6 | Nominal | 1–9 | 9 | Occupation |
| A8 | Binary | Nominal | 2 | Telephone listed (t/f) |
| A9 | Binary | Nominal | 2 | Foreign worker (t/f) |
| A11 | Binary | Nominal | 2 | Prior default history (t/f) |
| A12 | Ordinal | 1–3 | 3 | Employment status (s/g/p) |

### Data Quality
- **Missing Values**: 0 (100% complete)
- **Duplicates**: Not checked; assume unique records
- **Outliers**: A14 has extreme value (100,001); A7, A3 show right-skewed distributions
- **Preprocessing Applied**: Categorical one-hot encoding, stratified train/test split (80/20)

### Target Variable Analysis
- **Approved (1)**: 307 records (44.5%)
- **Denied (2)**: 383 records (55.5%)
- **Imbalance Ratio**: 1:1.24 (moderate); use stratified CV and class-weighted models

### Top Predictive Features
Ranked by absolute correlation with approval decision (A15):

1. **A8** (r = 0.720) — Telephone listed; strongest predictor
2. **A9** (r = 0.458) — Foreign worker status
3. **A10** (r = 0.406) — Months/tenure in account
4. **A5** (r = 0.374) — Purpose of credit
5. **A7** (r = 0.322) — Debt/loan metric

**Weak Predictors** (|r| < 0.15):
- A1, A11, A13, A14, A2, A12

### Data Files Generated

#### Processed Data
- **`outputs/train.csv`** — Training set (552 rows, 80%) with one-hot encoded features
- **`outputs/test.csv`** — Test set (138 rows, 20%) with one-hot encoded features
- **`outputs/data_encoded.csv`** — Full dataset with categorical features encoded (690 rows)
