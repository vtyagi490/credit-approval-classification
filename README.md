# Credit Approval Classification

## Project Type
**Individual Final Project**

## Author
**Vishal Tyagi**

---

## 1. Project Overview

This project focuses on developing and evaluating supervised machine learning models to predict **credit approval decisions** based on applicant financial and demographic information.  
The goal is to compare multiple classification algorithms and identify a model that balances **predictive performance**, **interpretability**, and **business relevance** in a credit risk assessment context.

Credit approval is a high-stakes decision-making problem where incorrect approvals can lead to financial loss, while incorrect rejections may result in lost business opportunities. This project emphasizes both statistical rigor and practical applicability.

---

## 2. Dataset Description

- **Dataset Name:** Statlog (Australian Credit Approval Dataset)
- **Source:** UCI Machine Learning Repository
- **Observations:** 690 records
- **Features:** 14 input attributes (mix of numerical and categorical)
- **Target Variable:** Binary class indicating credit approval (approved / not approved)

The dataset contains anonymized applicant information, including financial indicators and personal attributes. Several features contain missing values and require preprocessing before modeling.

---

## 3. Data Cleaning and Preparation

The following preprocessing steps were applied:

- Handling missing values using statistically appropriate imputation techniques
- Encoding categorical variables for compatibility with machine learning models
- Feature scaling for models sensitive to magnitude (e.g., Logistic Regression, SVM)
- Splitting the dataset into training and testing subsets to evaluate generalization performance

These steps ensured data consistency and improved model stability.

---

## 4. Exploratory Data Analysis (EDA)

Exploratory analysis was conducted to:

- Examine the distribution of the target variable
- Identify relationships between predictors and credit approval outcomes
- Detect correlations and potential multicollinearity
- Understand class balance and feature behavior

Insights from EDA informed feature selection and model choice.

---

## 5. Models Implemented

The following supervised learning models were trained and evaluated:

- **Logistic Regression**
- **Support Vector Machine (SVM)**
- **Random Forest Classifier**

Each model was selected to provide a balance between interpretability and predictive power, allowing meaningful comparison across linear and non-linear approaches.

---

## 6. Model Evaluation

Models were evaluated using multiple performance metrics:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

Given the business context of credit approval, special attention was paid to **false positives** (approving high-risk applicants) and **false negatives** (rejecting creditworthy applicants). This ensured the evaluation aligned with real-world risk considerations.

---

## 7. Key Findings

- Ensemble-based models demonstrated strong predictive performance.
- Logistic Regression provided greater interpretability and transparency.
- Certain financial attributes showed consistent importance across models.
- Trade-offs between accuracy and interpretability were observed and discussed.

---

## 8. Project Structure

```text
credit-approval-classification/
│
├── data/
│   ├── raw/
│   │   └── australian.dat
│   └── processed/
│       └── credit_approval_clean.csv
│
├── notebooks/
│   └── Credit_approval_model.ipynb
│
├── outputs/
│   ├── figures/
│   └── metrics/
│
├── README.md
├── requirements.txt
└── .gitignore
```

## 9. Technologies Used

- Python

- Pandas, NumPy

- Scikit-learn

- Matplotlib, Seaborn

- Jupyter Notebook

- Git and GitHub

## 10. How to Run the Project
 - 1. Clone the repository:

 git clone https://github.com/vtyagi490/credit-approval-classification.git

 - 2. Navigate to the project directory:

 cd credit-approval-classification

 - 3. Install dependencies:

 pip install -r requirements.txt

 - 4. Launch Jupyter Notebook and run all cells:

 jupyter notebook


## 11. Limitations and Future Work

- The dataset is relatively small and anonymized, limiting feature interpretability.

- Additional hyperparameter tuning could further improve performance.

- Future work may include cost-sensitive learning or threshold optimization.

- Model explainability techniques such as SHAP could enhance interpretability.

## 12.Conclusion

This project demonstrates an end-to-end machine learning workflow for a real-world credit approval problem. Through careful data preparation, exploratory analysis, and comparative modeling, the project highlights the strengths and limitations of different supervised learning approaches in financial decision-making contexts.