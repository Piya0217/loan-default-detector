# Loan Default Detector

An end-to-end supervised machine learning pipeline that predicts whether a loan applicant will default, built on the German Credit Dataset.

## Problem Statement
Banks lose significant revenue when borrowers default on loans. This project builds a data-driven system to predict default risk **before** loan disbursement — enabling faster, fairer, and more accurate credit decisions.

##  Tech Stack
- **Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, Imbalanced-learn, Matplotlib, Seaborn
- **Models:** Decision Tree, Random Forest
- **Techniques:** SMOTE, Label Encoding, Explainable AI, Fairness Audit

##  Project Structure
loan-default-detector/
├── data/                   # Dataset and generated outputs
├── 1_explore.py            # Data exploration and visualization
├── 2_preprocess.py         # Cleaning, encoding, SMOTE balancing
├── 3_train.py              # Model training
├── 4_evaluate.py           # Model evaluation and comparison
├── 5_explain.py            # Explainable AI and feature importance
├── 6_fairness.py           # Fairness audit across gender and age
└── report_summary.py       # Final project summary

##  How to Run
1. Clone the repository
   git clone https://github.com/Piya0217/loan-default-detector.git

2. Install dependencies
   pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn

3. Run scripts in order
   python 1_explore.py
   python 2_preprocess.py
   python 3_train.py
   python 4_evaluate.py
   python 5_explain.py
   python 6_fairness.py
   python report_summary.py

##  Results
| Metric | Decision Tree | Random Forest |
|--------|--------------|---------------|
| Accuracy | ~75% | ~78% |
| F1 Score | ~72% | ~76% |
| ROC-AUC | ~73% | ~77% |

 **Random Forest** selected as final model.

##  Key Findings
- **Top default predictors:** Credit Amount, Duration, Age
- **Fairness gap:** ~10% F1 difference between male and female applicants
- **Highest rejection rate:** Adult (26–35) age group at ~53%

##  Decision Framework
| Default Probability | Decision |
|---------------------|----------|
| Below 30% | ✅ Auto Approve |
| 30% - 70% | ⚠️ Manual Review |
| Above 70% | ❌ Auto Reject |

## 📜 License
MIT License
