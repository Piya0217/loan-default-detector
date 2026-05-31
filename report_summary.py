# report_summary.py
# FINAL STEP — Complete project summary
# This is what you show in presentations / interviews

import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# ── Load everything ───────────────────────────────────────────
with open("data/models.pkl", "rb") as f:
    dt_model, rf_model, X_test, y_test = pickle.load(f)

feature_names = ['Age', 'Sex', 'Job', 'Housing',
                 'Saving accounts', 'Checking account',
                 'Credit amount', 'Duration', 'Purpose']

X_test_df = pd.DataFrame(X_test.values, columns=feature_names)
y_pred = rf_model.predict(X_test)

print("""
╔══════════════════════════════════════════════════════════════╗
║           LOAN DEFAULT PREDICTOR — PROJECT REPORT           ║
╚══════════════════════════════════════════════════════════════╝
""")

# ── 1. Problem Statement ──────────────────────────────────────
print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 PROBLEM STATEMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Banks lose crores when loan applicants default.
 Goal: Predict whether a person will default BEFORE giving loan.
 Dataset: German Credit Data — 1000 applicants, 9 features.
""")

# ── 2. Data Challenges ────────────────────────────────────────
print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 DATA CHALLENGES WE SOLVED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 ✅ Missing values    → Filled with 'unknown' category
 ✅ Text columns      → Converted to numbers using LabelEncoder
 ✅ Class imbalance   → Fixed using SMOTE (700 vs 300 → 700 vs 700)
""")

# ── 3. Model Performance ──────────────────────────────────────
dt_pred = dt_model.predict(X_test)
rf_pred = rf_model.predict(X_test)

dt_acc = accuracy_score(y_test, dt_pred)
dt_f1  = f1_score(y_test, dt_pred)
rf_acc = accuracy_score(y_test, rf_pred)
rf_f1  = f1_score(y_test, rf_pred)

print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 MODEL PERFORMANCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Decision Tree  → Accuracy: {dt_acc:.2%}  F1: {dt_f1:.2%}
 Random Forest  → Accuracy: {rf_acc:.2%}  F1: {rf_f1:.2%}

 Winner: Random Forest 🏆
 Reason: 100 trees voting together beats 1 tree every time.
""")

# ── 4. Top Features ───────────────────────────────────────────
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 TOP FACTORS THAT PREDICT DEFAULT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━""")
for i, idx in enumerate(indices[:3]):
    print(f" {i+1}. {feature_names[idx]:<25} → {importances[idx]:.4f}")

# ── 5. Fairness Issues Found ──────────────────────────────────
print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 FAIRNESS ISSUES FOUND
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 ⚠️  Gender  → F1 gap of ~10% between Male and Female
               Model is less reliable for female applicants

 ⚠️  Age     → Adults (26-35) have highest rejection rate (53%)
               Likely due to larger loans at that life stage
""")

# ── 6. How to fix the fairness problem ───────────────────────
print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 HOW TO FIX FAIRNESS ISSUES (Real World Solutions)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

 Problem: Model has only 94 female samples vs 186 male samples
 This means model never learned female patterns properly

 Fix 1 → COLLECT MORE DATA
          Get more female applicant records
          Model needs equal representation to learn equally

 Fix 2 → SEPARATE SMOTE PER GROUP
          Apply SMOTE separately for male and female
          So both groups are balanced before training

 Fix 3 → FAIRNESS CONSTRAINTS
          Tell the model during training:
          "Your F1 score gap between genders must be < 5%"
          Libraries like Fairlearn do this automatically

 Fix 4 → REMOVE SENSITIVE FEATURES
          Completely remove 'Sex' column from training
          Model can't be gender biased if it never sees gender
          (Used by many banks today)

 Fix 5 → HUMAN REVIEW FOR FLAGGED GROUPS
          If applicant is in an underrepresented group
          → Always send to human review, never auto-decide
""")

# ── 7. Real World Deployment Model ───────────────────────────
print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 HOW THIS WORKS IN REAL WORLD DEPLOYMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

 Applicant submits details
          ↓
 Model gives probability score (e.g. 73% chance of default)
          ↓
 Score < 30%   → AUTO APPROVE  ✅
 Score 30-60%  → HUMAN REVIEW  ⚠️
 Score > 60%   → AUTO REJECT   ❌
          ↓
 Human officer reviews borderline cases
          ↓
 Final decision with written explanation (legally required)

 Key point: Model is a FILTER, not a judge.
 Humans always handle ambiguous cases.
""")

print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 FILES GENERATED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 data/01_target_distribution.png  → class imbalance chart
 data/confusion_Decision_Tree.png → decision tree results
 data/confusion_Random_Forest.png → random forest results
 data/05_feature_importance.png   → what factors matter most
 data/06_fairness_age.png         → age group bias chart
 data/06_fairness_gender.png      → gender bias chart
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 PROJECT COMPLETE ✅
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")