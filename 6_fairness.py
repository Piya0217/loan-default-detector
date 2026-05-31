# 6_fairness.py
# STEP 6 — Check if our model is FAIR
# Is it biased against women? Younger people? Certain job types?
# This is what AI Ethics researchers actually do

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score

# ── Load everything ───────────────────────────────────────────
with open("data/models.pkl", "rb") as f:
    dt_model, rf_model, X_test, y_test = pickle.load(f)

# Rebuild test dataframe with column names
feature_names = ['Age', 'Sex', 'Job', 'Housing',
                 'Saving accounts', 'Checking account',
                 'Credit amount', 'Duration', 'Purpose']

X_test_df = pd.DataFrame(X_test.values, columns=feature_names)
X_test_df['Actual'] = y_test.values
X_test_df['Predicted'] = rf_model.predict(X_test)

# ── Fairness Check 1: Gender Bias ─────────────────────────────
# Sex column: after encoding, 0=female, 1=male (check your encoding)
# Does model perform equally for both genders?

print("="*50)
print("FAIRNESS CHECK 1 — Gender Bias")
print("="*50)

for gender_code, gender_name in [(0, 'Female'), (1, 'Male')]:
    group = X_test_df[X_test_df['Sex'] == gender_code]
    if len(group) == 0:
        continue
    acc = accuracy_score(group['Actual'], group['Predicted'])
    f1  = f1_score(group['Actual'], group['Predicted'],
                   zero_division=0)
    print(f"\n{gender_name} (n={len(group)}):")
    print(f"  Accuracy : {acc:.2%}")
    print(f"  F1 Score : {f1:.2%}")

# ── Fairness Check 2: Age Group Bias ─────────────────────────
# Are younger people unfairly rejected more?

print("\n" + "="*50)
print("FAIRNESS CHECK 2 — Age Group Bias")
print("="*50)

# Create age buckets
X_test_df['Age Group'] = pd.cut(
    X_test_df['Age'],
    bins=[0, 25, 35, 50, 100],
    labels=['Young (≤25)', 'Adult (26-35)',
            'Middle (36-50)', 'Senior (50+)']
)

for age_group in ['Young (≤25)', 'Adult (26-35)',
                  'Middle (36-50)', 'Senior (50+)']:
    group = X_test_df[X_test_df['Age Group'] == age_group]
    if len(group) == 0:
        continue
    acc = accuracy_score(group['Actual'], group['Predicted'])

    # Rejection rate — what % did model predict as bad risk
    rejection_rate = (group['Predicted'] == 0).sum() / len(group)

    print(f"\n{age_group} (n={len(group)}):")
    print(f"  Accuracy       : {acc:.2%}")
    print(f"  Rejection Rate : {rejection_rate:.2%}")

# ── Fairness Check 3: Visualize rejection rates ───────────────
age_groups = []
rejection_rates = []

for age_group in ['Young (≤25)', 'Adult (26-35)',
                  'Middle (36-50)', 'Senior (50+)']:
    group = X_test_df[X_test_df['Age Group'] == age_group]
    if len(group) == 0:
        continue
    rejection_rate = (group['Predicted'] == 0).sum() / len(group)
    age_groups.append(age_group)
    rejection_rates.append(rejection_rate)

plt.figure(figsize=(8, 5))
colors = ['#e74c3c' if r > 0.4 else '#2ecc71' for r in rejection_rates]
plt.bar(age_groups, rejection_rates, color=colors)
plt.axhline(y=0.4, color='orange', linestyle='--',
            label='Fairness threshold (40%)')
plt.title("Rejection Rate by Age Group\n"
          "Red = potentially unfair (>40% rejection)")
plt.xlabel("Age Group")
plt.ylabel("Rejection Rate")
plt.legend()
plt.tight_layout()
plt.savefig("data/06_fairness_age.png")
plt.show()

# ── Fairness Check 1 Visual — Gender ─────────────────────────
genders = ['Female', 'Male']
accuracies = []
f1_scores = []

for gender_code in [0, 1]:
    group = X_test_df[X_test_df['Sex'] == gender_code]
    accuracies.append(accuracy_score(group['Actual'], group['Predicted']))
    f1_scores.append(f1_score(group['Actual'], group['Predicted'],
                              zero_division=0))

x = np.arange(len(genders))
width = 0.35

plt.figure(figsize=(7, 5))
plt.bar(x - width/2, accuracies, width, label='Accuracy', color='#3498db')
plt.bar(x + width/2, f1_scores,  width, label='F1 Score',  color='#e74c3c')
plt.xticks(x, genders)
plt.ylim(0, 1)
plt.title("Model Performance by Gender\n"
          "Large gap = potential bias")
plt.ylabel("Score")
plt.legend()
plt.tight_layout()
plt.savefig("data/06_fairness_gender.png")
plt.show()
print("Gender graph saved!")


# ── Summary ───────────────────────────────────────────────────
print("\n" + "="*50)
print("FAIRNESS SUMMARY")
print("="*50)
print("""
What we checked:
  ✅ Gender bias    — does model treat men/women equally?
  ✅ Age bias       — are younger people unfairly rejected?

What this means in real world:
  → If rejection rate differs hugely between groups = BIASED
  → Biased model = legal problems + ethical problems
  → Solution: collect more data from underrepresented groups
               or apply fairness constraints during training

This is why AI Ethics is a real job in every major bank.
""")