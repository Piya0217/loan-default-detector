# 5_explain.py
# STEP 5 — Explain WHY the model makes decisions
# This is called XAI — Explainable AI
# Real banks legally MUST explain why they rejected someone

import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ── Load model and test data ──────────────────────────────────
with open("data/models.pkl", "rb") as f:
    dt_model, rf_model, X_test, y_test = pickle.load(f)

# Load original data to get column names
df = pd.read_csv("data/german_credit_data.csv")
df = df.drop(columns=['Unnamed: 0'])

feature_names = ['Age', 'Sex', 'Job', 'Housing',
                 'Saving accounts', 'Checking account',
                 'Credit amount', 'Duration', 'Purpose']

# ── Part 1: Feature Importance ────────────────────────────────
# Which factors matter MOST to the model when deciding?
# Example: "Credit amount matters more than Age"

importances = rf_model.feature_importances_  # scores for each feature
indices = np.argsort(importances)[::-1]       # sort from highest to lowest

print("="*50)
print("FEATURE IMPORTANCE — What matters most?")
print("="*50)
for i, idx in enumerate(indices):
    print(f"{i+1}. {feature_names[idx]:<25} {importances[idx]:.4f}")

# Plot feature importance
plt.figure(figsize=(10, 6))
colors = sns.color_palette("RdYlGn", len(feature_names))
plt.barh(
    [feature_names[i] for i in indices[::-1]],
    importances[indices[::-1]],
    color=colors
)
plt.xlabel("Importance Score")
plt.title("Which Factors Matter Most for Loan Default Prediction?")
plt.tight_layout()
plt.savefig("data/05_feature_importance.png")
plt.show()
print("\nGraph saved!")

# ── Part 2: Predict a real example ───────────────────────────
# Let's take ONE person from test data and explain the decision

print("\n" + "="*50)
print("INDIVIDUAL PREDICTION EXAMPLE")
print("="*50)

# Take first person from test set
person = X_test.iloc[0]
actual = y_test.iloc[0]
prediction = rf_model.predict([person])[0]
probability = rf_model.predict_proba([person])[0]

print("\nPerson's Details:")
for name, value in zip(feature_names, person):
    print(f"  {name:<25}: {value}")

print(f"\nActual Risk:     {'BAD' if actual == 0 else 'GOOD'}")
print(f"Model Predicted: {'BAD' if prediction == 0 else 'GOOD'}")
print(f"Confidence:      {max(probability):.2%}")
print(f"  → Probability of GOOD: {probability[1]:.2%}")
print(f"  → Probability of BAD:  {probability[0]:.2%}")

if prediction == actual:
    print("\n✅ Model got this one RIGHT")
else:
    print("\n❌ Model got this one WRONG — this is a false prediction")
    print("   This is normal — no model is 100% accurate")
    print("   In real world: this case would go to human review")

# ── Part 3: Threshold analysis ────────────────────────────────
# Remember we discussed banks don't just say yes/no
# They use probability thresholds

print("\n" + "="*50)
print("THRESHOLD BASED DECISION")
print("="*50)

bad_probability = probability[0]

if bad_probability >= 0.70:
    print(f"  Default probability: {bad_probability:.2%}")
    print("  Decision: ❌ AUTO REJECT — too risky")
elif bad_probability >= 0.40:
    print(f"  Default probability: {bad_probability:.2%}")
    print("  Decision: ⚠️  MANUAL REVIEW — borderline case")
else:
    print(f"  Default probability: {bad_probability:.2%}")
    print("  Decision: ✅ AUTO APPROVE — low risk")