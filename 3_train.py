# 3_train.py
# STEP 3 — Train the model

import pickle
import numpy as np
from sklearn.model_selection import train_test_split  # splits data into train/test
from sklearn.tree import DecisionTreeClassifier       # our main model
from sklearn.ensemble import RandomForestClassifier   # stronger version of decision tree

# ── Load processed data ───────────────────────────────────────
with open("data/processed_data.pkl", "rb") as f:
    X, y = pickle.load(f)

print("Loaded data shape:", X.shape)

# ── Split data into train and test ────────────────────────────
# We train on 80% of data
# We test on remaining 20% — model has NEVER seen this data
# This is how we check if model actually learned or just memorized

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20% for testing
    random_state=42     # same split every time
)

print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples:  {X_test.shape[0]}")

# ── Model 1: Decision Tree ────────────────────────────────────
# Like a flowchart — asks yes/no questions to reach a decision
# Example: "Is credit amount > 5000? → Is duration > 24 months? → Bad risk"

dt_model = DecisionTreeClassifier(
    max_depth=5,        # limit how deep the tree goes — prevents overfitting
    random_state=42
)
dt_model.fit(X_train, y_train)  # TRAIN — model learns from data here
print("\n✅ Decision Tree trained!")

# ── Model 2: Random Forest ────────────────────────────────────
# Builds 100 decision trees and takes majority vote
# Much stronger and more reliable than a single tree

rf_model = RandomForestClassifier(
    n_estimators=100,   # 100 trees
    max_depth=10,
    random_state=42
)
rf_model.fit(X_train, y_train)
print("✅ Random Forest trained!")

# ── Save models + test data ───────────────────────────────────
with open("data/models.pkl", "wb") as f:
    pickle.dump((dt_model, rf_model, X_test, y_test), f)

print("\n✅ Models saved to data/models.pkl")
print("Ready for evaluation!")