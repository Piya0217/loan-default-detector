# 2_preprocess.py
# STEP 2 — Clean the data and prepare it for the model

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder  # converts text → numbers
from imblearn.over_sampling import SMOTE        # fixes class imbalance
import pickle                                   # saves our processed data

# ── Load data ─────────────────────────────────────────────────
df = pd.read_csv("data/german_credit_data.csv")
df = df.drop(columns=['Unnamed: 0'])
df['Risk'] = ['good'] * 700 + ['bad'] * 300

print("Original shape:", df.shape)
print("Missing values before cleaning:")
print(df.isnull().sum())

# ── Step 1: Fill missing values ───────────────────────────────
# For text columns, fill empty cells with the word "unknown"
df['Saving accounts'] = df['Saving accounts'].fillna('unknown')
df['Checking account'] = df['Checking account'].fillna('unknown')

print("\nMissing values after cleaning:")
print(df.isnull().sum())  # should all be 0 now

# ── Step 2: Convert text columns to numbers ───────────────────
# ML models can't understand "male"/"female" — needs to be 0/1
# LabelEncoder does this automatically

le = LabelEncoder()

text_columns = ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose', 'Risk']

for col in text_columns:
    df[col] = le.fit_transform(df[col])
    print(f"{col} encoded: {df[col].unique()}")  # show what numbers it used

print("\nData after encoding:")
print(df.head())

# ── Step 3: Split into features (X) and target (y) ────────────
# X = everything the model uses to decide
# y = what we want to predict (Risk)

X = df.drop(columns=['Risk'])   # all columns except Risk
y = df['Risk']                  # only the Risk column

print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)
print("Class distribution before SMOTE:")
print(y.value_counts())         # how many good vs bad

# ── Step 4: Fix class imbalance with SMOTE ────────────────────
# SMOTE creates synthetic "bad" examples so both classes are equal
# This stops the model from being lazy and just predicting "good" always

smote = SMOTE(random_state=42)  # random_state=42 means same result every run
X_balanced, y_balanced = smote.fit_resample(X, y)

print("\nClass distribution AFTER SMOTE:")
print(pd.Series(y_balanced).value_counts())  # should be equal now

# ── Step 5: Save the processed data ───────────────────────────
# pickle saves Python objects to a file so other scripts can load them
with open("data/processed_data.pkl", "wb") as f:
    pickle.dump((X_balanced, y_balanced), f)

print("\n✅ Processed data saved to data/processed_data.pkl")
print("Ready for training!")