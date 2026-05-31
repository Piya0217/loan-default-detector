# 1_explore.py
# STEP 1 — Understand what's inside our dataset
# Before building any model, we ALWAYS look at the data first

# ── Import libraries ──────────────────────────────────────────
import pandas as pd          # for reading and working with data
import matplotlib.pyplot as plt  # for drawing graphs
import seaborn as sns        # for prettier graphs

# ── Load the dataset ──────────────────────────────────────────
# pd.read_csv reads the CSV file and stores it as a "DataFrame"
# think of it like an Excel sheet in Python
df = pd.read_csv("data/german_credit_data.csv")

# This dataset has no Risk column — we create it from the index
# In the original German Credit dataset, first 700 are "Good", last 300 are "Bad"
df['Risk'] = ['good'] * 700 + ['bad'] * 300

# ── Basic info ────────────────────────────────────────────────
print("="*50)
print("SHAPE — rows and columns")
print(df.shape)              # how many people and how many features

print("\nCOLUMN NAMES")
print(df.columns.tolist())   # what information we have about each person

print("\nFIRST 5 ROWS")
print(df.head())             # look at first 5 people in the dataset

print("\nDATA TYPES")
print(df.dtypes)             # is each column a number or text?

print("\nMISSING VALUES")
print(df.isnull().sum())     # how many empty cells in each column?

print("\nBASIC STATISTICS")
print(df.describe())         # min, max, average of number columns

# ── Target column ─────────────────────────────────────────────
# "Risk" column = Good (will pay) or Bad (will default)
# This is what we are trying to predict
print("\n" + "="*50)
print("TARGET COLUMN — Risk")
print(df['Risk'].value_counts())  # how many Good vs Bad

# ── Visualize the imbalance ───────────────────────────────────
plt.figure(figsize=(6, 4))
sns.countplot(x='Risk', data=df, palette='Set2')
plt.title("Loan Default Distribution\n(Good = will pay, Bad = will default)")
plt.xlabel("Risk")
plt.ylabel("Number of People")
plt.tight_layout()
plt.savefig("data/01_target_distribution.png")  # saves the graph
plt.show()
print("\nGraph saved to data/01_target_distribution.png")