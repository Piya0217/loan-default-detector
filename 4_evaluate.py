# 4_evaluate.py
# STEP 4 — Evaluate how good our models are

import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,        # basic % correct
    f1_score,              # balance between precision and recall
    roc_auc_score,         # how well model separates good vs bad
    confusion_matrix,      # table showing all 4 outcomes
    classification_report  # full breakdown
)

# ── Load models and test data ─────────────────────────────────
with open("data/models.pkl", "rb") as f:
    dt_model, rf_model, X_test, y_test = pickle.load(f)

# ── Evaluate function ─────────────────────────────────────────
# We write a function so we don't repeat code for each model
def evaluate_model(model, X_test, y_test, model_name):
    print(f"\n{'='*50}")
    print(f"  {model_name}")
    print(f"{'='*50}")

    # Get predictions
    y_pred = model.predict(X_test)

    # Basic accuracy — how many did it get right overall
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy:  {acc:.2%}")

    # F1 Score — better metric when classes are imbalanced
    # combines precision (when it says bad, is it right?)
    # and recall (of all actual bad loans, how many did it catch?)
    f1 = f1_score(y_test, y_pred)
    print(f"F1 Score:  {f1:.2%}")

    # ROC-AUC — 0.5 = random guessing, 1.0 = perfect
    auc = roc_auc_score(y_test, y_pred)
    print(f"ROC-AUC:   {auc:.2%}")

    # Full report — precision, recall for each class
    print("\nDetailed Report:")
    print(classification_report(y_test, y_pred,
          target_names=['Bad Risk', 'Good Risk']))

    # Confusion matrix — the 4 outcome table we discussed
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn',
                xticklabels=['Predicted Bad', 'Predicted Good'],
                yticklabels=['Actual Bad', 'Actual Good'])
    plt.title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    plt.savefig(f"data/confusion_{model_name.replace(' ', '_')}.png")
    plt.show()
    print(f"Graph saved!")

    return acc, f1, auc

# ── Evaluate both models ──────────────────────────────────────
dt_acc, dt_f1, dt_auc = evaluate_model(
    dt_model, X_test, y_test, "Decision Tree"
)
rf_acc, rf_f1, rf_auc = evaluate_model(
    rf_model, X_test, y_test, "Random Forest"
)

# ── Compare both models ───────────────────────────────────────
print("\n" + "="*50)
print("  MODEL COMPARISON")
print("="*50)
print(f"{'Metric':<12} {'Decision Tree':>15} {'Random Forest':>15}")
print("-"*45)
print(f"{'Accuracy':<12} {dt_acc:>14.2%} {rf_acc:>14.2%}")
print(f"{'F1 Score':<12} {dt_f1:>14.2%} {rf_f1:>14.2%}")
print(f"{'ROC-AUC':<12} {dt_auc:>14.2%} {rf_auc:>14.2%}")

# ── Winner ────────────────────────────────────────────────────
if rf_f1 > dt_f1:
    print("\n🏆 Random Forest wins — we'll use this for deployment")
else:
    print("\n🏆 Decision Tree wins — we'll use this for deployment")