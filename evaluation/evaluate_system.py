import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support
)


# ============================================================
# FILE PATHS
# ============================================================

GOLDEN_PATH = "results/golden_set_reviewed.csv"
PREDICTIONS_PATH = "results/golden_predictions.csv"
OUTPUT_PATH = "results/evaluation_results.csv"


# ============================================================
# LOAD GOLDEN SET
# ============================================================

print("Loading Golden Set...")

golden = pd.read_csv(GOLDEN_PATH)

golden = golden.dropna(
    subset=["customer_text", "intent"]
)

golden = golden[
    golden["intent"].astype(str).str.strip() != ""
]

print("Golden examples:", len(golden))


# ============================================================
# LOAD MODEL PREDICTIONS
# ============================================================

print("\nLoading model predictions...")

predictions = pd.read_csv(PREDICTIONS_PATH)

predictions = predictions.dropna(
    subset=["customer_text", "intent", "predicted_intent"]
)

print("Prediction examples:", len(predictions))


# ============================================================
# CALCULATE MODEL METRICS
# ============================================================

y_true = predictions["intent"].astype(str)
y_pred = predictions["predicted_intent"].astype(str)


accuracy = accuracy_score(
    y_true,
    y_pred
)


precision, recall, f1, _ = precision_recall_fscore_support(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)


# ============================================================
# MAJORITY BASELINE
# ============================================================

majority_intent = golden["intent"].value_counts().idxmax()

majority_predictions = [
    majority_intent
] * len(y_true)


majority_accuracy = accuracy_score(
    y_true,
    majority_predictions
)


majority_precision, majority_recall, majority_f1, _ = (
    precision_recall_fscore_support(
        y_true,
        majority_predictions,
        average="macro",
        zero_division=0
    )
)


# ============================================================
# CREATE RESULTS TABLE
# ============================================================

results = pd.DataFrame([
    {
        "system": "Majority Baseline",
        "accuracy": round(majority_accuracy, 4),
        "macro_precision": round(majority_precision, 4),
        "macro_recall": round(majority_recall, 4),
        "macro_f1": round(majority_f1, 4)
    },
    {
        "system": "TF-IDF + Logistic Regression",
        "accuracy": round(accuracy, 4),
        "macro_precision": round(precision, 4),
        "macro_recall": round(recall, 4),
        "macro_f1": round(f1, 4)
    }
])


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n======================================")
print("SYSTEM EVALUATION")
print("======================================")

print(
    results.to_string(index=False)
)


# ============================================================
# IMPROVEMENT OVER MAJORITY BASELINE
# ============================================================

accuracy_improvement = (
    accuracy - majority_accuracy
)

f1_improvement = (
    f1 - majority_f1
)


print("\n======================================")
print("IMPROVEMENT")
print("======================================")

print(
    "Accuracy improvement:",
    round(accuracy_improvement, 4)
)

print(
    "Macro F1 improvement:",
    round(f1_improvement, 4)
)


# ============================================================
# PER-INTENT PERFORMANCE
# ============================================================

print("\n======================================")
print("PER-INTENT PERFORMANCE")
print("======================================")


_, _, f1_scores, support = precision_recall_fscore_support(
    y_true,
    y_pred,
    labels=sorted(y_true.unique()),
    zero_division=0
)


intent_names = sorted(
    y_true.unique()
)


intent_results = pd.DataFrame({
    "intent": intent_names,
    "f1": f1_scores,
    "support": support
})


intent_results["f1"] = intent_results["f1"].round(4)

print(
    intent_results.to_string(index=False)
)


# ============================================================
# SAVE RESULTS
# ============================================================

results.to_csv(
    OUTPUT_PATH,
    index=False
)


print("\n======================================")
print("Evaluation completed successfully!")
print("======================================")

print("\nResults saved to:")
print(OUTPUT_PATH)