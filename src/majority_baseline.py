import pandas as pd

from sklearn.metrics import accuracy_score, classification_report


# -----------------------------
# File path
# -----------------------------
DATA_PATH = "results/golden_set_reviewed.csv"


# -----------------------------
# Load Golden Set
# -----------------------------
print("Loading Golden Set...")

df = pd.read_csv(DATA_PATH)

df = df.dropna(subset=["customer_text", "intent"])
df = df[df["intent"].str.strip() != ""]

print("Examples available:", len(df))


# -----------------------------
# Find majority class
# -----------------------------
majority_intent = df["intent"].value_counts().idxmax()

majority_count = df["intent"].value_counts().max()

print("\nMajority intent:")
print(majority_intent)

print("Number of examples:")
print(majority_count)


# -----------------------------
# Train / Test split
# -----------------------------
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    df["customer_text"],
    df["intent"],
    test_size=0.20,
    random_state=42,
    stratify=df["intent"]
)


# -----------------------------
# Predict majority class
# -----------------------------
y_pred = [majority_intent] * len(y_test)


# -----------------------------
# Evaluation
# -----------------------------
accuracy = accuracy_score(y_test, y_pred)

print("\n======================================")
print("Majority Class Baseline Results")
print("======================================")

print("\nAccuracy:")
print(round(accuracy, 4))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)