import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# -----------------------------------
# File paths
# -----------------------------------
TRAINING_DATA = "results/training_data.csv"
GOLDEN_DATA = "results/golden_set_reviewed.csv"


# -----------------------------------
# Load training data
# -----------------------------------
print("Loading AmazonHelp conversations...")

train_df = pd.read_csv(TRAINING_DATA)

train_df = train_df.dropna(
    subset=["customer_text", "intent"]
)

train_df = train_df[
    train_df["intent"].str.strip() != ""
]

print("Training examples:", len(train_df))


# -----------------------------------
# Prepare training data
# -----------------------------------
X = train_df["customer_text"].astype(str)
y = train_df["intent"].astype(str)


# -----------------------------------
# Split training data
# -----------------------------------
X_train, X_unused, y_train, y_unused = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# -----------------------------------
# TF-IDF
# -----------------------------------
print("\nCreating TF-IDF features...")

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95
)

X_train_tfidf = vectorizer.fit_transform(X_train)


# -----------------------------------
# Train model
# -----------------------------------
print("Training Logistic Regression...")

model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced"
)

model.fit(X_train_tfidf, y_train)


# -----------------------------------
# Load Golden Set
# -----------------------------------
print("\nLoading Golden Set...")

golden_df = pd.read_csv(GOLDEN_DATA)

golden_df = golden_df.dropna(
    subset=["customer_text", "intent"]
)

golden_df = golden_df[
    golden_df["intent"].str.strip() != ""
]

X_golden = golden_df["customer_text"].astype(str)
y_golden = golden_df["intent"].astype(str)


# -----------------------------------
# Convert Golden Set to TF-IDF
# -----------------------------------
X_golden_tfidf = vectorizer.transform(X_golden)


# -----------------------------------
# Predict
# -----------------------------------
print("Evaluating on Golden Set...")

y_pred = model.predict(X_golden_tfidf)


# -----------------------------------
# Evaluation
# -----------------------------------
accuracy = accuracy_score(
    y_golden,
    y_pred
)

print("\n======================================")
print("Full Training Data Results")
print("======================================")

print("\nAccuracy:")
print(round(accuracy, 4))

print("\nClassification Report:")

print(
    classification_report(
        y_golden,
        y_pred,
        zero_division=0
    )
)


# -----------------------------------
# Save predictions
# -----------------------------------
golden_df["predicted_intent"] = y_pred

OUTPUT_PATH = "results/golden_predictions.csv"

golden_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nPredictions saved to:")
print(OUTPUT_PATH)