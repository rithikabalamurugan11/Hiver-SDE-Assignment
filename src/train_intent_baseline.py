import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score


# -----------------------------
# File paths
# -----------------------------
DATA_PATH = "results/golden_set_reviewed.csv"


# -----------------------------
# Load Golden Set
# -----------------------------
print("Loading Golden Set...")

df = pd.read_csv(DATA_PATH)

# Remove rows without labels
df = df.dropna(subset=["customer_text", "intent"])

# Remove empty labels
df = df[df["intent"].str.strip() != ""]

print("Examples available:", len(df))


# -----------------------------
# Input and target
# -----------------------------
X = df["customer_text"].astype(str)
y = df["intent"].astype(str)


# -----------------------------
# Train / Test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining examples:", len(X_train))
print("Testing examples:", len(X_test))


# -----------------------------
# TF-IDF Vectorizer
# -----------------------------
print("\nCreating TF-IDF features...")

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=1,
    max_df=0.95
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# -----------------------------
# Logistic Regression
# -----------------------------
print("Training Logistic Regression...")

model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced"
)

model.fit(X_train_tfidf, y_train)


# -----------------------------
# Predictions
# -----------------------------
print("Making predictions...")

y_pred = model.predict(X_test_tfidf)


# -----------------------------
# Evaluation
# -----------------------------
accuracy = accuracy_score(y_test, y_pred)

print("\n======================================")
print("TF-IDF + Logistic Regression Results")
print("======================================")

print("\nAccuracy:")
print(round(accuracy, 4))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))