import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DATA_PATH = "results/amazon_conversations.csv"


print("Loading AmazonHelp conversations...")

df = pd.read_csv(DATA_PATH)

df = df.dropna(
    subset=["customer_text", "amazon_reply"]
)

print("Conversations loaded:", len(df))


# ------------------------------------------------
# Create TF-IDF representation of customer messages
# ------------------------------------------------

print("\nCreating search index...")

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2,
    max_features=50000
)

customer_vectors = vectorizer.fit_transform(
    df["customer_text"].astype(str)
)

print("Search index created.")


# ------------------------------------------------
# Function to find similar historical conversations
# ------------------------------------------------

def find_similar_messages(query, top_k=5):

    query_vector = vectorizer.transform([query])

    similarities = cosine_similarity(
        query_vector,
        customer_vectors
    ).flatten()

    top_indices = similarities.argsort()[-top_k:][::-1]

    results = df.iloc[top_indices].copy()

    results["similarity"] = similarities[top_indices]

    return results


# ------------------------------------------------
# Test the retrieval system
# ------------------------------------------------

query = input(
    "\nEnter a customer message: "
)

results = find_similar_messages(
    query,
    top_k=5
)


print("\n======================================")
print("SIMILAR HISTORICAL CONVERSATIONS")
print("======================================")


for i, (_, row) in enumerate(
    results.iterrows(),
    start=1
):

    print(f"\n--- Result {i} ---")

    print("\nCustomer:")
    print(row["customer_text"])

    print("\nHistorical AmazonHelp Reply:")
    print(row["amazon_reply"])

    print("\nSimilarity:")
    print(round(row["similarity"], 4))


print("\n======================================")