import pandas as pd
from collections import Counter
import re

DATA_PATH = "results/amazon_conversations.csv"

print("Loading AmazonHelp conversations...")

df = pd.read_csv(DATA_PATH)

# Common words that do not help identify the customer problem
stopwords = {
    "this", "that", "have", "your", "with", "from", "they",
    "what", "when", "will", "been", "just", "service",
    "customer", "time", "today", "please", "help", "there",
    "back", "even", "need", "call", "says", "thanks", "item",
    "after", "want", "already", "same", "about", "again",
    "like", "amazon", "amazonhelp", "https"
}

texts = df["customer_text"].dropna().astype(str).str.lower()

words = []

for text in texts:
    text_words = re.findall(r"\b[a-z]{4,}\b", text)

    for word in text_words:
        if word not in stopwords:
            words.append(word)

word_counts = Counter(words)

print("\nUseful words in customer messages:\n")

for word, count in word_counts.most_common(50):
    print(f"{word}: {count}")