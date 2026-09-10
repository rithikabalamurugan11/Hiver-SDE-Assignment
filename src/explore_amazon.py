import pandas as pd

DATA_PATH = "data/twcs.csv"

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

# Get AmazonHelp tweets
amazon = df[df["author_id"] == "AmazonHelp"]

print("\nTotal AmazonHelp tweets:")
print(len(amazon))

print("\nSample AmazonHelp tweets:")
print(amazon[["tweet_id", "inbound", "text"]].head(20).to_string(index=False))