import pandas as pd

DATA_PATH = "data/twcs.csv"

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

# Get the tweet IDs of AmazonHelp tweets
amazon_ids = set(
    df.loc[df["author_id"] == "AmazonHelp", "tweet_id"].astype("int64")
)

# Convert response IDs to numbers
reply_to = pd.to_numeric(
    df["in_response_to_tweet_id"],
    errors="coerce"
)

# Keep only customer tweets that directly reply to AmazonHelp
amazon_customer_messages = df[
    (df["inbound"] == True) &
    (reply_to.isin(amazon_ids))
].copy()

print("\nAmazon customer messages found:")
print(len(amazon_customer_messages))

# Take a random sample of 200
sample = amazon_customer_messages.sample(
    n=min(200, len(amazon_customer_messages)),
    random_state=42
)

# Keep useful columns
sample = sample[
    ["tweet_id", "text", "in_response_to_tweet_id"]
]

# Save the sample
sample.to_csv(
    "results/amazon_customer_sample.csv",
    index=False
)

print("\nSample saved:")
print("results/amazon_customer_sample.csv")