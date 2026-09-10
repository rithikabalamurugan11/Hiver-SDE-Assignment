import pandas as pd

DATA_PATH = "data/twcs.csv"

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

# Keep only AmazonHelp replies
amazon_replies = df[
    (df["author_id"] == "AmazonHelp") &
    (df["inbound"] == False)
].copy()

# Create a lookup using tweet_id
tweets_by_id = df.set_index("tweet_id")

print("\nAmazonHelp replies found:")
print(len(amazon_replies))

print("\nSample customer + AmazonHelp conversations:\n")

count = 0

for _, reply in amazon_replies.iterrows():

    customer_id = reply["in_response_to_tweet_id"]

    if pd.isna(customer_id):
        continue

    if customer_id not in tweets_by_id.index:
        continue

    customer = tweets_by_id.loc[customer_id]

    if customer["inbound"] != True:
        continue

    print("CUSTOMER:")
    print(customer["text"])

    print("\nAMAZONHELP:")
    print(reply["text"])

    print("\n" + "=" * 80 + "\n")

    count += 1

    if count == 10:
        break