import pandas as pd

DATA_PATH = "data/twcs.csv"

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

# Keep AmazonHelp tweets
amazon = df[df["author_id"] == "AmazonHelp"].copy()

# Create lookup using tweet_id
tweets_by_id = df.set_index("tweet_id")

conversations = []

for _, reply in amazon.iterrows():

    # AmazonHelp reply should point to the customer's tweet
    customer_id = reply["in_response_to_tweet_id"]

    if pd.isna(customer_id):
        continue

    if customer_id not in tweets_by_id.index:
        continue

    customer = tweets_by_id.loc[customer_id]

    # Make sure the original tweet is from a customer
    if customer["inbound"] != True:
        continue

    conversations.append({
        "customer_text": customer["text"],
        "amazon_reply": reply["text"]
    })

# Convert to DataFrame
conversation_df = pd.DataFrame(conversations)

# Remove duplicates
conversation_df = conversation_df.drop_duplicates()

# Save
conversation_df.to_csv(
    "results/amazon_conversations.csv",
    index=False
)

print("\nConversations collected:")
print(len(conversation_df))

print("\nSaved to:")
print("results/amazon_conversations.csv")

print("\nSample conversations:\n")

print(
    conversation_df.head(10).to_string(index=False)
)