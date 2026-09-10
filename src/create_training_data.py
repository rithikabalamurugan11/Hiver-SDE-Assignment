import pandas as pd
import re


DATA_PATH = "results/amazon_conversations.csv"
OUTPUT_PATH = "results/training_data.csv"


print("Loading AmazonHelp conversations...")

df = pd.read_csv(DATA_PATH)

print("Total conversations:", len(df))


def classify_intent(text):
    text = str(text).lower()

    # Delivery Issue
    if any(word in text for word in [
        "late", "delayed", "delay", "not delivered",
        "didn't arrive", "didnt arrive", "never arrived",
        "missing package", "lost package", "delivery failed",
        "delivery driver", "delivered to neighbor",
        "delivery address", "delivery date", "package",
        "parcel", "courier"
    ]):
        return "Delivery Issue"

    # Order Status
    if any(word in text for word in [
        "where is my order", "order status", "tracking",
        "track my order", "when will my order",
        "when is my order", "shipping status",
        "out for delivery", "on the way", "dispatch"
    ]):
        return "Order Status"

    # Refund / Return
    if any(word in text for word in [
        "refund", "return", "replacement", "money back",
        "reimburse", "reimbursement", "cancel my order"
    ]):
        return "Refund / Return"

    # Payment / Charges
    if any(word in text for word in [
        "charged", "charge", "payment", "price",
        "cost", "invoice", "credit card", "debit card",
        "promo code", "discount code", "cashback"
    ]):
        return "Payment / Charges"

    # Account / Login
    if any(word in text for word in [
        "login", "log in", "sign in", "password",
        "account", "hacked", "phishing", "security",
        "unauthorized"
    ]):
        return "Account / Login"

    # Prime Membership
    if any(word in text for word in [
        "prime membership", "amazon prime", "prime member",
        "prime trial", "prime subscription", "prime charge"
    ]):
        return "Prime Membership"

    # Digital Content
    if any(word in text for word in [
        "prime video", "kindle", "audible", "fire tv",
        "ebook", "movie", "video", "music"
    ]):
        return "Digital Content"

    # Product / Seller
    if any(word in text for word in [
        "seller", "product", "wrong item", "wrong product",
        "defective", "broken product", "counterfeit",
        "sold by", "availability", "color", "model"
    ]):
        return "Product / Seller"

    # Technical Issue
    if any(word in text for word in [
        "app", "website", "error", "bug", "not working",
        "crash", "screen", "technical", "login error"
    ]):
        return "Technical Issue"

    # Customer Support / Escalation
    if any(word in text for word in [
        "customer service", "customer care", "support",
        "agent", "representative", "supervisor",
        "complaint", "escalate", "escalation",
        "contact amazon", "no response"
    ]):
        return "Customer Support / Escalation"

    return "Other"


print("\nCreating intent labels...")

df["intent"] = df["customer_text"].apply(classify_intent)


# Save only the required columns
training_df = df[
    ["customer_text", "amazon_reply", "intent"]
]

training_df.to_csv(
    OUTPUT_PATH,
    index=False
)


print("\n======================================")
print("Training Data Created")
print("======================================")

print("\nTotal examples:")
print(len(training_df))

print("\nIntent distribution:")
print(training_df["intent"].value_counts())

print("\nSaved to:")
print(OUTPUT_PATH)