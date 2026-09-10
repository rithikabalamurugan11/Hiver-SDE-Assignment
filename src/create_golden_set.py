import pandas as pd

INPUT_PATH = "results/golden_set.csv"
OUTPUT_PATH = "results/golden_set_reviewed.csv"

print("Loading existing Golden Set...")

df = pd.read_csv(INPUT_PATH)

print("Examples found:", len(df))


def classify_intent(text):
    text = str(text).lower()

    # ==================================================
    # 1. DELIVERY ISSUE
    # Actual delivery problem, delay, missing package,
    # wrong address, damaged delivery, etc.
    # ==================================================
    delivery_words = [
        "late",
        "delayed",
        "delay",
        "not delivered",
        "not arrive",
        "hasn't arrived",
        "hasnt arrived",
        "missing package",
        "missing parcel",
        "lost parcel",
        "lost order",
        "delivery failed",
        "delivery is late",
        "delivery date missed",
        "missed delivery",
        "delivery postponed",
        "wrong address",
        "delivered to neighbor",
        "delivered to neighbour",
        "courier",
        "parcel",
        "package",
        "delivery guy",
        "delivery driver",
        "throwing",
        "thrown",
        "mailbox",
        "can't deliver",
        "cant deliver",
        "unable to deliver",
        "delivery process",
        "shipping address",
        "didn't receive",
        "didnt receive",
        "never received",
        "only 1 delivered",
        "only one delivered",
        "won't be delivered",
        "wont be delivered",
        "delivery promise",
        "guaranteed delivery",
        "one day delivery",
        "next day delivery",
        "two-day delivery",
        "2day shipping",
        "same day delivery"
    ]

    if any(word in text for word in delivery_words):
        return "Delivery Issue"


    # ==================================================
    # 2. REFUND / RETURN
    # ==================================================
    refund_words = [
        "refund",
        "refunded",
        "return",
        "returned",
        "replacement",
        "money back",
        "return pickup",
        "return pick",
        "reverse pickup",
        "reimburse",
        "reimbursement",
        "cancel and refund",
        "cancelled and refund"
    ]

    if any(word in text for word in refund_words):
        return "Refund / Return"


    # ==================================================
    # 3. ACCOUNT / LOGIN
    # ==================================================
    account_words = [
        "account",
        "login",
        "log in",
        "sign in",
        "sign-in",
        "password",
        "locked account",
        "account hacked",
        "account on hold",
        "access to my account",
        "my account",
        "mobile number listed",
        "phone number",
        "account closed"
    ]

    if any(word in text for word in account_words):
        return "Account / Login"


    # ==================================================
    # 4. PRIME MEMBERSHIP
    # ==================================================
    prime_words = [
        "prime membership",
        "prime member",
        "prime subscription",
        "prime trial",
        "prime account",
        "cancel my prime",
        "cancelled my prime",
        "canceled my prime",
        "renew prime",
        "prime benefit",
        "prime benefits",
        "subscribing to prime",
        "subscribing",
        "amazon prime membership"
    ]

    if any(word in text for word in prime_words):
        return "Prime Membership"


    # ==================================================
    # 5. DIGITAL CONTENT
    # ==================================================
    digital_words = [
        "prime video",
        "amazon video",
        "streaming",
        "movie",
        "movies",
        "episode",
        "digital content",
        "kindle",
        "subtitle",
        "subtitles",
        "video",
        "watch"
    ]

    if any(word in text for word in digital_words):
        return "Digital Content"


    # ==================================================
    # 6. PAYMENT / CHARGES
    # ==================================================
    payment_words = [
        "charged",
        "charge",
        "payment",
        "paid",
        "credit card",
        "debit card",
        "billing",
        "fee",
        "cashback",
        "money deducted",
        "money was deducted",
        "transaction",
        "price",
        "priced",
        "price mismatch",
        "different prices",
        "voucher",
        "promo code",
        "payment failed",
        "payment has failed",
        "paid for"
    ]

    if any(word in text for word in payment_words):
        return "Payment / Charges"


    # ==================================================
    # 7. TECHNICAL ISSUE
    # ==================================================
    technical_words = [
        "app",
        "application",
        "website",
        "error",
        "technical",
        "not working",
        "doesn't work",
        "doesnt work",
        "broken",
        "fire tv",
        "firestick",
        "echo",
        "device",
        "blank screen",
        "screen",
        "slow",
        "bug",
        "system",
        "browser",
        "cache",
        "token",
        "software",
        "firmware"
    ]

    if any(word in text for word in technical_words):
        return "Technical Issue"


    # ==================================================
    # 8. PRODUCT / SELLER
    # ==================================================
    product_words = [
        "product",
        "seller",
        "third party",
        "third-party",
        "stock",
        "in stock",
        "out of stock",
        "item",
        "wrong product",
        "wrong console",
        "wrong color",
        "counterfeit",
        "fake product",
        "defective product",
        "product availability",
        "available now",
        "available",
        "drone",
        "tablet",
        "phone case"
    ]

    if any(word in text for word in product_words):
        return "Product / Seller"


    # ==================================================
    # 9. CUSTOMER SUPPORT / ESCALATION
    # ==================================================
    support_words = [
        "customer service",
        "customer care",
        "support",
        "representative",
        "agent",
        "supervisor",
        "helpline",
        "contact",
        "call",
        "no response",
        "nobody replied",
        "escalate",
        "escalation",
        "complaint",
        "complaining",
        "rude representative",
        "rude reps",
        "customer service reps"
    ]

    if any(word in text for word in support_words):
        return "Customer Support / Escalation"


    # ==================================================
    # 10. ORDER STATUS
    # ==================================================
    order_words = [
        "order",
        "ordered",
        "order id",
        "order number",
        "tracking",
        "status",
        "shipment",
        "shipping",
        "dispatch",
        "dispatching",
        "out for delivery",
        "on the way",
        "expected delivery",
        "delivery date",
        "arriving today",
        "arrive today"
    ]

    if any(word in text for word in order_words):
        return "Order Status"


    # ==================================================
    # 11. OTHER
    # ==================================================
    return "Other"


# Generate suggested labels
df["suggested_intent"] = df["customer_text"].apply(
    classify_intent
)

# Keep the existing label for comparison
df["previous_intent"] = df["intent"]

# Replace intent with the improved suggestion
df["intent"] = df["suggested_intent"]

# Remove temporary column
df = df.drop(columns=["suggested_intent"])

# Add notes column if it doesn't exist
if "label_notes" not in df.columns:
    df["label_notes"] = ""

# Save reviewed version
df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n======================================")
print("Golden Set review file created!")
print("======================================")

print("\nTotal examples:")
print(len(df))

print("\nNew intent distribution:")
print(df["intent"].value_counts())

print("\nSaved to:")
print(OUTPUT_PATH)

print("\nIMPORTANT:")
print("Review these labels manually before submission.")
print("The labels are suggestions, not ground truth.")