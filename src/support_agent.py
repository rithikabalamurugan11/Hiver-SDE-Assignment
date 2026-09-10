import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "results/amazon_conversations.csv"

TOP_K = 5


INTENTS = [
    "Delivery Issue",
    "Order Status",
    "Refund / Return",
    "Payment / Charges",
    "Account / Login",
    "Prime Membership",
    "Digital Content",
    "Product / Seller",
    "Technical Issue",
    "Customer Support / Escalation",
    "Other"
]


# ============================================================
# LOAD HISTORICAL AMAZON CONVERSATIONS
# ============================================================

print("Loading AmazonHelp conversations...")

df = pd.read_csv(DATA_PATH)

df = df.dropna(
    subset=[
        "customer_text",
        "amazon_reply"
    ]
).reset_index(drop=True)

df["customer_text"] = (
    df["customer_text"]
    .astype(str)
)

df["amazon_reply"] = (
    df["amazon_reply"]
    .astype(str)
)

print(
    f"Conversations loaded: {len(df)}"
)


# ============================================================
# BUILD TF-IDF RETRIEVAL INDEX
# ============================================================

print("Building retrieval index...")

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2,
    max_features=50000
)

customer_vectors = vectorizer.fit_transform(
    df["customer_text"]
)

print("Retrieval index ready.")


# ============================================================
# INTENT CLASSIFICATION
# ============================================================

def classify_intent(text):

    text = str(text).lower().strip()


    # ========================================================
    # DELIVERY ISSUE
    # ========================================================

    delivery_phrases = [

        "supposed to arrive",
        "was supposed to arrive",
        "supposed to be delivered",
        "should have arrived",
        "should've arrived",
        "was due to arrive",
        "due to arrive",
        "missed the delivery",
        "delivery was missed",
        "delivery is late",
        "delivery was late",
        "delivery is delayed",
        "delivery was delayed",
        "delivery delayed",
        "delivery delay",
        "package is late",
        "package was late",
        "order is late",
        "order was late",
        "still hasn't arrived",
        "still has not arrived",
        "still haven't received",
        "still have not received",
        "didn't arrive",
        "didnt arrive",
        "didn't come",
        "didnt come",
        "never arrived",
        "never came",
        "not delivered",
        "missing package",
        "missing parcel",
        "lost package",
        "lost parcel",
        "package missing",
        "parcel missing",
        "delivery failed",
        "delivery driver",
        "delivery address",
        "delivery date",
        "delivery postponed",
        "delivered to neighbor",
        "delivered to neighbour",
        "wrong delivery address",
        "delivered but not received",
        "only 1 delivered",
        "only one delivered",
        "not received"
    ]

    if any(
        phrase in text
        for phrase in delivery_phrases
    ):
        return "Delivery Issue"


    # ========================================================
    # REFUND / RETURN
    # ========================================================

    refund_phrases = [

        "refund",
        "refunded",
        "money back",
        "reimburse",
        "reimbursement",
        "return my item",
        "return this item",
        "return the item",
        "return order",
        "replacement",
        "replace my item",
        "cancel my order",
        "cancel the order",
        "want to return",
        "need to return"
    ]

    if any(
        phrase in text
        for phrase in refund_phrases
    ):
        return "Refund / Return"


    # ========================================================
    # PAYMENT / CHARGES
    # ========================================================

    payment_phrases = [

        "charged twice",
        "charged me twice",
        "wrong charge",
        "extra charge",
        "charged",
        "charge",
        "payment",
        "price",
        "cost",
        "invoice",
        "credit card",
        "debit card",
        "promo code",
        "discount code",
        "cashback",
        "billing"
    ]

    if any(
        phrase in text
        for phrase in payment_phrases
    ):
        return "Payment / Charges"


    # ========================================================
    # ACCOUNT / LOGIN
    # ========================================================

    account_phrases = [

        "login",
        "log in",
        "sign in",
        "password",
        "account",
        "hacked",
        "phishing",
        "security",
        "unauthorized",
        "someone accessed my account",
        "someone accessed my",
        "account compromised",
        "account was compromised",
        "mobile number",
        "phone number",
        "change my number"
    ]

    if any(
        phrase in text
        for phrase in account_phrases
    ):
        return "Account / Login"


    # ========================================================
    # PRIME MEMBERSHIP
    # ========================================================

    prime_phrases = [

        "prime membership",
        "amazon prime",
        "prime member",
        "prime trial",
        "prime subscription",
        "prime charge",
        "prime membership charge",
        "prime renewal",
        "prime subscription charge"
    ]

    if any(
        phrase in text
        for phrase in prime_phrases
    ):
        return "Prime Membership"


    # ========================================================
    # DIGITAL CONTENT
    # ========================================================

    digital_phrases = [

        "prime video",
        "kindle",
        "audible",
        "fire tv",
        "firestick",
        "fire stick",
        "ebook",
        "e-book",
        "movie",
        "music",
        "video"
    ]

    if any(
        phrase in text
        for phrase in digital_phrases
    ):
        return "Digital Content"


    # ========================================================
    # PRODUCT / SELLER
    # ========================================================

    product_phrases = [

        "seller",
        "product",
        "wrong item",
        "wrong product",
        "wrong console",
        "defective",
        "broken product",
        "counterfeit",
        "sold by",
        "availability",
        "out of stock",
        "color",
        "colour",
        "model",
        "preorder",
        "pre-order",
        "special offers",
        "special offer",
        "tablet",
        "headset"
    ]

    if any(
        phrase in text
        for phrase in product_phrases
    ):
        return "Product / Seller"


    # ========================================================
    # TECHNICAL ISSUE
    # ========================================================

    technical_phrases = [

        "app",
        "application",
        "website",
        "error",
        "bug",
        "not working",
        "doesn't work",
        "doesnt work",
        "crash",
        "crashing",
        "screen",
        "technical issue",
        "technical problem",
        "browser",
        "playstore",
        "play store",
        "android tv",
        "cannot open",
        "can't open"
    ]

    if any(
        phrase in text
        for phrase in technical_phrases
    ):
        return "Technical Issue"


    # ========================================================
    # CUSTOMER SUPPORT / ESCALATION
    # ========================================================

    support_phrases = [

        "customer service",
        "customer care",
        "customer support",
        "support",
        "agent",
        "representative",
        "supervisor",
        "manager",
        "complaint",
        "escalate",
        "escalation",
        "contact amazon",
        "call me",
        "talk to someone",
        "speak to someone",
        "speak to a human",
        "talk to a human",
        "real person",
        "human agent",
        "leadership team",
        "no response",
        "not responding"
    ]

    if any(
        phrase in text
        for phrase in support_phrases
    ):
        return "Customer Support / Escalation"


    # ========================================================
    # ORDER STATUS
    # ========================================================

    order_status_phrases = [

        "where is my order",
        "where's my order",
        "where is the order",
        "order status",
        "order number",
        "order no",
        "order details",
        "track order",
        "tracking number",
        "tracking",
        "delivery tracking",
        "delivery status",
        "shipment",
        "shipped",
        "shipping",
        "amazon shipping",
        "when will my order arrive",
        "when will my order come",
        "when is my order arriving",
        "has my order shipped"
    ]

    if any(
        phrase in text
        for phrase in order_status_phrases
    ):
        return "Order Status"


    # ========================================================
    # OTHER
    # ========================================================

    return "Other"


# ============================================================
# RETRIEVE SIMILAR HISTORICAL CONVERSATIONS
# ============================================================

def retrieve_similar(
    query,
    top_k=TOP_K,
    exclude_text=None
):

    query = str(query).strip()


    # --------------------------------------------------------
    # CONVERT QUERY INTO TF-IDF VECTOR
    # --------------------------------------------------------

    query_vector = vectorizer.transform(
        [query]
    )


    # --------------------------------------------------------
    # CALCULATE COSINE SIMILARITY
    # --------------------------------------------------------

    similarities = cosine_similarity(
        query_vector,
        customer_vectors
    ).flatten()


    # --------------------------------------------------------
    # EXCLUDE IDENTICAL CUSTOMER MESSAGES
    # --------------------------------------------------------
    #
    # This is important during evaluation.
    #
    # If the current golden-set message already exists in the
    # historical dataset, retrieving it would cause data leakage.
    #
    # We therefore remove ALL identical copies, not just one.
    # --------------------------------------------------------

    if exclude_text is not None:

        exclude_text = (
            str(exclude_text)
            .strip()
            .lower()
        )

        normalized_customer_text = (
            df["customer_text"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        matching_indices = df.index[
            normalized_customer_text
            == exclude_text
        ]

        similarities[
            matching_indices
        ] = -1.0


    # --------------------------------------------------------
    # SELECT TOP RESULTS
    # --------------------------------------------------------

    top_indices = np.argsort(
        similarities
    )[-top_k:][::-1]


    results = []


    for index in top_indices:

        similarity = float(
            similarities[index]
        )


        # Skip excluded records

        if similarity < 0:
            continue


        results.append({

            "customer_text":
                df.iloc[index]["customer_text"],

            "amazon_reply":
                df.iloc[index]["amazon_reply"],

            "similarity":
                similarity
        })


    return results


# ============================================================
# AUTO-HANDLE / ESCALATE DECISION
# ============================================================

def decide_action(
    customer_text,
    predicted_intent,
    best_similarity
):

    text = str(
        customer_text
    ).lower()


    # ========================================================
    # SECURITY / ACCOUNT RISK
    # ========================================================

    security_keywords = [

        "hacked",
        "hack",
        "phishing",
        "stolen",
        "unauthorized",
        "not me",
        "someone accessed",
        "someone has access",
        "security issue",
        "account compromised",
        "account was compromised",
        "identity theft"
    ]

    if predicted_intent == "Account / Login":

        if any(
            keyword in text
            for keyword in security_keywords
        ):

            return (
                "ESCALATE",
                "Possible account security issue requiring human investigation."
            )


    # ========================================================
    # CUSTOMER REQUESTS HUMAN SUPPORT
    # ========================================================

    human_request_keywords = [

        "speak to a human",
        "speak with a human",
        "talk to a human",
        "talk to someone",
        "speak to someone",
        "real person",
        "human agent",
        "customer service",
        "customer support",
        "customer care",
        "representative",
        "supervisor",
        "manager",
        "escalate",
        "escalation",
        "complaint",
        "call me",
        "leadership team"
    ]

    if any(
        keyword in text
        for keyword in human_request_keywords
    ):

        return (
            "ESCALATE",
            "Customer is requesting human support or escalation."
        )


    # ========================================================
    # SERIOUS DELIVERY PROBLEMS
    # ========================================================

    serious_delivery_keywords = [

        "lost package",
        "lost parcel",
        "missing package",
        "missing parcel",
        "stolen package",
        "stolen parcel",
        "delivered but not received",
        "not received",
        "never arrived",
        "never came",
        "wrong delivery address",
        "delivery failed",
        "delivery driver",
        "package damaged",
        "parcel damaged",
        "damaged package",
        "damaged parcel",
        "only 1 delivered",
        "only one delivered"
    ]

    if any(
        keyword in text
        for keyword in serious_delivery_keywords
    ):

        return (
            "ESCALATE",
            "The delivery issue may require human investigation."
        )


    # ========================================================
    # SENSITIVE PAYMENT ISSUES
    # ========================================================

    sensitive_payment_keywords = [

        "charged twice",
        "charged me twice",
        "wrong charge",
        "extra charge",
        "fraud",
        "chargeback",
        "compensation",
        "money stolen"
    ]

    if any(
        keyword in text
        for keyword in sensitive_payment_keywords
    ):

        return (
            "ESCALATE",
            "The payment issue may require account-level investigation."
        )


    # ========================================================
    # LOW HISTORICAL EVIDENCE
    # ========================================================

    if best_similarity < 0.30:

        return (
            "ESCALATE",
            "Low similarity to historical support cases; human review is safer."
        )


    # ========================================================
    # COMPLEX / SENSITIVE REQUESTS
    # ========================================================

    complex_keywords = [

        "legal",
        "lawsuit",
        "court",
        "police",
        "fraud",
        "identity theft",
        "damages"
    ]

    if any(
        keyword in text
        for keyword in complex_keywords
    ):

        return (
            "ESCALATE",
            "The request involves a potentially sensitive or complex issue."
        )


    # ========================================================
    # DEFAULT
    # ========================================================

    return (
        "AUTO-HANDLE",
        "The request matches a known support intent and has sufficient historical evidence."
    )


# ============================================================
# GENERATE HISTORICALLY GROUNDED REPLY
# ============================================================

def generate_reply(
    customer_text,
    predicted_intent,
    similar_cases=None
):

    historical_replies = []


    # --------------------------------------------------------
    # COLLECT VALID HISTORICAL REPLIES
    # --------------------------------------------------------

    if similar_cases:

        for case in similar_cases[:3]:

            reply = str(
                case.get(
                    "amazon_reply",
                    ""
                )
            ).strip()

            similarity = float(
                case.get(
                    "similarity",
                    0
                )
            )


            if (
                reply
                and similarity >= 0.20
            ):

                historical_replies.append(
                    reply
                )


    # --------------------------------------------------------
    # USE HISTORICAL REPLY
    # --------------------------------------------------------

    if historical_replies:

        best_reply = historical_replies[0]


        if len(best_reply) >= 20:

            return (
                "Based on similar historical Amazon support cases, "
                "a suitable response would be:\n\n"
                + best_reply
            )


    # ========================================================
    # FALLBACK REPLIES
    # ========================================================

    if predicted_intent == "Delivery Issue":

        return (
            "I'm sorry for the trouble with your delivery. "
            "Please check the latest tracking update and estimated "
            "delivery date. If the package has not arrived or the "
            "tracking information has not updated, our support team "
            "can investigate further."
        )


    if predicted_intent == "Order Status":

        return (
            "I'd be happy to help with your order. "
            "Please check the latest tracking information and "
            "estimated delivery date for the current status."
        )


    if predicted_intent == "Refund / Return":

        return (
            "I'm sorry for the trouble. Please check your order "
            "details for the available return or refund options. "
            "If the refund or return is not showing correctly, "
            "our support team can investigate further."
        )


    if predicted_intent == "Payment / Charges":

        return (
            "I'm sorry for the billing issue. Please review the "
            "charge in your order or payment details. If the charge "
            "appears incorrect or duplicated, our support team can "
            "investigate it further."
        )


    if predicted_intent == "Account / Login":

        return (
            "I'm sorry you're having trouble with your account. "
            "Please verify your account and login details. "
            "For security-related issues, please contact support "
            "through the official Amazon support channel."
        )


    if predicted_intent == "Prime Membership":

        return (
            "I can help with your Prime membership question. "
            "Please check your Prime membership settings and "
            "subscription details for the current status or charges."
        )


    if predicted_intent == "Digital Content":

        return (
            "I'm sorry you're having trouble with your digital "
            "content. Please check the relevant Amazon digital "
            "service and verify that your device or application "
            "is updated."
        )


    if predicted_intent == "Product / Seller":

        return (
            "I'm sorry for the trouble with the product or seller. "
            "Please check the order and product details for the "
            "available options. If the issue cannot be resolved "
            "there, our support team can investigate further."
        )


    if predicted_intent == "Technical Issue":

        return (
            "I'm sorry you're experiencing a technical issue. "
            "Please try refreshing the page or restarting the "
            "application and make sure it is updated. If the "
            "problem continues, our support team can investigate."
        )


    if predicted_intent == "Customer Support / Escalation":

        return (
            "I'm sorry for the experience you've had. "
            "This issue may require additional support. "
            "Please contact our customer support team so they "
            "can review the details and assist you further."
        )


    # ========================================================
    # OTHER
    # ========================================================

    return (
        "I'm sorry you're experiencing this issue. "
        "Please provide a few more details so our support team "
        "can better understand and assist with your request."
    )


# ============================================================
# INTERACTIVE AGENT
# ============================================================

def run_interactive_agent():

    print()
    print("=" * 70)
    print("HIVER AI SUPPORT AGENT")
    print("=" * 70)


    while True:

        customer_text = input(
            "\nEnter customer message: "
        ).strip()


        # ----------------------------------------------------
        # EXIT
        # ----------------------------------------------------

        if customer_text.lower() in [
            "q",
            "quit",
            "exit"
        ]:

            print(
                "\nExiting agent..."
            )

            break


        # ----------------------------------------------------
        # EMPTY INPUT
        # ----------------------------------------------------

        if not customer_text:

            print(
                "Please enter a customer message."
            )

            continue


        # ----------------------------------------------------
        # CLASSIFY
        # ----------------------------------------------------

        predicted_intent = classify_intent(
            customer_text
        )


        # ----------------------------------------------------
        # RETRIEVE
        # ----------------------------------------------------

        similar = retrieve_similar(
            customer_text,
            top_k=3
        )


        # ----------------------------------------------------
        # SIMILARITY
        # ----------------------------------------------------

        if len(similar) > 0:

            best_similarity = float(
                similar[0]["similarity"]
            )

        else:

            best_similarity = 0.0


        # ----------------------------------------------------
        # DECISION
        # ----------------------------------------------------

        action, reason = decide_action(
            customer_text,
            predicted_intent,
            best_similarity
        )


        # ----------------------------------------------------
        # REPLY
        # ----------------------------------------------------

        reply = generate_reply(
            customer_text,
            predicted_intent,
            similar
        )


        # ----------------------------------------------------
        # DISPLAY
        # ----------------------------------------------------

        print()
        print("-" * 70)

        print(
            f"Intent          : "
            f"{predicted_intent}"
        )

        print(
            f"Best similarity : "
            f"{best_similarity:.3f}"
        )

        print(
            f"Decision        : "
            f"{action}"
        )

        print(
            f"Reason          : "
            f"{reason}"
        )

        print()
        print("Suggested Reply:")
        print(reply)

        print()
        print("Historical Evidence:")


        for i, item in enumerate(
            similar,
            start=1
        ):

            print()

            print(
                f"[{i}] Similarity: "
                f"{item['similarity']:.3f}"
            )

            print(
                f"Customer: "
                f"{item['customer_text']}"
            )

            print(
                f"Amazon Reply: "
                f"{item['amazon_reply']}"
            )


        print(
            "-" * 70
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    run_interactive_agent()