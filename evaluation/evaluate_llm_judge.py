import pandas as pd
import os

INPUT_PATH = "results/agent_evaluation.csv"
OUTPUT_PATH = "results/llm_judge_results.csv"

print("=" * 80)
print("SUPPORT REPLY QUALITY EVALUATION")
print("=" * 80)

df = pd.read_csv(INPUT_PATH)

print(f"Examples available: {len(df)}")


def evaluate_reply(row):
    customer = str(row["customer_text"]).lower()
    reply = str(row["suggested_reply"]).lower()

    # -------------------------------------------------
    # 1. RELEVANCE
    # -------------------------------------------------

    customer_words = {
        word.strip(".,!?")
        for word in customer.split()
        if len(word.strip(".,!?")) >= 5
    }

    reply_words = {
        word.strip(".,!?")
        for word in reply.split()
        if len(word.strip(".,!?")) >= 5
    }

    overlap = len(customer_words.intersection(reply_words))

    if overlap >= 6:
        relevance = 5
    elif overlap >= 4:
        relevance = 4
    elif overlap >= 2:
        relevance = 3
    elif overlap >= 1:
        relevance = 2
    else:
        relevance = 1

    # -------------------------------------------------
    # 2. HELPFULNESS
    # -------------------------------------------------

    helpful_terms = [
        "check",
        "contact",
        "track",
        "refund",
        "return",
        "order",
        "account",
        "support",
        "delivery",
        "payment",
        "please",
        "help",
        "assist"
    ]

    helpful_count = sum(
        1 for term in helpful_terms
        if term in reply
    )

    helpfulness = min(5, max(1, helpful_count))

    # -------------------------------------------------
    # 3. GROUNDEDNESS
    # -------------------------------------------------

    # Current agent uses retrieved historical Amazon
    # support replies as evidence.

    if (
        "historical amazon support cases" in reply
        and len(reply) >= 40
    ):
        groundedness = 4
    elif len(reply) >= 40:
        groundedness = 3
    else:
        groundedness = 2

    # -------------------------------------------------
    # 4. PROFESSIONALISM
    # -------------------------------------------------

    professionalism = 5

    if len(reply.strip()) < 20:
        professionalism = 2
    elif len(reply.strip()) < 40:
        professionalism = 3

    # -------------------------------------------------
    # 5. SAFETY
    # -------------------------------------------------

    unsafe_terms = [
        "guarantee",
        "guaranteed",
        "100%",
        "definitely",
        "certainly",
        "will definitely"
    ]

    if any(term in reply for term in unsafe_terms):
        safety = 2
    else:
        safety = 5

    # -------------------------------------------------
    # 6. EVIDENCE SUPPORT
    # -------------------------------------------------

    evidence_supported = (
        "historical amazon support cases" in reply
    )

    # -------------------------------------------------
    # 7. OVERALL
    # -------------------------------------------------

    overall = round(
        (
            relevance
            + helpfulness
            + groundedness
            + professionalism
            + safety
        ) / 5,
        2
    )

    return {
        "relevance": relevance,
        "helpfulness": helpfulness,
        "groundedness": groundedness,
        "professionalism": professionalism,
        "safety": safety,
        "evidence_supported": evidence_supported,
        "overall_score": overall
    }


results = []

for i, row in df.iterrows():

    scores = evaluate_reply(row)

    results.append({
        "customer_text": row["customer_text"],
        "actual_intent": row["actual_intent"],
        "predicted_intent": row["predicted_intent"],
        "intent_correct": row["intent_correct"],
        "action": row["action"],
        "reason": row["reason"],
        "suggested_reply": row["suggested_reply"],

        "relevance": scores["relevance"],
        "helpfulness": scores["helpfulness"],
        "groundedness": scores["groundedness"],
        "professionalism": scores["professionalism"],
        "safety": scores["safety"],
        "evidence_supported": scores["evidence_supported"],
        "overall_score": scores["overall_score"]
    })

    print(
        f"Example {i + 1}: "
        f"Overall={scores['overall_score']}/5 | "
        f"Evidence={scores['evidence_supported']}"
    )


result_df = pd.DataFrame(results)

os.makedirs("results", exist_ok=True)

result_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n" + "=" * 80)
print("EVALUATION SUMMARY")
print("=" * 80)

if len(result_df) > 0:

    print(
        f"Average Relevance: "
        f"{result_df['relevance'].mean():.2f}/5"
    )

    print(
        f"Average Helpfulness: "
        f"{result_df['helpfulness'].mean():.2f}/5"
    )

    print(
        f"Average Groundedness: "
        f"{result_df['groundedness'].mean():.2f}/5"
    )

    print(
        f"Average Professionalism: "
        f"{result_df['professionalism'].mean():.2f}/5"
    )

    print(
        f"Average Safety: "
        f"{result_df['safety'].mean():.2f}/5"
    )

    print(
        f"Average Overall Score: "
        f"{result_df['overall_score'].mean():.2f}/5"
    )

    evidence_rate = (
        result_df["evidence_supported"].mean() * 100
    )

    print(
        f"Evidence-supported replies: "
        f"{evidence_rate:.1f}%"
    )

print("\nResults saved to:")
print(OUTPUT_PATH)

print("=" * 80)