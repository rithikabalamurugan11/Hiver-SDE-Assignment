
import pandas as pd
import sys
import os


# ============================================================
# IMPORT SUPPORT AGENT
# ============================================================

sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "src"
    )
)

from support_agent import (
    classify_intent,
    retrieve_similar,
    decide_action,
    generate_reply
)


# ============================================================
# FILE PATHS
# ============================================================

INPUT_PATH = "results/golden_set_human.csv"
OUTPUT_PATH = "results/agent_evaluation.csv"


# ============================================================
# START
# ============================================================

print("=" * 80)
print("HIVER AI SUPPORT AGENT EVALUATION")
print("=" * 80)


# ============================================================
# LOAD HUMAN-LABELLED GOLDEN SET
# ============================================================

print("\nLoading human-labelled golden set...")

df = pd.read_csv(INPUT_PATH)

df = df.dropna(
    subset=[
        "customer_text",
        "human_intent"
    ]
)

df["customer_text"] = (
    df["customer_text"]
    .astype(str)
    .str.strip()
)

df["human_intent"] = (
    df["human_intent"]
    .astype(str)
    .str.strip()
)

# Remove rows without a human label

df = df[
    df["human_intent"] != ""
].reset_index(drop=True)


print(
    f"Examples available: {len(df)}"
)


# ============================================================
# EVALUATE EACH EXAMPLE
# ============================================================

results = []


for index, row in df.iterrows():

    customer_text = str(
        row["customer_text"]
    ).strip()

    actual_intent = str(
        row["human_intent"]
    ).strip()


    # ========================================================
    # 1. INTENT CLASSIFICATION
    # ========================================================

    predicted_intent = classify_intent(
        customer_text
    )


    # ========================================================
    # 2. HISTORICAL RETRIEVAL
    # ========================================================
    #
    # IMPORTANT:
    #
    # exclude_text removes ALL historical records having
    # exactly the same customer message.
    #
    # This prevents evaluation leakage.
    # ========================================================

    similar = retrieve_similar(
        customer_text,
        top_k=3,
        exclude_text=customer_text
    )


    # ========================================================
    # 3. SAFETY CHECK FOR EXACT DUPLICATE
    # ========================================================

    normalized_query = (
        customer_text
        .strip()
        .lower()
    )

    leakage_detected = False

    for case in similar:

        retrieved_customer = (
            str(
                case.get(
                    "customer_text",
                    ""
                )
            )
            .strip()
            .lower()
        )

        if retrieved_customer == normalized_query:

            leakage_detected = True

            # Do not allow this result to influence evaluation

            break


    # ========================================================
    # 4. REMOVE ANY LEAKED RESULT
    # ========================================================

    if leakage_detected:

        similar = [
            case
            for case in similar
            if (
                str(
                    case.get(
                        "customer_text",
                        ""
                    )
                )
                .strip()
                .lower()
                != normalized_query
            )
        ]

        print(
            f"WARNING: Exact duplicate detected and removed "
            f"for example {index + 1}"
        )


    # ========================================================
    # 5. BEST SIMILARITY
    # ========================================================

    if len(similar) > 0:

        best_similarity = float(
            similar[0].get(
                "similarity",
                0.0
            )
        )

    else:

        best_similarity = 0.0


    # ========================================================
    # 6. HISTORICAL EVIDENCE
    # ========================================================

    evidence_customer = ""
    evidence_reply = ""
    evidence_similarity = 0.0


    if len(similar) > 0:

        evidence_customer = str(
            similar[0].get(
                "customer_text",
                ""
            )
        )

        evidence_reply = str(
            similar[0].get(
                "amazon_reply",
                ""
            )
        )

        evidence_similarity = float(
            similar[0].get(
                "similarity",
                0.0
            )
        )


    # ========================================================
    # 7. AUTO-HANDLE / ESCALATE DECISION
    # ========================================================

    action, reason = decide_action(
        customer_text,
        predicted_intent,
        best_similarity
    )


    # ========================================================
    # 8. REPLY GENERATION
    # ========================================================

    reply = generate_reply(
        customer_text,
        predicted_intent,
        similar
    )


    # ========================================================
    # 9. CHECK INTENT
    # ========================================================

    intent_correct = (
        predicted_intent
        == actual_intent
    )


    # ========================================================
    # 10. STORE RESULT
    # ========================================================

    results.append({

        "example_id":
            index + 1,

        "customer_text":
            customer_text,

        "actual_intent":
            actual_intent,

        "predicted_intent":
            predicted_intent,

        "intent_correct":
            intent_correct,

        "best_similarity":
            best_similarity,

        "action":
            action,

        "reason":
            reason,

        "suggested_reply":
            reply,

        "evidence_customer_text":
            evidence_customer,

        "evidence_amazon_reply":
            evidence_reply,

        "evidence_similarity":
            evidence_similarity,

        "leakage_detected":
            leakage_detected
    })


    # ========================================================
    # 11. PRINT PROGRESS
    # ========================================================

    print(
        f"Example {index + 1}: "
        f"{predicted_intent} | "
        f"{action} | "
        f"Similarity={best_similarity:.3f}"
    )


# ============================================================
# CREATE RESULTS DATAFRAME
# ============================================================

result_df = pd.DataFrame(
    results
)


# ============================================================
# CREATE RESULTS DIRECTORY
# ============================================================

os.makedirs(
    "results",
    exist_ok=True
)


# ============================================================
# SAVE RESULTS
# ============================================================

result_df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# CALCULATE METRICS
# ============================================================

total = len(
    result_df
)


if total > 0:

    # --------------------------------------------------------
    # INTENT ACCURACY
    # --------------------------------------------------------

    correct = int(
        result_df[
            "intent_correct"
        ].sum()
    )

    intent_accuracy = (
        correct / total
    )


    # --------------------------------------------------------
    # AUTO-HANDLE
    # --------------------------------------------------------

    auto_handle_count = int(
        (
            result_df["action"]
            == "AUTO-HANDLE"
        ).sum()
    )


    # --------------------------------------------------------
    # ESCALATE
    # --------------------------------------------------------

    escalate_count = int(
        (
            result_df["action"]
            == "ESCALATE"
        ).sum()
    )


    # --------------------------------------------------------
    # RATES
    # --------------------------------------------------------

    auto_handle_rate = (
        auto_handle_count / total
    )

    escalation_rate = (
        escalate_count / total
    )


    # --------------------------------------------------------
    # AVERAGE SIMILARITY
    # --------------------------------------------------------

    average_similarity = (
        result_df[
            "best_similarity"
        ].mean()
    )


    # --------------------------------------------------------
    # LEAKAGE COUNT
    # --------------------------------------------------------

    leakage_count = int(
        result_df[
            "leakage_detected"
        ].sum()
    )

else:

    correct = 0
    intent_accuracy = 0.0

    auto_handle_count = 0
    escalate_count = 0

    auto_handle_rate = 0.0
    escalation_rate = 0.0

    average_similarity = 0.0

    leakage_count = 0


# ============================================================
# FINAL RESULTS
# ============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "AGENT EVALUATION RESULTS"
)

print(
    "=" * 80
)


print(
    f"Total examples: "
    f"{total}"
)


print(
    f"Correct intents: "
    f"{correct}"
)


print(
    f"Intent accuracy: "
    f"{intent_accuracy:.3f}"
)


print(
    f"AUTO-HANDLE: "
    f"{auto_handle_count} "
    f"({auto_handle_rate:.1%})"
)


print(
    f"ESCALATE: "
    f"{escalate_count} "
    f"({escalation_rate:.1%})"
)


print(
    f"Average similarity: "
    f"{average_similarity:.3f}"
)


print(
    f"Exact-duplicate leakage detected: "
    f"{leakage_count}"
)


print(
    f"\nResults saved to: "
    f"{OUTPUT_PATH}"
)


print(
    "=" * 80
)


# ============================================================
# EVALUATION STATUS
# ============================================================

if leakage_count == 0:

    print(
        "\nPASS: No exact duplicate leakage detected."
    )

else:

    print(
        "\nWARNING: Some exact duplicates were detected "
        "and removed before evaluation."
    )

print(
    "=" * 80
)
