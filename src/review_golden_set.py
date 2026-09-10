import pandas as pd


INPUT_PATH = "results/golden_set_reviewed.csv"
OUTPUT_PATH = "results/golden_set_human.csv"


INTENTS = {
    "1": "Delivery Issue",
    "2": "Order Status",
    "3": "Refund / Return",
    "4": "Payment / Charges",
    "5": "Account / Login",
    "6": "Prime Membership",
    "7": "Digital Content",
    "8": "Product / Seller",
    "9": "Technical Issue",
    "10": "Customer Support / Escalation",
    "11": "Other"
}


# ============================================================
# LOAD DATA
# ============================================================

print("Loading Golden Set...")

df = pd.read_csv(INPUT_PATH)

df = df.dropna(
    subset=["customer_text", "amazon_reply"]
).reset_index(drop=True)


# ============================================================
# PREPARE COLUMNS
# ============================================================

# Force text columns to string/object type
# This prevents pandas dtype errors with empty values.

if "human_intent" not in df.columns:
    df["human_intent"] = ""
else:
    df["human_intent"] = df["human_intent"].fillna("").astype(str)

if "label_notes" not in df.columns:
    df["label_notes"] = ""
else:
    df["label_notes"] = df["label_notes"].fillna("").astype(str)


# ============================================================
# RESUME FROM PREVIOUS PROGRESS
# ============================================================

start_index = 0

for i in range(len(df)):

    value = str(
        df.loc[i, "human_intent"]
    ).strip()

    if value == "" or value.lower() == "nan":
        start_index = i
        break
else:
    start_index = len(df)


# ============================================================
# DISPLAY INTENTS
# ============================================================

print("\n======================================")
print("GOLDEN SET HUMAN REVIEW")
print("======================================")

print("\nChoose the correct intent:\n")

for number, intent in INTENTS.items():
    print(f"{number}. {intent}")


print("\n--------------------------------------")
print("Instructions:")
print("Enter the number of the correct intent.")
print("Enter 'q' to save and quit.")
print("--------------------------------------")


# ============================================================
# REVIEW EACH EXAMPLE
# ============================================================

for i in range(start_index, len(df)):

    row = df.iloc[i]

    print("\n")
    print("=" * 80)
    print(f"Example {i + 1} / {len(df)}")
    print("=" * 80)

    print("\nCUSTOMER MESSAGE:")
    print(row["customer_text"])

    print("\nHISTORICAL AMAZON REPLY:")
    print(row["amazon_reply"])

    while True:

        choice = input(
            "\nYour intent (1-11, or q to quit): "
        ).strip()

        if choice.lower() == "q":

            df.to_csv(
                OUTPUT_PATH,
                index=False
            )

            print("\nProgress saved!")
            print("Saved to:")
            print(OUTPUT_PATH)

            raise SystemExit

        if choice in INTENTS:

            selected_intent = INTENTS[choice]

            df.loc[
                i,
                "human_intent"
            ] = selected_intent

            note = input(
                "Optional note (press Enter to skip): "
            ).strip()

            df.loc[
                i,
                "label_notes"
            ] = note

            # Save after every example
            df.to_csv(
                OUTPUT_PATH,
                index=False
            )

            print(
                f"Saved: {selected_intent}"
            )

            break

        print(
            "Invalid choice. Please enter 1-11 or q."
        )


# ============================================================
# FINISHED
# ============================================================

print("\n======================================")
print("HUMAN REVIEW COMPLETED!")
print("======================================")

print("\nTotal reviewed:", len(df))

print("\nIntent distribution:")

print(
    df["human_intent"].value_counts()
)

print("\nSaved to:")
print(OUTPUT_PATH)