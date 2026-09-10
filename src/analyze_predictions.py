import pandas as pd


INPUT_PATH = "results/golden_predictions.csv"

print("Loading predictions...")

df = pd.read_csv(INPUT_PATH)

# Find incorrect predictions
wrong = df[
    df["intent"] != df["predicted_intent"]
].copy()

print("\n======================================")
print("Prediction Error Analysis")
print("======================================")

print("\nTotal examples:", len(df))
print("Incorrect predictions:", len(wrong))
print(
    "Correct predictions:",
    len(df) - len(wrong)
)

print("\nError rate:")
print(round(len(wrong) / len(df), 4))


# -----------------------------------
# Most common confusion pairs
# -----------------------------------

wrong["confusion"] = (
    wrong["intent"]
    + "  -->  "
    + wrong["predicted_intent"]
)

print("\n======================================")
print("Most Common Confusions")
print("======================================")

print(
    wrong["confusion"]
    .value_counts()
    .head(15)
)


# -----------------------------------
# Show real failure examples
# -----------------------------------

print("\n======================================")
print("Example Failure Cases")
print("======================================")

for _, row in wrong.head(15).iterrows():

    print("\nCUSTOMER:")
    print(row["customer_text"])

    print("\nTRUE INTENT:")
    print(row["intent"])

    print("\nPREDICTED INTENT:")
    print(row["predicted_intent"])

    print("\nAMAZON REPLY:")
    print(row["amazon_reply"])

    print("\n" + "-" * 80)


# -----------------------------------
# Save errors
# -----------------------------------

wrong[
    [
        "customer_text",
        "amazon_reply",
        "intent",
        "predicted_intent"
    ]
].to_csv(
    "results/prediction_errors.csv",
    index=False
)

print("\nError file saved to:")
print("results/prediction_errors.csv")