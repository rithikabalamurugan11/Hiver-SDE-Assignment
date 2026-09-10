import pandas as pd

# Location of our dataset
DATA_PATH = "data/twcs.csv"

# Read the dataset
df = pd.read_csv(DATA_PATH)

print("Dataset shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nNumber of tweets by author:")
print(df["author_id"].value_counts().head(15))