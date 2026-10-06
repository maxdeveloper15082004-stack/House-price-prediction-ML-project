import pandas as pd

df = pd.read_csv("data/cleaned/house_price_cleaned.csv")

print("Before Feature Engineering:")
print(df.head())

df["total_rooms"] = df["bedrooms"] + df["bathrooms"]

df["area_per_bedroom"] = df["area_sqft"] / df["bedrooms"]

print("\nAfter Feature Engineering:")
print(df.head())

print("\nNew Columns:")
print(df.columns)

df.to_csv(
    "data/cleaned/house_price_feature_engineered.csv",
    index=False
)

print("\nFeature engineered data saved successfully.")