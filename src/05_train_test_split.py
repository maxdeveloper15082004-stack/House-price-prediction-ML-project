import pandas as pd

from sklearn.model_selection import train_test_split

df = pd.read_csv(
    "data/cleaned/house_price_feature_engineered.csv"
)

print("Dataset shape:", df.shape)

X = df[
    [
        "area_sqft",
        "bedrooms",
        "bathrooms",
        "age_years",
        "parking",
        "floor",
        "location_score",
        "total_rooms",
        "area_per_bedroom"
    ]
]

y = df["price"]

print("\nX shape:", X.shape)
print("y shape:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining data:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\nTesting data:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)