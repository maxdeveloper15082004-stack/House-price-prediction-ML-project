import pandas as pd
import joblib
import os

from sklearn.linear_model import LinearRegression

df = pd.read_csv(
    "data/cleaned/house_price_feature_engineered.csv"
)

features = [
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

X = df[features]
y = df["price"]

model = LinearRegression()

model.fit(X, y)

joblib.dump(
    model, 
    "models/house_price_model.pkl"
)

print("Final model trained successfully.")
print("Model saved successfully.")