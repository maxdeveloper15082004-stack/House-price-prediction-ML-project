import pandas as pd
import joblib

model = joblib.load(
    "models/house_price_model.pkl"
)

new_house = pd.DataFrame({
    "area_sqft": [1800],
    "bedrooms": [3],
    "bathrooms": [2],
    "age_years": [5],
    "parking": [1],
    "floor": [2],
    "location_score": [8]
})

new_house["total_rooms"] = (
    new_house["bedrooms"] +
    new_house["bathrooms"]
)

new_house["area_per_bedroom"] = (
    new_house["area_sqft"] /
    new_house["bedrooms"]
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

prediction = model.predict(
    new_house[features]
)

print("Predicted House Price:", prediction[0])