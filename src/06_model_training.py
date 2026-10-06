import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv(
    "data/cleaned/house_price_feature_engineered.csv"
)

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

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train)

print("Model training completed!")

print("\nModel coefficients:")
print(model.coef_)

print("\nModel intercept:")
print(model.intercept_)

y_pred = model.predict(X_test)

comparison = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": y_pred
})

print("\nActual vs Predicted:")
print(comparison.head(10))