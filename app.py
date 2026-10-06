import streamlit as st
import pandas as pd
import joblib

model = joblib.load("models/house_price_model.pkl")

st.title("🏠 House Price Prediction")

st.write("Enter the house details below to predict the price.")

area_sqft = st.number_input(
    "Area (sqft)",
    min_value=100,
    value=1800
)

bedrooms = st.number_input(
    "Bedrooms",
    min_value=1,
    value=3
)

bathrooms = st.number_input(
    "Bathrooms",
    min_value=1,
    value=2
)

age_years = st.number_input(
    "Age of House (years)",
    min_value=0,
    value=5
)

parking = st.number_input(
    "Parking Spaces",
    min_value=0,
    value=1
)

floor = st.number_input(
    "Floor",
    min_value=0,
    value=2
)

location_score = st.number_input(
    "Location Score",
    min_value=1,
    max_value=10,
    value=8
)

if st.button("Predict Price"):

    total_rooms = bedrooms + bathrooms

    area_per_bedroom = area_sqft / bedrooms

    new_house = pd.DataFrame({
        "area_sqft": [area_sqft],
        "bedrooms": [bedrooms],
        "bathrooms": [bathrooms],
        "age_years": [age_years],
        "parking": [parking],
        "floor": [floor],
        "location_score": [location_score],
        "total_rooms": [total_rooms],
        "area_per_bedroom": [area_per_bedroom]
    })

    prediction = model.predict(new_house)

    st.success(
        f"Predicted House Price: ₹{prediction[0]:,.2f}"
    )