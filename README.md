# House Price Prediction ML Project

## Project Overview

This is an end-to-end Machine Learning project that predicts house prices using property features.

**Workflow:** Data Collection → Data Understanding → Data Cleaning → EDA → Feature Engineering → Train/Test Split → Model Training → Model Evaluation → Model Improvement → Model Saving → Prediction → Streamlit Deployment

## Problem Statement

Predict the price of a house using:

- Area in square feet
- Bedrooms
- Bathrooms
- Age of the house
- Parking spaces
- Floor
- Location score

## Dataset

The project uses a synthetic dataset containing 100 house records.

### Features

| Feature | Description |
|---|---|
| `area_sqft` | House area in square feet |
| `bedrooms` | Number of bedrooms |
| `bathrooms` | Number of bathrooms |
| `age_years` | Age of the house |
| `parking` | Number of parking spaces |
| `floor` | Floor number |
| `location_score` | Location quality score |
| `price` | Target house price |

## Feature Engineering

Two features were created:

- `total_rooms` = `bedrooms + bathrooms`
- `area_per_bedroom` = `area_sqft / bedrooms`

## Models

The project compares:

1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor

Evaluation metrics:

- MAE — Mean Absolute Error
- RMSE — Root Mean Squared Error
- R² Score

Cross-validation and GridSearchCV were also used for model improvement.

## Final Model

Based on the current evaluation, Linear Regression performed best on the test dataset and was selected as the final model for this learning project.

Saved model:

```text
models/house_price_model.pkl
```

> The dataset is synthetic and contains only 100 records, so the results should not be treated as real-world house-price prediction performance.

## Project Structure

```text
house predict project/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── raw/
│   │   └── house_price_raw_100.csv
│   └── cleaned/
│       ├── house_price_cleaned.csv
│       └── house_price_feature_engineered.csv
│
├── models/
│   └── house_price_model.pkl
│
└── src/
    ├── 01_load_and_understand_data.py
    ├── 02_data_cleaning.py
    ├── 03_eda.py
    ├── 04_feature_engineering.py
    ├── 05_train_test_split.py
    ├── 06_model_training.py
    ├── 07_model_evaluation.py
    ├── 08_model_improvement.py
    ├── 09_save_final_model.py
    └── 10_predict.py
```

## Installation

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Run Prediction

```bash
python src/10_predict.py
```

## Run Streamlit App

```bash
streamlit run app.py
```

The application allows users to enter house details and receive a predicted house price.

## Example Input

```text
Area: 1800 sqft
Bedrooms: 3
Bathrooms: 2
Age: 5 years
Parking: 1
Floor: 2
Location Score: 8
```

## Skills Demonstrated

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Data Cleaning
- Exploratory Data Analysis
- Feature Engineering
- Regression
- Train/Test Split
- Model Evaluation
- Cross Validation
- Hyperparameter Tuning
- Joblib
- Streamlit
- End-to-End Machine Learning

## Future Improvements

- Use a larger real-world dataset
- Add more property and location features
- Compare additional regression models
- Build a stronger preprocessing pipeline
- Improve model validation
- Deploy the application online

## Disclaimer

This project is for learning and portfolio purposes. The dataset is synthetic and should not be used for real-world property valuation.
