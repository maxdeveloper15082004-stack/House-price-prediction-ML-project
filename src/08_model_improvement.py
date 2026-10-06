import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


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


# Linear Regression

linear_model = LinearRegression()

linear_model.fit(
    X_train,
    y_train
)

linear_pred = linear_model.predict(
    X_test
)

linear_mae = mean_absolute_error(
    y_test,
    linear_pred
)

linear_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        linear_pred
    )
)

linear_r2 = r2_score(
    y_test,
    linear_pred
)


# Decision Tree

tree_model = DecisionTreeRegressor(
    random_state=42
)

tree_model.fit(
    X_train,
    y_train
)

tree_pred = tree_model.predict(
    X_test
)

tree_mae = mean_absolute_error(
    y_test,
    tree_pred
)

tree_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        tree_pred
    )
)

tree_r2 = r2_score(
    y_test,
    tree_pred
)


# Random Forest

forest_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

forest_model.fit(
    X_train,
    y_train
)

forest_pred = forest_model.predict(
    X_test
)

forest_mae = mean_absolute_error(
    y_test,
    forest_pred
)

forest_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        forest_pred
    )
)

forest_r2 = r2_score(
    y_test,
    forest_pred
)


# Model Comparison

results = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Decision Tree",
        "Random Forest"
    ],
    "MAE": [
        linear_mae,
        tree_mae,
        forest_mae
    ],
    "RMSE": [
        linear_rmse,
        tree_rmse,
        forest_rmse
    ],
    "R2": [
        linear_r2,
        tree_r2,
        forest_r2
    ]
})

print("\nModel Comparison")
print(results)


# Cross Validation

cv_scores = cross_val_score(
    forest_model,
    X,
    y,
    cv=5,
    scoring="r2"
)

print("\nCross Validation R2 Scores:")
print(cv_scores)

print(
    "Average Cross Validation R2:",
    cv_scores.mean()
)


# Hyperparameter Tuning

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5]
}

grid_search = GridSearchCV(
    RandomForestRegressor(
        random_state=42
    ),
    param_grid,
    cv=5,
    scoring="r2",
    n_jobs=-1
)

grid_search.fit(
    X_train,
    y_train
)


print("\nBest Parameters:")
print(grid_search.best_params_)


# Best Model

best_model = grid_search.best_estimator_

final_pred = best_model.predict(
    X_test
)


# Final Evaluation

final_mae = mean_absolute_error(
    y_test,
    final_pred
)

final_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        final_pred
    )
)

final_r2 = r2_score(
    y_test,
    final_pred
)


print("\nFinal Model Results")
print("-------------------")

print("MAE:", final_mae)
print("RMSE:", final_rmse)
print("R2:", final_r2)