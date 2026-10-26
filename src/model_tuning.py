import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# LOAD DATA

df = pd.read_csv(
    "data/ml_ready_social_media_engagement.csv"
)

print("========== MODEL HYPERPARAMETER TUNING ==========")


# FEATURES AND TARGET

X = df.drop(columns=["Engagement_Rate"])
y = df["Engagement_Rate"]


# FEATURE TYPES

categorical_features = [
    "Platform",
    "Content_Type",
    "Category",
    "Day_of_Week",
    "Sentiment",
    "Influencer_Tier"
]

numerical_features = [
    "Hour_of_Day",
    "Hashtag_Count",
    "Content_Length",
    "Follower_Count",
    "Year",
    "Month",
    "Day_Number",
    "Month_Number"
]


# TRAIN TEST SPLIT


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# ENCODING

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_features
        )
    ],
    remainder="passthrough"
)


X_train_encoded = preprocessor.fit_transform(X_train)

X_test_encoded = preprocessor.transform(X_test)


# DECISION TREE TUNING

print("\n========== TUNING DECISION TREE ==========")

decision_tree = DecisionTreeRegressor(
    random_state=42
)

dt_parameters = {
    "max_depth": [5, 8, 12, 16, 20, None],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf": [1, 2, 5, 10]
}


dt_grid = GridSearchCV(
    estimator=decision_tree,
    param_grid=dt_parameters,
    cv=5,
    scoring="neg_mean_absolute_error",
    n_jobs=-1
)


dt_grid.fit(
    X_train_encoded,
    y_train
)


print("\nBest Decision Tree parameters:")
print(dt_grid.best_params_)

print(
    "Best CV MAE:",
    -dt_grid.best_score_
)


# EVALUATE DECISION TREE

best_dt = dt_grid.best_estimator_

dt_predictions = best_dt.predict(
    X_test_encoded
)

dt_mae = mean_absolute_error(
    y_test,
    dt_predictions
)

dt_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        dt_predictions
    )
)

dt_r2 = r2_score(
    y_test,
    dt_predictions
)


print("\nDecision Tree test performance:")
print(f"MAE  : {dt_mae:.4f}")
print(f"RMSE : {dt_rmse:.4f}")
print(f"R²   : {dt_r2:.4f}")


# GRADIENT BOOSTING TUNING

print("\n========== TUNING GRADIENT BOOSTING ==========")

gradient_boosting = GradientBoostingRegressor(
    random_state=42
)


gb_parameters = {
    "n_estimators": [100, 200],
    "learning_rate": [0.03, 0.05, 0.1],
    "max_depth": [2, 3, 4],
    "min_samples_leaf": [2, 5, 10]
}


gb_grid = GridSearchCV(
    estimator=gradient_boosting,
    param_grid=gb_parameters,
    cv=5,
    scoring="neg_mean_absolute_error",
    n_jobs=-1
)


gb_grid.fit(
    X_train_encoded,
    y_train
)


print("\nBest Gradient Boosting parameters:")
print(gb_grid.best_params_)

print(
    "Best CV MAE:",
    -gb_grid.best_score_
)


# EVALUATE GRADIENT BOOSTING

best_gb = gb_grid.best_estimator_

gb_predictions = best_gb.predict(
    X_test_encoded
)

gb_mae = mean_absolute_error(
    y_test,
    gb_predictions
)

gb_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        gb_predictions
    )
)

gb_r2 = r2_score(
    y_test,
    gb_predictions
)


print("\nGradient Boosting test performance:")
print(f"MAE  : {gb_mae:.4f}")
print(f"RMSE : {gb_rmse:.4f}")
print(f"R²   : {gb_r2:.4f}")


# FINAL COMPARISON

comparison = pd.DataFrame({
    "Model": [
        "Tuned Decision Tree",
        "Tuned Gradient Boosting"
    ],
    "MAE": [
        dt_mae,
        gb_mae
    ],
    "RMSE": [
        dt_rmse,
        gb_rmse
    ],
    "R2": [
        dt_r2,
        gb_r2
    ]
})


print("\n========== TUNED MODEL COMPARISON ==========")

print(
    comparison.to_string(
        index=False,
        formatters={
            "MAE": "{:.4f}".format,
            "RMSE": "{:.4f}".format,
            "R2": "{:.4f}".format
        }
    )
)


# SAVE RESULTS


comparison.to_csv(
    "data/tuned_model_results.csv",
    index=False
)


print("\nTuned model results saved successfully!")

print("\n========== MODEL TUNING COMPLETED ==========")