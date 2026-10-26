import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# LOAD DATA

df = pd.read_csv(
    "data/ml_ready_social_media_engagement.csv"
)

print("========== MODEL COMPARISON ==========")


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


print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)


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


print("Encoded training data:", X_train_encoded.shape)
print("Encoded testing data:", X_test_encoded.shape)


# DEFINE MODELS

models = {

    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    ),

    "Extra Trees": ExtraTreesRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
}


# TRAIN AND EVALUATE

results = []


for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(
        X_train_encoded,
        y_train
    )

    y_pred = model.predict(
        X_test_encoded
    )

    mae = mean_absolute_error(
        y_test,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            y_pred
        )
    )

    r2 = r2_score(
        y_test,
        y_pred
    )

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })


# RESULTS TABLE

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="R2",
    ascending=False
)


print("\n========== MODEL COMPARISON RESULTS ==========")

print(
    results_df.to_string(
        index=False,
        formatters={
            "MAE": "{:.4f}".format,
            "RMSE": "{:.4f}".format,
            "R2": "{:.4f}".format
        }
    )
)


# BEST MODEL

best_model = results_df.iloc[0]

print("\n========== BEST MODEL ==========")

print("Model:", best_model["Model"])
print("MAE :", f"{best_model['MAE']:.4f}")
print("RMSE:", f"{best_model['RMSE']:.4f}")
print("R²  :", f"{best_model['R2']:.4f}")


# SAVE RESULTS

results_df.to_csv(
    "data/model_comparison_results.csv",
    index=False
)

print("\nModel comparison results saved successfully!")

print("\n========== MODEL COMPARISON COMPLETED ==========")