import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



# LOAD DATA

df = pd.read_csv(
    "data/ml_ready_social_media_engagement.csv"
)

print("========== FINAL MODEL TRAINING ==========")


# FEATURES AND TARGET

X = df.drop(
    columns=["Engagement_Rate"]
)

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


# PREPROCESSING

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


# FINAL MODEL

model = DecisionTreeRegressor(
    random_state=42
)


# COMPLETE PIPELINE

final_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            model
        )
    ]
)


# TRAIN TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# TRAIN

print("\nTraining final model...")

final_pipeline.fit(
    X_train,
    y_train
)

print("Training completed!")


# PREDICTION

y_pred = final_pipeline.predict(
    X_test
)



# EVALUATION

mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = mean_squared_error(
    y_test,
    y_pred
) ** 0.5

r2 = r2_score(
    y_test,
    y_pred
)


print("\n========== FINAL MODEL PERFORMANCE ==========")

print(
    "MAE :",
    round(mae, 4)
)

print(
    "RMSE:",
    round(rmse, 4)
)

print(
    "R²  :",
    round(r2, 4)
)


# CREATE MODELS FOLDER

os.makedirs(
    "models",
    exist_ok=True
)


# SAVE MODEL

model_path = (
    "models/final_model.pkl"
)

joblib.dump(
    final_pipeline,
    model_path
)


print(
    "\nFinal model saved successfully!"
)

print(
    "Location:",
    model_path
)

print(
    "\n========== FINAL MODEL READY =========="
)