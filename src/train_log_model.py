import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# LOAD DATA

df = pd.read_csv(
    "data/ml_ready_social_media_engagement.csv"
)

print("========== LOG-TARGET RANDOM FOREST ==========")


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


# LOG TRANSFORM TARGET

y_train_log = np.log1p(y_train)


print("\nOriginal target statistics:")
print("Mean:", y_train.mean())
print("Median:", y_train.median())
print("Maximum:", y_train.max())

print("\nLog-transformed target statistics:")
print("Mean:", y_train_log.mean())
print("Median:", y_train_log.median())
print("Maximum:", y_train_log.max())


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


# RANDOM FOREST

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)


print("\nTraining log-target Random Forest...")

model.fit(
    X_train_encoded,
    y_train_log
)

print("Model training completed!")


# PREDICTION

predicted_log = model.predict(X_test_encoded)


# Convert predictions back to original scale
y_pred = np.expm1(predicted_log)


# EVALUATION

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


# RESULTS

print("\n========== LOG MODEL PERFORMANCE ==========")

print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")


print("\n========== LOG MODEL COMPLETED ==========")