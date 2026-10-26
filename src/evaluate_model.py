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

print("========== MODEL EVALUATION ==========")


# FEATURES AND TARGET


X = df.drop(columns=["Engagement_Rate"])
y = df["Engagement_Rate"]


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



# TRAIN MODEL


model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_encoded, y_train)



# PREDICTIONS


y_pred = model.predict(X_test_encoded)



# EVALUATION


mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

r2 = r2_score(y_test, y_pred)


print("\n========== MODEL PERFORMANCE ==========")

print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")

# CREATE PREDICTION RESULTS


results = X_test.copy()

results["Actual_Engagement_Rate"] = y_test.values
results["Predicted_Engagement_Rate"] = y_pred

results["Prediction_Error"] = (
    results["Actual_Engagement_Rate"]
    - results["Predicted_Engagement_Rate"]
)

results["Absolute_Error"] = (
    results["Prediction_Error"].abs()
)



# SORT BY LARGEST ERROR


largest_errors = results.sort_values(
    by="Absolute_Error",
    ascending=False
)


print("\n========== 10 LARGEST PREDICTION ERRORS ==========")

print(
    largest_errors[
        [
            "Platform",
            "Content_Type",
            "Influencer_Tier",
            "Actual_Engagement_Rate",
            "Predicted_Engagement_Rate",
            "Absolute_Error"
        ]
    ].head(10).to_string(index=False)
)



# SAVE RESULTS


results.to_csv(
    "data/model_predictions.csv",
    index=False
)

print("\nPrediction results saved successfully!")

print("\n========== MODEL EVALUATION COMPLETED ==========")