import pandas as pd
import joblib
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# LOAD TREND DATA

df = pd.read_csv(
    "data/trend_prediction_data.csv"
)

df["Date"] = pd.to_datetime(df["Date"])

print("==========================================")
print("       TREND PREDICTION MODEL")
print("==========================================")

print("\nDataset shape:", df.shape)


# SORT CHRONOLOGICALLY

df = df.sort_values("Date").reset_index(drop=True)


# FEATURES AND TARGET

features = [
    "Day",
    "Month",
    "Year",
    "Day_Number",
    "Previous_Day_Engagement",
    "Previous_2_Day_Engagement",
    "Previous_3_Day_Engagement",
    "Rolling_3_Day_Average",
    "Post_Count"
]

target = "Average_Engagement_Rate"


X = df[features]
y = df[target]


# TIME-BASED TRAIN / TEST SPLIT

split_index = int(len(df) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]


print("\nTraining records:", len(X_train))
print("Testing records :", len(X_test))

print("\nTraining period:")
print(df["Date"].iloc[0], "to", df["Date"].iloc[split_index - 1])

print("\nTesting period:")
print(df["Date"].iloc[split_index], "to", df["Date"].iloc[-1])


# TRAIN MODEL

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)


print("\nTraining trend prediction model...")

model.fit(
    X_train,
    y_train
)

print("Training completed!")


# PREDICTION

y_pred = model.predict(X_test)


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


print("\n==========================================")
print("       TREND MODEL PERFORMANCE")
print("==========================================")

print("MAE :", round(mae, 4))
print("RMSE:", round(rmse, 4))
print("R²  :", round(r2, 4))


# SAVE PREDICTIONS

results = df.iloc[split_index:].copy()

results["Actual_Engagement"] = y_test.values

results["Predicted_Engagement"] = y_pred

results["Prediction_Error"] = (
    results["Actual_Engagement"]
    - results["Predicted_Engagement"]
)

results.to_csv(
    "data/trend_model_predictions.csv",
    index=False
)


# SAVE MODEL

os.makedirs(
    "models",
    exist_ok=True
)

joblib.dump(
    model,
    "models/trend_model.pkl"
)


# FINAL MESSAGE

print("\nTrend predictions saved successfully!")

print(
    "File: data/trend_model_predictions.csv"
)

print(
    "Model: models/trend_model.pkl"
)

print("\n==========================================")
print("       TRAINING TREND MODEL COMPLETE")
print("==========================================")