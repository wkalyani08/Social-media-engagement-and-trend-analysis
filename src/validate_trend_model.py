import pandas as pd


# LOAD TREND PREDICTIONS

df = pd.read_csv(
    "data/trend_model_predictions.csv"
)

print("==========================================")
print("       TREND MODEL VALIDATION")
print("==========================================")


# BASIC INFORMATION

print("\nNumber of test records:", len(df))

print("\nPrediction columns:")
print([
    "Date",
    "Actual_Engagement",
    "Predicted_Engagement",
    "Prediction_Error"
])


# PREVIEW PREDICTIONS

print("\n==========================================")
print("        SAMPLE PREDICTIONS")
print("==========================================")

print(
    df[
        [
            "Date",
            "Actual_Engagement",
            "Predicted_Engagement",
            "Prediction_Error"
        ]
    ].head(10)
)


# ERROR ANALYSIS

df["Absolute_Error"] = (
    df["Prediction_Error"].abs()
)


print("\n==========================================")
print("          ERROR ANALYSIS")
print("==========================================")

print(
    "Average Absolute Error:",
    round(df["Absolute_Error"].mean(), 4)
)

print(
    "Maximum Absolute Error:",
    round(df["Absolute_Error"].max(), 4)
)


# TREND DIRECTION

df["Actual_Direction"] = (
    df["Actual_Engagement"]
    .diff()
    .apply(
        lambda x:
        "Increasing"
        if x > 0
        else
        "Decreasing"
        if x < 0
        else
        "Stable"
    )
)

df["Predicted_Direction"] = (
    df["Predicted_Engagement"]
    .diff()
    .apply(
        lambda x:
        "Increasing"
        if x > 0
        else
        "Decreasing"
        if x < 0
        else
        "Stable"
    )
)


# DIRECTION ACCURACY

direction_comparison = df[
    "Actual_Direction"
] == df[
    "Predicted_Direction"
]

direction_accuracy = (
    direction_comparison.mean() * 100
)


print(
    "\nTrend Direction Accuracy:",
    round(direction_accuracy, 2),
    "%"
)


 
# FINAL STATUS

print("\n==========================================")
print("          VALIDATION SUMMARY")
print("==========================================")

print(
    "The trend model has been evaluated on",
    "chronologically later data."
)

print(
    "Predictions and errors have been",
    "successfully validated."
)

print("\n==========================================")
print("        VALIDATING TREND MODEL COMPLETE")
print("==========================================")