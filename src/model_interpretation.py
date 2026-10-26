import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# LOAD DATA

df = pd.read_csv(
    "data/ml_ready_social_media_engagement.csv"
)

print("========== MODEL INTERPRETATION ==========")


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


# TRAIN DECISION TREE

model = DecisionTreeRegressor(
    random_state=42
)

model.fit(
    X_train_encoded,
    y_train
)


# PREDICTIONS

y_pred = model.predict(X_test_encoded)


# MODEL METRICS

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


print("\n========== MODEL PERFORMANCE ==========")

print("MAE :", round(mae, 4))
print("RMSE:", round(rmse, 4))
print("R²  :", round(r2, 4))



# CREATE RESULTS DATAFRAME

results = X_test.copy()

results["Actual_Engagement_Rate"] = y_test.values
results["Predicted_Engagement_Rate"] = y_pred

results["Error"] = (
    results["Actual_Engagement_Rate"]
    - results["Predicted_Engagement_Rate"]
)

results["Absolute_Error"] = (
    results["Error"].abs()
)


# SAVE RESULTS

results.to_csv(
    "data/model_interpretation_results.csv",
    index=False
)

print(
    "\nModel interpretation results saved successfully!"
)


# CREATE VISUALIZATION FOLDER

os.makedirs(
    "visualizations",
    exist_ok=True
)


# CHART 1 — ACTUAL VS PREDICTED

plt.figure(figsize=(9, 7))

sns.scatterplot(
    x=results["Actual_Engagement_Rate"],
    y=results["Predicted_Engagement_Rate"]
)

plt.title(
    "Actual vs Predicted Engagement Rate"
)

plt.xlabel(
    "Actual Engagement Rate"
)

plt.ylabel(
    "Predicted Engagement Rate"
)

plt.tight_layout()

plt.savefig(
    "visualizations/actual_vs_predicted.png",
    dpi=300
)

plt.show()


# CHART 2 — ERROR DISTRIBUTION

plt.figure(figsize=(9, 6))

sns.histplot(
    results["Error"],
    bins=40,
    kde=True
)

plt.axvline(
    0,
    linestyle="--"
)

plt.title(
    "Distribution of Prediction Errors"
)

plt.xlabel(
    "Prediction Error"
)

plt.ylabel(
    "Frequency"
)

plt.tight_layout()

plt.savefig(
    "visualizations/error_distribution.png",
    dpi=300
)

plt.show()


# CHART 3 — PERFORMANCE BY INFLUENCER TIER

tier_performance = (
    results
    .groupby("Influencer_Tier")
    .agg(
        Actual_Mean=(
            "Actual_Engagement_Rate",
            "mean"
        ),
        Predicted_Mean=(
            "Predicted_Engagement_Rate",
            "mean"
        ),
        MAE=(
            "Absolute_Error",
            "mean"
        )
    )
    .reset_index()
)


print(
    "\n========== PERFORMANCE BY INFLUENCER TIER =========="
)

print(
    tier_performance.to_string(
        index=False
    )
)


plt.figure(figsize=(10, 6))

tier_plot = tier_performance.melt(
    id_vars="Influencer_Tier",
    value_vars=[
        "Actual_Mean",
        "Predicted_Mean"
    ],
    var_name="Type",
    value_name="Engagement_Rate"
)

sns.barplot(
    data=tier_plot,
    x="Influencer_Tier",
    y="Engagement_Rate",
    hue="Type"
)

plt.title(
    "Actual vs Predicted Engagement by Influencer Tier"
)

plt.xlabel(
    "Influencer Tier"
)

plt.ylabel(
    "Average Engagement Rate"
)

plt.tight_layout()

plt.savefig(
    "visualizations/tier_actual_vs_predicted.png",
    dpi=300
)

plt.show()


# LARGEST PREDICTION ERRORS

print(
    "\n========== TOP 10 LARGEST PREDICTION ERRORS =========="
)

print(
    results[
        [
            "Platform",
            "Content_Type",
            "Influencer_Tier",
            "Actual_Engagement_Rate",
            "Predicted_Engagement_Rate",
            "Absolute_Error"
        ]
    ]
    .sort_values(
        by="Absolute_Error",
        ascending=False
    )
    .head(10)
    .to_string(index=False)
)


print(
    "\n========== MODEL INTERPRETATION COMPLETED =========="
)

print(
    "Saved: data/model_interpretation_results.csv"
)

print(
    "Saved: visualizations/actual_vs_predicted.png"
)

print(
    "Saved: visualizations/error_distribution.png"
)

print(
    "Saved: visualizations/tier_actual_vs_predicted.png"
)