import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeRegressor


# LOAD DATA

df = pd.read_csv(
    "data/ml_ready_social_media_engagement.csv"
)

print("========== FEATURE IMPORTANCE ANALYSIS ==========")


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


# TRAIN DECISION TREE

model = DecisionTreeRegressor(
    random_state=42
)

model.fit(
    X_train_encoded,
    y_train
)


# GET FEATURE NAMES

encoded_feature_names = (
    preprocessor
    .get_feature_names_out()
)


# FEATURE IMPORTANCE

importance_df = pd.DataFrame({
    "Feature": encoded_feature_names,
    "Importance": model.feature_importances_
})


importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)


# DISPLAY TOP 20

print("\n========== TOP 20 IMPORTANT FEATURES ==========")

print(
    importance_df.head(20).to_string(
        index=False
    )
)


# SAVE RESULTS

importance_df.to_csv(
    "data/feature_importance.csv",
    index=False
)


# VISUALIZATION

os.makedirs(
    "visualizations",
    exist_ok=True
)


top_features = importance_df.head(15)


plt.figure(figsize=(10, 7))

sns.barplot(
    data=top_features,
    x="Importance",
    y="Feature"
)

plt.title(
    "Top 15 Features Influencing Engagement Rate"
)

plt.xlabel("Feature Importance")

plt.ylabel("Feature")

plt.tight_layout()

plt.savefig(
    "visualizations/feature_importance.png",
    dpi=300
)

plt.show()


print(
    "\nFeature importance chart saved successfully!"
)

print(
    "Saved: data/feature_importance.csv"
)

print(
    "Saved: visualizations/feature_importance.png"
)

print(
    "\n========== FEATURE IMPORTANCE COMPLETED =========="
)