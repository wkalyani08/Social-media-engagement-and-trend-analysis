import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


# LOAD DATA

df = pd.read_csv(
    "data/ml_ready_social_media_engagement.csv"
)

print("========== ML PREPROCESSING ==========")

# Separate features and target

X = df.drop(columns=["Engagement_Rate"])
y = df["Engagement_Rate"]


# DEFINE FEATURE TYPES

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


print("\nCategorical features:")
print(categorical_features)

print("\nNumerical features:")
print(numerical_features)


# TRAIN TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


print("\n========== TRAIN TEST SPLIT ==========")

print("Training features:", X_train.shape)
print("Testing features:", X_test.shape)

print("Training target:", y_train.shape)
print("Testing target:", y_test.shape)


# ONE-HOT ENCODING

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


# Fit on training data
X_train_encoded = preprocessor.fit_transform(X_train)

# Transform testing data
X_test_encoded = preprocessor.transform(X_test)


# CHECK ENCODED DATA

print("\n========== ENCODING COMPLETED ==========")

print("Encoded training shape:", X_train_encoded.shape)
print("Encoded testing shape:", X_test_encoded.shape)

print("\nOriginal feature count:", X.shape[1])
print("Encoded feature count:", X_train_encoded.shape[1])


print("\n========== ML PREPROCESSING COMPLETED ==========")