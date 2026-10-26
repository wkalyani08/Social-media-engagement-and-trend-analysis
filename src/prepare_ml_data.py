import pandas as pd

# Load feature-engineered dataset
df = pd.read_csv(
    "data/feature_engineered_social_media_engagement.csv"
)

print("========== ML DATA PREPARATION ==========")

# Convert timestamp
df["Timestamp"] = pd.to_datetime(df["Timestamp"])

# Features that can reasonably be known before/around posting
feature_columns = [
    "Platform",
    "Content_Type",
    "Category",
    "Hour_of_Day",
    "Day_of_Week",
    "Hashtag_Count",
    "Content_Length",
    "Sentiment",
    "Influencer_Tier",
    "Has_Media",
    "Is_Verified",
    "Follower_Count",
    "Year",
    "Month",
    "Day_Number",
    "Month_Number"
]

target_column = "Engagement_Rate"

X = df[feature_columns]
y = df[target_column]

print("\nFeatures:")
print(X.columns.tolist())

print("\nTarget:")
print(target_column)

print("\nFeature shape:", X.shape)
print("Target shape:", y.shape)

print("\nMissing values in features:")
print(X.isnull().sum())

print("\nMissing values in target:")
print(y.isnull().sum())

# Save ML-ready dataset
ml_data = df[feature_columns + [target_column]]

ml_data.to_csv(
    "data/ml_ready_social_media_engagement.csv",
    index=False
)

print("\nML-ready dataset saved successfully!")

print("\n========== ML DATA PREPARATION COMPLETED ==========")