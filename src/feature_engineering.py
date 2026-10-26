import pandas as pd

# Load cleaned dataset
df = pd.read_csv("data/cleaned_social_media_engagement.csv")

print("========== BEFORE FEATURE ENGINEERING ==========")
print("Shape:", df.shape)

# Convert Timestamp to datetime
df["Timestamp"] = pd.to_datetime(df["Timestamp"])


# DATE & TIME FEATURES

df["Post_Date"] = df["Timestamp"].dt.date

df["Year"] = df["Timestamp"].dt.year

df["Month"] = df["Timestamp"].dt.month

df["Day"] = df["Timestamp"].dt.day

df["Day_Number"] = df["Timestamp"].dt.dayofweek

df["Month_Number"] = df["Timestamp"].dt.month

# TOTAL ENGAGEMENT

df["Total_Engagement"] = (
    df["Likes"]
    + df["Comments"]
    + df["Shares"]
    + df["Saves"]
)

# ENGAGEMENT PER VIEW

df["Engagement_Per_View"] = (
    df["Total_Engagement"] / df["Views"]
)

# DISPLAY NEW FEATURES

print("\n========== NEW FEATURES ==========")

print(
    df[
        [
            "Timestamp",
            "Post_Date",
            "Year",
            "Month",
            "Day",
            "Day_Number",
            "Total_Engagement",
            "Engagement_Per_View"
        ]
    ].head()
)

# FINAL SHAPE

print("\n========== AFTER FEATURE ENGINEERING ==========")
print("Shape:", df.shape)

# Save feature-engineered dataset
df.to_csv(
    "data/feature_engineered_social_media_engagement.csv",
    index=False
)

print("\nFeature-engineered dataset saved successfully!")