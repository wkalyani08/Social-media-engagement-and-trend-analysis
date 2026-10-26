import pandas as pd

# Load the raw dataset
df = pd.read_csv("data/social_media_engagement_dataset.csv")

print("========== BEFORE CLEANING ==========")
print("Rows and columns:", df.shape)


# 1. Check duplicate rows


print("\nDuplicate rows:", df.duplicated().sum())

# Remove completely duplicate rows
df = df.drop_duplicates()


# 2. Check duplicate Post IDs


print("Duplicate Post_IDs:", df["Post_ID"].duplicated().sum())


# 3. Convert Timestamp to datetime

df["Timestamp"] = pd.to_datetime(df["Timestamp"])


# 4. Check numerical columns for negative values


numeric_columns = [
    "Likes",
    "Comments",
    "Shares",
    "Views",
    "Saves",
    "Follower_Count",
    "Engagement_Rate",
    "Hour_of_Day",
    "Hashtag_Count",
    "Content_Length"
]

print("\nNegative values:")
for column in numeric_columns:
    print(column, (df[column] < 0).sum())


# 5. Check categorical values


print("\nPlatforms:")
print(df["Platform"].value_counts())

print("\nContent Types:")
print(df["Content_Type"].value_counts())

print("\nSentiment:")
print(df["Sentiment"].value_counts())

print("\nInfluencer Tiers:")
print(df["Influencer_Tier"].value_counts())


# 6. Final shape


print("\n========== AFTER CLEANING ==========")
print("Rows and columns:", df.shape)


# 7. Save cleaned dataset


df.to_csv("data/cleaned_social_media_engagement.csv", index=False)

print("\nCleaned dataset saved successfully!")