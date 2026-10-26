import pandas as pd

# Load feature-engineered dataset
df = pd.read_csv(
    "data/feature_engineered_social_media_engagement.csv"
)

print("========== EDA STARTED ==========")


# 1. Dataset information


print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
print(df.info())


# 2. Platform analysis


print("\n========== PLATFORM ANALYSIS ==========")

platform_engagement = (
    df.groupby("Platform")["Engagement_Rate"]
    .mean()
    .sort_values(ascending=False)
)

print(platform_engagement)


# 3. Content type analysis

print("\n========== CONTENT TYPE ANALYSIS ==========")

content_engagement = (
    df.groupby("Content_Type")["Engagement_Rate"]
    .mean()
    .sort_values(ascending=False)
)

print(content_engagement)

# 4. Day of week analysis

print("\n========== DAY OF WEEK ANALYSIS ==========")

day_engagement = (
    df.groupby("Day_of_Week")["Engagement_Rate"]
    .mean()
    .sort_values(ascending=False)
)

print(day_engagement)

# 5. Hour analysis


print("\n========== HOUR ANALYSIS ==========")

hour_engagement = (
    df.groupby("Hour_of_Day")["Engagement_Rate"]
    .mean()
    .sort_values(ascending=False)
)

print(hour_engagement.head(10))


# 6. Sentiment analysis

print("\n========== SENTIMENT ANALYSIS ==========")

sentiment_engagement = (
    df.groupby("Sentiment")["Engagement_Rate"]
    .mean()
    .sort_values(ascending=False)
)

print(sentiment_engagement)

# 7. Influencer tier analysis

print("\n========== INFLUENCER TIER ANALYSIS ==========")

influencer_engagement = (
    df.groupby("Influencer_Tier")["Engagement_Rate"]
    .mean()
    .sort_values(ascending=False)
)

print(influencer_engagement)

print("\n========== EDA COMPLETED ==========")