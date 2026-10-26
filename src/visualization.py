import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load dataset
df = pd.read_csv(
    "data/feature_engineered_social_media_engagement.csv"
)

# Create visualization folder if it doesn't exist
os.makedirs("visualizations", exist_ok=True)

# 1. PLATFORM VS ENGAGEMENT

platform_data = (
    df.groupby("Platform")["Engagement_Rate"]
    .mean()
    .sort_values(ascending=False)
)

plt.figure(figsize=(10, 6))
sns.barplot(
    x=platform_data.index,
    y=platform_data.values
)

plt.title("Average Engagement Rate by Platform")
plt.xlabel("Platform")
plt.ylabel("Average Engagement Rate (%)")
plt.xticks(rotation=30)
plt.tight_layout()

plt.savefig(
    "visualizations/platform_engagement.png",
    dpi=300
)

plt.show()
plt.close()


# 2. CONTENT TYPE VS ENGAGEMENT

content_data = (
    df.groupby("Content_Type")["Engagement_Rate"]
    .mean()
    .sort_values(ascending=False)
)

plt.figure(figsize=(12, 7))
sns.barplot(
    x=content_data.values,
    y=content_data.index
)

plt.title("Average Engagement Rate by Content Type")
plt.xlabel("Average Engagement Rate (%)")
plt.ylabel("Content Type")
plt.tight_layout()

plt.savefig(
    "visualizations/content_type_engagement.png",
    dpi=300
)

plt.show()
plt.close()


# 3. DAY OF WEEK VS ENGAGEMENT

day_data = (
    df.groupby("Day_of_Week")["Engagement_Rate"]
    .mean()
)

day_order = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
]

day_data = day_data.reindex(day_order)

plt.figure(figsize=(10, 6))
sns.barplot(
    x=day_data.index,
    y=day_data.values
)

plt.title("Average Engagement Rate by Day of Week")
plt.xlabel("Day of Week")
plt.ylabel("Average Engagement Rate (%)")
plt.xticks(rotation=30)
plt.tight_layout()

plt.savefig(
    "visualizations/day_engagement.png",
    dpi=300
)

plt.show()
plt.close()


# 4. HOUR OF DAY VS ENGAGEMENT

hour_data = (
    df.groupby("Hour_of_Day")["Engagement_Rate"]
    .mean()
    .sort_index()
)

plt.figure(figsize=(12, 6))
sns.lineplot(
    x=hour_data.index,
    y=hour_data.values,
    marker="o"
)

plt.title("Average Engagement Rate by Hour of Day")
plt.xlabel("Hour of Day")
plt.ylabel("Average Engagement Rate (%)")
plt.xticks(range(0, 24))
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "visualizations/hour_engagement.png",
    dpi=300
)

plt.show()
plt.close()


# 5. SENTIMENT VS ENGAGEMENT

sentiment_data = (
    df.groupby("Sentiment")["Engagement_Rate"]
    .mean()
    .sort_values(ascending=False)
)

plt.figure(figsize=(8, 6))
sns.barplot(
    x=sentiment_data.index,
    y=sentiment_data.values
)

plt.title("Average Engagement Rate by Sentiment")
plt.xlabel("Sentiment")
plt.ylabel("Average Engagement Rate (%)")
plt.tight_layout()

plt.savefig(
    "visualizations/sentiment_engagement.png",
    dpi=300
)

plt.show()
plt.close()


# 6. INFLUENCER TIER VS ENGAGEMENT

influencer_data = (
    df.groupby("Influencer_Tier")["Engagement_Rate"]
    .mean()
    .sort_values(ascending=False)
)

plt.figure(figsize=(9, 6))
sns.barplot(
    x=influencer_data.index,
    y=influencer_data.values
)

plt.title("Average Engagement Rate by Influencer Tier")
plt.xlabel("Influencer Tier")
plt.ylabel("Average Engagement Rate (%)")
plt.tight_layout()

plt.savefig(
    "visualizations/influencer_tier_engagement.png",
    dpi=300
)

plt.show()
plt.close()


print("\n===================================")
print("ALL VISUALIZATIONS CREATED!")
print("===================================")