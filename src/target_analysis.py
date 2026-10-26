import pandas as pd

df = pd.read_csv(
    "data/ml_ready_social_media_engagement.csv"
)

print("========== TARGET VARIABLE ANALYSIS ==========")

target = df["Engagement_Rate"]


# BASIC STATISTICS

print("\n========== ENGAGEMENT RATE STATISTICS ==========")

print("Minimum :", target.min())
print("Maximum :", target.max())
print("Mean    :", target.mean())
print("Median  :", target.median())
print("Std Dev :", target.std())

print("\nQuartiles:")
print(target.quantile([0.25, 0.50, 0.75, 0.90, 0.95, 0.99]))


# TOP 10

print("\n========== TOP 10 ENGAGEMENT RATES ==========")

print(
    df[
        [
            "Platform",
            "Content_Type",
            "Influencer_Tier",
            "Follower_Count",
            "Engagement_Rate"
        ]
    ]
    .sort_values(
        by="Engagement_Rate",
        ascending=False
    )
    .head(10)
    .to_string(index=False)
)


# BOTTOM 10

print("\n========== BOTTOM 10 ENGAGEMENT RATES ==========")

print(
    df[
        [
            "Platform",
            "Content_Type",
            "Influencer_Tier",
            "Follower_Count",
            "Engagement_Rate"
        ]
    ]
    .sort_values(
        by="Engagement_Rate",
        ascending=True
    )
    .head(10)
    .to_string(index=False)
)


# INFLUENCER TIER STATISTICS

print("\n========== ENGAGEMENT RATE BY INFLUENCER TIER ==========")

tier_stats = (
    df.groupby("Influencer_Tier")["Engagement_Rate"]
    .agg(["count", "mean", "median", "min", "max"])
    .sort_values("mean", ascending=False)
)

print(tier_stats)


# EXTREME VALUES

print("\n========== EXTREME VALUE CHECK ==========")

print(
    "Values above 100:",
    (target > 100).sum()
)

print(
    "Values above 500:",
    (target > 500).sum()
)

print(
    "Values above 1000:",
    (target > 1000).sum()
)

print(
    "Values above 2000:",
    (target > 2000).sum()
)


print("\n========== TARGET ANALYSIS COMPLETED ==========")