import pandas as pd
import numpy as np


# LOAD DATA


df = pd.read_csv(
    "data/social_media_engagement_dataset.csv"
)

print("========== ENGAGEMENT RATE FORMULA ANALYSIS ==========")



# CALCULATE TOTAL ENGAGEMENT

df["Calculated_Total_Engagement"] = (
    df["Likes"]
    + df["Comments"]
    + df["Shares"]
    + df["Saves"]
)


# POSSIBLE ENGAGEMENT RATE FORMULAS

# Formula 1:
# Total Engagement / Followers × 100

df["Formula_1"] = (
    df["Calculated_Total_Engagement"]
    / df["Follower_Count"]
) * 100


# Formula 2:
# Total Engagement / Views × 100

df["Formula_2"] = (
    df["Calculated_Total_Engagement"]
    / df["Views"]
) * 100


# Formula 3:
# Likes / Followers × 100

df["Formula_3"] = (
    df["Likes"]
    / df["Follower_Count"]
) * 100


# Formula 4:
# Likes / Views × 100

df["Formula_4"] = (
    df["Likes"]
    / df["Views"]
) * 100


# COMPARE FORMULAS WITH ACTUAL ENGAGEMENT RATE

print("\n========== CORRELATION WITH ACTUAL ENGAGEMENT RATE ==========")

comparison_columns = [
    "Engagement_Rate",
    "Formula_1",
    "Formula_2",
    "Formula_3",
    "Formula_4"
]

print(
    df[comparison_columns]
    .corr()["Engagement_Rate"]
    .sort_values(ascending=False)
)


# SHOW EXAMPLES

print("\n========== SAMPLE COMPARISON ==========")

print(
    df[
        [
            "Likes",
            "Comments",
            "Shares",
            "Saves",
            "Views",
            "Follower_Count",
            "Engagement_Rate",
            "Formula_1",
            "Formula_2",
            "Formula_3",
            "Formula_4"
        ]
    ]
    .head(10)
    .to_string(index=False)
)


# DIFFERENCE FROM FORMULA 1

df["Formula_1_Difference"] = (
    df["Engagement_Rate"] - df["Formula_1"]
).abs()

print("\n========== FORMULA 1 DIFFERENCE ==========")

print(
    "Average absolute difference:",
    df["Formula_1_Difference"].mean()
)

print(
    "Maximum absolute difference:",
    df["Formula_1_Difference"].max()
)


print("\n========== FORMULA ANALYSIS COMPLETED ==========")