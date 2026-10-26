import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


# LOAD DATA

df = pd.read_csv(
    "data/ml_ready_social_media_engagement.csv"
)

target = df["Engagement_Rate"]


# CREATE VISUALIZATION FOLDER

os.makedirs("visualizations", exist_ok=True)


# FULL DISTRIBUTION

plt.figure(figsize=(10, 6))

sns.histplot(
    target,
    bins=50,
    kde=True
)

plt.title("Distribution of Engagement Rate")
plt.xlabel("Engagement Rate")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig(
    "visualizations/engagement_rate_distribution.png",
    dpi=300
)

plt.show()


# LOG-SCALE DISTRIBUTION

plt.figure(figsize=(10, 6))

sns.histplot(
    target,
    bins=50,
    kde=True
)

plt.xscale("log")

plt.title("Engagement Rate Distribution - Log Scale")
plt.xlabel("Engagement Rate (Log Scale)")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig(
    "visualizations/engagement_rate_log_distribution.png",
    dpi=300
)

plt.show()


print("========== TARGET DISTRIBUTION ANALYSIS ==========")

print("Distribution charts created successfully!")

print("\nSaved:")
print("visualizations/engagement_rate_distribution.png")
print("visualizations/engagement_rate_log_distribution.png")

print("\n========== TARGET DISTRIBUTION ANALYSIS COMPLETED ==========")