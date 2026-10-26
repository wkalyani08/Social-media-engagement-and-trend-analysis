import pandas as pd

# Load feature-engineered dataset
df = pd.read_csv(
    "data/feature_engineered_social_media_engagement.csv"
)

print("========== CORRELATION ANALYSIS ==========")

# Select numerical columns
numeric_columns = [
    "Likes",
    "Comments",
    "Shares",
    "Views",
    "Saves",
    "Follower_Count",
    "Engagement_Rate",
    "Hashtag_Count",
    "Content_Length",
    "Total_Engagement",
    "Engagement_Per_View"
]

# Create correlation matrix
correlation_matrix = df[numeric_columns].corr()

print("\nCorrelation with Engagement Rate:")

engagement_correlation = (
    correlation_matrix["Engagement_Rate"]
    .sort_values(ascending=False)
)

print(engagement_correlation)

print("\n========== CORRELATION ANALYSIS COMPLETED ==========")