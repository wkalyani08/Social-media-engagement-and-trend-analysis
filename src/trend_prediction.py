import pandas as pd
import numpy as np


# LOAD DATA

df = pd.read_csv(
    "data/feature_engineered_social_media_engagement.csv"
)

print("==========================================")
print("       TREND PREDICTION - DATA PREP")
print("==========================================")

print("\nOriginal Shape:", df.shape)


# CONVERT TIMESTAMP

df["Timestamp"] = pd.to_datetime(df["Timestamp"])

print("\nDate Range:")
print("Start:", df["Timestamp"].min())
print("End  :", df["Timestamp"].max())


# CREATE DATE COLUMN

df["Date"] = df["Timestamp"].dt.date


# DAILY ENGAGEMENT

daily_trend = (
    df.groupby("Date")
      .agg(
          Average_Engagement_Rate=("Engagement_Rate", "mean"),
          Total_Engagement=("Total_Engagement", "sum"),
          Post_Count=("Post_ID", "count")
      )
      .reset_index()
)


# SORT BY DATE

daily_trend["Date"] = pd.to_datetime(
    daily_trend["Date"]
)

daily_trend = daily_trend.sort_values(
    "Date"
).reset_index(drop=True)


# CREATE TIME FEATURES

daily_trend["Day"] = daily_trend["Date"].dt.day

daily_trend["Month"] = daily_trend["Date"].dt.month

daily_trend["Year"] = daily_trend["Date"].dt.year

daily_trend["Day_Number"] = (
    daily_trend["Date"].dt.dayofweek
)


# CREATE LAG FEATURES

daily_trend["Previous_Day_Engagement"] = (
    daily_trend["Average_Engagement_Rate"].shift(1)
)

daily_trend["Previous_2_Day_Engagement"] = (
    daily_trend["Average_Engagement_Rate"].shift(2)
)

daily_trend["Previous_3_Day_Engagement"] = (
    daily_trend["Average_Engagement_Rate"].shift(3)
)


# ROLLING AVERAGE

daily_trend["Rolling_3_Day_Average"] = (
    daily_trend["Average_Engagement_Rate"]
    .shift(1)
    .rolling(window=3)
    .mean()
)


# DISPLAY RESULTS

print("\n==========================================")
print("        DAILY TREND DATA")
print("==========================================")

print(
    daily_trend.head(10)
)


print("\n==========================================")
print("        TREND DATA INFORMATION")
print("==========================================")

print(
    "\nNumber of daily records:",
    len(daily_trend)
)


print("\nMissing values after lag creation:")

print(
    daily_trend.isnull().sum()
)


# REMOVE ROWS WITH MISSING LAGS

daily_trend_model = daily_trend.dropna().copy()


print("\n==========================================")
print("       FINAL TREND MODEL DATA")
print("==========================================")

print(
    "Shape:",
    daily_trend_model.shape
)


# SAVE DATA

daily_trend_model.to_csv(
    "data/trend_prediction_data.csv",
    index=False
)


print("\nTrend prediction data saved successfully!")

print(
    "File: data/trend_prediction_data.csv"
)

print("\n==========================================")
print("          TREND PREDICTION COMPLETE")
print("==========================================")
