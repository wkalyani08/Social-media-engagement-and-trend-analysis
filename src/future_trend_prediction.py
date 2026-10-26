import pandas as pd
import joblib


# LOAD DATA AND MODEL

df = pd.read_csv(
    "data/trend_prediction_data.csv"
)

df["Date"] = pd.to_datetime(df["Date"])

model = joblib.load(
    "models/trend_model.pkl"
)


print("==========================================")
print("       FUTURE TREND PREDICTION")
print("==========================================")


# SORT DATA

df = df.sort_values(
    "Date"
).reset_index(drop=True)


# GET MOST RECENT DATA

latest = df.iloc[-1]


previous_day = latest["Average_Engagement_Rate"]

previous_2_day = df.iloc[-2][
    "Average_Engagement_Rate"
]

previous_3_day = df.iloc[-3][
    "Average_Engagement_Rate"
]


rolling_average = (
    previous_day
    + previous_2_day
    + previous_3_day
) / 3


# CREATE NEXT-DAY FEATURES

next_date = latest["Date"] + pd.Timedelta(days=1)

next_day = next_date.day

next_month = next_date.month

next_year = next_date.year

next_day_number = next_date.dayofweek


# ESTIMATE POST COUNT

post_count = latest["Post_Count"]


# CREATE INPUT

future_data = pd.DataFrame([{

    "Day": next_day,

    "Month": next_month,

    "Year": next_year,

    "Day_Number": next_day_number,

    "Previous_Day_Engagement":
        previous_day,

    "Previous_2_Day_Engagement":
        previous_2_day,

    "Previous_3_Day_Engagement":
        previous_3_day,

    "Rolling_3_Day_Average":
        rolling_average,

    "Post_Count":
        post_count

}])



# PREDICT

prediction = model.predict(
    future_data
)[0]


# DETERMINE TREND

if prediction > previous_day * 1.05:

    trend = "Increasing"

elif prediction < previous_day * 0.95:

    trend = "Decreasing"

else:

    trend = "Stable"


# DISPLAY RESULT

print("\n==========================================")
print("          FUTURE PREDICTION")
print("==========================================")

print(
    "\nLatest Date:",
    latest["Date"].date()
)

print(
    "Next Date:",
    next_date.date()
)

print(
    "\nPrevious Day Engagement:",
    round(previous_day, 2)
)

print(
    "Predicted Engagement:",
    round(prediction, 2)
)

print(
    "Predicted Trend:",
    trend
)


print("\n==========================================")
print("          TREND SUMMARY")
print("==========================================")

if trend == "Increasing":

    print(
        "Expected engagement is likely to increase."
    )

elif trend == "Decreasing":

    print(
        "Expected engagement is likely to decrease."
    )

else:

    print(
        "Expected engagement is likely to remain stable."
    )


print("\n==========================================")
print("       PREDICTION COMPLETE")
print("==========================================")