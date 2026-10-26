import pandas as pd
import joblib


# LOAD FINAL MODEL

model = joblib.load("models/final_model.pkl")

print("\n==========================================")
print("   SOCIAL MEDIA ENGAGEMENT PREDICTOR")
print("==========================================\n")


# VALID OPTIONS

platforms = [
    "Facebook",
    "Instagram",
    "LinkedIn",
    "TikTok",
    "Twitter",
    "YouTube"
]

content_types = [
    "Article",
    "Carousel",
    "Community Post",
    "Document",
    "Duet",
    "Live",
    "Photo",
    "Poll",
    "Post",
    "Reel",
    "Retweet",
    "Short",
    "Stitch",
    "Story",
    "Thread",
    "Tweet",
    "Video"
]

sentiments = [
    "Positive",
    "Negative",
    "Neutral"
]

influencer_tiers = [
    "Nano",
    "Micro",
    "Mid-tier",
    "Macro"
]

categories = [
    "Entertainment",
    "Education",
    "Technology",
    "Lifestyle",
    "Sports",
    "Gaming",
    "Travel",
    "Business",
    "Fashion",
    "Food",
    "Health",
    "News",
    "Music"
]

days = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
]


# INPUT FUNCTION

def get_choice(message, options):

    while True:

        value = input(message).strip()

        for option in options:

            if value.lower() == option.lower():
                return option

        print("Invalid choice. Please select a valid option.")
        print("Available options:")
        print(", ".join(options))
        print()


# GET POST INFORMATION

platform = get_choice(
    "Platform: ",
    platforms
)

content_type = get_choice(
    "Content Type: ",
    content_types
)

category = get_choice(
    "Category: ",
    categories
)

hour = int(
    input("Hour of Day (0-23): ")
)

day = get_choice(
    "Day of Week: ",
    days
)

hashtag_count = int(
    input("Hashtag Count: ")
)

content_length = int(
    input("Content Length: ")
)

sentiment = get_choice(
    "Sentiment: ",
    sentiments
)

influencer_tier = get_choice(
    "Influencer Tier: ",
    influencer_tiers
)

has_media = input(
    "Has Media? (True/False): "
).strip().lower() == "true"

is_verified = input(
    "Is Verified? (True/False): "
).strip().lower() == "true"

follower_count = int(
    input("Follower Count: ")
)

year = int(
    input("Year: ")
)

month = int(
    input("Month (1-12): ")
)

day_number = int(
    input("Day Number (0=Monday, 6=Sunday): ")
)

month_number = month


# CREATE INPUT DATAFRAME

input_data = pd.DataFrame([{

    "Platform": platform,
    "Content_Type": content_type,
    "Category": category,
    "Hour_of_Day": hour,
    "Day_of_Week": day,
    "Hashtag_Count": hashtag_count,
    "Content_Length": content_length,
    "Sentiment": sentiment,
    "Influencer_Tier": influencer_tier,
    "Has_Media": has_media,
    "Is_Verified": is_verified,
    "Follower_Count": follower_count,
    "Year": year,
    "Month": month,
    "Day_Number": day_number,
    "Month_Number": month_number

}])


# PREDICTION

prediction = model.predict(input_data)[0]


# ENGAGEMENT LEVEL

if prediction < 2:

    level = "Low"

elif prediction < 10:

    level = "Moderate"

elif prediction < 50:

    level = "High"

else:

    level = "Very High"


# DISPLAY RESULT

print("\n==========================================")
print("          PREDICTION RESULT")
print("==========================================")

print(
    f"\nPredicted Engagement Rate: {prediction:.2f}"
)

print(
    f"Predicted Engagement Level: {level}"
)

print("\n==========================================")
print("             POST DETAILS")
print("==========================================")

print("Platform        :", platform)
print("Content Type    :", content_type)
print("Category        :", category)
print("Hour            :", hour)
print("Day             :", day)
print("Sentiment       :", sentiment)
print("Influencer Tier :", influencer_tier)
print("Followers       :", follower_count)

print("\nPrediction completed successfully! ")