
# Social Media Engagement Analysis and Trend Prediction

## Project Overview

This project analyzes social media engagement patterns and uses machine learning to predict engagement and future engagement trends.

The project focuses on understanding how platform, content type, posting time, audience size, sentiment, influencer tier and other factors are associated with social media engagement.

---

## Objectives

- Analyze social media engagement patterns.
- Compare engagement across different platforms.
- Analyze engagement by content type.
- Identify high-performing posting times.
- Study relationships between engagement-related variables.
- Identify important factors associated with engagement.
- Build a machine learning model for engagement-rate prediction.
- Build a trend prediction model for future engagement behavior.
- Generate useful analytical insights and recommendations.

---

## Dataset

Dataset: Social Media Engagement Dataset

Records: 5,000 social media posts

Original Features: 20

Data period: 2024–2025

The dataset contains information about platforms, content types, categories, likes, comments, shares, views, saves, followers, engagement rate, posting time, sentiment and influencer tier.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter
- Power BI
- VS Code

---

## Project Workflow

1. Data Loading
2. Data Cleaning
3. Feature Engineering
4. Exploratory Data Analysis
5. Data Visualization
6. Correlation Analysis
7. Machine Learning Preparation
8. Model Training
9. Model Comparison
10. Model Tuning
11. Feature Importance Analysis
12. Engagement Prediction
13. Trend Prediction
14. Model Evaluation
15. Dashboard and Insights

---

## Feature Engineering

Additional features were created from the timestamp and engagement variables, including:

- Post Date
- Year
- Month
- Day
- Day Number
- Total Engagement
- Engagement Per View

---

## Machine Learning

The engagement prediction task uses features that are available before or around the time of publishing.

Outcome variables such as likes, comments, shares, saves, views and derived engagement totals were excluded from the prediction features to reduce target leakage.

Several regression models were compared:

- Linear Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- Extra Trees

The original Decision Tree model achieved:

- MAE: 8.2284
- RMSE: 84.9621
- R²: 0.5195

A separate Gradient Boosting model achieved the lowest MAE among the compared models:

- MAE: 7.9169

Therefore, model selection depends on the evaluation metric used.

---

## Feature Importance

The Decision Tree model identified the following major features:

- Follower Count
- TikTok platform indicator
- Content Length
- Month
- Hour of Day

Feature importance indicates model reliance and should not be interpreted as proof of causation.

---

## Trend Prediction

A separate time-based trend prediction model was developed.

The data was ordered chronologically and divided into earlier training data and later testing data.

Lag-based features were created, including:

- Previous Day Engagement
- Previous 2 Day Engagement
- Previous 3 Day Engagement
- Rolling 3 Day Average

The trend model predicts future engagement behavior using historical engagement patterns.

---

## Prediction System

The project includes an interactive engagement prediction system.

Users can enter post characteristics such as:

- Platform
- Content Type
- Category
- Posting Hour
- Day
- Hashtag Count
- Content Length
- Sentiment
- Influencer Tier
- Follower Count

The trained model then predicts the expected engagement rate.

---

## Dashboard

The project includes a Power BI dashboard for exploring:

- Platform performance
- Content-type performance
- Posting-time patterns
- Sentiment
- Influencer tiers
- Engagement trends

---

## Important Dataset Limitation

The engagement-rate target in the supplied dataset is highly right-skewed and is strongly associated with engagement-related variables.

Analysis also showed that the engagement rate is extremely closely related to a formula based on likes, comments, shares, saves and follower count.

Therefore, the machine learning results should be interpreted as patterns within this dataset rather than proof of real-world causal relationships.

The trend prediction component should also be considered an analytical forecasting exercise rather than a guarantee of future social media performance.

---

## Conclusion

The project transforms raw social media data into analytical insights and machine-learning-based predictions.

It demonstrates a complete data science workflow covering data preprocessing, feature engineering, exploratory analysis, visualization, machine learning, model evaluation, prediction and trend analysis.
=======
# Social-media-engagement-and-trend-analysis
Social Media Analysis is a data analytics project that uses Python to analyze social media engagement, including likes, comments, shares, and trending hashtags. It uses Pandas, NumPy, Matplotlib, and Seaborn to visualize data, identify trends, understand user behavior, and generate insights for better content performance and decision-making.

The main objective is to identify trends, measure audience engagement, and gain meaningful insights from social media activities. Using tools such as Python, Pandas, NumPy, Matplotlib, and Seaborn, the project presents data through charts and visualizations to support better decision-making.

This project demonstrates the practical application of data analytics in understanding social media trends and evaluating content performance.

**Key Features:**
- Social media data collection and preprocessing
- Analysis of likes, comments, shares, and engagement rates
- Identification of popular posts, hashtags, and trends
- Data visualization using graphs and charts
- Generation of insights to improve content performance

**Technologies Used:** Python, Pandas, NumPy, Matplotlib, Seaborn, and Jupyter Notebook.

**Outcome:** The project helps transform raw social media data into meaningful insights that can support content optimization, audience understanding, and data-driven marketing decisions.
>>>>>>> a2b8683f5fa799f5b2c58bb98baf27f721e55592
