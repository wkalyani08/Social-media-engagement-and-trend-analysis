# SOCIAL MEDIA ENGAGEMENT ANALYSIS AND TREND PREDICTION

## 1. Introduction

Social media platforms generate large amounts of user engagement data through likes, comments, shares, views and saves.

This project focuses on analyzing social media engagement patterns across different platforms and content types. It also studies time-based patterns, audience-related factors and other characteristics associated with engagement.

Machine learning techniques are used to predict engagement rates and analyze future engagement trends.

---

# 2. Problem Statement

Social media platforms contain large volumes of engagement data, making it difficult to manually identify important patterns.

The problem addressed by this project is to analyze historical social media data and identify factors associated with higher engagement, while developing machine-learning-based prediction and trend-analysis capabilities.

---

# 3. Objectives

The major objectives are:

1. Analyze social media engagement data.
2. Compare engagement across platforms.
3. Analyze engagement across different content types.
4. Identify time-based engagement patterns.
5. Study the relationship between audience size and engagement.
6. Analyze sentiment and influencer-tier patterns.
7. Identify important features associated with engagement.
8. Develop a machine learning model for engagement-rate prediction.
9. Develop a trend prediction model using historical engagement patterns.
10. Present insights through a dashboard.

---

# 4. Dataset Description

The project uses a Social Media Engagement Dataset containing 5,000 social media posts and 20 original features.

The dataset contains information such as:

- Platform
- Content Type
- Category
- Likes
- Comments
- Shares
- Views
- Saves
- Follower Count
- Engagement Rate
- Posting Hour
- Day of Week
- Hashtag Count
- Content Length
- Sentiment
- Influencer Tier
- Media availability
- Verification status

The dataset covers social media activity from 2024–2025.

---

# 5. Technologies Used

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

# 6. Data Preprocessing

The dataset was inspected for:

- Missing values
- Duplicate records
- Data types
- Invalid numerical values
- Categorical values
- Timestamp consistency

The Timestamp column was converted into datetime format.

The cleaned dataset was saved for further processing.

---

# 7. Feature Engineering

Additional features were generated from the original dataset.

These included:

- Post Date
- Year
- Month
- Day
- Day Number
- Total Engagement
- Engagement Per View

Time-related information was extracted from the Timestamp column to support time-based analysis.

---

# 8. Exploratory Data Analysis

Exploratory analysis was performed to understand engagement patterns.

The analysis included:

- Platform-wise engagement
- Content-type engagement
- Day-wise engagement
- Hour-wise engagement
- Sentiment-wise engagement
- Influencer-tier engagement
- Correlation analysis
- Engagement-rate distribution
- Outlier analysis

---

# 9. Major Analytical Findings

## Platform

TikTok showed the highest average engagement rate in the dataset, while LinkedIn showed the lowest average engagement rate.

## Content Type

Stitch and Duet content showed particularly high average engagement rates in the dataset.

## Time

Engagement varied considerably across posting hours. Several high-engagement periods were observed, including morning and evening hours.

## Sentiment

Positive, negative and neutral posts showed different engagement levels, with positive sentiment having the highest average engagement among the three categories.

## Influencer Tier

Nano influencers showed substantially higher engagement rates than larger influencer tiers in this dataset.

These findings represent associations within the dataset and should not automatically be interpreted as causal relationships.

---

# 10. Machine Learning

Machine learning was used to predict Engagement Rate.

The following models were compared:

- Linear Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- Extra Trees

The evaluation metrics used were:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

---

# 11. Model Comparison

The original model comparison produced the following results:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Decision Tree | 8.2284 | 84.9621 | 0.5195 |
| Gradient Boosting | 7.9169 | 87.8218 | 0.4866 |
| Random Forest | 7.9484 | 97.7204 | 0.3643 |
| Extra Trees | 8.0476 | 100.3832 | 0.3292 |
| Linear Regression | 15.2102 | 112.8637 | 0.1520 |

The Decision Tree achieved the highest R² and lowest RMSE among these original models.

Gradient Boosting achieved the lowest MAE among the compared models.

For the final engagement prediction system, the Decision Tree model was selected based on its overall R² and RMSE performance.

---

# 12. Feature Importance

The Decision Tree model identified the following important features:

- Follower Count
- TikTok platform indicator
- Content Length
- Month
- Hour of Day

Follower Count had the largest individual model importance.

Feature importance represents the features used by the model for prediction and does not establish causation.

---

# 13. Engagement Prediction System

An interactive prediction system was developed using the trained Decision Tree pipeline.

The system accepts post characteristics including:

- Platform
- Content Type
- Category
- Posting Hour
- Day
- Hashtag Count
- Content Length
- Sentiment
- Influencer Tier
- Media status
- Verification status
- Follower Count

The system then produces:

- Predicted Engagement Rate
- Engagement Level

---

# 14. Trend Prediction

A separate trend prediction component was developed.

Historical engagement data was aggregated by date.

Lag features were created using:

- Previous Day Engagement
- Previous 2 Day Engagement
- Previous 3 Day Engagement
- Rolling 3 Day Average

The dataset was split chronologically so that earlier observations were used for training and later observations were used for testing.

A Random Forest regression model was used to predict future average engagement behavior.

The trend component also classifies the expected movement as:

- Increasing
- Decreasing
- Stable

---

# 15. Dashboard

A Power BI dashboard was developed to provide an interactive view of the analysis.

The dashboard presents information related to:

- Platform performance
- Content type
- Posting time
- Engagement patterns
- Sentiment
- Influencer tier
- Other engagement characteristics

The dashboard allows users to understand important patterns without directly working with the raw dataset.

---

# 16. Data Leakage Consideration

The engagement prediction model intentionally excludes post-outcome variables such as:

- Likes
- Comments
- Shares
- Saves
- Views
- Total Engagement
- Engagement Per View

These variables are closely related to the engagement outcome and could introduce target leakage if used as predictors for a pre-publishing prediction scenario.

---

# 17. Dataset Limitation

The Engagement Rate variable is highly right-skewed.

The analysis also showed that Engagement Rate is extremely strongly associated with a formula involving likes, comments, shares, saves and follower count.

Therefore, the model results should be interpreted as patterns within the supplied dataset.

The prediction system should not be interpreted as a guarantee of actual real-world social media performance.

Similarly, trend predictions represent analytical forecasts based on historical patterns rather than guaranteed future outcomes.

---

# 18. Conclusion

The project demonstrates an end-to-end data science workflow for social media engagement analysis.

The workflow includes:

Data Loading → Data Cleaning → Feature Engineering → EDA → Visualization → Correlation Analysis → Machine Learning → Model Comparison → Prediction → Trend Prediction → Dashboard

The project successfully transforms raw social media data into meaningful analytical insights and machine-learning-based predictions.

---

# 19. Future Scope

Future improvements could include:

- Larger and more diverse real-world datasets
- Real-time social media APIs
- Advanced time-series forecasting models
- Hyperparameter optimization
- Deep learning models
- Real-time dashboards
- Automated recommendation systems
- Platform-specific prediction models
- Continuous model retraining

---

# 20. Project Outcome

The final system provides:

1. Social media engagement analysis.
2. Platform and content comparison.
3. Time-based engagement analysis.
4. Feature importance analysis.
5. Engagement-rate prediction.
6. Future trend prediction.
7. Interactive dashboard-based visualization.
8. A reproducible Python-based data science workflow.