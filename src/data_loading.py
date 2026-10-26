import pandas as pd

# Load dataset
df = pd.read_csv("data/social_media_engagement_dataset.csv")

print("========== DATASET OVERVIEW ==========")

# Number of rows and columns
print("\nDataset Shape:")
print(df.shape)

# Column names
print("\nColumn Names:")
print(df.columns.tolist())

# Data types
print("\nData Types:")
print(df.dtypes)

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# First 5 rows
print("\nFirst 5 Rows:")
print(df.head())

# Statistical summary
print("\nStatistical Summary:")
print(df.describe())