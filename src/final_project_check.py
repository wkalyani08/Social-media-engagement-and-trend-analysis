import os


print("==========================================")
print("       FINAL PROJECT VERIFICATION")
print("==========================================\n")


required_files = [

    # Data
    "data/social_media_engagement_dataset.csv",
    "data/cleaned_social_media_engagement.csv",
    "data/feature_engineered_social_media_engagement.csv",
    "data/ml_ready_social_media_engagement.csv",

    # Models
    "models/final_model.pkl",
    "models/trend_model.pkl",

    # Documentation
    "README.md",
    "requirements.txt",

    # Main scripts
    "src/data_loading.py",
    "src/data_cleaning.py",
    "src/feature_engineering.py",
    "src/eda.py",
    "src/visualization.py",
    "src/correlation_analysis.py",
    "src/prepare_ml_data.py",
    "src/ml_preprocessing.py",
    "src/train_model.py",
    "src/evaluate_model.py",
    "src/model_comparison.py",
    "src/model_tuning.py",
    "src/feature_importance.py",
    "src/model_interpretation.py",
    "src/save_final_model.py",
    "src/predict_engagement.py",

    # Trend prediction
    "src/trend_prediction.py",
    "src/train_trend_model.py",
    "src/future_trend_prediction.py",
    "src/validate_trend_model.py"
]


print("Checking required project files...\n")


missing_files = []


for file in required_files:

    if os.path.exists(file):

        print("[OK]     ", file)

    else:

        print("[MISSING]", file)

        missing_files.append(file)


print("\n==========================================")
print("             CHECK SUMMARY")
print("==========================================")


if len(missing_files) == 0:

    print("\nSUCCESS! All required project files exist.")

else:

    print(
        "\nMissing files:",
        len(missing_files)
    )

    print("\nPlease check:")

    for file in missing_files:

        print("-", file)


print("\n==========================================")
print("       FINAL CHECK COMPLETED")
print("==========================================")