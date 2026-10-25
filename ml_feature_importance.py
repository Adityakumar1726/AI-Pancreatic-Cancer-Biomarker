import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier

BASE = r"C:\Users\adity\OneDrive\Desktop\AI-Pancreatic-Cancer-Biomarker"

INPUT = os.path.join(BASE, "results", "ml_biomarker_dataset.csv")
OUTPUT = os.path.join(BASE, "results", "rf_feature_importance.csv")

# Load dataset
df = pd.read_csv(INPUT)

# Remove label and sample ID from ML features
X = df.drop(columns=["label", "sample_id"], errors="ignore")
y = df["label"]

print("Dataset shape:", X.shape)
print("Number of features:", X.shape[1])

# Random Forest model
model = RandomForestClassifier(
    n_estimators=1000,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

# Train model
model.fit(X, y)

# Calculate feature importance
importance = pd.DataFrame({
    "Biomarker": X.columns,
    "Importance": model.feature_importances_
})

# Sort from highest to lowest
importance = importance.sort_values(
    "Importance",
    ascending=False
)

# Add rank
importance["Rank"] = range(1, len(importance) + 1)

# Arrange columns
importance = importance[
    ["Rank", "Biomarker", "Importance"]
]

# Save results
importance.to_csv(OUTPUT, index=False)

# Display results
print("\n==============================")
print("RANDOM FOREST FEATURE IMPORTANCE")
print("==============================")
print(importance.to_string(index=False))

print("\nSaved to:")
print(OUTPUT)