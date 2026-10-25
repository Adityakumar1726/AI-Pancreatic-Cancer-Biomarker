import pandas as pd
import numpy as np
import os

from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

BASE = r"C:\Users\adity\OneDrive\Desktop\AI-Pancreatic-Cancer-Biomarker"

INPUT = os.path.join(BASE, "results", "ml_biomarker_dataset.csv")
OUTPUT = os.path.join(BASE, "results", "ml_model_comparison.csv")

# Load dataset
df = pd.read_csv(INPUT)

X = df.drop(columns=["label"])
X = X.select_dtypes(include=[np.number])

y = df["label"].astype(int)

print("Dataset:", X.shape)
print("Tumor:", sum(y == 1))
print("Normal:", sum(y == 0))

# Models
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            class_weight="balanced",
            max_iter=5000,
            random_state=42
        ))
    ]),

    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(
            class_weight="balanced",
            probability=True,
            random_state=42
        ))
    ]),

    "Random Forest": RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
}

loo = LeaveOneOut()

results = []

for name, model in models.items():

    print("\nTraining:", name)

    predictions = cross_val_predict(
        model,
        X,
        y,
        cv=loo,
        method="predict",
        n_jobs=-1
    )

    probabilities = cross_val_predict(
        model,
        X,
        y,
        cv=loo,
        method="predict_proba",
        n_jobs=-1
    )[:, 1]

    accuracy = accuracy_score(y, predictions)
    precision = precision_score(y, predictions, zero_division=0)
    recall = recall_score(y, predictions, zero_division=0)
    f1 = f1_score(y, predictions, zero_division=0)
    auc = roc_auc_score(y, probabilities)

    cm = confusion_matrix(y, predictions)

    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))
    print("ROC-AUC  :", round(auc, 4))
    print("Confusion Matrix:")
    print(cm)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC_AUC": auc
    })

# Save comparison
results_df = pd.DataFrame(results)
results_df = results_df.sort_values("ROC_AUC", ascending=False)

results_df.to_csv(OUTPUT, index=False)

print("\n==============================")
print("MODEL COMPARISON")
print("==============================")
print(results_df.to_string(index=False))

print("\nBest model by ROC-AUC:")
print(results_df.iloc[0]["Model"])

print("\nSaved to:")
print(OUTPUT)