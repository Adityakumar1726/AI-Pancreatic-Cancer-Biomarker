import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt

from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

BASE = r"C:\Users\adity\OneDrive\Desktop\AI-Pancreatic-Cancer-Biomarker"

INPUT = os.path.join(BASE, "results", "ml_biomarker_dataset.csv")
IMPORTANCE_FILE = os.path.join(BASE, "results", "rf_feature_importance.csv")

ROC_OUTPUT = os.path.join(BASE, "results", "roc_curve_comparison.png")
CM_OUTPUT = os.path.join(BASE, "results", "random_forest_confusion_matrix.png")
FI_OUTPUT = os.path.join(BASE, "results", "random_forest_feature_importance.png")

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

df = pd.read_csv(INPUT)

X = df.drop(columns=["label", "sample_id"], errors="ignore")
y = df["label"].astype(int)

print("Dataset:", X.shape)
print("Tumor samples:", sum(y == 1))
print("Normal samples:", sum(y == 0))

# --------------------------------------------------
# MODELS
# --------------------------------------------------

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
        n_estimators=1000,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
}

loo = LeaveOneOut()

# --------------------------------------------------
# ROC CURVE
# --------------------------------------------------

plt.figure(figsize=(9, 7))

for name, model in models.items():

    print("\nProcessing:", name)

    probabilities = cross_val_predict(
        model,
        X,
        y,
        cv=loo,
        method="predict_proba",
        n_jobs=-1
    )[:, 1]

    fpr, tpr, _ = roc_curve(y, probabilities)
    auc = roc_auc_score(y, probabilities)

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"{name} (AUC = {auc:.3f})"
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    linewidth=1
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison of ML Models")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(ROC_OUTPUT, dpi=300)
plt.close()

print("\nROC curve saved:")
print(ROC_OUTPUT)

# --------------------------------------------------
# RANDOM FOREST CONFUSION MATRIX
# --------------------------------------------------

rf = models["Random Forest"]

rf_predictions = cross_val_predict(
    rf,
    X,
    y,
    cv=loo,
    method="predict",
    n_jobs=-1
)

cm = confusion_matrix(y, rf_predictions)

plt.figure(figsize=(7, 6))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Normal", "Tumor"]
)

disp.plot(
    values_format="d",
    cmap="Blues"
)

plt.title("Random Forest Confusion Matrix")
plt.tight_layout()

plt.savefig(CM_OUTPUT, dpi=300)
plt.close()

print("\nConfusion matrix saved:")
print(CM_OUTPUT)

# --------------------------------------------------
# FEATURE IMPORTANCE
# --------------------------------------------------

importance = pd.read_csv(IMPORTANCE_FILE)

# Sort by importance
importance = importance.sort_values(
    "Importance",
    ascending=True
)

plt.figure(figsize=(10, 8))

plt.barh(
    importance["Biomarker"],
    importance["Importance"]
)

plt.xlabel("Random Forest Feature Importance")
plt.ylabel("Biomarker")
plt.title("Biomarker Feature Importance")

plt.tight_layout()
plt.savefig(FI_OUTPUT, dpi=300)
plt.close()

print("\nFeature importance graph saved:")
print(FI_OUTPUT)

# --------------------------------------------------
# COMPLETE
# --------------------------------------------------

print("\n==============================")
print("VISUALIZATION COMPLETE")
print("==============================")

print("\nGenerated files:")

print("1.", ROC_OUTPUT)
print("2.", CM_OUTPUT)
print("3.", FI_OUTPUT)