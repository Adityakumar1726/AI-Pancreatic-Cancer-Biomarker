
import os
import warnings
import numpy as np
import pandas as pd

from scipy.stats import ttest_ind
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.abspath(__file__))
MRNA_FILE = os.path.join(
    BASE, "data", "processed", "mrna_expression_counts.csv"
)
LABEL_FILE = os.path.join(
    BASE, "data", "processed", "de_groups.tsv"
)
OUT_DIR = os.path.join(BASE, "results")
os.makedirs(OUT_DIR, exist_ok=True)

# --------------------------------------------------
# 1. Load expression data and labels
# --------------------------------------------------
print("Loading expression data and labels...")

expression = pd.read_csv(MRNA_FILE)
labels = pd.read_csv(LABEL_FILE, sep="\t")

id_col = expression.columns[0]
expression[id_col] = expression[id_col].astype(str)

# Convert expression matrix to samples x genes
X_all = expression.set_index(id_col).T
X_all.index = X_all.index.astype(str)
X_all.index.name = "Sample ID"

labels["Sample ID"] = labels["Sample ID"].astype(str)

label_map = {
    "tumor": 1,
    "tumour": 1,
    "normal": 0,
    "normal tissue": 0
}

labels["label"] = (
    labels["Tissue Type"].astype(str).str.lower().map(label_map)
)

labels = labels.dropna(subset=["label"])
labels["label"] = labels["label"].astype(int)
labels = labels.drop_duplicates(subset=["Sample ID"])

# Align sample IDs
common = X_all.index.intersection(labels["Sample ID"])
X_all = X_all.loc[common]

y = (
    labels.set_index("Sample ID")
    .loc[common, "label"]
    .astype(int)
)

# Numeric expression values
X_all = X_all.apply(pd.to_numeric, errors="coerce")
X_all = X_all.replace([np.inf, -np.inf], np.nan)

# Remove genes with no usable expression
X_all = X_all.dropna(axis=1, how="all")
X_all = X_all.loc[:, X_all.nunique(dropna=True) > 1]

# Count-based expression is right-skewed; log transform
X_all = np.log2(X_all.clip(lower=0) + 1)

print("Samples:", len(y))
print("Genes available:", X_all.shape[1])
print("Tumor samples:", int((y == 1).sum()))
print("Normal samples:", int((y == 0).sum()))

if y.nunique() != 2:
    raise ValueError("Both tumor and normal samples are required.")

if (y == 0).sum() < 2:
    raise ValueError("Too few normal samples for evaluation.")

# --------------------------------------------------
# 2. Models
# Feature selection happens inside each pipeline.
# --------------------------------------------------
k = min(50, X_all.shape[1])

models = {
    "Logistic Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("selector", SelectKBest(score_func=f_classif, k=k)),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            class_weight="balanced",
            max_iter=5000,
            random_state=42
        ))
    ]),

    "SVM": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("selector", SelectKBest(score_func=f_classif, k=k)),
        ("scaler", StandardScaler()),
        ("model", SVC(
            class_weight="balanced",
            probability=True,
            random_state=42
        ))
    ]),

    "Random Forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("selector", SelectKBest(score_func=f_classif, k=k)),
        ("model", RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ])
}

# --------------------------------------------------
# 3. Leave-One-Out cross-validation
# --------------------------------------------------
cv = LeaveOneOut()
results = []
prediction_tables = []

for name, model in models.items():
    print("\nEvaluating:", name)

    predictions = cross_val_predict(
        clone(model), X_all, y,
        cv=cv, method="predict", n_jobs=1
    )

    probabilities = cross_val_predict(
        clone(model), X_all, y,
        cv=cv, method="predict_proba", n_jobs=1
    )[:, 1]

    tn, fp, fn, tp = confusion_matrix(
        y, predictions, labels=[0, 1]
    ).ravel()

    specificity = tn / (tn + fp) if (tn + fp) else np.nan

    # AUC is not reliable if either class is absent;
    # both classes exist in the full label vector here.
    auc = roc_auc_score(y, probabilities)

    row = {
        "Model": name,
        "Accuracy": accuracy_score(y, predictions),
        "Balanced_Accuracy": balanced_accuracy_score(y, predictions),
        "Precision": precision_score(
            y, predictions, zero_division=0
        ),
        "Sensitivity": recall_score(
            y, predictions, zero_division=0
        ),
        "Specificity": specificity,
        "F1": f1_score(y, predictions, zero_division=0),
        "ROC_AUC": auc,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp)
    }

    results.append(row)

    pred_df = pd.DataFrame({
        "Sample_ID": y.index,
        "Actual": y.values,
        "Predicted": predictions,
        "Tumor_Probability": probabilities,
        "Model": name
    })
    prediction_tables.append(pred_df)

    print("Accuracy:", round(row["Accuracy"], 4))
    print("Balanced accuracy:", round(row["Balanced_Accuracy"], 4))
    print("Sensitivity:", round(row["Sensitivity"], 4))
    print("Specificity:", round(row["Specificity"], 4))
    print("F1:", round(row["F1"], 4))
    print("ROC-AUC:", round(row["ROC_AUC"], 4))
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print([[tn, fp], [fn, tp]])

# --------------------------------------------------
# 4. Save outputs
# --------------------------------------------------
results_df = pd.DataFrame(results)
results_df.to_csv(
    os.path.join(OUT_DIR, "ml_corrected_model_comparison.csv"),
    index=False
)

predictions_df = pd.concat(prediction_tables, ignore_index=True)
predictions_df.to_csv(
    os.path.join(OUT_DIR, "ml_corrected_predictions.csv"),
    index=False
)

print("\n====================================")
print("CORRECTED MODEL EVALUATION COMPLETE")
print("====================================")
print(results_df.round(4).to_string(index=False))
print("\nSaved files:")
print(os.path.join(OUT_DIR, "ml_corrected_model_comparison.csv"))
print(os.path.join(OUT_DIR, "ml_corrected_predictions.csv"))

print(
    "\nCAUTION: Only four normal samples are available. "
    "Metrics are highly uncertain and are not evidence of "
    "clinical diagnostic performance."
)
