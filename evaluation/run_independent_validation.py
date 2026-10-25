import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
from pathlib import Path

matrix_path = Path("data/validation/GSE183795_normalized_matrix.txt.gz")
metadata_path = Path("data/validation/GSE183795_sample_mapping.csv")
candidates_path = Path("results/FINAL_TOP20_WITH_GENE_NAMES.csv")
output_path = Path("results/INDEPENDENT_BIOMARKER_VALIDATION.csv")

# Load metadata and expression matrix
metadata = pd.read_csv(metadata_path)
matrix = pd.read_csv(matrix_path, sep="\t", compression="gzip")

metadata["tissue_type"] = metadata["characteristics"].str.extract(
    r"tissue:\s*([^;]+)", expand=False
)

# Keep only tumor and adjacent non-tumor samples
metadata = metadata[
    metadata["tissue_type"].isin(["Tumor", "adjacent non-tumor"])
].copy()

tumor_samples = metadata.loc[
    metadata["tissue_type"] == "Tumor", "title"
].tolist()

normal_samples = metadata.loc[
    metadata["tissue_type"] == "adjacent non-tumor", "title"
].tolist()

# Matrix columns are CEL filenames
tumor_cols = [s + ".CEL" for s in tumor_samples]
normal_cols = [s + ".CEL" for s in normal_samples]

missing = [
    s for s in tumor_cols + normal_cols
    if s not in matrix.columns
]
if missing:
    raise ValueError(f"Missing expression columns, examples: {missing[:10]}")

# Load candidate genes and identify symbols
candidates = pd.read_csv(candidates_path)
print("Candidate file columns:", candidates.columns.tolist())

symbol_col = next(
    (c for c in ["gene_symbol", "Gene_Symbol", "gene_name", "symbol"]
     if c in candidates.columns),
    None
)

if symbol_col is None:
    print("No recognized gene-symbol column found.")
    print("Please inspect the candidate file before proceeding.")
    raise SystemExit(1)

symbols = candidates[symbol_col].dropna().astype(str).unique().tolist()
print("Candidate gene symbols:", symbols)

# The matrix ID column contains gene symbols
matrix["ID"] = matrix["ID"].astype(str).str.upper()
results = []

for symbol in symbols:
    if not symbol or symbol.upper() in ["NAN", "NA"]:
        continue

    matching = matrix[matrix["ID"] == symbol.upper()]
    if matching.empty:
        results.append({
            "gene_symbol": symbol,
            "tumor_mean": np.nan,
            "adjacent_mean": np.nan,
            "mean_difference": np.nan,
            "p_value": np.nan,
            "status": "Gene symbol not found in matrix"
        })
        continue

    # If multiple rows map to the same symbol, average their expression
    row = matching[tumor_cols + normal_cols].apply(
        pd.to_numeric, errors="coerce"
    ).mean(axis=0)

    tumor = row[tumor_cols].dropna().to_numpy()
    normal = row[normal_cols].dropna().to_numpy()

    if len(tumor) < 2 or len(normal) < 2:
        p_value = np.nan
        status = "Insufficient expression values"
    else:
        p_value = float(ttest_ind(
            tumor, normal, equal_var=False, nan_policy="omit"
        ).pvalue)
        status = "Tested"

    results.append({
        "gene_symbol": symbol,
        "tumor_n": len(tumor),
        "adjacent_n": len(normal),
        "tumor_mean": float(np.mean(tumor)) if len(tumor) else np.nan,
        "adjacent_mean": float(np.mean(normal)) if len(normal) else np.nan,
        "mean_difference": (
            float(np.mean(tumor) - np.mean(normal))
            if len(tumor) and len(normal) else np.nan
        ),
        "p_value": p_value,
        "status": status
    })

result_df = pd.DataFrame(results)

# Benjamini-Hochberg FDR correction across tested candidates
valid = result_df["p_value"].notna()
p = result_df.loc[valid, "p_value"].to_numpy()
if len(p):
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    q = np.empty(len(p))
    q[order] = np.minimum(adjusted, 1.0)
    result_df.loc[valid, "FDR"] = q

result_df.to_csv(output_path, index=False)

print("\nTumor samples:", len(tumor_cols))
print("Adjacent non-tumor samples:", len(normal_cols))
print("\nValidation results:")
print(result_df.to_string(index=False))
print("\nSaved to:", output_path)
print("Note: this validates expression differences, not clinical diagnostic performance.")
