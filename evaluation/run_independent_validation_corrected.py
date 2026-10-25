import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
from pathlib import Path

matrix_path = "data/validation/GSE183795_normalized_matrix.txt.gz"
metadata_path = "data/validation/GSE183795_sample_mapping.csv"
output_path = "results/INDEPENDENT_BIOMARKER_VALIDATION_CORRECTED.csv"

genes = [
    "MROH6", "SPINK4", "IL10RA", "FCGR1BP",
    "HOXA5", "MS4A14", "NME8", "OSCAR", "CLIC2"
]

metadata = pd.read_csv(metadata_path)
metadata["tissue_type"] = metadata["characteristics"].str.extract(
    r"tissue:\s*([^;]+)", expand=False
)

tumor = metadata[metadata["tissue_type"] == "Tumor"]["title"].tolist()
adjacent = metadata[
    metadata["tissue_type"] == "adjacent non-tumor"
]["title"].tolist()

tumor_cols = [s + ".CEL" for s in tumor]
adjacent_cols = [s + ".CEL" for s in adjacent]

matrix = pd.read_csv(matrix_path, sep="\t", compression="gzip")
matrix["ID"] = matrix["ID"].astype(str).str.upper()

results = []

for gene in genes:
    rows = matrix[matrix["ID"] == gene]

    if rows.empty:
        results.append({
            "gene_symbol": gene,
            "tumor_n": len(tumor),
            "adjacent_n": len(adjacent),
            "tumor_mean": np.nan,
            "adjacent_mean": np.nan,
            "mean_difference": np.nan,
            "p_value": np.nan,
            "status": "Not found by gene symbol"
        })
        continue

    expression = rows[tumor_cols + adjacent_cols].apply(
        pd.to_numeric, errors="coerce"
    ).mean(axis=0)

    t = expression[tumor_cols].dropna().to_numpy()
    a = expression[adjacent_cols].dropna().to_numpy()

    p = ttest_ind(t, a, equal_var=False, nan_policy="omit").pvalue

    results.append({
        "gene_symbol": gene,
        "tumor_n": len(t),
        "adjacent_n": len(a),
        "tumor_mean": np.mean(t),
        "adjacent_mean": np.mean(a),
        "mean_difference": np.mean(t) - np.mean(a),
        "p_value": p,
        "status": "Tested"
    })

df = pd.DataFrame(results)

# Benjamini-Hochberg correction over genes successfully tested
valid = df["p_value"].notna()
pvals = df.loc[valid, "p_value"].to_numpy()
if len(pvals):
    order = np.argsort(pvals)
    sorted_p = pvals[order]
    adjusted = sorted_p * len(sorted_p) / np.arange(1, len(sorted_p) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    qvals = np.empty(len(pvals))
    qvals[order] = np.minimum(adjusted, 1.0)
    df.loc[valid, "FDR"] = qvals

df.to_csv(output_path, index=False)

print(df.to_string(index=False))
print("\nSaved to:", output_path)
