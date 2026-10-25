import os
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = r"C:\Users\adity\OneDrive\Desktop\AI-Pancreatic-Cancer-Biomarker"

PROCESSED_DIR = os.path.join(
    PROJECT_DIR,
    "data",
    "processed"
)

RESULTS_DIR = os.path.join(
    PROJECT_DIR,
    "results"
)

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# FILES
# ============================================================

GROUP_FILE = os.path.join(
    PROCESSED_DIR,
    "de_groups.tsv"
)

COMMON_FILE = os.path.join(
    PROCESSED_DIR,
    "common_samples_182.txt"
)

MRNA_FILE = os.path.join(
    PROCESSED_DIR,
    "mrna_expression_counts.csv"
)

LNCRNA_FILE = os.path.join(
    PROCESSED_DIR,
    "lncrna_expression_counts.csv"
)

MIRNA_FILE = os.path.join(
    PROCESSED_DIR,
    "mirna_expression_counts.csv"
)


# ============================================================
# READ SAMPLE INFORMATION
# ============================================================

print("Reading sample information...")

groups = pd.read_csv(
    GROUP_FILE,
    sep="\t"
)

common_samples = pd.read_csv(
    COMMON_FILE
)["Sample_ID"].tolist()

groups = groups[
    groups["Sample ID"].isin(common_samples)
].copy()

print("Total samples:", len(groups))

print("\nSample groups:")
print(groups["Tissue Type"].value_counts())


# ============================================================
# DEFINE TUMOR AND NORMAL SAMPLES
# ============================================================

tumor_samples = groups.loc[
    groups["Tissue Type"] == "Tumor",
    "Sample ID"
].tolist()

normal_samples = groups.loc[
    groups["Tissue Type"] == "Normal",
    "Sample ID"
].tolist()

print("\nTumor samples:", len(tumor_samples))
print("Normal samples:", len(normal_samples))


# ============================================================
# DIFFERENTIAL EXPRESSION FUNCTION
# ============================================================

def calculate_de(expression_file, output_name, significant_name):

    print("\n==========================================")
    print("Processing:", output_name)
    print("==========================================")

    # --------------------------------------------------------
    # Read expression matrix
    # --------------------------------------------------------

    expression = pd.read_csv(
        expression_file,
        index_col=0
    )

    print("Original features:", expression.shape[0])

    # --------------------------------------------------------
    # IMPORTANT:
    # Combine duplicate RNA IDs
    # --------------------------------------------------------

    expression = expression.groupby(
        level=0
    ).sum()

    print("Features after duplicate removal:", expression.shape[0])

    # --------------------------------------------------------
    # Keep only common samples
    # --------------------------------------------------------

    available_samples = [
        s for s in common_samples
        if s in expression.columns
    ]

    expression = expression[available_samples]

    # --------------------------------------------------------
    # Separate tumor and normal
    # --------------------------------------------------------

    tumor_columns = [
        s for s in tumor_samples
        if s in expression.columns
    ]

    normal_columns = [
        s for s in normal_samples
        if s in expression.columns
    ]

    tumor = expression[tumor_columns]

    normal = expression[normal_columns]

    print("Tumor columns:", tumor.shape[1])
    print("Normal columns:", normal.shape[1])
    print("Features:", expression.shape[0])

    results = []

    # --------------------------------------------------------
    # Calculate statistics for every RNA
    # --------------------------------------------------------

    for i, gene in enumerate(expression.index, start=1):

        tumor_values = pd.to_numeric(
            tumor.loc[gene],
            errors="coerce"
        ).fillna(0).values

        normal_values = pd.to_numeric(
            normal.loc[gene],
            errors="coerce"
        ).fillna(0).values

        # ----------------------------------------------------
        # Log2 transformation
        # ----------------------------------------------------

        tumor_log = np.log2(
            tumor_values + 1
        )

        normal_log = np.log2(
            normal_values + 1
        )

        # ----------------------------------------------------
        # Mean expression
        # ----------------------------------------------------

        tumor_mean = np.mean(
            tumor_log
        )

        normal_mean = np.mean(
            normal_log
        )

        # ----------------------------------------------------
        # log2 Fold Change
        # ----------------------------------------------------

        log2fc = (
            tumor_mean -
            normal_mean
        )

        # ----------------------------------------------------
        # Welch's t-test
        # ----------------------------------------------------

        try:

            statistic, pvalue = ttest_ind(
                tumor_log,
                normal_log,
                equal_var=False,
                nan_policy="omit"
            )

        except Exception:

            pvalue = np.nan

        results.append([
            gene,
            tumor_mean,
            normal_mean,
            log2fc,
            pvalue
        ])

        # Progress message
        if i % 5000 == 0:

            print(
                f"Processed {i}/{len(expression.index)} features"
            )

    # ========================================================
    # CREATE RESULT TABLE
    # ========================================================

    result = pd.DataFrame(
        results,
        columns=[
            "RNA_ID",
            "Tumor_Mean_Log2",
            "Normal_Mean_Log2",
            "log2FC",
            "p_value"
        ]
    )

    # --------------------------------------------------------
    # Convert p-values to numeric
    # --------------------------------------------------------

    result["p_value"] = pd.to_numeric(
        result["p_value"],
        errors="coerce"
    )

    valid = result["p_value"].notna()

    # --------------------------------------------------------
    # FDR correction
    # Benjamini-Hochberg
    # --------------------------------------------------------

    result["FDR"] = np.nan

    if valid.sum() > 0:

        result.loc[
            valid,
            "FDR"
        ] = multipletests(
            result.loc[
                valid,
                "p_value"
            ],
            method="fdr_bh"
        )[1]

    # ========================================================
    # SIGNIFICANT RNAs
    #
    # Base-paper threshold:
    # p-value < 0.05
    # |log2FC| > 1
    # ========================================================

    significant = result[
        (result["p_value"] < 0.05) &
        (result["log2FC"].abs() > 1)
    ].copy()

    # --------------------------------------------------------
    # Sort results
    # --------------------------------------------------------

    result = result.sort_values(
        by="p_value",
        ascending=True
    )

    significant = significant.sort_values(
        by="p_value",
        ascending=True
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    result_file = os.path.join(
        RESULTS_DIR,
        output_name
    )

    significant_file = os.path.join(
        RESULTS_DIR,
        significant_name
    )

    result.to_csv(
        result_file,
        index=False
    )

    significant.to_csv(
        significant_file,
        index=False
    )

    print("\nResults saved:")
    print(result_file)

    print("\nSignificant RNAs saved:")
    print(significant_file)

    print(
        "\nSignificant RNAs:",
        len(significant)
    )

    return result, significant


# ============================================================
# mRNA
# ============================================================

mrna_results, mrna_significant = calculate_de(
    MRNA_FILE,
    "mrna_differential_expression.csv",
    "mrna_significant.csv"
)


# ============================================================
# lncRNA
# ============================================================

lncrna_results, lncrna_significant = calculate_de(
    LNCRNA_FILE,
    "lncrna_differential_expression.csv",
    "lncrna_significant.csv"
)


# ============================================================
# miRNA
# ============================================================

mirna_results, mirna_significant = calculate_de(
    MIRNA_FILE,
    "mirna_differential_expression.csv",
    "mirna_significant.csv"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("==========================================")
print("DIFFERENTIAL EXPRESSION COMPLETED")
print("==========================================")

print(
    "mRNA significant:",
    len(mrna_significant)
)

print(
    "lncRNA significant:",
    len(lncrna_significant)
)

print(
    "miRNA significant:",
    len(mirna_significant)
)

print("\nAll DE results are stored in:")
print(RESULTS_DIR)