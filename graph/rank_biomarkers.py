import os
import pandas as pd

PROJECT = r"C:\Users\adity\OneDrive\Desktop\AI-Pancreatic-Cancer-Biomarker"

CENTRALITY_FILE = os.path.join(
    PROJECT,
    "data",
    "interactions",
    "node_centrality.csv"
)

RESULTS_DIR = os.path.join(
    PROJECT,
    "results"
)

OUT_FILE = os.path.join(
    RESULTS_DIR,
    "candidate_biomarkers.csv"
)

print("Loading network centrality...")
centrality = pd.read_csv(CENTRALITY_FILE)

print("Network nodes:", len(centrality))
print()

# --------------------------------------------------
# Load differential-expression results
# --------------------------------------------------

files = {
    "mRNA": "mrna_significant.csv",
    "lncRNA": "lncrna_significant.csv",
    "miRNA": "mirna_significant.csv"
}

all_de = []

for rna_type, filename in files.items():

    path = os.path.join(
        RESULTS_DIR,
        filename
    )

    print("Loading:", filename)

    df = pd.read_csv(path)

    print("  Rows:", len(df))
    print("  Columns:", list(df.columns))

    df["rna_type"] = rna_type

    all_de.append(df)

print()

de = pd.concat(
    all_de,
    ignore_index=True
)

# --------------------------------------------------
# Standardize RNA ID
# --------------------------------------------------

if "RNA_ID" in de.columns:

    de["rna_id"] = (
        de["RNA_ID"]
        .astype(str)
        .str.strip()
    )

elif "gene_id" in de.columns:

    de["rna_id"] = (
        de["gene_id"]
        .astype(str)
        .str.strip()
    )

elif "Gene" in de.columns:

    de["rna_id"] = (
        de["Gene"]
        .astype(str)
        .str.strip()
    )

else:

    raise ValueError(
        "Could not identify RNA ID column"
    )

# --------------------------------------------------
# Standardize statistics
# --------------------------------------------------

if "FDR" in de.columns:

    de["fdr"] = pd.to_numeric(
        de["FDR"],
        errors="coerce"
    )

elif "padj" in de.columns:

    de["fdr"] = pd.to_numeric(
        de["padj"],
        errors="coerce"
    )

else:

    raise ValueError(
        "Could not identify FDR/padj column"
    )

if "log2FC" in de.columns:

    de["log2fc"] = pd.to_numeric(
        de["log2FC"],
        errors="coerce"
    )

elif "log2FoldChange" in de.columns:

    de["log2fc"] = pd.to_numeric(
        de["log2FoldChange"],
        errors="coerce"
    )

else:

    raise ValueError(
        "Could not identify log2FC column"
    )

# --------------------------------------------------
# Merge DE with network
# --------------------------------------------------

print("Merging DE results with network centrality...")

merged = centrality.merge(
    de[
        [
            "rna_id",
            "rna_type",
            "log2fc",
            "fdr"
        ]
    ],
    on=["rna_id", "rna_type"],
    how="inner"
)

print(
    "Matched DE + network nodes:",
    len(merged)
)

# --------------------------------------------------
# Remove invalid statistics
# --------------------------------------------------

merged = merged.dropna(
    subset=[
        "log2fc",
        "fdr",
        "degree",
        "betweenness",
        "pagerank"
    ]
)

merged = merged[
    merged["fdr"] > 0
]

# --------------------------------------------------
# Calculate biological scores
# --------------------------------------------------

merged["abs_log2fc"] = (
    merged["log2fc"].abs()
)

merged["-log10_fdr"] = (
    -merged["fdr"].apply(
        lambda x: __import__("math").log10(x)
    )
)

# --------------------------------------------------
# Normalize scores
# --------------------------------------------------

def normalize(series):

    minimum = series.min()
    maximum = series.max()

    if maximum == minimum:
        return pd.Series(
            0.5,
            index=series.index
        )

    return (
        (series - minimum)
        / (maximum - minimum)
    )

merged["de_score"] = (
    0.5 * normalize(
        merged["abs_log2fc"]
    )
    +
    0.5 * normalize(
        merged["-log10_fdr"]
    )
)

merged["network_score"] = (
    0.4 * normalize(
        merged["degree"]
    )
    +
    0.3 * normalize(
        merged["betweenness"]
    )
    +
    0.3 * normalize(
        merged["pagerank"]
    )
)

# --------------------------------------------------
# Final biomarker score
# --------------------------------------------------

merged["biomarker_score"] = (
    0.6 * merged["de_score"]
    +
    0.4 * merged["network_score"]
)

# --------------------------------------------------
# Ranking
# --------------------------------------------------

merged = merged.sort_values(
    "biomarker_score",
    ascending=False
)

merged["biomarker_rank"] = range(
    1,
    len(merged) + 1
)

# --------------------------------------------------
# Save complete ranking
# --------------------------------------------------

merged.to_csv(
    OUT_FILE,
    index=False
)

print()
print("======================================")
print("BIOMARKER RANKING COMPLETED")
print("======================================")

print()
print("Total candidates:", len(merged))

print()
print("Output:")
print(OUT_FILE)

print()
print("TOP 20 CANDIDATE BIOMARKERS")
print("--------------------------------------")

print(
    merged[
        [
            "biomarker_rank",
            "rna_id",
            "rna_type",
            "log2fc",
            "fdr",
            "degree",
            "betweenness",
            "pagerank",
            "biomarker_score"
        ]
    ]
    .head(20)
    .to_string(index=False)
)

print()
print("TOP lncRNA CANDIDATES")
print("--------------------------------------")

print(
    merged[
        merged["rna_type"] == "lncRNA"
    ][
        [
            "biomarker_rank",
            "rna_id",
            "log2fc",
            "fdr",
            "degree",
            "biomarker_score"
        ]
    ]
    .head(10)
    .to_string(index=False)
)

print()
print("TOP miRNA CANDIDATES")
print("--------------------------------------")

print(
    merged[
        merged["rna_type"] == "miRNA"
    ][
        [
            "biomarker_rank",
            "rna_id",
            "log2fc",
            "fdr",
            "degree",
            "biomarker_score"
        ]
    ]
    .head(10)
    .to_string(index=False)
)

print()
print("TOP mRNA CANDIDATES")
print("--------------------------------------")

print(
    merged[
        merged["rna_type"] == "mRNA"
    ][
        [
            "biomarker_rank",
            "rna_id",
            "log2fc",
            "fdr",
            "degree",
            "biomarker_score"
        ]
    ]
    .head(10)
    .to_string(index=False)
)