
import os
import re
import numpy as np
import pandas as pd

# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

DQN_FILE = os.path.join(RESULTS_DIR, "dqn_driver_candidates.csv")
RGCN_FILE = os.path.join(RESULTS_DIR, "rgcn_node_embeddings.csv")

DE_FILES = [
    os.path.join(RESULTS_DIR, "mrna_differential_expression.csv"),
    os.path.join(RESULTS_DIR, "lncrna_differential_expression.csv"),
    os.path.join(RESULTS_DIR, "mirna_differential_expression.csv"),
]

FINAL_FILE = os.path.join(RESULTS_DIR, "FINAL_RGCN_DQN_BIOMARKERS.csv")
TOP10_FILE = os.path.join(RESULTS_DIR, "FINAL_TOP10_BIOMARKERS.csv")
TOP20_FILE = os.path.join(RESULTS_DIR, "FINAL_TOP20_BIOMARKERS.csv")
REPORT_FILE = os.path.join(RESULTS_DIR, "FINAL_RGCN_DQN_REPORT.txt")


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def normalize_name(name):
    """Normalize a column name for flexible matching."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def find_column(df, candidates, required=True):
    """Find a column regardless of capitalization or underscores."""
    normalized = {normalize_name(col): col for col in df.columns}

    for candidate in candidates:
        key = normalize_name(candidate)
        if key in normalized:
            return normalized[key]

    if required:
        raise ValueError(
            f"Could not find any of these columns: {candidates}\n"
            f"Available columns: {list(df.columns)}"
        )

    return None


def clean_ids(series):
    """Normalize RNA identifiers without changing their biological IDs."""
    return series.astype(str).str.strip()


def minmax(series):
    """Scale values between 0 and 1."""
    series = pd.to_numeric(series, errors="coerce").fillna(0.0)
    low = series.min()
    high = series.max()

    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return pd.Series(0.0, index=series.index)

    return (series - low) / (high - low)


def load_de_file(path, rna_type):
    """Load and standardize one differential-expression file."""
    if not os.path.exists(path):
        print(f"WARNING: Missing DE file: {os.path.basename(path)}")
        return pd.DataFrame(columns=["rna_id", "rna_type", "log2fc", "fdr"])

    df = pd.read_csv(path)
    print(f"Loaded {os.path.basename(path)}: {len(df)} rows")

    id_col = find_column(
        df, ["RNA_ID", "rna_id", "gene_id", "gene", "Gene", "ID", "id"]
    )
    fc_col = find_column(
        df, ["log2FC", "log2fc", "logFC", "log2_fold_change"]
    )
    fdr_col = find_column(
        df, ["FDR", "fdr", "padj", "adjusted_p_value", "adj_p_value"],
        required=False
    )

    # FDR is preferred. If unavailable, use p_value as a fallback.
    if fdr_col is None:
        fdr_col = find_column(df, ["p_value", "pvalue", "pval"])

    result = pd.DataFrame({
        "rna_id": clean_ids(df[id_col]),
        "rna_type": rna_type,
        "log2fc": pd.to_numeric(df[fc_col], errors="coerce"),
        "fdr": pd.to_numeric(df[fdr_col], errors="coerce"),
    })

    result = result.dropna(subset=["rna_id"])
    result = result.drop_duplicates(subset=["rna_id"], keep="first")
    return result


# ============================================================
# 3. CHECK INPUT FILES
# ============================================================

print("=" * 70)
print("FINAL RGCN + DQN PANCREATIC CANCER BIOMARKER RANKING")
print("=" * 70)

for path in [DQN_FILE, RGCN_FILE]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required input file not found: {path}")

dqn = pd.read_csv(DQN_FILE)
rgcn = pd.read_csv(RGCN_FILE)

print(f"\nDQN rows: {len(dqn)}")
print(f"RGCN embedding rows: {len(rgcn)}")


# ============================================================
# 4. STANDARDIZE DQN COLUMNS
# ============================================================

dqn_id_col = find_column(
    dqn, ["rna_id", "RNA_ID", "gene_id", "gene", "node_id"]
)

dqn = dqn.rename(columns={dqn_id_col: "rna_id"})
dqn["rna_id"] = clean_ids(dqn["rna_id"])

dqn_type_col = find_column(
    dqn, ["rna_type", "RNA_type", "type"], required=False
)

if dqn_type_col and dqn_type_col != "rna_type":
    dqn = dqn.rename(columns={dqn_type_col: "rna_type"})

degree_col = find_column(
    dqn, ["network_degree", "degree", "node_degree"], required=False
)

if degree_col:
    dqn["network_degree"] = pd.to_numeric(
        dqn[degree_col], errors="coerce"
    ).fillna(0)
else:
    dqn["network_degree"] = 0.0

q_col = find_column(
    dqn,
    ["dqn_q_value", "q_value", "qvalue", "action_q_value", "q_score"],
    required=False
)

if q_col:
    dqn["dqn_q_value"] = pd.to_numeric(
        dqn[q_col], errors="coerce"
    ).fillna(0)
else:
    print("WARNING: No Q-value column found; using equal DQN scores.")
    dqn["dqn_q_value"] = 0.0

degree_score_col = find_column(
    dqn, ["degree_score"], required=False
)

if degree_score_col:
    dqn["degree_score"] = pd.to_numeric(
        dqn[degree_score_col], errors="coerce"
    ).fillna(0)
else:
    dqn["degree_score"] = minmax(dqn["network_degree"])

# Keep one row per RNA ID.
dqn = dqn.drop_duplicates(subset=["rna_id"], keep="first")


# ============================================================
# 5. LOAD DIFFERENTIAL-EXPRESSION DATA
# ============================================================

de_tables = []

for path in DE_FILES:
    name = os.path.basename(path).lower()

    if "mrna" in name:
        rna_type = "mRNA"
    elif "lncrna" in name:
        rna_type = "lncRNA"
    else:
        rna_type = "miRNA"

    table = load_de_file(path, rna_type)

    if not table.empty:
        de_tables.append(table)

if not de_tables:
    raise ValueError("No differential-expression data could be loaded.")

de = pd.concat(de_tables, ignore_index=True)

# If an ID appears in more than one file, keep its first occurrence.
de = de.drop_duplicates(subset=["rna_id"], keep="first")

print(f"\nUnique DE RNA IDs: {len(de)}")


# ============================================================
# 6. STANDARDIZE RGCN INFORMATION
# ============================================================

rgcn_id_col = find_column(
    rgcn, ["rna_id", "RNA_ID", "gene_id", "gene", "node_id"]
)

rgcn = rgcn.rename(columns={rgcn_id_col: "rna_id"})
rgcn["rna_id"] = clean_ids(rgcn["rna_id"])

rgcn_type_col = find_column(
    rgcn, ["rna_type", "RNA_type", "type"], required=False
)

if rgcn_type_col and rgcn_type_col != "rna_type":
    rgcn = rgcn.rename(columns={rgcn_type_col: "rna_type"})

# Accept common RGCN embedding-column formats.
embedding_columns = [
    col for col in rgcn.columns
    if (
        normalize_name(col).startswith("rgcnembedding")
        or normalize_name(col).startswith("embedding")
    )
]

if not embedding_columns:
    # Also support columns such as x0, x1, ... only if present together.
    possible = [
        col for col in rgcn.columns
        if re.fullmatch(r"x\d+", str(col).lower())
    ]
    if possible:
        embedding_columns = possible

if embedding_columns:
    for col in embedding_columns:
        rgcn[col] = pd.to_numeric(rgcn[col], errors="coerce").fillna(0)

    rgcn["rgcn_embedding_norm"] = np.sqrt(
        (rgcn[embedding_columns] ** 2).sum(axis=1)
    )
else:
    print("WARNING: RGCN embedding columns were not recognized.")
    rgcn["rgcn_embedding_norm"] = 0.0

rgcn_columns = ["rna_id", "rgcn_embedding_norm"]

if "rna_type" in rgcn.columns:
    rgcn_columns.append("rna_type")

rgcn_small = rgcn[rgcn_columns].drop_duplicates(
    subset=["rna_id"], keep="first"
)


# ============================================================
# 7. MERGE DQN, DE, AND RGCN
# ============================================================

merged = dqn.merge(de, on="rna_id", how="left", suffixes=("", "_de"))

# Use DE type if DQN did not supply a type.
if "rna_type" not in merged.columns:
    merged["rna_type"] = merged["rna_type_de"]
elif "rna_type_de" in merged.columns:
    merged["rna_type"] = merged["rna_type"].fillna(merged["rna_type_de"])

merged = merged.drop(columns=["rna_type_de"], errors="ignore")

merged = merged.merge(
    rgcn_small,
    on="rna_id",
    how="left",
    suffixes=("", "_rgcn")
)

if "rna_type_rgcn" in merged.columns:
    if "rna_type" in merged.columns:
        merged["rna_type"] = merged["rna_type"].fillna(
            merged["rna_type_rgcn"]
        )
    else:
        merged["rna_type"] = merged["rna_type_rgcn"]

merged = merged.drop(columns=["rna_type_rgcn"], errors="ignore")

if "rna_type" not in merged.columns:
    merged["rna_type"] = "Unknown"

merged["rna_type"] = merged["rna_type"].fillna("Unknown")

print(f"Merged DQN candidates: {len(merged)}")


# ============================================================
# 8. CLEAN VALUES AND CALCULATE SCORES
# ============================================================

merged["log2fc"] = pd.to_numeric(
    merged["log2fc"], errors="coerce"
).fillna(0.0)

merged["fdr"] = pd.to_numeric(
    merged["fdr"], errors="coerce"
).fillna(1.0)

merged["network_degree"] = pd.to_numeric(
    merged["network_degree"], errors="coerce"
).fillna(0.0)

merged["dqn_q_value"] = pd.to_numeric(
    merged["dqn_q_value"], errors="coerce"
).fillna(0.0)

merged["rgcn_embedding_norm"] = pd.to_numeric(
    merged["rgcn_embedding_norm"], errors="coerce"
).fillna(0.0)

# Differential-expression score.
merged["de_score"] = minmax(merged["log2fc"].abs())

# FDR significance score.
merged["fdr_score"] = -np.log10(
    merged["fdr"].clip(lower=1e-300, upper=1.0)
)
merged["fdr_score_norm"] = minmax(merged["fdr_score"])

# DQN score. Higher Q-values are treated as better candidates.
merged["dqn_score"] = minmax(merged["dqn_q_value"])

# Network connectivity score.
merged["network_score"] = minmax(merged["network_degree"])

# RGCN embedding-magnitude score.
merged["rgcn_score"] = minmax(merged["rgcn_embedding_norm"])


# ============================================================
# 9. COMBINED RANKING SCORE
# ============================================================

# These are project-defined weights, not the exact base-paper formula.
merged["combined_biomarker_score"] = (
    0.35 * merged["dqn_score"]
    + 0.20 * merged["rgcn_score"]
    + 0.20 * merged["de_score"]
    + 0.10 * merged["fdr_score_norm"]
    + 0.15 * merged["network_score"]
)

# Only keep connected nodes.
merged = merged[merged["network_degree"] > 0].copy()

if merged.empty:
    raise ValueError(
        "No connected candidates remain. Check the DQN network-degree column."
    )

merged = merged.sort_values(
    "combined_biomarker_score", ascending=False
).reset_index(drop=True)

merged.insert(0, "biomarker_rank", np.arange(1, len(merged) + 1))


# ============================================================
# 10. SAVE RESULTS
# ============================================================

final_columns = [
    "biomarker_rank",
    "rna_id",
    "rna_type",
    "log2fc",
    "fdr",
    "network_degree",
    "degree_score",
    "dqn_q_value",
    "dqn_score",
    "rgcn_score",
    "de_score",
    "fdr_score_norm",
    "network_score",
    "combined_biomarker_score",
]

final_columns = [col for col in final_columns if col in merged.columns]
final_df = merged[final_columns].copy()

final_df.to_csv(FINAL_FILE, index=False)

top10 = final_df.head(10)
top20 = final_df.head(20)

top10.to_csv(TOP10_FILE, index=False)
top20.to_csv(TOP20_FILE, index=False)


# ============================================================
# 11. WRITE REPORT
# ============================================================

with open(REPORT_FILE, "w", encoding="utf-8") as report:
    report.write("FINAL RGCN + DQN PANCREATIC CANCER BIOMARKER REPORT\n")
    report.write("=" * 70 + "\n\n")
    report.write(f"DQN candidates loaded: {len(dqn)}\n")
    report.write(f"RGCN embedding rows: {len(rgcn)}\n")
    report.write(f"Final connected candidates ranked: {len(final_df)}\n")
    report.write("Ranking weights: DQN 35%, RGCN 20%, DE 20%, FDR 10%, network 15%\n")
    report.write(
        "Note: The combined score is a project-defined prioritization score. "
        "Candidates require independent biological and clinical validation.\n\n"
    )
    report.write("TOP 20 CANDIDATES\n")
    report.write("-" * 70 + "\n")

    for _, row in top20.iterrows():
        report.write(
            f"{int(row['biomarker_rank'])}. {row['rna_id']} | "
            f"{row['rna_type']} | log2FC={row['log2fc']:.4f} | "
            f"FDR={row['fdr']:.6g} | "
            f"Degree={row['network_degree']:.0f} | "
            f"Score={row['combined_biomarker_score']:.6f}\n"
        )


# ============================================================
# 12. DISPLAY COMPLETION
# ============================================================

print("\n" + "=" * 70)
print("TOP 20 FINAL BIOMARKER CANDIDATES")
print("=" * 70)
print(top20.to_string(index=False))

print("\nSaved files:")
print(FINAL_FILE)
print(TOP10_FILE)
print(TOP20_FILE)
print(REPORT_FILE)

print("\n" + "=" * 70)
print("FINAL BIOMARKER RANKING COMPLETED")
print("=" * 70)
