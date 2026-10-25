import os
import pandas as pd
import numpy as np


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

INTERACTION_DIR = os.path.join(
    PROJECT_DIR,
    "data",
    "interactions"
)

os.makedirs(INTERACTION_DIR, exist_ok=True)


# ============================================================
# INPUT FILES
# ============================================================

MRNA_EXPRESSION = os.path.join(
    PROCESSED_DIR,
    "mrna_expression_counts.csv"
)

LNCRNA_EXPRESSION = os.path.join(
    PROCESSED_DIR,
    "lncrna_expression_counts.csv"
)

MIRNA_EXPRESSION = os.path.join(
    PROCESSED_DIR,
    "mirna_expression_counts.csv"
)

MRNA_SIGNIFICANT = os.path.join(
    RESULTS_DIR,
    "mrna_significant.csv"
)

LNCRNA_SIGNIFICANT = os.path.join(
    RESULTS_DIR,
    "lncrna_significant.csv"
)

MIRNA_SIGNIFICANT = os.path.join(
    RESULTS_DIR,
    "mirna_significant.csv"
)


# ============================================================
# OUTPUT FILES
# ============================================================

NODES_OUTPUT = os.path.join(
    INTERACTION_DIR,
    "nodes.csv"
)

EDGES_OUTPUT = os.path.join(
    INTERACTION_DIR,
    "edges.csv"
)

SUMMARY_OUTPUT = os.path.join(
    INTERACTION_DIR,
    "network_summary.txt"
)


# ============================================================
# PARAMETERS
# ============================================================

PCC_THRESHOLD = 0.80


# ============================================================
# LOAD SIGNIFICANT RNA LISTS
# ============================================================

print("==============================================")
print("BUILDING HETEROGENEOUS RNA INTERACTION GRAPH")
print("==============================================")

print("\nReading significant RNA lists...")


mrna_sig = pd.read_csv(MRNA_SIGNIFICANT)
lncrna_sig = pd.read_csv(LNCRNA_SIGNIFICANT)
mirna_sig = pd.read_csv(MIRNA_SIGNIFICANT)


mrna_ids = set(
    mrna_sig["RNA_ID"].astype(str)
)

lncrna_ids = set(
    lncrna_sig["RNA_ID"].astype(str)
)

mirna_ids = set(
    mirna_sig["RNA_ID"].astype(str)
)


print("Significant mRNA:", len(mrna_ids))
print("Significant lncRNA:", len(lncrna_ids))
print("Significant miRNA:", len(mirna_ids))


# ============================================================
# LOAD EXPRESSION MATRICES
# ============================================================

print("\nLoading expression matrices...")


mrna = pd.read_csv(
    MRNA_EXPRESSION,
    index_col=0
)

lncrna = pd.read_csv(
    LNCRNA_EXPRESSION,
    index_col=0
)

mirna = pd.read_csv(
    MIRNA_EXPRESSION,
    index_col=0
)


# Convert IDs to strings

mrna.index = mrna.index.astype(str)
lncrna.index = lncrna.index.astype(str)
mirna.index = mirna.index.astype(str)


# ============================================================
# KEEP ONLY SIGNIFICANT RNAs
# ============================================================

mrna = mrna[
    mrna.index.isin(mrna_ids)
].copy()

lncrna = lncrna[
    lncrna.index.isin(lncrna_ids)
].copy()

mirna = mirna[
    mirna.index.isin(mirna_ids)
].copy()


print("\nExpression features after filtering:")
print("mRNA:", mrna.shape)
print("lncRNA:", lncrna.shape)
print("miRNA:", mirna.shape)


# ============================================================
# CREATE NODE TABLE
# ============================================================

print("\nCreating RNA nodes...")


nodes = []


for rna_id in mrna.index:

    nodes.append([
        "mRNA_" + str(rna_id),
        str(rna_id),
        "mRNA"
    ])


for rna_id in lncrna.index:

    nodes.append([
        "lncRNA_" + str(rna_id),
        str(rna_id),
        "lncRNA"
    ])


for rna_id in mirna.index:

    nodes.append([
        "miRNA_" + str(rna_id),
        str(rna_id),
        "miRNA"
    ])


nodes_df = pd.DataFrame(
    nodes,
    columns=[
        "node_id",
        "rna_id",
        "rna_type"
    ]
)


nodes_df.to_csv(
    NODES_OUTPUT,
    index=False
)


print("Total nodes:", len(nodes_df))


# ============================================================
# CORRELATION EDGE FUNCTION
# ============================================================

def create_correlation_edges(
    expression,
    rna_type,
    threshold=0.80
):

    print(
        f"\nCalculating {rna_type}-{rna_type} "
        f"Pearson correlations..."
    )

    if expression.shape[0] < 2:
        return []

    # Make sure values are numeric

    expression = expression.apply(
        pd.to_numeric,
        errors="coerce"
    ).fillna(0)

    # Convert genes x samples to numpy

    values = expression.values.astype(
        np.float32
    )

    # Pearson correlation between rows

    correlation = np.corrcoef(values)

    genes = expression.index.tolist()

    edges = []

    for i in range(len(genes)):

        if i % 100 == 0:
            print(
                f"  Processing {i}/{len(genes)} "
                f"{rna_type} nodes"
            )

        for j in range(i + 1, len(genes)):

            pcc = correlation[i, j]

            if not np.isfinite(pcc):
                continue

            if abs(pcc) >= threshold:

                source = (
                    rna_type + "_" +
                    str(genes[i])
                )

                target = (
                    rna_type + "_" +
                    str(genes[j])
                )

                edges.append([
                    source,
                    target,
                    rna_type + "_" + rna_type,
                    float(pcc)
                ])

    print(
        f"{rna_type}-{rna_type} edges:",
        len(edges)
    )

    return edges


# ============================================================
# BUILD SAME-TYPE CORRELATION EDGES
# ============================================================

all_edges = []


# mRNA-mRNA

mrna_edges = create_correlation_edges(
    mrna,
    "mRNA",
    PCC_THRESHOLD
)

all_edges.extend(mrna_edges)


# lncRNA-lncRNA

lncrna_edges = create_correlation_edges(
    lncrna,
    "lncRNA",
    PCC_THRESHOLD
)

all_edges.extend(lncrna_edges)


# miRNA-miRNA

mirna_edges = create_correlation_edges(
    mirna,
    "miRNA",
    PCC_THRESHOLD
)

all_edges.extend(mirna_edges)


# ============================================================
# OPTIONAL CROSS-TYPE BIOLOGICAL INTERACTIONS
# ============================================================

print("\nChecking for biological interaction files...")


def load_cross_type_interactions(
    filename,
    relation_type,
    source_prefix,
    target_prefix
):

    filepath = os.path.join(
        INTERACTION_DIR,
        filename
    )

    if not os.path.exists(filepath):

        print(
            f"Not found: {filename}"
        )

        return []

    print(
        f"Loading: {filename}"
    )

    df = pd.read_csv(filepath)

    required_columns = [
        "source",
        "target"
    ]

    if not all(
        column in df.columns
        for column in required_columns
    ):

        print(
            f"Skipping {filename}: "
            "required columns are source,target"
        )

        return []

    edges = []

    for _, row in df.iterrows():

        source = str(
            row["source"]
        )

        target = str(
            row["target"]
        )

        source_node = (
            source_prefix + "_" +
            source
        )

        target_node = (
            target_prefix + "_" +
            target
        )

        # Only keep interactions
        # between significant RNAs

        source_exists = (
            source_node in
            set(nodes_df["node_id"])
        )

        target_exists = (
            target_node in
            set(nodes_df["node_id"])
        )

        if not source_exists or not target_exists:
            continue

        edges.append([
            source_node,
            target_node,
            relation_type,
            1.0
        ])

    print(
        f"Valid {relation_type} edges:",
        len(edges)
    )

    return edges


# ------------------------------------------------------------
# These files will be added later after obtaining
# biological interaction data.
# ------------------------------------------------------------

cross_edges = []


cross_edges.extend(
    load_cross_type_interactions(
        "mirna_mrna.csv",
        "miRNA_mRNA",
        "miRNA",
        "mRNA"
    )
)


cross_edges.extend(
    load_cross_type_interactions(
        "lncrna_mirna.csv",
        "lncRNA_miRNA",
        "lncRNA",
        "miRNA"
    )
)


cross_edges.extend(
    load_cross_type_interactions(
        "lncrna_mrna.csv",
        "lncRNA_mRNA",
        "lncRNA",
        "mRNA"
    )
)


all_edges.extend(
    cross_edges
)


# ============================================================
# CREATE EDGE TABLE
# ============================================================

edges_df = pd.DataFrame(
    all_edges,
    columns=[
        "source",
        "target",
        "relation",
        "weight"
    ]
)


# Remove duplicate edges

if len(edges_df) > 0:

    edges_df = edges_df.drop_duplicates(
        subset=[
            "source",
            "target",
            "relation"
        ]
    )


# ============================================================
# SAVE EDGES
# ============================================================

edges_df.to_csv(
    EDGES_OUTPUT,
    index=False
)


# ============================================================
# NETWORK SUMMARY
# ============================================================

relation_counts = (
    edges_df["relation"].value_counts()
    if len(edges_df) > 0
    else pd.Series(dtype=int)
)


with open(
    SUMMARY_OUTPUT,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "PANCREATIC CANCER RNA NETWORK SUMMARY\n"
    )

    f.write(
        "=====================================\n\n"
    )

    f.write(
        f"mRNA nodes: {len(mrna)}\n"
    )

    f.write(
        f"lncRNA nodes: {len(lncrna)}\n"
    )

    f.write(
        f"miRNA nodes: {len(mirna)}\n"
    )

    f.write(
        f"Total nodes: {len(nodes_df)}\n\n"
    )

    f.write(
        f"PCC threshold: {PCC_THRESHOLD}\n\n"
    )

    f.write(
        f"Total edges: {len(edges_df)}\n\n"
    )

    f.write(
        "Edges by relation:\n"
    )

    for relation, count in relation_counts.items():

        f.write(
            f"{relation}: {count}\n"
        )


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n==============================================")
print("RNA INTERACTION NETWORK COMPLETED")
print("==============================================")

print("\nNodes saved to:")
print(NODES_OUTPUT)

print("\nEdges saved to:")
print(EDGES_OUTPUT)

print("\nNetwork summary saved to:")
print(SUMMARY_OUTPUT)

print("\nTotal nodes:", len(nodes_df))
print("Total edges:", len(edges_df))

print("\nEdges by relation:")

if len(relation_counts) > 0:

    print(relation_counts)

else:

    print("No edges generated.")