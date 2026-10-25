import os
import requests
import pandas as pd

PROJECT = r"C:\Users\adity\OneDrive\Desktop\AI-Pancreatic-Cancer-Biomarker"

INPUT = os.path.join(
    PROJECT,
    "results",
    "final_candidate_summary.csv"
)

OUTPUT = os.path.join(
    PROJECT,
    "results",
    "final_annotated_biomarkers.csv"
)

df = pd.read_csv(INPUT)

# Keep statistically significant + network-supported candidates
df = df[
    (df["fdr"] < 0.05) &
    (df["degree"] > 0)
].copy()

# Select top 5 from each RNA type
selected = []

for rna_type in ["miRNA", "lncRNA", "mRNA"]:

    part = (
        df[df["rna_type"] == rna_type]
        .sort_values("biomarker_score", ascending=False)
        .head(5)
        .copy()
    )

    selected.append(part)

final = pd.concat(
    selected,
    ignore_index=True
)

# Only Ensembl IDs need annotation
ensembl_ids = (
    final.loc[
        final["rna_type"].isin(["mRNA", "lncRNA"]),
        "rna_id"
    ]
    .astype(str)
    .unique()
    .tolist()
)

print("Ensembl IDs to annotate:", len(ensembl_ids))

symbols = {}

if ensembl_ids:

    response = requests.post(
        "https://mygene.info/v3/gene",
        params={
            "ids": ",".join(ensembl_ids),
            "fields": "symbol,name",
            "species": "human"
        },
        timeout=120
    )

    print("MyGene HTTP status:", response.status_code)

    response.raise_for_status()

    for item in response.json():

        query = str(
            item.get("query", "")
        )

        symbol = item.get(
            "symbol",
            ""
        )

        name = item.get(
            "name",
            ""
        )

        symbols[query] = {
            "gene_symbol": symbol,
            "gene_name": name
        }

# Add annotation
final["gene_symbol"] = final["rna_id"].map(
    lambda x: symbols.get(
        str(x),
        {}
    ).get("gene_symbol", "")
)

final["gene_name"] = final["rna_id"].map(
    lambda x: symbols.get(
        str(x),
        {}
    ).get("gene_name", "")
)

# For miRNAs, use their existing ID
final.loc[
    final["rna_type"] == "miRNA",
    "gene_symbol"
] = final.loc[
    final["rna_type"] == "miRNA",
    "rna_id"
]

final = final.sort_values(
    "biomarker_score",
    ascending=False
)

final.to_csv(
    OUTPUT,
    index=False
)

print()
print("======================================")
print("FINAL BIOMARKER ANNOTATION COMPLETED")
print("======================================")

print()
print(
    final[
        [
            "rna_type",
            "rna_id",
            "gene_symbol",
            "gene_name",
            "log2fc",
            "fdr",
            "degree",
            "biomarker_score"
        ]
    ].to_string(index=False)
)

print()
print("Saved:")
print(OUTPUT)