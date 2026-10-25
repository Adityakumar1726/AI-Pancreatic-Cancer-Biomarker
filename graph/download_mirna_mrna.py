import os
import time
import requests
import pandas as pd
from io import StringIO

PROJECT = r"C:\Users\adity\OneDrive\Desktop\AI-Pancreatic-Cancer-Biomarker"

SIG_FILE = os.path.join(
    PROJECT, "results", "mirna_significant.csv"
)

NODES_FILE = os.path.join(
    PROJECT, "data", "interactions", "nodes.csv"
)

OUT_FILE = os.path.join(
    PROJECT, "data", "interactions", "mirna_mrna.csv"
)

MAP_FILE = os.path.join(
    PROJECT, "data", "interactions", "mirna_mature_mapping.csv"
)

API = "https://rnasysu.com/encori/api/miRNATarget/"


# ============================================================
# LOAD PROJECT DATA
# ============================================================

sig = pd.read_csv(SIG_FILE)

project_mirnas = (
    sig["RNA_ID"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

nodes = pd.read_csv(NODES_FILE)

project_mrna = set(
    nodes.loc[
        nodes["rna_type"].eq("mRNA"),
        "rna_id"
    ].astype(str)
)

print("Project miRNAs:", len(project_mirnas))
print("Project mRNA nodes:", len(project_mrna))
print()


# ============================================================
# STORAGE
# ============================================================

all_edges = []
mapping = []


# ============================================================
# QUERY EACH PROJECT miRNA
# ============================================================

for precursor in project_mirnas:

    # --------------------------------------------------------
    # Convert project naming:
    # hsa-mir-223 -> hsa-miR-223
    # --------------------------------------------------------

    base = precursor.replace(
        "hsa-mir-",
        "hsa-miR-"
    )

    # --------------------------------------------------------
    # Remove precursor copy suffix where necessary
    #
    # hsa-miR-196a-1 -> hsa-miR-196a
    # hsa-miR-196a-2 -> hsa-miR-196a
    # --------------------------------------------------------

    if base.endswith("-1") or base.endswith("-2"):
        base_no_copy = base.rsplit("-", 1)[0]
    else:
        base_no_copy = base

    # --------------------------------------------------------
    # Query both mature arms
    # --------------------------------------------------------

    candidates = [
        base_no_copy + "-3p",
        base_no_copy + "-5p"
    ]

    for mature in candidates:

        print(
            f"Querying {precursor} -> {mature}"
        )

        params = {
            "assembly": "hg38",
            "geneType": "mRNA",
            "miRNA": mature,
            "clipExpNum": 1,
            "degraExpNum": 0,
            "pancancerNum": 0,
            "programNum": 1,
            "program": "None",
            "target": "all",
            "cellType": "all"
        }

        try:

            response = requests.get(
                API,
                params=params,
                timeout=120
            )

            print(
                "  HTTP:",
                response.status_code
            )

            if response.status_code != 200:
                print("  HTTP error")
                continue

            text = response.text.strip()

            if not text:
                print("  No data returned")
                continue

            # =================================================
            # FIND THE REAL TABLE HEADER
            #
            # ENCORI returns citation/comment lines before
            # the actual tab-separated table.
            # =================================================

            lines = text.splitlines()

            header_index = None

            for i, line in enumerate(lines):

                if (
                    "miRNAid" in line
                    and "geneID" in line
                    and "geneType" in line
                ):
                    header_index = i
                    break

            if header_index is None:

                print(
                    "  Could not find ENCORI table header"
                )

                continue

            # Keep only the actual table
            table_text = "\n".join(
                lines[header_index:]
            )

            # Read the real table
            df = pd.read_csv(
                StringIO(table_text),
                sep="\t"
            )

            # =================================================
            # CHECK geneID
            # =================================================

            if "geneID" not in df.columns:

                print(
                    "  geneID column not found"
                )

                continue

            # =================================================
            # KEEP ONLY mRNAs PRESENT IN OUR PROJECT
            # =================================================

            df["geneID"] = (
                df["geneID"]
                .astype(str)
                .str.strip()
            )

            df = df[
                df["geneID"].isin(project_mrna)
            ]

            if len(df) == 0:

                print(
                    "  0 project-mRNA matches"
                )

                continue

            print(
                "  Project-mRNA matches:",
                len(df)
            )

            # =================================================
            # CREATE EDGES
            #
            # IMPORTANT:
            # source = PROJECT miRNA ID
            # NOT mature ENCORI ID
            #
            # Example:
            # hsa-mir-223 -> ENSG000...
            #
            # This matches our existing nodes.csv:
            # miRNA_hsa-mir-223
            # =================================================

            for gene_id in df["geneID"]:

                all_edges.append({
                    "source": precursor,
                    "target": gene_id
                })

            # Save mapping information

            mapping.append({
                "project_miRNA": precursor,
                "encori_mature_miRNA": mature,
                "project_mRNA_matches": len(df)
            })

        except Exception as e:

            print(
                "  ERROR:",
                str(e)
            )

        # Small delay to avoid sending requests too quickly
        time.sleep(1)


# ============================================================
# REMOVE DUPLICATES
# ============================================================

if all_edges:

    edges = pd.DataFrame(
        all_edges,
        columns=[
            "source",
            "target"
        ]
    )

    edges = edges.drop_duplicates()

else:

    edges = pd.DataFrame(
        columns=[
            "source",
            "target"
        ]
    )


# ============================================================
# SAVE INTERACTION FILE
# ============================================================

edges.to_csv(
    OUT_FILE,
    index=False
)


# ============================================================
# SAVE MATURE miRNA MAPPING
# ============================================================

pd.DataFrame(mapping).to_csv(
    MAP_FILE,
    index=False
)


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("======================================")
print("DONE")
print("======================================")

print(
    "Unique interactions:",
    len(edges)
)

print(
    "Output:",
    OUT_FILE
)

print(
    "Mapping:",
    MAP_FILE
)

print()

print("First 10 interactions:")

if len(edges) > 0:
    print(
        edges.head(10).to_string(
            index=False
        )
    )
else:
    print(
        "NO INTERACTIONS FOUND"
    )