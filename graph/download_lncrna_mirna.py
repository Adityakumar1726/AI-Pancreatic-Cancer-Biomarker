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
    PROJECT, "data", "interactions", "lncrna_mirna.csv"
)

MAP_FILE = os.path.join(
    PROJECT, "data", "interactions", "mirna_lncRNA_mapping.csv"
)

API = "https://rnasysu.com/encori/api/miRNATarget/"


# ============================================================
# LOAD PROJECT miRNAs
# ============================================================

sig = pd.read_csv(SIG_FILE)

project_mirnas = (
    sig["RNA_ID"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)


# ============================================================
# LOAD PROJECT lncRNA NODES
# ============================================================

nodes = pd.read_csv(NODES_FILE)

project_lncrna = set(
    nodes.loc[
        nodes["rna_type"].eq("lncRNA"),
        "rna_id"
    ].astype(str)
)

print("Project miRNAs:", len(project_mirnas))
print("Project lncRNA nodes:", len(project_lncrna))
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

    # Example:
    # hsa-mir-223 -> hsa-miR-223

    base = precursor.replace(
        "hsa-mir-",
        "hsa-miR-"
    )

    # Remove precursor copy suffix
    # hsa-miR-196a-1 -> hsa-miR-196a

    if base.endswith("-1") or base.endswith("-2"):
        base_no_copy = base.rsplit("-", 1)[0]
    else:
        base_no_copy = base

    # Query both mature arms

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
            "geneType": "lncRNA",
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
            # FIND ACTUAL TABLE HEADER
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

            table_text = "\n".join(
                lines[header_index:]
            )

            df = pd.read_csv(
                StringIO(table_text),
                sep="\t"
            )

            if "geneID" not in df.columns:

                print(
                    "  geneID column not found"
                )

                continue

            # =================================================
            # KEEP ONLY lncRNAs IN OUR PROJECT
            # =================================================

            df["geneID"] = (
                df["geneID"]
                .astype(str)
                .str.strip()
            )

            df = df[
                df["geneID"].isin(project_lncrna)
            ]

            if len(df) == 0:

                print(
                    "  0 project-lncRNA matches"
                )

                continue

            print(
                "  Project-lncRNA matches:",
                len(df)
            )

            # =================================================
            # IMPORTANT:
            #
            # build_interaction_network.py expects:
            #
            # source = lncRNA
            # target = miRNA
            #
            # Therefore we reverse the API orientation.
            # =================================================

            for lncrna_id in df["geneID"]:

                all_edges.append({
                    "source": lncrna_id,
                    "target": precursor
                })

            mapping.append({
                "project_miRNA": precursor,
                "encori_mature_miRNA": mature,
                "project_lncRNA_matches": len(df)
            })

        except Exception as e:

            print(
                "  ERROR:",
                str(e)
            )

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
# SAVE
# ============================================================

edges.to_csv(
    OUT_FILE,
    index=False
)

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
    "Unique lncRNA-miRNA interactions:",
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