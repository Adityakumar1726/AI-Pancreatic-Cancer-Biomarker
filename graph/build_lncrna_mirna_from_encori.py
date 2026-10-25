import os
import pandas as pd

PROJECT = r"C:\Users\adity\OneDrive\Desktop\AI-Pancreatic-Cancer-Biomarker"

INPUT = os.path.join(
    PROJECT,
    "data", "raw", "encori", "mirna_lncrna",
    "lncRNA", "hg38_miRlncRNA.txt"
)

NODES = os.path.join(
    PROJECT,
    "data", "interactions", "nodes.csv"
)

OUTPUT = os.path.join(
    PROJECT,
    "data", "interactions", "lncrna_mirna.csv"
)

MAPPING = os.path.join(
    PROJECT,
    "data", "interactions", "mirna_lncrna_mapping.csv"
)

# ---------------------------------------
# Load project nodes
# ---------------------------------------
nodes = pd.read_csv(NODES)

project_lnc = set(
    nodes.loc[nodes["rna_type"].eq("lncRNA"), "rna_id"]
    .astype(str)
)

project_mirnas = (
    nodes.loc[nodes["rna_type"].eq("miRNA"), "rna_id"]
    .astype(str)
    .unique()
    .tolist()
)

print("Project lncRNA nodes:", len(project_lnc))
print("Project miRNA nodes:", len(project_mirnas))

# ---------------------------------------
# Build mature-miRNA -> project-miRNA map
# ---------------------------------------
mature_to_project = {}

for precursor in project_mirnas:

    base = precursor.replace("hsa-mir-", "hsa-miR-")

    # Remove genomic copy suffix only for matching mature miRNA
    if base.endswith("-1") or base.endswith("-2"):
        family = base.rsplit("-", 1)[0]
    else:
        family = base

    for arm in ["3p", "5p"]:
        mature = family + "-" + arm

        if mature not in mature_to_project:
            mature_to_project[mature] = []

        mature_to_project[mature].append(precursor)

# ---------------------------------------
# Read the 380 MB ENCORI file in chunks
# ---------------------------------------
edges = set()
mapping_rows = []

print()
print("Reading ENCORI miRNA-lncRNA data...")
print("This may take a little time because the file is ~380 MB.")
print()

for chunk in pd.read_csv(
    INPUT,
    sep="\t",
    dtype=str,
    chunksize=200000,
    low_memory=False
):

    required = {"miRNAname", "geneID", "geneType"}

    if not required.issubset(chunk.columns):
        print("ERROR: Required columns not found.")
        print(chunk.columns.tolist())
        raise SystemExit

    # Keep only lncRNA records
    chunk = chunk[
        chunk["geneType"].astype(str).str.lower().eq("lncrna")
    ]

    # Keep only project lncRNAs
    chunk = chunk[
        chunk["geneID"].astype(str).isin(project_lnc)
    ]

    if len(chunk) == 0:
        continue

    for _, row in chunk.iterrows():

        mature = str(row["miRNAname"]).strip()
        lnc = str(row["geneID"]).strip()

        if mature not in mature_to_project:
            continue

        project_matches = mature_to_project[mature]

        # Do not invent an assignment when the same mature miRNA
        # corresponds to multiple project precursor nodes.
        if len(project_matches) != 1:
            continue

        project_mirna = project_matches[0]

        edges.add((lnc, project_mirna))

# ---------------------------------------
# Save
# ---------------------------------------
result = pd.DataFrame(
    sorted(edges),
    columns=["source", "target"]
)

result.to_csv(OUTPUT, index=False)

print()
print("======================================")
print("DONE")
print("======================================")
print("Unique lncRNA-miRNA interactions:", len(result))
print("Output:", OUTPUT)

print()
print("First interactions:")

if len(result) > 0:
    print(result.head(10).to_string(index=False))
else:
    print("NO MATCHES FOUND")
