import os
import time
import requests
import pandas as pd

PROJECT = r"C:\Users\adity\OneDrive\Desktop\AI-Pancreatic-Cancer-Biomarker"

NODES_FILE = os.path.join(
    PROJECT,
    "data",
    "interactions",
    "nodes.csv"
)

MAPPING_FILE = os.path.join(
    PROJECT,
    "data",
    "interactions",
    "lncrna_id_name_mapping.csv"
)

OUT_FILE = os.path.join(
    PROJECT,
    "data",
    "interactions",
    "lncrna_mrna.csv"
)

API = "https://rnasysu.com/encori/api/RNARNA/"

# -----------------------------
# Load project nodes
# -----------------------------

nodes = pd.read_csv(NODES_FILE)

project_mrna = set(
    nodes.loc[
        nodes["rna_type"].eq("mRNA"),
        "rna_id"
    ]
    .astype(str)
)

# -----------------------------
# Load lncRNA ID -> name mapping
# -----------------------------

mapping = pd.read_csv(MAPPING_FILE, dtype=str).fillna("")

mapping = mapping[
    mapping["gene_name"].str.strip() != ""
]

print("Project mRNA nodes:", len(project_mrna))
print("lncRNAs with gene names:", len(mapping))
print()

edges = set()

session = requests.Session()

# -----------------------------
# Query ENCORI
# -----------------------------

for i, row in enumerate(
    mapping.iterrows(),
    start=1
):

    _, record = row

    lnc_id = str(record["rna_id"]).strip()
    lnc_name = str(record["gene_name"]).strip()

    print(
        f"[{i}/{len(mapping)}] "
        f"Querying {lnc_name} "
        f"({lnc_id})"
    )

    params = {
        "assembly": "hg38",
        "geneType": "lncRNA",
        "RNA": lnc_name,
        "interNum": 1,
        "expNum": 1,
        "cellType": "all"
    }

    try:

        response = session.get(
            API,
            params=params,
            timeout=120
        )

        if response.status_code != 200:

            print(
                "  HTTP error:",
                response.status_code
            )

            continue

        lines = [
            line.strip()
            for line in response.text.splitlines()
            if line.strip()
        ]

        # -----------------------------
        # Extract 18-column records
        # -----------------------------

        records = []

        for line in lines:

            fields = line.split("\t")

            if len(fields) == 18:
                records.append(fields)

        if not records:

            print("  No ENCORI records")

            continue

        matches = 0

        for fields in records:

            gene_id = str(fields[0]).strip()
            gene_name = str(fields[1]).strip()
            gene_type = str(fields[2]).strip()

            pair_gene_id = str(fields[3]).strip()
            pair_gene_name = str(fields[4]).strip()
            pair_gene_type = str(fields[5]).strip()

            # Make sure this is an lncRNA interaction
            if gene_type != "lncRNA":
                continue

            # We only want project mRNAs
            if pair_gene_type not in [
                "protein_coding",
                "mRNA"
            ]:
                continue

            if pair_gene_id not in project_mrna:
                continue

            # Store project lncRNA ID,
            # not the ENCORI query name
            edges.add(
                (
                    lnc_id,
                    pair_gene_id
                )
            )

            matches += 1

        if matches > 0:

            print(
                "  Project-mRNA matches:",
                matches
            )

    except Exception as e:

        print(
            "  ERROR:",
            str(e)
        )

    # Small delay to avoid hammering API
    time.sleep(0.3)

# -----------------------------
# Save result
# -----------------------------

result = pd.DataFrame(
    sorted(edges),
    columns=[
        "source",
        "target"
    ]
)

result = result.drop_duplicates()

result.to_csv(
    OUT_FILE,
    index=False
)

# -----------------------------
# Final report
# -----------------------------

print()
print("======================================")
print("DONE")
print("======================================")

print(
    "Unique lncRNA-mRNA interactions:",
    len(result)
)

print(
    "Output:",
    OUT_FILE
)

print()

if len(result) > 0:

    print("First 20 interactions:")
    print(
        result.head(20)
        .to_string(index=False)
    )

else:

    print("NO MATCHES FOUND")