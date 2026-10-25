import os
import requests
import pandas as pd

PROJECT = r"C:\Users\adity\OneDrive\Desktop\AI-Pancreatic-Cancer-Biomarker"

NODES_FILE = os.path.join(
    PROJECT,
    "data",
    "interactions",
    "nodes.csv"
)

OUT_FILE = os.path.join(
    PROJECT,
    "data",
    "interactions",
    "lncrna_id_name_mapping.csv"
)

nodes = pd.read_csv(NODES_FILE)

lncrnas = (
    nodes.loc[
        nodes["rna_type"].eq("lncRNA"),
        "rna_id"
    ]
    .astype(str)
    .unique()
    .tolist()
)

print("Total lncRNA IDs:", len(lncrnas))
print()

url = "https://mygene.info/v3/gene"

params = {
    "ids": ",".join(lncrnas),
    "fields": "symbol,name,ensembl",
    "species": "human"
}

response = requests.post(
    url,
    params=params,
    timeout=120
)

print("HTTP status:", response.status_code)

response.raise_for_status()

data = response.json()

mapping = {}

for item in data:

    query_id = str(item.get("query", ""))

    symbol = item.get("symbol", "")

    if not symbol:
        symbol = item.get("name", "")

    mapping[query_id] = symbol

rows = []

for rna_id in lncrnas:

    rows.append({
        "rna_id": rna_id,
        "gene_name": mapping.get(rna_id, "")
    })

result = pd.DataFrame(rows)

result.to_csv(
    OUT_FILE,
    index=False
)

print()
print("======================================")
print("DONE")
print("======================================")

print("Total IDs:", len(result))

print(
    "With gene names:",
    (result["gene_name"].astype(str).str.strip() != "").sum()
)

print("Output:", OUT_FILE)

print()
print("First 20 mappings:")
print(
    result.head(20).to_string(index=False)
)