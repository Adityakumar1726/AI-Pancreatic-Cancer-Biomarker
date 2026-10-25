
import os
import pandas as pd
import urllib.parse
import urllib.request
import json
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

INPUT_FILE = os.path.join(RESULTS_DIR, "FINAL_TOP20_BIOMARKERS.csv")
OUTPUT_FILE = os.path.join(RESULTS_DIR, "FINAL_TOP20_WITH_GENE_NAMES.csv")

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"Missing input file: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)

if "rna_id" not in df.columns:
    raise ValueError("The input CSV must contain an rna_id column.")

# Keep Ensembl gene IDs only; preserve all rows in the original ranking.
ensembl_ids = [
    str(value).strip().split(".")[0]
    for value in df["rna_id"].dropna().unique()
    if str(value).strip().startswith("ENSG")
]

symbols = {}
server = "https://rest.ensembl.org"

for gene_id in ensembl_ids:
    url = f"{server}/lookup/id/{urllib.parse.quote(gene_id)}?content-type=application/json"
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "PancreaticCancerBiomarkerProject/1.0"}
    )

    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            data = json.loads(response.read().decode("utf-8"))

        symbols[gene_id] = {
            "gene_symbol": data.get("display_name", "Not available"),
            "gene_description": data.get("description", "Not available")
        }
        print(gene_id, "->", symbols[gene_id]["gene_symbol"])

    except Exception as error:
        symbols[gene_id] = {
            "gene_symbol": "Lookup unavailable",
            "gene_description": str(error)
        }
        print("Lookup failed for", gene_id, "-", error)

    time.sleep(0.1)

df["ensembl_id"] = df["rna_id"].astype(str).str.split(".").str[0]
df["gene_symbol"] = df["ensembl_id"].map(
    lambda gene_id: symbols.get(gene_id, {}).get(
        "gene_symbol", "Non-Ensembl RNA ID"
    )
)
df["gene_description"] = df["ensembl_id"].map(
    lambda gene_id: symbols.get(gene_id, {}).get(
        "gene_description", "Not available"
    )
)

df.to_csv(OUTPUT_FILE, index=False)

print("\nGene annotation completed.")
print("Saved:", OUTPUT_FILE)
print(df[["biomarker_rank", "rna_id", "gene_symbol", "gene_description"]].to_string(index=False))
