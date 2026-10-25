
import os
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(BASE, "results")

INPUT = os.path.join(RESULTS, "FINAL_TOP20_WITH_GENE_NAMES.csv")
OUTPUT = os.path.join(RESULTS, "FINAL_BIOMARKER_VALIDATION_TABLE.csv")

df = pd.read_csv(INPUT)

# Supplement the two IDs whose Ensembl lookups previously timed out.
recovered = {
    "ENSG00000204839": ("MROH6", "Maestro heat-like repeat family member 6"),
    "ENSG00000122711": ("SPINK4", "Serine peptidase inhibitor Kazal type 4"),
}

for idx, row in df.iterrows():
    gene_id = str(row.get("rna_id", "")).split(".")[0]

    if gene_id in recovered:
        df.at[idx, "gene_symbol"] = recovered[gene_id][0]
        df.at[idx, "gene_description"] = recovered[gene_id][1]

# Literature notes from the preliminary review.
evidence = {
    "MROH6": (
        "Human Protein Atlas lists a prognostic association in pancreatic "
        "adenocarcinoma; independent validation is still needed.",
        "https://www.proteinatlas.org/ENSG00000204839-MROH6"
    ),
    "HOXA5": (
        "Published review discusses pancreatic precursor-lesion biology; "
        "Human Protein Atlas validation reports no prognostic association.",
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC6562939/"
    ),
    "SPINK4": (
        "Detected in pancreatic adenocarcinoma datasets; this does not by "
        "itself establish a diagnostic biomarker.",
        "https://www.proteinatlas.org/ENSG00000122711-SPINK4/cancer"
    ),
}

def evidence_note(symbol):
    if symbol in evidence:
        return evidence[symbol][0]
    return "Specific pancreatic-cancer literature review still required."

def evidence_link(symbol):
    if symbol in evidence:
        return evidence[symbol][1]
    return ""

df["literature_evidence"] = df["gene_symbol"].map(evidence_note)
df["literature_source"] = df["gene_symbol"].map(evidence_link)

df["fdr"] = pd.to_numeric(df["fdr"], errors="coerce")
df["statistical_status"] = df["fdr"].apply(
    lambda value: (
        "FDR < 0.05"
        if pd.notna(value) and value < 0.05
        else "Not significant at FDR < 0.05"
        if pd.notna(value)
        else "FDR unavailable"
    )
)

df["clinical_validation_status"] = (
    "Computational candidate; clinical validation not established"
)

df.to_csv(OUTPUT, index=False)

print("Validation table saved:")
print(OUTPUT)
print("\nCandidate count:", len(df))
print(
    df[[
        "biomarker_rank",
        "rna_id",
        "gene_symbol",
        "fdr",
        "statistical_status",
        "literature_evidence"
    ]].to_string(index=False)
)
