import os
import pandas as pd

# Project paths
PROJECT_DIR = r"C:\Users\adity\OneDrive\Desktop\AI-Pancreatic-Cancer-Biomarker"

RNA_DIR = os.path.join(PROJECT_DIR, "data", "raw", "rna_seq")
SAMPLE_FILE = os.path.join(PROJECT_DIR, "data", "raw", "de_rna_samples.tsv")
PROCESSED_DIR = os.path.join(PROJECT_DIR, "data", "processed")

os.makedirs(PROCESSED_DIR, exist_ok=True)

print("Reading sample information...")

samples = pd.read_csv(SAMPLE_FILE, sep="\t")

# File Name -> Sample ID mapping
file_to_sample = dict(zip(samples["File Name"], samples["Sample ID"]))

print("Samples found:", len(samples))

# Find RNA files
rna_files = []

for root, dirs, files in os.walk(RNA_DIR):
    for file in files:
        if file.endswith(".rna_seq.augmented_star_gene_counts.tsv"):
            rna_files.append(os.path.join(root, file))

print("RNA files found:", len(rna_files))

if len(rna_files) == 0:
    raise FileNotFoundError("No RNA expression files found.")

# Store mRNA and lncRNA data
mrna_data = {}
lncrna_data = {}

for i, filepath in enumerate(rna_files, start=1):

    filename = os.path.basename(filepath)
    file_name = filename

    if file_name not in file_to_sample:
        continue

    sample_id = file_to_sample[file_name]

    print(f"Processing {i}/{len(rna_files)}: {sample_id}")

    df = pd.read_csv(
        filepath,
        sep="\t",
        comment="#"
    )

    # Remove rows without gene IDs
    df = df[df["gene_id"].notna()]

    # Remove GDC summary rows
    df = df[~df["gene_id"].astype(str).str.startswith("N_")]

    # Remove Ensembl version number
    df["gene_id"] = df["gene_id"].astype(str).str.split(".").str[0]

    # Keep unstranded read counts
    df["unstranded"] = pd.to_numeric(
        df["unstranded"],
        errors="coerce"
    ).fillna(0)

    # mRNA = protein coding
    mrna = df[df["gene_type"] == "protein_coding"]

    # lncRNA-related GENCODE categories
    lncrna_types = [
        "lncRNA",
        "antisense",
        "sense_intronic",
        "sense_overlapping",
        "processed_transcript",
        "3prime_overlapping_ncRNA",
        "bidirectional_promoter_lncRNA",
        "macro_lncRNA"
    ]

    lncrna = df[df["gene_type"].isin(lncrna_types)]

    mrna_data[sample_id] = (
        mrna.set_index("gene_id")["unstranded"]
    )

    lncrna_data[sample_id] = (
        lncrna.set_index("gene_id")["unstranded"]
    )


# Convert to matrices
mrna_matrix = pd.DataFrame(mrna_data).fillna(0)
lncrna_matrix = pd.DataFrame(lncrna_data).fillna(0)

# Sort genes
mrna_matrix = mrna_matrix.sort_index()
lncrna_matrix = lncrna_matrix.sort_index()

# Save
mrna_output = os.path.join(
    PROCESSED_DIR,
    "mrna_expression_counts.csv"
)

lncrna_output = os.path.join(
    PROCESSED_DIR,
    "lncrna_expression_counts.csv"
)

mrna_matrix.to_csv(mrna_output)
lncrna_matrix.to_csv(lncrna_output)

print("\n====================================")
print("PREPROCESSING COMPLETED")
print("====================================")

print("mRNA saved to:")
print(mrna_output)

print("\nlncRNA saved to:")
print(lncrna_output)

print("\nFinal dimensions:")
print("mRNA:", mrna_matrix.shape)
print("lncRNA:", lncrna_matrix.shape)