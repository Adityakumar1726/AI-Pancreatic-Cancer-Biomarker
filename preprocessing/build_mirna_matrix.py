import os
import pandas as pd

PROJECT_DIR = r"C:\Users\adity\OneDrive\Desktop\AI-Pancreatic-Cancer-Biomarker"

MIRNA_DIR = os.path.join(PROJECT_DIR, "data", "raw", "mirna_seq")
SAMPLE_FILE = os.path.join(PROJECT_DIR, "data", "raw", "mirna_sample_sheet.tsv")
PROCESSED_DIR = os.path.join(PROJECT_DIR, "data", "processed")

os.makedirs(PROCESSED_DIR, exist_ok=True)

print("Reading miRNA sample information...")

samples = pd.read_csv(SAMPLE_FILE, sep="\t")

file_to_sample = dict(
    zip(samples["File Name"], samples["Sample ID"])
)

print("Samples found:", len(samples))

mirna_files = []

for root, dirs, files in os.walk(MIRNA_DIR):
    for file in files:
        if file.endswith(".mirbase21.mirnas.quantification.txt"):
            mirna_files.append(os.path.join(root, file))

print("miRNA files found:", len(mirna_files))

if len(mirna_files) == 0:
    raise FileNotFoundError("No miRNA files found.")

mirna_data = {}

for i, filepath in enumerate(mirna_files, start=1):

    filename = os.path.basename(filepath)

    if filename not in file_to_sample:
        continue

    sample_id = file_to_sample[filename]

    print(f"Processing {i}/{len(mirna_files)}: {sample_id}")

    df = pd.read_csv(
        filepath,
        sep="\t"
    )

    df["read_count"] = pd.to_numeric(
        df["read_count"],
        errors="coerce"
    ).fillna(0)

    df = df[["miRNA_ID", "read_count"]]

    df = df.drop_duplicates(
        subset="miRNA_ID"
    )

    mirna_data[sample_id] = df.set_index(
        "miRNA_ID"
    )["read_count"]

mirna_matrix = pd.DataFrame(
    mirna_data
).fillna(0)

mirna_matrix = mirna_matrix.sort_index()

output = os.path.join(
    PROCESSED_DIR,
    "mirna_expression_counts.csv"
)

mirna_matrix.to_csv(output)

print("\n====================================")
print("miRNA PREPROCESSING COMPLETED")
print("====================================")

print("miRNA saved to:")
print(output)

print("\nFinal dimensions:")
print("miRNA:", mirna_matrix.shape)