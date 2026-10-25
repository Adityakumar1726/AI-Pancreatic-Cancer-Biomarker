import pandas as pd
import os

BASE = r"C:\Users\adity\OneDrive\Desktop\AI-Pancreatic-Cancer-Biomarker"

# Files
mrna_file = os.path.join(BASE, "data", "processed", "mrna_expression_counts.csv")
lncrna_file = os.path.join(BASE, "data", "processed", "lncrna_expression_counts.csv")
mirna_file = os.path.join(BASE, "data", "processed", "mirna_expression_counts.csv")
label_file = os.path.join(BASE, "data", "processed", "de_groups.tsv")

# Load data
mrna = pd.read_csv(mrna_file)
lncrna = pd.read_csv(lncrna_file)
mirna = pd.read_csv(mirna_file)
labels = pd.read_csv(label_file, sep="\t")

print("mRNA:", mrna.shape)
print("lncRNA:", lncrna.shape)
print("miRNA:", mirna.shape)
print("Labels:", labels.shape)

print("\nLabel columns:")
print(labels.columns.tolist())

# Detect sample and group columns
group_col = None
sample_col = None

for col in labels.columns:
    vals = labels[col].astype(str).str.lower()
    if vals.isin(["tumor", "normal", "normal tissue", "tumour"]).sum() > 0:
        group_col = col
        break

for col in labels.columns:
    if col != group_col:
        sample_col = col
        break

print("\nDetected sample column:", sample_col)
print("Detected group column:", group_col)

if group_col is None:
    raise ValueError("Could not detect Tumor/Normal column.")

# Selected biomarkers
mrna_candidates = [
    "ENSG00000090402",
    "ENSG00000043355",
    "ENSG00000167613",
    "ENSG00000163283",
    "ENSG00000077152"
]

lncrna_candidates = [
    "ENSG00000242136",
    "ENSG00000260484",
    "ENSG00000249395",
    "ENSG00000274414",
    "ENSG00000255197"
]

mirna_candidates = [
    "hsa-mir-342",
    "hsa-mir-223",
    "hsa-mir-5586",
    "hsa-mir-196a-1"
]

# Find ID column
def find_id_column(df):
    for c in df.columns:
        if str(c).lower() in ["rna_id", "gene_id", "ensembl_id", "id"]:
            return c
    return df.columns[0]

mrna_id = find_id_column(mrna)
lncrna_id = find_id_column(lncrna)
mirna_id = find_id_column(mirna)

print("\nID columns:")
print("mRNA:", mrna_id)
print("lncRNA:", lncrna_id)
print("miRNA:", mirna_id)

# Keep only selected biomarkers
mrna_sel = mrna[mrna[mrna_id].astype(str).isin(mrna_candidates)].copy()
lncrna_sel = lncrna[lncrna[lncrna_id].astype(str).isin(lncrna_candidates)].copy()
mirna_sel = mirna[mirna[mirna_id].astype(str).isin(mirna_candidates)].copy()

print("\nSelected rows:")
print("mRNA:", len(mrna_sel))
print("lncRNA:", len(lncrna_sel))
print("miRNA:", len(mirna_sel))

# Convert each matrix to sample x biomarker
def transpose_matrix(df, id_col, prefix):
    x = df.set_index(id_col).T
    x.index.name = "sample_id"
    x = x.add_prefix(prefix)
    return x

X_mrna = transpose_matrix(mrna_sel, mrna_id, "mRNA_")
X_lncrna = transpose_matrix(lncrna_sel, lncrna_id, "lncRNA_")
X_mirna = transpose_matrix(mirna_sel, mirna_id, "miRNA_")

# Combine
X = X_mrna.join(X_lncrna, how="inner")
X = X.join(X_mirna, how="inner")

# Labels
y = labels[[sample_col, group_col]].copy()
y.columns = ["sample_id", "label"]
y["sample_id"] = y["sample_id"].astype(str)

# Match sample IDs
X.index = X.index.astype(str)

dataset = X.join(y.set_index("sample_id"), how="inner")

# Convert labels
dataset["label"] = (
    dataset["label"]
    .astype(str)
    .str.lower()
    .map({
        "normal": 0,
        "normal tissue": 0,
        "tumor": 1,
        "tumour": 1
    })
)

dataset = dataset.dropna(subset=["label"])
dataset["label"] = dataset["label"].astype(int)

# Save
out = os.path.join(BASE, "results", "ml_biomarker_dataset.csv")
dataset.to_csv(out)

print("\n==============================")
print("ML DATASET CREATED")
print("==============================")
print("Shape:", dataset.shape)
print("\nClass counts:")
print(dataset["label"].value_counts())
print("\nFeatures:")
print(dataset.columns.tolist())
print("\nSaved to:")
print(out)