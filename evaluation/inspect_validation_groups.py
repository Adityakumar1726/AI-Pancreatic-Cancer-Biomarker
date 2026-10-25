import pandas as pd
from pathlib import Path

metadata_path = Path("data/validation/GSE183795_sample_metadata.csv")
matrix_path = Path("data/validation/GSE183795_normalized_matrix.txt.gz")

metadata = pd.read_csv(metadata_path)
print("Metadata columns:", metadata.columns.tolist())
print("Total metadata samples:", len(metadata))

metadata["tissue_type"] = metadata["characteristics"].str.extract(
    r"tissue:\s*([^;]+)", expand=False
)

print("\nTissue group counts:")
print(metadata["tissue_type"].value_counts(dropna=False))

print("\nChecking expression matrix...")
matrix = pd.read_csv(matrix_path, sep="\t", compression="gzip", nrows=5)
print("First four matrix columns:", matrix.columns[:4].tolist())
print("First five gene rows:")
print(matrix.iloc[:, :6].to_string(index=False))

print(
    "\nNext: match matrix sample columns to official GEO sample titles "
    "before running statistical tests."
)
