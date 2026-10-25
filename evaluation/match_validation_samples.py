import pandas as pd
from pathlib import Path

metadata_path = Path("data/validation/GSE183795_sample_metadata.csv")
matrix_path = Path("data/validation/GSE183795_normalized_matrix.txt.gz")
output_path = Path("data/validation/GSE183795_sample_mapping.csv")

metadata = pd.read_csv(metadata_path)
matrix_columns = pd.read_csv(
    matrix_path, sep="\t", compression="gzip", nrows=0
).columns.tolist()

# Extract sample title and tissue type from GEO metadata
metadata["tissue_type"] = metadata["characteristics"].str.extract(
    r"tissue:\s*([^;]+)", expand=False
)

# CEL filenames in the matrix correspond to the GEO sample titles
metadata["matrix_sample"] = (
    metadata["title"].astype(str) + ".CEL"
)

matrix_samples = set(matrix_columns[4:])
metadata["matched"] = metadata["matrix_sample"].isin(matrix_samples)

metadata.to_csv(output_path, index=False)

print("Metadata samples:", len(metadata))
print("Expression matrix sample columns:", len(matrix_samples))
print("Matched samples:", int(metadata["matched"].sum()))
print("Unmatched metadata samples:", int((~metadata["matched"]).sum()))

print("\nMatched tissue groups:")
print(
    metadata.loc[metadata["matched"], "tissue_type"]
    .value_counts(dropna=False)
)

print("\nExamples of unmatched titles:")
print(
    metadata.loc[~metadata["matched"], "title"]
    .head(10).to_string(index=False)
)

print("\nMapping saved to:", output_path)
