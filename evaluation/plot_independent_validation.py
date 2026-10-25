import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

matrix_path = "data/validation/GSE183795_normalized_matrix.txt.gz"
metadata_path = "data/validation/GSE183795_sample_mapping.csv"
output_dir = Path("results")
output_dir.mkdir(exist_ok=True)

metadata = pd.read_csv(metadata_path)
metadata["tissue_type"] = metadata["characteristics"].str.extract(
    r"tissue:\s*([^;]+)", expand=False
)

tumor = metadata.loc[
    metadata["tissue_type"] == "Tumor", "title"
].tolist()
adjacent = metadata.loc[
    metadata["tissue_type"] == "adjacent non-tumor", "title"
].tolist()

matrix = pd.read_csv(matrix_path, sep="\t", compression="gzip")
matrix["ID"] = matrix["ID"].astype(str).str.upper()

genes = ["MROH6", "OSCAR"]
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

for ax, gene in zip(axes, genes):
    rows = matrix[matrix["ID"] == gene]
    if rows.empty:
        ax.set_title(f"{gene}: not found")
        continue

    row = rows.iloc[0]
    tumor_values = pd.to_numeric(
        row[[s + ".CEL" for s in tumor]], errors="coerce"
    ).dropna()
    adjacent_values = pd.to_numeric(
        row[[s + ".CEL" for s in adjacent]], errors="coerce"
    ).dropna()

    ax.boxplot(
        [adjacent_values, tumor_values],
        tick_labels=["Adjacent\nnon-tumor", "Tumor"]
    )
    ax.set_title(gene)
    ax.set_ylabel("Normalized expression")
    ax.grid(axis="y", alpha=0.25)

fig.suptitle("Independent Expression Validation — GSE183795")
fig.tight_layout()

path = output_dir / "INDEPENDENT_VALIDATION_MROH6_OSCAR.png"
fig.savefig(path, dpi=300, bbox_inches="tight")
plt.close(fig)
print("Plot saved to:", path)
