
import gzip
import pandas as pd

path = r"data\validation\GSE183795_normalized_matrix.txt.gz"

with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
    header = f.readline().rstrip("\n").split("\t")

samples = header[4:]

print("Total columns:", len(header))
print("Sample columns:", len(samples))
print("\nFirst 30 sample names:")

for sample in samples[:30]:
    print(sample)

print("\nSample-name patterns:")
print("Names containing T:", sum("_T" in s or "-T" in s for s in samples))
print("Names containing N:", sum("_N" in s or "-N" in s for s in samples))
print("Names containing R:", sum("_R" in s or "-R" in s for s in samples))
