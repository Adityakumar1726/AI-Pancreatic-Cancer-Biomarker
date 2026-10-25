import gzip
import urllib.request
import csv
from pathlib import Path

url = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE183nnn/GSE183795/soft/GSE183795_family.soft.gz"
out_dir = Path("data/validation")
out_dir.mkdir(parents=True, exist_ok=True)
download_path = out_dir / "GSE183795_family.soft.gz"
output_path = out_dir / "GSE183795_sample_metadata.csv"

print("Downloading official GEO metadata...")
try:
    urllib.request.urlretrieve(url, download_path)
except Exception as e:
    print("Download failed:", e)
    raise SystemExit(1)

records = []
current = None

with gzip.open(download_path, "rt", encoding="utf-8", errors="replace") as f:
    for line in f:
        line = line.rstrip("\n")
        if line.startswith("^SAMPLE = "):
            if current:
                records.append(current)
            current = {
                "geo_accession": line.split("=", 1)[1].strip(),
                "title": "",
                "characteristics": []
            }
        elif current and line.startswith("!Sample_title = "):
            current["title"] = line.split("=", 1)[1].strip().strip('"')
        elif current and line.startswith("!Sample_characteristics_ch1 = "):
            current["characteristics"].append(
                line.split("=", 1)[1].strip().strip('"')
            )

if current:
    records.append(current)

with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(
        f, fieldnames=["geo_accession", "title", "characteristics"]
    )
    writer.writeheader()
    for r in records:
        r["characteristics"] = "; ".join(r["characteristics"])
        writer.writerow(r)

print("GEO samples downloaded:", len(records))
print("Saved to:", output_path)
print("First 10 sample labels:")
for r in records[:10]:
    print(r["geo_accession"], "|", r["title"], "|", r["characteristics"])
