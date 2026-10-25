
import json
import urllib.request

ids = [
    "ENSG00000204839",
    "ENSG00000122711"
]

for gene_id in ids:
    url = (
        "https://rest.ensembl.org/lookup/id/"
        + gene_id
        + "?content-type=application/json"
    )

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "BiomarkerResearch/1.0"}
    )

    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            data = json.loads(response.read().decode("utf-8"))

        print("\nEnsembl ID:", gene_id)
        print("Gene symbol:", data.get("display_name", "Not available"))
        print("Description:", data.get("description", "Not available"))

    except Exception as error:
        print("\nEnsembl ID:", gene_id)
        print("Lookup failed:", error)
