
import os
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_FILE = os.path.join(
    BASE,
    "results",
    "INDEPENDENT_BIOMARKER_VALIDATION_CORRECTED.csv"
)

OUTPUT_FILE = os.path.join(
    BASE,
    "results",
    "INDEPENDENT_VALIDATION_SUMMARY.csv"
)

FDR_THRESHOLD = 0.05

# Load independent validation results
df = pd.read_csv(INPUT_FILE)

# Keep candidates with measured expression and valid FDR
tested = df[
    (df["status"] == "Tested") &
    df["FDR"].notna() &
    df["mean_difference"].notna()
].copy()

# Determine direction of expression change
tested["Direction"] = tested["mean_difference"].apply(
    lambda x: (
        "Higher in tumors" if x > 0
        else "Lower in tumors" if x < 0
        else "No mean difference"
    )
)

# Mark statistical significance
tested["Statistical_Result"] = tested["FDR"].apply(
    lambda x: (
        "Significant (FDR < 0.05)"
        if x < FDR_THRESHOLD
        else "Not significant (FDR >= 0.05)"
    )
)

# Sort by FDR
tested = tested.sort_values("FDR").reset_index(drop=True)

# Add a simple rank
tested.insert(0, "Rank_by_FDR", range(1, len(tested) + 1))

# Save all tested candidates
tested.to_csv(OUTPUT_FILE, index=False)

# Display summary
significant = tested[tested["FDR"] < FDR_THRESHOLD]

print("=" * 55)
print("INDEPENDENT BIOMARKER VALIDATION SUMMARY")
print("=" * 55)

print(f"Tested candidates: {len(tested)}")
print(f"Significant candidates (FDR < {FDR_THRESHOLD}): {len(significant)}")

if not significant.empty:
    print("\nSignificant candidates:")
    print(
        significant[
            ["gene_symbol", "mean_difference", "p_value", "FDR", "Direction"]
        ].to_string(index=False)
    )
else:
    print("\nNo candidates met the significance threshold.")

print("\nInterpretation:")
print("Positive mean difference = higher expression in tumors.")
print("Negative mean difference = lower expression in tumors.")
print("Statistical significance does not establish clinical validity.")

print("\nSaved summary to:")
print(OUTPUT_FILE)
