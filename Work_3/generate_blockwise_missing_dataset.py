"""
Synthetic multi-source block-wise missing dataset generator.

Clean focused version:
- No scaled files.
- No has_* columns.
- No profile column in the CSV files.
- No repeated disease_status in source-level files.
- Source-level files contain only:
    patient_id + source-specific features
- labels.csv contains:
    patient_id + disease_status
- Combined files contain:
    patient_id + disease_status + all source features

Generated files:
1. labels.csv
2. complete_multisource_dataset.csv
3. blockwise_missing_multisource_dataset.csv
4. source1_clinical_complete.csv
5. source1_clinical_blockwise_missing.csv
6. source2_imaging_complete.csv
7. source2_imaging_blockwise_missing.csv
8. source3_proteomics_complete.csv
9. source3_proteomics_blockwise_missing.csv
10. notes.md
"""

from pathlib import Path
import numpy as np
import pandas as pd

# ---------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------

RANDOM_SEED = 42
N_SAMPLES = 500

OUTPUT_DIR = Path("synthetic_bwm_data")
OUTPUT_DIR.mkdir(exist_ok=True)

rng = np.random.default_rng(RANDOM_SEED)


# ---------------------------------------------------------
# 2. Helper functions
# ---------------------------------------------------------

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def apply_internal_missingness(df, feature_cols, rate=0.02):
    """
    Adds a very small amount of scattered missingness inside observed source blocks.

    The main missingness is still block-wise.
    Set INTERNAL_MISSING_RATE = 0.00 if you want only source-level block missingness.
    """
    df_missing = df.copy()
    mask = rng.random((df_missing.shape[0], len(feature_cols))) < rate
    df_missing.loc[:, feature_cols] = df_missing.loc[:, feature_cols].mask(mask)
    return df_missing


# ---------------------------------------------------------
# 3. Generate shared sample-level latent variables
# ---------------------------------------------------------

patient_id = np.arange(1, N_SAMPLES + 1)

latent_risk = rng.normal(0.0, 1.0, N_SAMPLES)
latent_inflammation = 0.65 * latent_risk + rng.normal(0, 0.75, N_SAMPLES)
latent_metabolic = 0.55 * latent_risk + rng.normal(0, 0.80, N_SAMPLES)
latent_neuro = 0.70 * latent_risk + rng.normal(0, 0.70, N_SAMPLES)


# ---------------------------------------------------------
# 4. Generate binary label
# ---------------------------------------------------------

logit = (
    -0.15
    + 1.10 * latent_risk
    + 0.65 * latent_inflammation
    + 0.45 * latent_metabolic
    + rng.normal(0, 0.55, N_SAMPLES)
)

prob_high_risk = sigmoid(logit)
disease_status = rng.binomial(n=1, p=prob_high_risk, size=N_SAMPLES)

labels = pd.DataFrame({
    "patient_id": patient_id,
    "disease_status": disease_status
})


# ---------------------------------------------------------
# 5. Source 1: clinical/laboratory features
# ---------------------------------------------------------

source1_clinical = pd.DataFrame({
    "patient_id": patient_id,
    "age_years": np.clip(rng.normal(58 + 7.5 * latent_risk, 8.0, N_SAMPLES), 35, 90),
    "bmi": np.clip(rng.normal(27 + 1.8 * latent_metabolic, 3.5, N_SAMPLES), 17, 45),
    "systolic_bp": np.clip(rng.normal(122 + 8.0 * latent_metabolic, 12.0, N_SAMPLES), 85, 190),
    "diastolic_bp": np.clip(rng.normal(78 + 4.5 * latent_metabolic, 8.0, N_SAMPLES), 50, 120),
    "fasting_glucose": np.clip(rng.normal(95 + 9.0 * latent_metabolic, 12.0, N_SAMPLES), 60, 220),
    "hba1c": np.clip(rng.normal(5.4 + 0.35 * latent_metabolic, 0.45, N_SAMPLES), 4.2, 9.5),
    "crp": np.clip(rng.lognormal(mean=0.15 + 0.30 * latent_inflammation, sigma=0.45), 0.1, 20),
    "ldl_cholesterol": np.clip(rng.normal(115 + 9.0 * latent_metabolic, 25.0, N_SAMPLES), 45, 230),
    "hdl_cholesterol": np.clip(rng.normal(52 - 3.0 * latent_metabolic, 12.0, N_SAMPLES), 20, 100),
    "triglycerides": np.clip(rng.normal(135 + 18.0 * latent_metabolic, 35.0, N_SAMPLES), 45, 350),
})


# ---------------------------------------------------------
# 6. Source 2: imaging-derived features
# ---------------------------------------------------------

source2_imaging = pd.DataFrame({
    "patient_id": patient_id,
    "hippocampal_volume": np.clip(rng.normal(6.8 - 0.45 * latent_neuro, 0.55, N_SAMPLES), 3.8, 8.5),
    "cortical_thickness": np.clip(rng.normal(2.65 - 0.12 * latent_neuro, 0.18, N_SAMPLES), 1.8, 3.2),
    "white_matter_lesion_volume": np.clip(rng.lognormal(mean=0.25 + 0.42 * latent_neuro, sigma=0.55), 0.05, 25),
    "ventricle_volume": np.clip(rng.normal(32 + 4.5 * latent_neuro, 7.0, N_SAMPLES), 12, 70),
    "bbb_permeability_index": np.clip(rng.normal(0.85 + 0.22 * latent_inflammation, 0.18, N_SAMPLES), 0.25, 1.8),
    "cerebral_blood_flow": np.clip(rng.normal(52 - 3.5 * latent_neuro, 6.0, N_SAMPLES), 25, 75),
    "gray_matter_density": np.clip(rng.normal(0.72 - 0.05 * latent_neuro, 0.06, N_SAMPLES), 0.45, 0.90),
    "microbleed_count": np.clip(rng.poisson(lam=np.exp(-0.45 + 0.30 * latent_neuro)), 0, 15),
})


# ---------------------------------------------------------
# 7. Source 3: proteomic/molecular features
# ---------------------------------------------------------

proteomic_features = {}

for j in range(1, 26):
    if j <= 6:
        values = rng.normal(0.8 * latent_inflammation + 0.35 * latent_risk, 0.75, N_SAMPLES)
    elif j <= 12:
        values = rng.normal(0.45 * latent_metabolic + 0.20 * latent_risk, 0.85, N_SAMPLES)
    elif j <= 18:
        values = rng.normal(-0.35 * latent_risk, 0.90, N_SAMPLES)
    else:
        values = rng.normal(0.0, 1.0, N_SAMPLES)

    proteomic_features[f"protein_marker_{j:02d}"] = values

source3_proteomics = pd.DataFrame({
    "patient_id": patient_id,
    **proteomic_features
})


# ---------------------------------------------------------
# 8. Define feature columns
# ---------------------------------------------------------

source1_cols = [c for c in source1_clinical.columns if c != "patient_id"]
source2_cols = [c for c in source2_imaging.columns if c != "patient_id"]
source3_cols = [c for c in source3_proteomics.columns if c != "patient_id"]

all_feature_cols = source1_cols + source2_cols + source3_cols


# ---------------------------------------------------------
# 9. Build complete combined multi-source dataset
# ---------------------------------------------------------

complete_multisource = (
    labels
    .merge(source1_clinical, on="patient_id")
    .merge(source2_imaging, on="patient_id")
    .merge(source3_proteomics, on="patient_id")
)


# ---------------------------------------------------------
# 10. Create internal source availability patterns
# ---------------------------------------------------------

patterns = np.array([
    [1, 1, 1],
    [1, 1, 0],
    [1, 0, 1],
    [0, 1, 1],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
])

pattern_labels = np.array(["111", "110", "101", "011", "100", "010", "001"])

pattern_probs = np.array([
    0.42,
    0.16,
    0.12,
    0.10,
    0.08,
    0.08,
    0.04,
])

assigned_pattern_idx = rng.choice(
    len(patterns),
    size=N_SAMPLES,
    p=pattern_probs
)

assigned_patterns = patterns[assigned_pattern_idx]
assigned_profile_labels = pattern_labels[assigned_pattern_idx]


# ---------------------------------------------------------
# 11. Apply block-wise missingness
# ---------------------------------------------------------

source1_clinical_blockwise = source1_clinical.copy()
source2_imaging_blockwise = source2_imaging.copy()
source3_proteomics_blockwise = source3_proteomics.copy()

for row_idx in range(N_SAMPLES):
    source1_available = assigned_patterns[row_idx, 0]
    source2_available = assigned_patterns[row_idx, 1]
    source3_available = assigned_patterns[row_idx, 2]

    if source1_available == 0:
        source1_clinical_blockwise.loc[row_idx, source1_cols] = np.nan

    if source2_available == 0:
        source2_imaging_blockwise.loc[row_idx, source2_cols] = np.nan

    if source3_available == 0:
        source3_proteomics_blockwise.loc[row_idx, source3_cols] = np.nan


# Optional small scattered missingness inside available blocks
INTERNAL_MISSING_RATE = 0.02

source1_clinical_blockwise = apply_internal_missingness(
    source1_clinical_blockwise,
    source1_cols,
    rate=INTERNAL_MISSING_RATE
)

source2_imaging_blockwise = apply_internal_missingness(
    source2_imaging_blockwise,
    source2_cols,
    rate=INTERNAL_MISSING_RATE
)

source3_proteomics_blockwise = apply_internal_missingness(
    source3_proteomics_blockwise,
    source3_cols,
    rate=INTERNAL_MISSING_RATE
)


# ---------------------------------------------------------
# 12. Build combined block-wise missing multi-source dataset
# ---------------------------------------------------------

blockwise_missing_multisource = (
    labels
    .merge(source1_clinical_blockwise, on="patient_id")
    .merge(source2_imaging_blockwise, on="patient_id")
    .merge(source3_proteomics_blockwise, on="patient_id")
)


# ---------------------------------------------------------
# 13. Save only the required CSV files
# ---------------------------------------------------------

required_files = {
    "labels.csv": labels,
    "complete_multisource_dataset.csv": complete_multisource,
    "blockwise_missing_multisource_dataset.csv": blockwise_missing_multisource,
    "source1_clinical_complete.csv": source1_clinical,
    "source1_clinical_blockwise_missing.csv": source1_clinical_blockwise,
    "source2_imaging_complete.csv": source2_imaging,
    "source2_imaging_blockwise_missing.csv": source2_imaging_blockwise,
    "source3_proteomics_complete.csv": source3_proteomics,
    "source3_proteomics_blockwise_missing.csv": source3_proteomics_blockwise,
}

for filename, df in required_files.items():
    df.to_csv(OUTPUT_DIR / filename, index=False)


# ---------------------------------------------------------
# 14. Create one notes.md file
# ---------------------------------------------------------

profile_counts = pd.Series(assigned_profile_labels).value_counts().sort_index()

profile_meanings = {
    "111": "source 1 clinical + source 2 imaging + source 3 proteomics available",
    "110": "source 1 clinical + source 2 imaging available; source 3 proteomics blank",
    "101": "source 1 clinical + source 3 proteomics available; source 2 imaging blank",
    "011": "source 2 imaging + source 3 proteomics available; source 1 clinical blank",
    "100": "source 1 clinical only; source 2 and source 3 blank",
    "010": "source 2 imaging only; source 1 and source 3 blank",
    "001": "source 3 proteomics only; source 1 and source 2 blank",
}

profile_lines = [
    "| Internal pattern | Meaning | Count | Percentage |",
    "|---|---|---:|---:|",
]

for profile, count in profile_counts.items():
    profile_lines.append(
        f"| `{profile}` | {profile_meanings[profile]} | {int(count)} | {100 * count / N_SAMPLES:.2f}% |"
    )

source1_column_lines = [
    "| Column | Meaning |",
    "|---|---|",
    "| `patient_id` | Unique sample identifier used to link this source with labels and other sources. |",
]
for col in source1_cols:
    source1_column_lines.append(f"| `{col}` | Clinical/laboratory feature. |")

source2_column_lines = [
    "| Column | Meaning |",
    "|---|---|",
    "| `patient_id` | Unique sample identifier used to link this source with labels and other sources. |",
]
for col in source2_cols:
    source2_column_lines.append(f"| `{col}` | Imaging-derived feature. |")

source3_column_lines = [
    "| Column | Meaning |",
    "|---|---|",
    "| `patient_id` | Unique sample identifier used to link this source with labels and other sources. |",
]
for col in source3_cols:
    source3_column_lines.append(f"| `{col}` | Proteomic/molecular feature. |")

notes = f"""# Synthetic Multi-Source Block-Wise Missing Dataset

## Purpose

This folder contains a focused synthetic dataset for testing block-wise missing data algorithms.

There are three separate data sources:

1. **Source 1:** clinical/laboratory features
2. **Source 2:** imaging-derived features
3. **Source 3:** proteomic/molecular features

Each source has its own unique feature columns. The only repeated column across source files is `patient_id`, because it is needed to match the same sample across the different sources.

The class label is stored separately in:

`labels.csv`

## Label

| Column | Meaning |
|---|---|
| `patient_id` | Unique sample identifier. |
| `disease_status` | Binary label. `0` = low-risk/control-like sample; `1` = high-risk/disease-like sample. |

## Dataset size

| Item | Value |
|---|---:|
| Number of samples | {N_SAMPLES} |
| Source 1 clinical features | {len(source1_cols)} |
| Source 2 imaging features | {len(source2_cols)} |
| Source 3 proteomic features | {len(source3_cols)} |
| Total features across all sources | {len(all_feature_cols)} |
| Class 0 samples | {int((disease_status == 0).sum())} |
| Class 1 samples | {int((disease_status == 1).sum())} |
| Small internal missingness inside available blocks | {INTERNAL_MISSING_RATE * 100:.1f}% |

## Files generated

| File | Meaning |
|---|---|
| `labels.csv` | Contains only `patient_id` and `disease_status`. |
| `complete_multisource_dataset.csv` | Combined complete dataset: `patient_id`, `disease_status`, and all source features. |
| `blockwise_missing_multisource_dataset.csv` | Combined block-wise missing dataset: `patient_id`, `disease_status`, and all source features with missing source blocks left blank. |
| `source1_clinical_complete.csv` | Complete source 1 file: `patient_id` and clinical/laboratory features only. |
| `source1_clinical_blockwise_missing.csv` | Source 1 file with clinical feature cells left blank where source 1 is missing. |
| `source2_imaging_complete.csv` | Complete source 2 file: `patient_id` and imaging-derived features only. |
| `source2_imaging_blockwise_missing.csv` | Source 2 file with imaging feature cells left blank where source 2 is missing. |
| `source3_proteomics_complete.csv` | Complete source 3 file: `patient_id` and proteomic/molecular features only. |
| `source3_proteomics_blockwise_missing.csv` | Source 3 file with proteomic feature cells left blank where source 3 is missing. |

## Important structure

The source files do not contain `disease_status`. This avoids repeating the label in every source file.

The source files also do not contain `has_clinical`, `has_imaging`, `has_proteomics`, or `profile`.

If a source is missing for a sample, the feature cells are simply blank.

## Internal block-wise missingness pattern

The script uses these patterns internally to create missing source blocks. These pattern codes are not saved as columns in the CSV files.

The order is:

`source 1 clinical, source 2 imaging, source 3 proteomics`

{chr(10).join(profile_lines)}

## Source 1 columns: clinical/laboratory

{chr(10).join(source1_column_lines)}

## Source 2 columns: imaging-derived

{chr(10).join(source2_column_lines)}

## Source 3 columns: proteomic/molecular

{chr(10).join(source3_column_lines)}

## Recommended files for experiments

For complete-data baseline experiments, use:

`complete_multisource_dataset.csv`

For block-wise missing data experiments, use:

`blockwise_missing_multisource_dataset.csv`

For source-level experiments, use the individual source files together with `labels.csv`.
"""

(OUTPUT_DIR / "notes.md").write_text(notes, encoding="utf-8")


# ---------------------------------------------------------
# 15. Print output summary
# ---------------------------------------------------------

print("\nSynthetic multi-source block-wise missing dataset generated successfully.")
print(f"Output folder: {OUTPUT_DIR.resolve()}")

print("\nFiles created:")
for file in sorted(OUTPUT_DIR.glob("*")):
    print(f" - {file.name}")

print("\nClass distribution:")
print(labels["disease_status"].value_counts().sort_index())

print("\nInternal source pattern used to create blank source blocks:")
for profile, count in profile_counts.items():
    print(f"  {profile}: {count} samples - {profile_meanings[profile]}")
