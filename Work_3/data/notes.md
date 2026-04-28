# Synthetic Multi-Source Block-Wise Missing Dataset

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
| Number of samples | 500 |
| Source 1 clinical features | 10 |
| Source 2 imaging features | 8 |
| Source 3 proteomic features | 25 |
| Total features across all sources | 43 |
| Class 0 samples | 265 |
| Class 1 samples | 235 |
| Small internal missingness inside available blocks | 2.0% |

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

| Internal pattern | Meaning | Count | Percentage |
|---|---|---:|---:|
| `001` | source 3 proteomics only; source 1 and source 2 blank | 27 | 5.40% |
| `010` | source 2 imaging only; source 1 and source 3 blank | 27 | 5.40% |
| `011` | source 2 imaging + source 3 proteomics available; source 1 clinical blank | 61 | 12.20% |
| `100` | source 1 clinical only; source 2 and source 3 blank | 41 | 8.20% |
| `101` | source 1 clinical + source 3 proteomics available; source 2 imaging blank | 56 | 11.20% |
| `110` | source 1 clinical + source 2 imaging available; source 3 proteomics blank | 75 | 15.00% |
| `111` | source 1 clinical + source 2 imaging + source 3 proteomics available | 213 | 42.60% |

## Source 1 columns: clinical/laboratory

| Column | Meaning |
|---|---|
| `patient_id` | Unique sample identifier used to link this source with labels and other sources. |
| `age_years` | Clinical/laboratory feature. |
| `bmi` | Clinical/laboratory feature. |
| `systolic_bp` | Clinical/laboratory feature. |
| `diastolic_bp` | Clinical/laboratory feature. |
| `fasting_glucose` | Clinical/laboratory feature. |
| `hba1c` | Clinical/laboratory feature. |
| `crp` | Clinical/laboratory feature. |
| `ldl_cholesterol` | Clinical/laboratory feature. |
| `hdl_cholesterol` | Clinical/laboratory feature. |
| `triglycerides` | Clinical/laboratory feature. |

## Source 2 columns: imaging-derived

| Column | Meaning |
|---|---|
| `patient_id` | Unique sample identifier used to link this source with labels and other sources. |
| `hippocampal_volume` | Imaging-derived feature. |
| `cortical_thickness` | Imaging-derived feature. |
| `white_matter_lesion_volume` | Imaging-derived feature. |
| `ventricle_volume` | Imaging-derived feature. |
| `bbb_permeability_index` | Imaging-derived feature. |
| `cerebral_blood_flow` | Imaging-derived feature. |
| `gray_matter_density` | Imaging-derived feature. |
| `microbleed_count` | Imaging-derived feature. |

## Source 3 columns: proteomic/molecular

| Column | Meaning |
|---|---|
| `patient_id` | Unique sample identifier used to link this source with labels and other sources. |
| `protein_marker_01` | Proteomic/molecular feature. |
| `protein_marker_02` | Proteomic/molecular feature. |
| `protein_marker_03` | Proteomic/molecular feature. |
| `protein_marker_04` | Proteomic/molecular feature. |
| `protein_marker_05` | Proteomic/molecular feature. |
| `protein_marker_06` | Proteomic/molecular feature. |
| `protein_marker_07` | Proteomic/molecular feature. |
| `protein_marker_08` | Proteomic/molecular feature. |
| `protein_marker_09` | Proteomic/molecular feature. |
| `protein_marker_10` | Proteomic/molecular feature. |
| `protein_marker_11` | Proteomic/molecular feature. |
| `protein_marker_12` | Proteomic/molecular feature. |
| `protein_marker_13` | Proteomic/molecular feature. |
| `protein_marker_14` | Proteomic/molecular feature. |
| `protein_marker_15` | Proteomic/molecular feature. |
| `protein_marker_16` | Proteomic/molecular feature. |
| `protein_marker_17` | Proteomic/molecular feature. |
| `protein_marker_18` | Proteomic/molecular feature. |
| `protein_marker_19` | Proteomic/molecular feature. |
| `protein_marker_20` | Proteomic/molecular feature. |
| `protein_marker_21` | Proteomic/molecular feature. |
| `protein_marker_22` | Proteomic/molecular feature. |
| `protein_marker_23` | Proteomic/molecular feature. |
| `protein_marker_24` | Proteomic/molecular feature. |
| `protein_marker_25` | Proteomic/molecular feature. |

## Recommended files for experiments

For complete-data baseline experiments, use:

`complete_multisource_dataset.csv`

For block-wise missing data experiments, use:

`blockwise_missing_multisource_dataset.csv`

For source-level experiments, use the individual source files together with `labels.csv`.
