# Blockwise Missing Multi-Source iSFS Logistic Model

## Overview

This repository implements the current modular version of the incomplete-source
feature-selection (iSFS) model from Section 3 of `Multi_source_BWM.pdf`, with a
binary-classification extension based on logistic loss.

The implementation is organized around the same core blocks that appear in the
paper:

- missingness-profile construction from source availability
- profile-specific source weights `alpha_m`
- shared source coefficients `beta^i`
- alternating optimization over `alpha` and `beta`

This repository also includes the cost-sensitive extension discussed in
`notes.tex`, where the class-cost vector `C` is updated with a trust-region
step inside an outer block-coordinate-descent (BCD) loop.

The active path is the modular implementation:

- [missingness_profiles.py](missingness_profiles.py)
- [loss_functions.py](loss_functions.py)
- [regularization.py](regularization.py)
- [alpha_update.py](alpha_update.py)
- [beta_update.py](beta_update.py)
- [costs.py](costs.py)
- [prediction.py](prediction.py)
- [train_model.py](train_model.py)

The older file [blockwise_logistic_model.py](blockwise_logistic_model.py) is a
legacy all-in-one wrapper and is not the current source of truth.

## What Has Been Implemented

The current codebase implements the following project stages:

1. Missingness-profile construction from the paper.
2. Stable binary cross-entropy (BCE) from logits.
3. Exact-profile prediction for incomplete-source inference.
4. Cost-sensitive BCE with a trust-region update for `C`.
5. Outer BCD training loop:
   fix `C`, solve `alpha` and `beta`; fix `alpha` and `beta`, update `C`.
6. Paper-aligned `l1` regularization path:
   `R_alpha(alpha_m) = ||alpha_m||_1` and
   `R_beta(beta) = sum_i ||beta^i||_1`.

Important modeling note:

- The printed equations in the paper are derived with least squares.
- This repository keeps the Section 3 structure but replaces the data-fit term
  with logistic loss for binary classification.
- So the implementation is structurally faithful to the paper and follows the
  paper's own remark that the method can be extended to logistic regression.

## Mathematical Summary

For one optimization group `m`, the model score is

```text
z_m = sum_{i in m} alpha_m^i X_m^i beta^i
```

For binary classification, the loss for sample `j` in group `m` is

```text
ell(z_{m,j}, y_{m,j}) = log(1 + exp(z_{m,j})) - y_{m,j} z_{m,j}
```

With class costs `C`, the group objective used in code is

```text
f_m^C(alpha, beta, C)
= (1 / n_m) * sum_{j in G_m} C_{y_{m,j}} * ell(z_{m,j}, y_{m,j})
```

The fixed-cost Block A objective is

```text
Phi(alpha, beta ; C)
= (1 / |pf|) * sum_{m in pf} f_m^C(alpha, beta, C)
+ lambda_beta * sum_i ||beta^i||_1
```

subject to

```text
||alpha_m||_1 <= r_alpha
```

The outer BCD loop alternates between:

```text
Block A:
    fix C and solve for alpha, beta

Block B:
    fix alpha, beta and update C by trust region
```

For the cost block, the classwise loss vector is

```text
L_k(alpha, beta)
= (1 / |pf|) * sum_{m in pf} (1 / n_m)
  * sum_{j in G_m : y_{m,j} = k} ell(z_{m,j}, y_{m,j})
```

and the cost-only objective becomes

```text
Phi_C(C) = L(alpha, beta)^T C
```

The trust-region step updates `C` subject to nonnegativity and radius
constraints defined in `notes.tex`.

## Data Layout

The repository currently uses the real blockwise-missing CSV files in `data/`:

- [data/source1_clinical_blockwise_missing.csv](data/source1_clinical_blockwise_missing.csv)
- [data/source2_imaging_blockwise_missing.csv](data/source2_imaging_blockwise_missing.csv)
- [data/source3_proteomics_blockwise_missing.csv](data/source3_proteomics_blockwise_missing.csv)
- [data/labels.csv](data/labels.csv)

Current real-data dimensions:

| Item | Value |
|---|---:|
| Samples | 500 |
| Source 1 features | 10 |
| Source 2 features | 8 |
| Source 3 features | 25 |
| Labels | binary |
| Class 0 count | 265 |
| Class 1 count | 235 |

Important data assumption used by the code:

- the three source CSV files are row-aligned with `labels.csv`
- the source CSV files currently contain feature columns only
- the training code does not merge on `patient_id`
- a source is treated as missing for sample `j` only when the entire row of that
  source is `NaN`
- the default `real` training path uses the blockwise-missing source files above,
  not the `*_complete.csv` files and not `blockwise_missing_multisource_dataset.csv`

The `real` dataset option therefore means the separate source files with
missing source blocks:

```text
source1_clinical_blockwise_missing.csv
source2_imaging_blockwise_missing.csv
source3_proteomics_blockwise_missing.csv
labels.csv
```

The `synthetic` dataset option builds a small in-memory blockwise-missing
dataset for debugging. It does not read files from `data/`.

The current exact participant profiles in the real dataset are:

| Profile code | Meaning | Count |
|---|---|---:|
| `001` | source 3 only | 27 |
| `010` | source 2 only | 27 |
| `011` | source 2 + source 3 | 61 |
| `100` | source 1 only | 41 |
| `101` | source 1 + source 3 | 56 |
| `110` | source 1 + source 2 | 75 |
| `111` | source 1 + source 2 + source 3 | 213 |

The current overlapping iSFS optimization groups are:

| Group code | Count |
|---|---:|
| `001` | 357 |
| `010` | 376 |
| `011` | 274 |
| `100` | 385 |
| `101` | 269 |
| `110` | 288 |
| `111` | 213 |

## Repository Structure

### Core model modules

| File | Role |
|---|---|
| [missingness_profiles.py](missingness_profiles.py) | Builds exact participant profiles and overlapping paper-style optimization groups from source availability. |
| [loss_functions.py](loss_functions.py) | Implements numerically stable BCE from logits and its gradient. |
| [loss.py](loss.py) | Compatibility shim that re-exports the loss helpers used by older modules. |
| [regularization.py](regularization.py) | Implements the active paper-aligned `l1` regularization operators for `alpha` and `beta`. |
| [alpha_update.py](alpha_update.py) | Solves the fixed-`beta`, fixed-`C` alpha subproblem with projected gradient on the `l1` ball. |
| [beta_update.py](beta_update.py) | Solves the fixed-`alpha`, fixed-`C` beta subproblem with proximal gradient and soft-thresholding. |
| [costs.py](costs.py) | Implements cost-sensitive BCE, the class-loss vector `L(alpha, beta)`, and the trust-region cost step for `C`. |
| [prediction.py](prediction.py) | Computes logits, probabilities, and class predictions using exact profiles at inference time. |
| [train_model.py](train_model.py) | Runs the full outer BCD training loop and returns the trained model dictionary. |

### Runnable scripts

| File | Role |
|---|---|
| [run_training.py](run_training.py) | Main user-facing training entrypoint for real or synthetic data. |
| [run_training.sh](run_training.sh) | Thin shell wrapper around `run_training.py`; easiest way to launch training. |
| [generate_results_report.py](generate_results_report.py) | Reads one saved `training_outputs/...` run directory and generates compact Markdown, LaTeX, and PDF result tables for sharing. |
| [example_run.py](example_run.py) | Small synthetic example that trains the model and prints learned parameters. |
| [test_real_data_checks.py](test_real_data_checks.py) | Real-data audit script for profiles, loss, costs, prediction, and regularization. |
| [tests/test_loss_and_profiles.py](tests/test_loss_and_profiles.py) | Unit tests and numerical gradient checks. |

### Data and utilities

| File | Role |
|---|---|
| [generate_blockwise_missing_dataset.py](generate_blockwise_missing_dataset.py) | Utility script that generates a synthetic blockwise-missing dataset and companion notes. |
| [data/notes.md](data/notes.md) | Dataset notes describing the synthetic blockwise-missing data design. |
| [Multi_source_BWM.pdf](Multi_source_BWM.pdf) | Research-paper reference for the Section 3 model structure. |
| [notes.tex](notes.tex) | Trust-region and cost-extension notes used for the `C` block. |

### Legacy file

| File | Role |
|---|---|
| [blockwise_logistic_model.py](blockwise_logistic_model.py) | Older monolithic educational wrapper. Not the active modular training path. |

## How Training Works

The default runner first creates a held-out evaluation split, then trains only
on the training partition:

1. Read the source matrices and labels.
2. Build a 70/30 train/test split stratified by label and exact missingness profile.
3. Fit preprocessing on the training rows only.
4. Transform train and test rows with the train-fitted preprocessing parameters.
5. Train the model on the training rows only.
6. Evaluate and save metrics for train and test rows separately.

The optimization loop used by [train_model.py](train_model.py) is:

1. Build exact profiles and overlapping optimization groups on the training rows.
2. Initialize `beta`, `alpha`, and the current cost vector `C`.
3. Solve Block A with fixed `C`.
4. Compute the class-loss vector `L(alpha, beta)`.
5. Solve the trust-region cost subproblem to get a trial `C`.
6. Re-solve Block A at the trial `C`.
7. Accept or reject the cost step using the trust-region ratio.
8. Repeat until the outer stopping condition is met.

Inside Block A, the code follows the paper's alternating structure:

1. With `beta` fixed, approximately solve each `alpha_m` subproblem.
2. With `alpha` fixed, approximately solve the shared `beta` subproblem.
3. Repeat until the fixed-cost Block A objective stabilizes.

```mermaid
flowchart TD
    A[Load selected dataset] --> B[Build 70/30 stratified train/test split]
    B --> C[Fit preprocessing on train rows only]
    C --> D[Transform train and test rows]
    D --> E[Build train exact profiles and optimization groups]
    E --> F[Initialize C, alpha, beta]
    F --> G[Initial Block A solve at fixed C]
    G --> H{Outer BCD iteration}
    H --> I[Compute class-loss vector L(alpha, beta)]
    I --> J[Solve trust-region step for trial C]
    J --> K[Re-solve Block A at trial C]
    K --> L{Trust-region ratio accepted?}
    L -- yes --> M[Keep trial C, alpha, beta]
    L -- no --> N[Keep previous C, alpha, beta]
    M --> O[Update trust-region radius]
    N --> O
    O --> P{Reached max iter or tolerance?}
    P -- no --> H
    P -- yes --> Q[Evaluate train rows and held-out test rows]
    Q --> R[Save JSON, CSV, Markdown, LaTeX, and PDF outputs]
```

## Quick Start

Run the normal training flow on the real blockwise-missing CSV dataset:

```bash
./run_training.sh
```

This default flow trains on 70% of the rows, evaluates on the held-out 30%,
uses train-only preprocessing, and saves train/test outputs under
`training_outputs/`.

Run the Python entrypoint directly with the same default flow:

```bash
python3 run_training.py --dataset real
```

Generate a compact shareable results report from a saved training run:

```bash
python3 generate_results_report.py --run-dir training_outputs/<run_folder>
```

Run the old in-sample evaluation mode only when you need to reproduce older
behavior:

```bash
python3 run_training.py --dataset real --in-sample-evaluation
```

In-sample evaluation trains on all rows and evaluates on those same rows. It is
not a held-out test.

Run a quick debug run with fewer iterations:

```bash
python3 run_training.py --dataset real --max-iter 1 --block-a-max-iter 1 --alpha-subproblem-max-iter 2 --beta-subproblem-max-iter 2 --run-name quick_debug
```

This command is sometimes called a smoke run: it is a short run used to check
that the code path, output files, and report generation work. It is not a final
experiment.

Train on the built-in synthetic dataset:

```bash
./run_training.sh --dataset synthetic
```

Run the small synthetic example:

```bash
python3 example_run.py
```

Run all real-data checks:

```bash
python3 test_real_data_checks.py --target all
```

Run the unit tests:

```bash
python3 -m unittest discover -s tests -v
```

## Paper Comparison Runners

The two paper folders have their own concise READMEs:

- [paper_informs_reduced/README.md](paper_informs_reduced/README.md)
- [paper_plos_one_bwm/README.md](paper_plos_one_bwm/README.md)

For the shared `main_data/` comparison workflow, run:

```bash
Rscript paper_informs_reduced/scripts/run_informs_main_data.R --dataset all
Rscript paper_plos_one_bwm/scripts/run_plos_bwm_main_data.R --dataset all
Rscript scripts/collect_main_data_comparison.R
```

The collector writes `comparison_results/main_data_model_metrics.csv`.
If local `Rscript` is unavailable, use the Docker alternatives documented in
the paper-folder READMEs.

## Main Training Scripts

### `run_training.sh`

This is the simplest launcher. It runs:

```bash
python3 run_training.py \
  --dataset real \
  --max-iter 12 \
  --block-a-max-iter 25 \
  --alpha-subproblem-max-iter 25 \
  --beta-subproblem-max-iter 25 \
  "$@"
```

So by default it trains on the real dataset with all major iteration limits
above 10. Any extra command-line options you pass to the shell script are
forwarded to `run_training.py` and can override those defaults.

Examples:

```bash
./run_training.sh
./run_training.sh --in-sample-evaluation
./run_training.sh --max-iter 1 --block-a-max-iter 1 --alpha-subproblem-max-iter 2 --beta-subproblem-max-iter 2 --run-name quick_debug
./run_training.sh --dataset synthetic
```

### `run_training.py`

This is the main training CLI. It:

- loads the selected dataset
- prints the profile structure
- trains the model with the full BCD loop
- explains the BCD fields while it runs
- prints the learned cost vector
- prints the learned `alpha` and `beta`
- prints detailed train and test metrics
- prints exact-profile and optimization-group metrics for train and test rows
- saves CSV and JSON reports for the run unless disabled

Example:

```bash
python3 run_training.py --dataset real --max-iter 12 --block-a-max-iter 25
```

### `generate_results_report.py`

This script turns one saved run directory produced by `run_training.py` into a
compact results document. It reads the saved JSON and CSV files from one
`training_outputs/...` directory and writes:

- `results_report.md`
- `results_report.tex`
- `results_report.pdf`

inside that same run directory.

Example:

```bash
python3 generate_results_report.py --run-dir training_outputs/real_20260424_010309
```

## `run_training.py` Parameters

### Main run modes

| Mode | Command | Meaning |
|---|---|---|
| Normal training flow | `python3 run_training.py --dataset real` | Use the real blockwise-missing source files, split rows 70/30, fit preprocessing on the 70% training rows only, train on training rows, and evaluate on the held-out 30% test rows. |
| Old in-sample behavior | `python3 run_training.py --dataset real --in-sample-evaluation` | Use all rows for fitting and evaluate on those same rows. This reproduces the older behavior and should not be described as held-out test performance. |
| Quick debug run | `python3 run_training.py --dataset real --max-iter 1 --block-a-max-iter 1 --alpha-subproblem-max-iter 2 --beta-subproblem-max-iter 2 --run-name quick_debug` | Run the same code path with very small iteration counts to check that training and output generation work. This is not a final experiment. |
| Synthetic debug run | `python3 run_training.py --dataset synthetic` | Use the small synthetic in-memory dataset instead of the CSV files in `data/`. |

### Dataset and initialization parameters

| Parameter | Meaning | Default |
|---|---|---|
| `--dataset` | Which dataset to use. `real` reads the separate blockwise-missing source CSV files in `data/`; `synthetic` creates a small in-memory blockwise-missing dataset. | `real` |
| `--random-state` | Random seed used for initialization and synthetic-data generation. | `42` |
| `--split-random-state` | Random seed used for the train/test split. | `42` |
| `--test-size` | Holdout fraction for the stratified test split. `0.30` means 70% train and 30% test. | `0.30` |
| `--decision-threshold` | Probability threshold used for class prediction and reported metrics. | `0.5` |
| `--initial-cost-vector` | Initial class-cost vector `C^(0)`, passed as comma-separated text. | `"0.5,0.5"` |
| `--baseline-cost-vector` | Baseline trust-region center `C^0`, passed as comma-separated text. | `"0.5,0.5"` |
| `--in-sample-evaluation`, `--no-test-split` | Train and evaluate on all rows, matching the older behavior. | off |
| `--no-standardize`, `--no-preprocess` | Disable train-only partial-NaN imputation and feature standardization. | off |

Examples:

```bash
python3 run_training.py --dataset real
python3 run_training.py --dataset synthetic --random-state 7
python3 run_training.py --initial-cost-vector 0.6,0.4 --baseline-cost-vector 0.5,0.5
```

### Alpha and beta optimization parameters

| Parameter | Meaning | Default |
|---|---|---|
| `--learning-rate-alpha` | Step size for projected-gradient updates in the alpha subproblem. | `0.01` |
| `--learning-rate-beta` | Step size for proximal-gradient updates in the beta subproblem. | `0.01` |
| `--lambda-beta` | Weight of the `l1` penalty on `beta`. | `0.01` |
| `--alpha-l1-radius` | Radius `r_alpha` of the `l1` ball used to constrain each `alpha_m`. | `1.0` |
| `--alpha-subproblem-max-iter` | Maximum projected-gradient iterations used inside each alpha solve while beta and `C` are fixed. | `25` |
| `--alpha-subproblem-tol` | Convergence tolerance for the alpha subproblem solver. | `1e-6` |
| `--beta-subproblem-max-iter` | Maximum proximal-gradient iterations used inside each beta solve while alpha and `C` are fixed. | `25` |
| `--beta-subproblem-tol` | Convergence tolerance for the beta subproblem solver. | `1e-6` |

Examples:

```bash
python3 run_training.py --learning-rate-alpha 0.005 --learning-rate-beta 0.005
python3 run_training.py --lambda-beta 0.02 --alpha-l1-radius 1.0
python3 run_training.py --alpha-subproblem-max-iter 50 --beta-subproblem-max-iter 50
```

### Block A and outer BCD parameters

| Parameter | Meaning | Default |
|---|---|---|
| `--block-a-max-iter` | Maximum alpha/beta alternating passes used to solve Block A at fixed cost `C`. Each pass can run an alpha subproblem solve and a beta subproblem solve. | `25` |
| `--block-a-tol` | Convergence tolerance for the Block A solver. | `1e-6` |
| `--max-iter` | Maximum outer BCD iterations over the cost vector `C`. Each outer iteration proposes a trust-region update for `C`, re-solves Block A at the trial `C`, then accepts or rejects the trial. | `12` |
| `--tol` | Outer stopping tolerance used in the BCD loop. | `1e-6` |

Examples:

```bash
python3 run_training.py --max-iter 20
python3 run_training.py --block-a-max-iter 40 --block-a-tol 1e-7
```

Default iteration counts:

| Loop | Parameter | Default |
|---|---|---:|
| Outer cost-vector BCD loop | `--max-iter` | 12 |
| Fixed-cost Block A alpha/beta alternation | `--block-a-max-iter` | 25 |
| Alpha projected-gradient subproblem | `--alpha-subproblem-max-iter` | 25 |
| Beta proximal-gradient subproblem | `--beta-subproblem-max-iter` | 25 |

### Trust-region cost-update parameters

| Parameter | Meaning | Default |
|---|---|---|
| `--initial-trust-region-radius` | Initial working trust-region radius for the `C` update. | `0.02` |
| `--max-trust-region-radius` | Maximum allowed working trust-region radius. | `0.125` |
| `--accept-threshold` | Minimum trust-region ratio needed to accept a cost step. | `0.25` |
| `--expand-threshold` | Ratio threshold above which the radius is expanded. | `0.75` |
| `--shrink-factor` | Multiplicative factor used when shrinking the trust-region radius. | `0.25` |
| `--expand-factor` | Multiplicative factor used when expanding the trust-region radius. | `2.0` |

Examples:

```bash
python3 run_training.py --initial-trust-region-radius 0.01 --max-trust-region-radius 0.10
python3 run_training.py --accept-threshold 0.20 --expand-threshold 0.80
python3 run_training.py --shrink-factor 0.5 --expand-factor 1.5
```

### Reporting and saved-output parameters

| Parameter | Meaning | Default |
|---|---|---|
| `--output-dir` | Base directory where per-run reports are saved. | `training_outputs` |
| `--run-name` | Optional suffix appended to the saved run directory. | `None` |
| `--no-save-results` | Disable CSV/JSON report writing for that run. | off |

Examples:

```bash
python3 run_training.py --output-dir my_runs
python3 run_training.py --run-name experiment_a
python3 run_training.py --no-save-results
```

## What `run_training.py` Prints

The training runner prints:

- dataset and optimization configuration
- a field guide for interpreting BCD output
- exact missingness profiles
- overlapping iSFS optimization groups
- initial Block A progress
- each outer BCD iteration for the cost update
- train and test summaries
- learned cost vector
- learned `alpha` values by profile
- learned `beta` shapes by source
- exact-profile metric tables for train and test rows
- optimization-group metric tables for train and test rows

The final summary includes:

- number of samples
- number of valid predictions
- train and test accuracy
- precision
- recall / sensitivity
- specificity
- F1 score
- balanced accuracy
- GMEAN, computed as `sqrt(recall * specificity)`
- negative predictive value
- false positive and false negative rates
- Matthews correlation coefficient
- Brier score
- log loss
- confusion counts
- probability range
- BCD history size
- accepted trust-region cost steps
- final reduced objective and trust-region statistics

## Saved Run Reports

By default, every `run_training.py` or `run_training.sh` call writes a
timestamped run directory inside `training_outputs/`.

Typical saved files are:

| File | Meaning |
|---|---|
| `config.json` | CLI arguments used for the run. |
| `split_metadata.json` | Train/test indices and label/profile stratum counts. |
| `split_summary.json` | Train/test indices and label/profile stratum counts. |
| `preprocessing.json` | Train-fitted feature means and standard deviations used for scaling. |
| `preprocessing_summary.json` | Train-fitted feature means and standard deviations used for scaling. |
| `train_summary.json` | Overall metrics on rows used for fitting. |
| `test_summary.json` | Overall metrics on held-out rows. |
| `training_summary.json` | Backward-compatible alias for `train_summary.json`. |
| `bcd_history.csv` | One row per outer BCD iteration. |
| `block_a_history.csv` | One row per inner Block A iteration across all solves. |
| `block_a_solve_summary.csv` | One summary row per Block A solve. |
| `train_exact_profile_metrics.csv` | Train metrics for each exact profile. |
| `test_exact_profile_metrics.csv` | Test metrics for each exact profile. |
| `train_optimization_group_metrics.csv` | Train metrics for each overlapping optimization group. |
| `test_optimization_group_metrics.csv` | Test metrics for each overlapping optimization group. |
| `exact_profile_metrics.csv` | Backward-compatible alias for train exact-profile metrics. |
| `optimization_group_metrics.csv` | Backward-compatible alias for train optimization-group metrics. |
| `final_parameters.json` | Final saved cost vector, alpha values, and beta values. |
| `parameter_history.json` | Saved parameter snapshots across the training run. |

After a run is saved, you can convert those files into a compact shareable
table report with:

```bash
python3 generate_results_report.py --run-dir training_outputs/<run_folder>
```

This creates three report files inside the same run directory:

| File | Meaning |
|---|---|
| `results_report.md` | Compact markdown summary with tables for setup, metrics, BCD, alpha, beta, exact profiles, and optimization groups. |
| `results_report.tex` | LaTeX version of the same summary table report. |
| `results_report.pdf` | PDF version of the same summary table report. |

## Prediction Path

[prediction.py](prediction.py) uses exact participant profiles at inference
time.

This is important:

- training uses overlapping optimization groups
- prediction does not use overlapping groups
- each sample belongs to exactly one exact observed-source profile at inference

For a sample with available sources `A_j`, the predictor is

```text
z_j = sum_{i in A_j} alpha_{A_j}^i x_j^i beta^i
```

Probabilities are then computed with the sigmoid link, and class labels are
obtained by thresholding.

The learned cost vector does not appear directly in the prediction equation.
Costs affect the trained `alpha` and `beta`, but inference itself uses the
learned parameters in the paper-style linear score.

## Diagnostics and Test Scripts

### `test_real_data_checks.py`

This script runs real-data checks on the active modules. It uses your actual
CSV files in `data/`, not a toy dataset.

Usage:

```bash
python3 test_real_data_checks.py --target profiles
python3 test_real_data_checks.py --target loss
python3 test_real_data_checks.py --target costs
python3 test_real_data_checks.py --target prediction
python3 test_real_data_checks.py --target regularization
python3 test_real_data_checks.py --target all
```

Parameters:

| Parameter | Meaning |
|---|---|
| `--target profiles` | Print exact profiles and overlapping optimization groups on the real data. |
| `--target loss` | Check real-data BCE values and a finite-difference gradient sanity check. |
| `--target costs` | Check the class-loss vector and one trust-region cost step on the real data. |
| `--target prediction` | Check that exact-profile prediction returns finite probabilities on the real data. |
| `--target regularization` | Check alpha feasibility and beta regularization measurements on the real data. |
| `--target all` | Run every real-data check in sequence. |

### `tests/test_loss_and_profiles.py`

This file contains unit tests and numerical audits, including:

- loss validation tests
- missingness-profile tests
- exact-profile prediction tests
- regularization tests
- finite-difference gradient checks for the alpha subproblem
- finite-difference gradient checks for the beta subproblem

Run it with:

```bash
python3 -m unittest discover -s tests -v
```

## Synthetic Utilities

### `example_run.py`

This is a compact demonstration script. It creates a synthetic blockwise-missing
dataset in memory, trains the model, and prints:

- final accuracy
- learned alpha values
- final cost vector
- alpha `l1` norms
- beta dimensions

Run:

```bash
python3 example_run.py
```

### `generate_blockwise_missing_dataset.py`

This utility generates a synthetic multi-source blockwise-missing dataset and
writes multiple CSV files plus notes. It is a dataset-generation utility, not
the core model trainer.

Run:

```bash
python3 generate_blockwise_missing_dataset.py
```

### `generate_results_report.py`

This utility reads one saved run directory and produces a short table-focused
results document for sharing.

Run:

```bash
python3 generate_results_report.py --run-dir training_outputs/<run_folder>
```

## Returned Model Structure

The model returned by [train_model.py](train_model.py) is a Python dictionary
with the main entries:

| Key | Meaning |
|---|---|
| `betas` | Learned shared source coefficients. |
| `alphas` | Learned profile-specific source weights. |
| `costs` | Final learned class-cost vector. |
| `baseline_costs` | Baseline trust-region center. |
| `lambda_beta` | Active beta regularization weight. |
| `alpha_l1_radius` | Active alpha constraint radius. |
| `source_order` | Canonical source order used for profile codes and prediction. |
| `profiles` | Optimization groups used in training. |
| `exact_profiles` | Exact profiles built from the training data. |
| `optimization_groups` | Same optimization groups stored explicitly for convenience. |
| `loss_history` | Outer BCD history for the cost block. |
| `block_a_history` | Histories of the inner fixed-cost Block A solves. |
| `predict_proba` | Callable probability predictor. |
| `predict` | Callable class predictor. |

## Important Assumptions and Current Scope

- The active implementation is the modular path, not the legacy monolithic
  wrapper.
- The source CSV files are assumed to be row-aligned with `labels.csv`.
- A source is treated as missing only when the whole row is `NaN`.
- Prediction uses exact profiles, not the overlapping optimization groups.
- The regularization path is currently fixed to the paper-aligned `l1` choice.
- The model currently prints results to the terminal; it does not yet save a
  trained checkpoint file automatically.

## Recommended Commands

If you want the shortest working workflow, use these:

```bash
./run_training.sh
python3 generate_results_report.py --run-dir training_outputs/<run_folder>
python3 test_real_data_checks.py --target all
python3 -m unittest discover -s tests -v
```

If you want a quick debug run:

```bash
./run_training.sh --max-iter 1 --block-a-max-iter 1 --alpha-subproblem-max-iter 2 --beta-subproblem-max-iter 2 --run-name quick_debug
```

If you want a synthetic sanity run:

```bash
./run_training.sh --dataset synthetic
```
