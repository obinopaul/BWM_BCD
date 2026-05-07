# PLOS One BWM Package

This folder contains the PLOS One block-wise missing-data R package. The original package code is in `R/`; the added wrappers run that package on the local `main_data/` CSVs.

The package API is:

- `bwm.train(p, X, y, lambda, ...)`
- `bwm.predict(model, X, p, ...)`
- `evaluation.model.param(y.test, y.pred, eval = ...)`
- `get_profile(p, X)`

`X` must be numeric. `p` is the source-size vector: for example, `p = c(4, 6, 2)` means columns `1:4` are source 1, columns `5:10` are source 2, and columns `11:12` are source 3. For a row, `get_profile()` treats a source as observed only when every column in that source block is non-missing.

For the detailed mechanics of missingness generation, train/test splitting, profile construction, training, and prediction, see [MODEL_AND_MISSINGNESS.md](MODEL_AND_MISSINGNESS.md).

## Which Runner To Use

Use the canonical long-format runner when you want outputs ready for the root comparison collector:

```bash
Rscript paper_plos_one_bwm/scripts/run_plos_bwm_main_data.R --dataset all
```

Use the Docker runner when local R or package dependencies are not installed:

```bash
./paper_plos_one_bwm/run_from_repo.sh --dataset all
```

The Docker runner calls `paper_plos_one_bwm/run_main_data_experiment.R`, which writes a wide CSV. The root collector can read both output formats. The Docker runner installs common Linux build libraries and missing R packages by default; set `PLOS_INSTALL_SYSTEM_DEPS=0` or `PLOS_INSTALL_DEPS=0` only when the image already has those dependencies.

Smoke run for quick validation:

```bash
Rscript paper_plos_one_bwm/scripts/run_plos_bwm_main_data.R \
  --dataset bike \
  --missing-props 0.1 \
  --max-iter 5 \
  --max-iter-alpha 3 \
  --max-iter-beta 3
```

Selected datasets and settings:

```bash
Rscript paper_plos_one_bwm/scripts/run_plos_bwm_main_data.R \
  --dataset bike,adult \
  --missing-props 0.1,0.3,0.5 \
  --lambda-grid 0,1e-4,1e-3,1e-2
```

## Runner Parameters

Canonical runner: `scripts/run_plos_bwm_main_data.R`

| Parameter | Default | Meaning |
|---|---|---|
| `--dataset` | `all` | Dataset selector. Use `bike`, `house`, `adult`, `all`, or a comma-separated list such as `bike,adult`. |
| `--missing-props` | `0.1,0.2,...,0.9` | Generated block-missingness rates. For each value `r`, the wrapper samples `floor(r * n)` rows without replacement for each targeted source block and sets every variable in that source block to `NA` for those rows. This is generated source-block missingness, not raw CSV missingness and not independent cellwise missingness. |
| `--lambda-grid` | `0,1e-4,1e-3,1e-2,1e-1` | Regularization values passed to `bwm.train(lambda=...)`. The package tries the grid and keeps the lambda with the best objective. `0` means no beta shrinkage; larger values apply stronger shrinkage. |
| `--max-iter` | `100` | Maximum outer alpha/beta alternations inside `bwm.train()` for each lambda value. |
| `--max-iter-alpha` | `20` | Maximum iterations for the alpha subproblem. Alpha weights control how much each available source contributes inside each missingness profile. |
| `--max-iter-beta` | `20` | Maximum iterations for the beta subproblem. Beta values are the feature coefficients within sources. |
| `--tol` | `1e-8` | Convergence tolerance. Smaller values require smaller objective changes before stopping. |
| `--normalize` | `true` | Whether to call `bwm.train(to.normalize=TRUE)`. This rescales numeric columns inside the package while preserving `NA` profiles. |
| `--seed` | `1234` | General run seed. The missingness generators and train/test split also use fixed paper-style seeds so results are reproducible. |
| `--results-dir` | `paper_plos_one_bwm/results/run_<timestamp>` | Output directory for long-format metrics and run config. |

Docker/root runner parameter names:

| Docker/root parameter | Canonical equivalent | Meaning |
|---|---|---|
| `--missing-rates` | `--missing-props` | Same concept: generated block-missingness rates. |
| `--lambda` | `--lambda-grid` | Same concept: regularization grid passed to `bwm.train()`. |
| `--output-dir` | `--results-dir` | Output directory. |
| `--data-dir` | fixed to `main_data` by default | Directory containing `bike_all.csv`, `house.csv`, and `adult.csv`. |
| `--test-fraction` | fixed 25% test split in canonical runner | Held-out test fraction. |
| `--normalize` / `--no-normalize` | `--normalize true/false` | Toggle package normalization. The canonical runner defaults to `true`; the Docker/root runner defaults to `FALSE` unless `--normalize` is passed. |

## Why These Values

- `--missing-props 0.1,...,0.9` is a stress-test grid. It starts with mild generated block missingness at 10% and ends with severe missingness at 90%.
- `--lambda-grid 0,1e-4,1e-3,1e-2,1e-1` is a small regularization grid. `0` keeps the unregularized model available; the positive values add increasingly strong beta shrinkage.
- `--max-iter 100`, `--max-iter-alpha 20`, and `--max-iter-beta 20` are production-oriented defaults for the canonical runner. The smoke-run values are intentionally smaller so you can check wiring quickly; they should not be treated as final experimental settings.
- The Docker/root runner has lower internal defaults (`--lambda 0.01`, `--max-iter 30`, `--max-iter-alpha 10`, `--max-iter-beta 10`, `--no-normalize`) because it was added as a practical repo-level wrapper. Override those values when you want it to match the canonical runner more closely.
- `--tol 1e-8` is a practical convergence tolerance. Smaller values run longer; larger values may stop before the objective has stabilized.
- `--normalize true` is useful because the source blocks mix variables on very different scales, such as counts, temperatures, square footage, and one-hot columns.

## What `--missing-props` Means

The raw CSVs are not already blockwise missing datasets. Bike and house have no raw missing cells. Adult has raw missing placeholders, but the current wrappers only read the blank `education-num` entries as R `NA`; `?` markers are treated as factor levels and recoded. The scripts create the blockwise `NA` pattern at runtime so the same CSV can be evaluated at different missingness levels. The full reproducibility description is in [MODEL_AND_MISSINGNESS.md](MODEL_AND_MISSINGNESS.md).

For example, on bike:

- `--missing-props 0.1` selects about 10% of rows and sets Source 1 variables to `NA`.
- It independently selects about 10% of rows and sets Source 2 variables to `NA`.
- Additional per-column noise is added to match the INFORMS generator.

So `0.1,0.3,0.5` means run three separate experiments: one at 10%, one at 30%, and one at 50% generated block missingness.

## Data, Sources, And Labels

The detailed dataset inventory is in [main_data/README.md](../main_data/README.md).

Summary:

| Dataset | Label/outcome | Task | Sources before one-hot encoding |
|---|---|---|---|
| Bike | `cnt` | Regression/count prediction | Source 1: `windspeed`, `hum`, `ToD`, `weekday`; Source 2: `hr`, `temp`, `temp2`, `weathersit`; Source 3: `season`, `mnth` |
| House | `price / 1000` | Regression | Source 1: `bedrooms`, `bathrooms`; Source 2: `sqft_living`, `sqft_lot`; Source 3: `floors`, `view`; Source 4: `condition`, `yr_built` |
| Adult | `salary` as `0/1` | Binary classification | Source 1: `age`, `workclass`, `fnlwgt`; Source 2: `marital.status`, `occupation`, `relationship`; Source 3: remaining predictors |

Categorical predictors are one-hot encoded before training. If any original variable in a source is missing for a row, the wrapper preserves that source as unavailable in the encoded matrix.

## Metrics Saved

The canonical runner writes long-format metrics to:

```text
paper_plos_one_bwm/results/run_<timestamp>/plos_bwm_main_data_metrics.csv
paper_plos_one_bwm/results/run_<timestamp>/run_config.csv
```

Regression metrics:

- `rmse`
- `mae`
- `mape`
- `smape`
- `r2`

Classification metrics:

- `accuracy`
- `balanced_accuracy`
- `recall` (sensitivity)
- `specificity`
- `precision`
- `f1`
- `gmean`
- `mcc`
- `hard_label_auc`

`bwm.predict()` returns class labels, not probabilities. Therefore `hard_label_auc` is computed from hard labels and equals balanced accuracy for binary 0/1 predictions.

Each metric row also includes `n_train`, `n_test`, `n_invalid_predictions`, `best_lambda`, and `source_count`.

## Combine Comparison Results

After running this PLOS workflow and the INFORMS workflow, collect comparison results from the repo root:

```bash
Rscript scripts/collect_main_data_comparison.R
```

Output:

```text
comparison_results/main_data_model_metrics.csv
```

The collector gathers PLOS BWM rows plus INFORMS benchmark rows such as BRM, BRM_nov, listwise, columnwise, MI, SI, CART, GAIN, iMSF, DISCOM, REFE, and CART-indicator where those files exist.

## Package Files

- `R/bwm_train.R`: training entrypoint.
- `R/bwm_predict.R`: prediction entrypoint.
- `R/get_profile.R`: profile detection from `X` and `p`.
- `R/evaluation_model_param.R`: built-in metric helper.
- `man/*.Rd`: generated R package help files, not diff files.
- `.Rproj.user/`: RStudio IDE state, not needed for command-line runs.

## Validation Status

In the current shell, `docker` is installed but local `Rscript` is not. Shell syntax checks pass for `run_from_repo.sh`.

A Docker smoke run reached the R dependency installation stage. The first attempt showed that the plain `rocker/r-ver:4.3.3` image is missing `cmake`, which is needed by the `nloptr` dependency pulled in by the package stack. `run_from_repo.sh` now installs `cmake` and other common R build system libraries before installing R packages. I stopped that long source-build smoke run after confirming the dependency issue, so the full model experiment still needs to be run in Docker or in a local R environment with the listed packages installed.
