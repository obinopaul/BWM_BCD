# INFORMS Reduced Modeling Code

This folder contains the reduced-modeling paper's R capsule. The original paper
code is preserved in `code/`; repo-specific runners live in `scripts/` and use
the shared `main_data/` CSVs without editing the original files.

## What It Runs

The original scripts start from complete datasets:

- `bike_all.csv`, outcome `cnt`
- `house.csv`, outcome `price`
- `adult.csv`, outcome `salary`

For each dataset, the paper code generates blockwise missing versions at
missing proportions `0.1` through `0.9`, splits each version into 75% train and
25% test with seed `1234`, trains BRM plus benchmark methods, predicts on the
held-out rows, and computes metrics.

The benchmark suite includes BRM, non-overlapping BRM, listwise deletion,
columnwise deletion, multiple imputation, softImpute, CART, GAIN, iMSF, DISCOM,
REFE, CART with missingness indicators, and lazy-model placeholders. Many calls
are wrapped in `try(...)` because the original paper reports that some methods
fail for some missingness settings.

## Canonical Commands

Use this runner for comparison work because it writes normalized outputs under a
timestamped run folder and is compatible with the root comparison collector.

```bash
Rscript paper_informs_reduced/scripts/run_informs_main_data.R --dataset all
```

Run selected datasets:

```bash
Rscript paper_informs_reduced/scripts/run_informs_main_data.R --dataset adult
Rscript paper_informs_reduced/scripts/run_informs_main_data.R --dataset bike,house
```

If local R packages are difficult, use the Docker wrapper:

```bash
paper_informs_reduced/scripts/run_informs_docker.sh --dataset all
```

Useful options:

```bash
Rscript paper_informs_reduced/scripts/run_informs_main_data.R --dataset all --include-scalability
Rscript paper_informs_reduced/scripts/run_informs_main_data.R --dataset all --install-python-deps
Rscript paper_informs_reduced/scripts/run_informs_main_data.R --dataset all --keep-runtime
```

`--install-python-deps` allows the original GAIN path to try miniconda and
TensorFlow setup. Without it, GAIN-related failures are still caught by the
original `try(...)` blocks and the rest of the benchmark suite can continue.

## Outputs

The canonical runner writes:

```text
paper_informs_reduced/results/run_<timestamp>/
```

Typical outputs include:

- `rmse_reg_bike.csv`, `mae_reg_bike.csv`, `smape_reg_bike.csv`
- `rmse_tree_house.csv`, `mae_tree_house.csv`, `smape_tree_house.csv`
- `acc_reg_adult.csv`, `auc_reg_adult.csv`, `f1_reg_adult.csv`
- `metadata_<dataset>.csv`

After running both paper workflows, collect comparison results from the repo
root with:

```bash
Rscript scripts/collect_main_data_comparison.R
```

That writes:

```text
comparison_results/main_data_model_metrics.csv
```

## Alternate Runner

`paper_informs_reduced/run_from_repo.sh` is a standalone Docker runner that
mounts the original CodeOcean-style paths directly and can run `bike`, `house`,
`adult`, `scalability`, or `all`:

```bash
./paper_informs_reduced/run_from_repo.sh bike
```

Use the canonical `scripts/run_informs_main_data.R` path for comparison tables.
Use `run_from_repo.sh` when you want a closer direct run of the original capsule.

## Notes

- The original scripts contain hard-coded `/root/capsule/...` paths.
- `scripts/run_informs_main_data.R` stages patched runtime copies under
  `paper_informs_reduced/.runtime/` and removes them unless `--keep-runtime` is
  passed.
- The source files in `code/` are not modified by either runner.
- `main_data/` and `paper_informs_reduced/data/` contain the same three CSVs.

