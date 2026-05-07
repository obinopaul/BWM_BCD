# Main Data For Paper Comparisons

This folder contains the complete benchmark CSVs used by the INFORMS reduced-modeling paper code and by the PLOS One BWM wrapper. These CSVs are single-table datasets, not already multi-source datasets. The experiment scripts convert each CSV into a multi-source/blockwise-missing problem by grouping columns into source blocks and then generating blockwise `NA` values.

## Raw CSV Summary

| File | Rows | Columns | Raw missing placeholders | Raw missing percent | Outcome column | Task |
|---|---:|---:|---:|---:|---|---|
| `bike_all.csv` | 17,379 | 15 | 0 | 0.00% | `cnt` | Regression/count prediction |
| `house.csv` | 21,597 | 17 | 0 | 0.00% | `price` | Regression |
| `adult.csv` | 32,561 | 15 | 5,238 | 1.07% | `salary` | Binary classification |

Raw adult placeholder columns:

| Column | Placeholder rows | Placeholder percent |
|---|---:|---:|
| `workclass` | 1,836 | 5.64% |
| `education-num` | 487 | 1.50% |
| `occupation` | 2,332 | 7.16% |
| `native-country` | 583 | 1.79% |

The bike and house CSVs are complete before the experiment scripts add missingness. Adult contains 5,238 raw missing placeholders when counting blank fields and `?` markers. In the current R wrappers, only the 487 blank `education-num` entries are read as actual `NA`; the `?` markers are read as factor levels and are later recoded with other infrequent levels. Therefore, the blockwise missingness used by BWM is mainly the generated `NA` pattern.

## How One CSV Becomes Multi-Source

The PLOS One BWM package expects a numeric matrix `X` plus a source-size vector `p`. The wrappers therefore preprocess each CSV, choose source blocks, one-hot encode categorical predictors, and pass the encoded matrix to `bwm.train()`.

For a row, a source is considered available only if every column in that source block is non-missing. If one original variable in a source is missing, the wrapper marks the encoded source block as unavailable for that row.

The exact generated missingness algorithm and BWM profile logic are documented in [paper_plos_one_bwm/MODEL_AND_MISSINGNESS.md](../paper_plos_one_bwm/MODEL_AND_MISSINGNESS.md).

Current feature counts:

| Dataset | Original predictors before one-hot | Encoded numeric columns passed to BWM | Source-size vector `p` after encoding |
|---|---:|---:|---|
| Bike | 10 | 25 | `c(14, 6, 5)` |
| House | 8 | 8 | `c(2, 2, 2, 2)` |
| Adult | 12 | 30 | `c(4, 14, 12)` |

The Docker/root runner also writes the actual `p` vector to its wide result CSV. The canonical runner writes `source_count`; the source definitions below explain what each source contains before one-hot encoding.

## Bike

Input file: `bike_all.csv`

Outcome: `cnt`

Preprocessing:

- Drop `instant`, `dteday`, `atemp`, `yr`, `workingday`, and `holiday`.
- Convert `weekday`, `weathersit`, and `season` to categorical variables.
- Recode `weathersit = 4` to `3`.
- Add `ToD` from `hr` using time-of-day bins.
- Rescale `temp` to the original temperature scale and add `temp2 = temp^2`.

Predictors used before one-hot encoding: 10

| Source | Original predictors |
|---|---|
| Source 1 | `windspeed`, `hum`, `ToD`, `weekday` |
| Source 2 | `hr`, `temp`, `temp2`, `weathersit` |
| Source 3 | `season`, `mnth` |

Generated missingness:

- `--missing-props 0.10` means about 10% of rows are selected for Source 1 block missingness and about 10% are selected independently for Source 2 block missingness.
- Source 3 is not directly block-missing in the generator.
- An additional 5% random per-column noise is added to the variables in Sources 1 and 2.

## House

Input file: `house.csv`

Outcome: `price`, scaled as `price / 1000`

Preprocessing:

- Drop `sqft_basement`, `yr_renovated`, `zip`, `sqft_above`, `sqft_lot15`, `grade`, `waterfront`, and `sqft_living15`.
- Remove high-price outliers with `price >= 95th percentile`.
- Rows after filtering: 20,513.

Predictors used before one-hot encoding: 8

| Source | Original predictors |
|---|---|
| Source 1 | `bedrooms`, `bathrooms` |
| Source 2 | `sqft_living`, `sqft_lot` |
| Source 3 | `floors`, `view` |
| Source 4 | `condition`, `yr_built` |

Generated missingness:

- `--missing-props r` selects about `r` of rows for Source 1 block missingness.
- It independently selects about `r` of rows for Source 2 block missingness.
- Source 3 has fixed block missingness for rows `2001:6000` after preprocessing.
- An additional 2% random per-column noise is added to predictors.

## Adult

Input file: `adult.csv`

Outcome: `salary`, converted to `1` for `>=50k` and `0` otherwise

Preprocessing:

- Convert categorical columns to factors.
- Recode infrequent factor levels to `other`.
- Drop `capital.gain` and `capital.loss`.
- The original paper script attempts to drop `fnlwt`, but this local CSV column is named `fnlwgt`, so `fnlwgt` remains as a predictor.

Predictors used before one-hot encoding: 12

| Source | Original predictors |
|---|---|
| Source 1 | `age`, `workclass`, `fnlwgt` |
| Source 2 | `marital.status`, `occupation`, `relationship` |
| Source 3 | `education`, `education.num`, `race`, `sex`, `hours.per.week`, `native.country` |

Generated missingness:

- `--missing-props r` selects about `r` of rows for Source 1 block missingness.
- It independently selects about `r` of rows for Source 2 block missingness.
- Source 3 is not directly block-missing in the generator, but the 487 blank `education-num` values are read as `NA` and can make Source 3 unavailable for those rows.
- The `?` placeholders in `workclass`, `occupation`, and `native-country` are not treated as `NA` by these wrappers; they are categorical levels before recoding.
- An additional 2% random per-column noise is added to predictors.

## Missingness Rate Examples

The missingness rate parameter controls generated block missingness. It is not the raw CSV missingness percentage.

| Rate | Bike rows selected per targeted source | House rows selected per targeted source | Adult rows selected per targeted source |
|---:|---:|---:|---:|
| 0.10 | about 1,737 | about 2,051 | about 3,256 |
| 0.30 | about 5,213 | about 6,153 | about 9,768 |
| 0.50 | about 8,689 | about 10,256 | about 16,280 |
| 0.90 | about 15,641 | about 18,461 | about 29,304 |

The exact row identities are controlled by fixed seeds in the wrappers so the same missingness pattern can be regenerated.
