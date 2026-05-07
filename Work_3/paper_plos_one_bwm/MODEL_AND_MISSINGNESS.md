# PLOS One BWM Model And Missingness Details

This note explains the exact mechanics used by the local PLOS One BWM wrappers. It is based on the code in:

- [scripts/run_plos_bwm_main_data.R](scripts/run_plos_bwm_main_data.R)
- [run_main_data_experiment.R](run_main_data_experiment.R)
- [R/bwm_train.R](R/bwm_train.R)
- [R/bwm_predict.R](R/bwm_predict.R)
- [R/get_profile.R](R/get_profile.R)
- [R/getBlockSamples.R](R/getBlockSamples.R)

## What Generated Block Missingness Means

The CSV files in `main_data/` are not already organized as multiple sources with blockwise missingness. The wrapper first preprocesses a CSV into ordinary predictors, assigns predictors to source blocks, and then artificially creates missing values. This generated missingness is used so the same dataset can be tested at controlled missingness levels.

For a missingness rate `r`, such as `0.10`, the code does not set 10% of all cells missing independently. Instead, it samples row indices and removes whole source blocks for those rows.

The common rule is:

```text
n = number of rows after preprocessing
k = floor(r * n)
A_s = sample({1, ..., n}, size = k, replace = FALSE)

For every row i in A_s and every predictor j in targeted source s:
  X[i, j] = NA
```

The row sample is simple random sampling without replacement. Because the code resets the random seed before each sampled source, the generated rows are reproducible.

Each value in `--missing-props 0.1,0.2,...,0.9` creates a separate experiment. It is a stress-test grid:

- `0.1`: mild block missingness, about 10% of rows in each targeted source block.
- `0.5`: severe block missingness, about half the rows in each targeted source block.
- `0.9`: extreme block missingness, about 90% of rows in each targeted source block.

These values are not fitted or tuned by the model. They are experimental conditions.

## Dataset-Specific Missingness Rules

### Bike

Code locations:

- `gen_missing_bike()` in [scripts/run_plos_bwm_main_data.R](scripts/run_plos_bwm_main_data.R)
- `gen_missing_bike()` in [run_main_data_experiment.R](run_main_data_experiment.R)

Sources before one-hot encoding:

- Source 1: `windspeed`, `hum`, `ToD`, `weekday`
- Source 2: `hr`, `temp`, `temp2`, `weathersit`
- Source 3: `season`, `mnth`

Generated block missingness:

```text
set.seed(1234)
A_1 = sample rows of size floor(r * n)
Set Source 1 columns to NA for rows in A_1

set.seed(123)
A_2 = sample rows of size floor(r * n)
Set Source 2 columns to NA for rows in A_2
```

Then the code adds smaller per-column noise:

```text
For each column in Source 1 and Source 2:
  set.seed(12 + column_index)
  sample floor(0.05 * n) rows
  set that one column to NA
```

Source 3 is not directly block-missing in the bike generator.

Ignoring the added 5% per-column noise, the two block samples are drawn independently. So a row can lose Source 1, Source 2, both, or neither. Approximately:

```text
P(Source 1 missing) = r
P(Source 2 missing) = r
P(Source 1 and Source 2 missing) ~= r^2
P(Source 1 and Source 2 both observed) ~= (1 - r)^2
```

The actual counts are deterministic after the seeds are set.

### House

Sources before one-hot encoding:

- Source 1: `bedrooms`, `bathrooms`
- Source 2: `sqft_living`, `sqft_lot`
- Source 3: `floors`, `view`
- Source 4: `condition`, `yr_built`

Generated block missingness:

```text
set.seed(1234)
A_1 = sample rows of size floor(r * n)
Set columns 1:2, Source 1, to NA for rows in A_1

set.seed(111)
A_2 = sample rows of size floor(r * n)
Set columns 3:4, Source 2, to NA for rows in A_2

Set rows 2001:6000, columns 5:6, Source 3, to NA
```

Then the code adds smaller per-column noise:

```text
For every predictor column, excluding the outcome:
  set.seed(12 + column_index)
  sample floor(0.02 * n) rows
  set that one column to NA
```

The `r` parameter controls only the Source 1 and Source 2 block samples. Source 3 has a fixed missing block, and all predictors can receive 2% per-column noise.

### Adult

Sources before one-hot encoding:

- Source 1: `age`, `workclass`, `fnlwgt`
- Source 2: `marital.status`, `occupation`, `relationship`
- Source 3: `education`, `education.num`, `race`, `sex`, `hours.per.week`, `native.country`

Generated block missingness:

```text
set.seed(1234)
A_1 = sample rows of size floor(r * n)
Set columns 1:3, Source 1, to NA for rows in A_1

set.seed(111)
A_2 = sample rows of size floor(r * n)
Set columns 6:8, Source 2, to NA for rows in A_2
```

Then the code adds smaller per-column noise:

```text
For every predictor column, excluding the outcome:
  set.seed(12 + column_index)
  sample floor(0.02 * n) rows
  set that one column to NA
```

The raw adult file also contains missing placeholders. In the current wrappers, blank `education-num` values become R `NA`, but `?` markers in categorical columns are treated as factor levels and then recoded with other rare categories.

## Train/Test Protocol

There is a train/test split. There is no cross-validation used for the final reported metrics.

The order is:

1. Load and preprocess the full CSV.
2. Generate missingness for one missingness rate `r`.
3. Split the generated missing dataset into train and test.
4. Encode train and test predictors into numeric BWM matrices.
5. Train `bwm.train()` on the training matrix.
6. Predict with `bwm.predict()` on the test matrix.
7. Compute metrics on the test predictions.

The canonical runner [scripts/run_plos_bwm_main_data.R](scripts/run_plos_bwm_main_data.R) uses:

```text
train size = floor(0.75 * n)
test size = n - train size
set.seed(1234)
train rows = sample({1, ..., n}, train size)
test rows = all remaining rows
```

The Docker/root runner [run_main_data_experiment.R](run_main_data_experiment.R) uses the same default 75/25 split, but exposes it as:

```text
--test-fraction 0.25
--seed 1234
```

The split is row-based, not source-based. The test set still contains missing source blocks. BWM predicts from whatever sources are available for each test row.

The PLOS wrapper does not use k-fold cross-validation to estimate test metrics. Also, the lambda grid is not selected by validation-set or test-set performance. `bwm.train()` tries each lambda, optimizes on the training data, and stores the lambda that gives the best training objective value inside the BWM training routine.

The package contains a `cv.glmnet()` branch inside `beta.initialization()` for a different initialization mode, but these wrappers do not activate that mode. With the current wrappers, bike and house use the regression path, and adult uses the binary logistic path.

## Profiles

BWM represents each row by a missingness profile. A source is observed for a row only if every encoded column in that source has no `NA`.

For row `i` and source `s`:

```text
O[i, s] = 1 if all columns in source s are observed for row i
O[i, s] = 0 otherwise
```

The profile is encoded as a binary number. With `S` sources:

```text
profile[i] = sum over s=1..S of O[i, s] * 2^(S - s)
```

For a 3-source dataset:

| Observed sources | Binary | Profile number |
|---|---:|---:|
| none | `000` | 0 |
| Source 3 only | `001` | 1 |
| Source 2 only | `010` | 2 |
| Sources 2 and 3 | `011` | 3 |
| Source 1 only | `100` | 4 |
| Sources 1 and 3 | `101` | 5 |
| Sources 1 and 2 | `110` | 6 |
| Sources 1, 2, and 3 | `111` | 7 |

This is implemented in [R/get_profile.R](R/get_profile.R).

A profile is assigned to every row, but every row does not get a unique profile. Many rows can share the same profile. With `S` sources, the maximum number of possible profiles is `2^S`:

```text
S = 3  -> max 8 profiles: 0,...,7
S = 4  -> max 16 profiles: 0,...,15
S = 7  -> max 128 profiles: 0,...,127
```

The code only creates model parameters for profiles that actually appear in the training set: `levels(pf.vec)`. In the local wrappers, the source count is small: bike and adult use 3 sources, and house uses 4 sources.

## Blocks Used During Training

The function [R/getBlockSamples.R](R/getBlockSamples.R) is subtle. For a profile `m`, it does not return only rows with exactly profile `m`. It returns rows whose available sources include all sources required by profile `m`.

If `A(m)` is the set of sources observed in profile `m`, then:

```text
B_m = { row i : A(m) is a subset of A(profile[i]) }
```

Example with three sources:

- For profile `011`, Sources 2 and 3 are observed.
- `getBlockSamples()` returns rows with profile `011` and rows with profile `111`, because both contain Sources 2 and 3.

More generally:

| Requested profile `m` | Required sources `A(m)` | Training rows included in block `B_m` |
|---:|---|---|
| `001` | Source 3 | `001`, `011`, `101`, `111` |
| `010` | Source 2 | `010`, `011`, `110`, `111` |
| `011` | Sources 2 and 3 | `011`, `111` |
| `100` | Source 1 | `100`, `101`, `110`, `111` |
| `101` | Sources 1 and 3 | `101`, `111` |
| `110` | Sources 1 and 2 | `110`, `111` |
| `111` | Sources 1, 2, and 3 | `111` |

This is the combination step. The model has profile-specific weights, but the training block for a less-complete profile borrows all rows that have at least those required sources. It does not only use rows with the exact same missingness pattern.

If `m = 0`, no sources are available. In R, `getBlockSamples()` treats the empty source requirement as satisfied by all rows. At prediction time, however, no source contributes to the sum because there are no observed sources.

## What The Model Learns

BWM does not train a fully separate independent model for every block. It learns:

- `beta_s`: feature coefficients for each source `s`.
- `alpha_m`: profile-specific source weights for each observed profile `m`.

For regression, the prediction score for a row with profile `m` is:

```text
y_hat = sum over s in A(m) of alpha[m, s] * X_s * beta_s
```

Here `X_s * beta_s` means the linear predictor for source `s`.

For binary classification, the same weighted linear score is computed. The code converts class labels to `-1` and `1` during training, then predicts the class with the smaller logistic loss:

```text
loss(class c) = log(1 + exp(-score * c))
predicted class = class with smaller loss
```

For adult, the two class labels are `0` and `1`, so a positive score maps to class `1` and a negative score maps to class `0`.

## Training Algorithm

Training happens in [R/bwm_train.R](R/bwm_train.R).

The high-level loop is:

1. Optionally normalize each numeric feature column using training-data min/max, preserving `NA`.
2. Decide the loss:
   - bike and house use least-squares regression.
   - adult uses binary logistic loss.
3. Compute each training row's profile with `get_profile()`.
4. Initialize `alpha` from the observed profiles.
5. Initialize `beta` source by source:
   - regression uses robust linear regression through `MASS::rlm()`;
   - binary classification uses logistic regression through `glm()`.
6. For each lambda in the supplied lambda grid:
   - update `alpha` with beta fixed;
   - update `beta` with alpha fixed using proximal gradient updates;
   - repeat up to `maxIter` times or until the objective changes by less than `tol`.
7. Keep the alpha, beta, and lambda combination with the best training objective value.

The alpha update is profile-specific. The beta update is shared across sources. This is why a future row does not select an entirely separate model file; it selects a profile-specific alpha vector and combines it with the shared beta coefficients.

## Prediction On New Rows

Prediction happens in [R/bwm_predict.R](R/bwm_predict.R).

For each new row:

1. Apply the training normalization if the model was trained with normalization.

The normalization step is not recomputed on the test data. During training, `bwm.train()` stores one `translation` and `scale` value per encoded feature column. In this code, `translation` is the training-column minimum and `scale` is the training-column range. During prediction, `bwm.predict()` applies:

```text
X_new_scaled[, j] = (X_new[, j] - training_min[j]) / training_range[j]
```

This keeps train and test rows on the same scale. Missing values stay missing because arithmetic with `NA` remains `NA` in R.

2. Compute the row's observed-source profile with `get_profile()`.
3. Look for the same profile among profiles seen during training:

```text
model.profile.index = which(levels(model$profile.vector) == row_profile)
```

This line is copied from the package prediction code. It appears in [R/bwm_predict.R](R/bwm_predict.R) as:

```r
model.profile.index <- which(levels(model$profile.vector) == m)
```

In plain words:

- `model$profile.vector` is the training rows' profile vector stored inside the fitted model.
- `levels(model$profile.vector)` is the list of unique profile numbers that were present during training.
- `row_profile`, called `m` in the code, is the profile number for the new/test row.
- `which(...)` returns the position of that row profile in the training profile list.

Example:

```text
Training profiles seen: levels(model$profile.vector) = c("3", "5", "7")
New row profile: row_profile = 5
which(c("3", "5", "7") == 5) returns 2
```

So the model uses `model$alpha[[2]]`, the alpha vector learned for profile `5`.

4. If that exact profile was not seen during training, the prediction is `NA`.
5. If the profile was seen, use only the sources observed in that row:

```text
sources.profile = which(as.binary(row_profile, n = S))
```

Here "sources observed in that row" means the sources whose full source block is present for the new row. If a source has any missing value in that row, it is not used for that row's prediction.

Example with 3 sources:

```text
New row profile = 101
Observed sources = Source 1 and Source 3
Missing source = Source 2
```

For this row, Source 2 cannot contribute because its source block contains missing values. The model does not impute Source 2. It makes the prediction from Source 1 and Source 3 only.

6. Combine the source predictions with the stored `alpha` weights for that profile and the shared `beta` coefficients:

```text
source_score_s = X_s * beta_s
y_hat = sum over observed sources s of alpha[m, s] * source_score_s
```

In code, for regression, this is:

```r
y.pred[i] <- y.pred[i] +
  model$alpha[[model.profile.index]][j] *
  X[i, col:nextCol] %*% model$beta[col:nextCol]
```

For a row with profile `101`, the equation is:

```text
y_hat =
  alpha[profile 101, Source 1] * (X_Source1 * beta_Source1)
  +
  alpha[profile 101, Source 3] * (X_Source3 * beta_Source3)
```

Source 2 is excluded from the sum.

So for a new row, the selected "model" is determined by its missingness profile. The code does not impute the missing sources and does not delete the row. It computes the prediction from the sources that are present, using the alpha weights learned for that exact observed-source profile.

## Practical Reproducibility Notes

- Missingness is generated before the train/test split.
- Source-block row samples are simple random samples without replacement.
- The important missingness seeds are fixed in the generator functions:
  - Source 1 block sample: `1234`
  - Source 2 block sample: `123` for bike, `111` for house/adult
  - Noise columns: `12 + column_index`
- The canonical train/test split seed is fixed at `1234`.
- The Docker/root runner exposes the train/test split seed as `--seed`, default `1234`.
- There is no final metric cross-validation in these wrappers.
- Test rows with profiles never seen in training receive `NA` predictions and are counted as invalid predictions by the wrappers.
