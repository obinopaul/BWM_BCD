#!/usr/bin/env Rscript

options(stringsAsFactors = FALSE)
options(repos = c(CRAN = "https://cloud.r-project.org"))

file_args <- grep("^--file=", commandArgs(FALSE), value = TRUE)
if (length(file_args) == 0) {
  stop("Could not determine script path from commandArgs(). Run this file with Rscript.")
}
script_path <- sub("^--file=", "", file_args[1])
script_dir <- dirname(normalizePath(script_path, mustWork = TRUE))
repo_root <- normalizePath(file.path(script_dir, ".."), mustWork = TRUE)

parse_args <- function(args) {
  opts <- list(
    dataset = "all",
    data_dir = file.path(repo_root, "main_data"),
    output_dir = file.path(script_dir, "results"),
    missing_rates = "0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90",
    seed = "1234",
    test_fraction = "0.25",
    lambda = "0.01",
    max_iter = "30",
    max_iter_alpha = "10",
    max_iter_beta = "10",
    tol = "1e-8",
    normalize = "FALSE"
  )

  i <- 1
  while (i <= length(args)) {
    arg <- args[[i]]
    if (arg %in% c("-h", "--help")) {
      opts$help <- TRUE
      i <- i + 1
      next
    }
    if (!grepl("^--", arg)) {
      stop("Unexpected positional argument: ", arg)
    }

    if (grepl("=", arg, fixed = TRUE)) {
      key <- sub("^--", "", sub("=.*$", "", arg))
      value <- sub("^[^=]*=", "", arg)
    } else {
      key <- sub("^--", "", arg)
      if (key == "normalize") {
        value <- "TRUE"
      } else if (key == "no-normalize") {
        key <- "normalize"
        value <- "FALSE"
      } else {
        if (i == length(args)) {
          stop("Missing value for argument: ", arg)
        }
        value <- args[[i + 1]]
        i <- i + 1
      }
    }

    key <- gsub("-", "_", key)
    opts[[key]] <- value
    i <- i + 1
  }

  opts
}

usage <- function() {
  cat("
Usage:
  Rscript paper_plos_one_bwm/run_main_data_experiment.R [options]

Options:
  --dataset DATASET          bike, house, adult, all, or comma-separated list.
                             Default: all
  --data-dir PATH            Directory containing bike_all.csv, house.csv,
                             and adult.csv. Default: ../main_data
  --output-dir PATH          Directory for metrics CSV output.
                             Default: paper_plos_one_bwm/results
  --missing-rates LIST       Comma-separated rates. Default: 0.10,...,0.90
  --seed INT                 Train/test split seed. Default: 1234
  --test-fraction FLOAT      Held-out test fraction. Default: 0.25
  --lambda LIST              Comma-separated lambda values for bwm.train.
                             Default: 0.01
  --max-iter INT             Outer alpha/beta iterations. Default: 30
  --max-iter-alpha INT       Alpha subproblem iterations. Default: 10
  --max-iter-beta INT        Beta subproblem iterations. Default: 10
  --normalize                Use bwm.train(to.normalize = TRUE).
  --no-normalize             Use bwm.train(to.normalize = FALSE).

Examples:
  Rscript paper_plos_one_bwm/run_main_data_experiment.R --dataset bike --missing-rates 0.10
  Rscript paper_plos_one_bwm/run_main_data_experiment.R --dataset all --max-iter 50
")
}

parse_numeric_list <- function(text) {
  values <- trimws(strsplit(text, ",", fixed = TRUE)[[1]])
  as.numeric(values[nzchar(values)])
}

parse_dataset_list <- function(text) {
  values <- trimws(strsplit(text, ",", fixed = TRUE)[[1]])
  values <- values[nzchar(values)]
  if (length(values) == 1 && values == "all") {
    return(c("bike", "house", "adult"))
  }
  values
}

parse_bool <- function(text) {
  value <- tolower(trimws(as.character(text)))
  if (value %in% c("true", "t", "1", "yes", "y")) {
    return(TRUE)
  }
  if (value %in% c("false", "f", "0", "no", "n")) {
    return(FALSE)
  }
  stop("Could not parse logical value: ", text)
}

source_bwm_package_files <- function() {
  r_dir <- file.path(script_dir, "R")
  files <- sort(list.files(r_dir, pattern = "\\.R$", full.names = TRUE))
  for (path in files) {
    source(path, chdir = TRUE)
  }
}

sample_rows <- function(n, prop, seed) {
  set.seed(seed)
  sample(seq_len(n), size = floor(prop * n), replace = FALSE)
}

split_train_test <- function(data, test_fraction, seed) {
  train_fraction <- 1 - test_fraction
  set.seed(seed)
  train_ind <- sample(seq_len(nrow(data)), size = floor(train_fraction * nrow(data)))
  list(
    train = data[train_ind, , drop = FALSE],
    test = data[-train_ind, , drop = FALSE],
    train_indices = train_ind,
    test_indices = setdiff(seq_len(nrow(data)), train_ind)
  )
}

recode_levels <- function(ds, mp = 0.10, ml = 5) {
  factor_cols <- names(ds)[vapply(ds, is.factor, logical(1))]
  n <- nrow(ds)

  for (name in factor_cols) {
    counts <- table(ds[[name]])
    keep <- names(counts)[counts > mp * n]
    levels(ds[[name]])[!(levels(ds[[name]]) %in% keep)] <- "other"
  }

  for (name in factor_cols) {
    counts <- sort(table(ds[[name]]), decreasing = TRUE)
    keep <- names(counts)[seq_len(min(ml, length(counts)))]
    levels(ds[[name]])[!(levels(ds[[name]]) %in% keep)] <- "other"
  }

  ds
}

gen_missing_bike <- function(bike, prop_missing) {
  miss_1 <- sample_rows(nrow(bike), prop_missing, 1234)
  miss_2 <- sample_rows(nrow(bike), prop_missing, 123)

  bike_mis <- bike
  block_1 <- c("windspeed", "hum", "ToD", "weekday")
  block_2 <- c("hr", "temp", "temp2", "weathersit")
  block_vars <- c(block_1, block_2)

  bike_mis[miss_1, names(bike_mis) %in% block_1] <- NA
  bike_mis[miss_2, names(bike_mis) %in% block_2] <- NA

  miss_3 <- bike_mis[, names(bike_mis) %in% block_vars, drop = FALSE]
  miss_4 <- bike_mis[, !(names(bike_mis) %in% block_vars), drop = FALSE]

  for (col in seq_len(ncol(miss_3))) {
    set.seed(12 + col)
    miss_temp <- sample(seq_len(nrow(miss_3)), floor(nrow(miss_3) * 0.05))
    miss_3[miss_temp, col] <- NA
  }

  cbind(miss_3, miss_4)
}

load_bike_spec <- function(data_dir) {
  data_all <- read.csv(file.path(data_dir, "bike_all.csv"))
  bike <- data_all[, !names(data_all) %in% c(
    "instant", "dteday", "atemp", "yr", "workingday", "holiday"
  )]
  bike$weekday <- as.factor(bike$weekday)
  bike$weathersit[bike$weathersit == 4] <- 3
  bike$weathersit <- as.factor(bike$weathersit)
  bike$season <- as.factor(bike$season)
  bike$ToD <- cut(bike$hr, breaks = c(-0.01, 5, 10, 16, 19, 24))
  bike$temp <- bike$temp * (39 + 8) - 8
  bike$temp2 <- bike$temp^2

  source_1 <- c("windspeed", "hum", "ToD", "weekday")
  source_2 <- c("hr", "temp", "temp2", "weathersit")
  predictors <- setdiff(names(bike), "cnt")
  source_3 <- setdiff(predictors, c(source_1, source_2))

  list(
    data = bike,
    outcome = "cnt",
    task = "regression",
    generator = gen_missing_bike,
    source_groups = list(source1 = source_1, source2 = source_2, source3 = source_3)
  )
}

gen_missing_house <- function(house, prop_missing) {
  miss_1 <- sample_rows(nrow(house), prop_missing, 1234)
  miss_2 <- sample_rows(nrow(house), prop_missing, 111)

  house_mis <- house
  house_mis[miss_1, 1:2] <- NA
  house_mis[miss_2, 3:4] <- NA

  if (nrow(house_mis) >= 2001) {
    fixed_rows <- 2001:min(6000, nrow(house_mis))
    house_mis[fixed_rows, 5:6] <- NA
  }

  for (col in seq_len(ncol(house_mis) - 1)) {
    set.seed(12 + col)
    miss_temp <- sample(seq_len(nrow(house_mis)), floor(nrow(house_mis) * 0.02))
    house_mis[miss_temp, col] <- NA
  }

  house_mis
}

load_house_spec <- function(data_dir) {
  data_all <- read.csv(file.path(data_dir, "house.csv"))
  house <- data_all[, !names(data_all) %in% c(
    "sqft_basement", "yr_renovated", "zip", "sqft_above",
    "sqft_lot15", "grade", "waterfront", "sqft_living15"
  )]
  house <- house[house$price < quantile(house$price, 0.95), ]
  house$price <- house$price / 1000

  predictors <- setdiff(names(house), "price")
  source_1 <- names(house)[1:2]
  source_2 <- names(house)[3:4]
  source_3 <- names(house)[5:6]
  source_4 <- setdiff(predictors, c(source_1, source_2, source_3))

  list(
    data = house,
    outcome = "price",
    task = "regression",
    generator = gen_missing_house,
    source_groups = list(
      source1 = source_1,
      source2 = source_2,
      source3 = source_3,
      source4 = source_4
    )
  )
}

gen_missing_adult <- function(adult, prop_missing) {
  miss_1 <- sample_rows(nrow(adult), prop_missing, 1234)
  miss_2 <- sample_rows(nrow(adult), prop_missing, 111)

  adult_mis <- adult
  adult_mis[miss_1, 1:3] <- NA
  adult_mis[miss_2, 6:8] <- NA

  for (col in seq_len(ncol(adult_mis) - 1)) {
    set.seed(12 + col)
    miss_temp <- sample(seq_len(nrow(adult_mis)), floor(nrow(adult_mis) * 0.02))
    adult_mis[miss_temp, col] <- NA
  }

  adult_mis
}

load_adult_spec <- function(data_dir) {
  adult <- read.csv(file.path(data_dir, "adult.csv"), check.names = TRUE)
  adult <- as.data.frame(unclass(adult), stringsAsFactors = TRUE)

  salary_text <- tolower(trimws(as.character(adult$salary)))
  adult$salary <- ifelse(salary_text %in% c(">=50k", ">50k", ">=50k."), 1, 0)
  adult <- adult[, !names(adult) %in% c("fnlwt", "capital.gain", "capital.loss")]
  adult <- recode_levels(adult)

  predictors <- setdiff(names(adult), "salary")
  source_1 <- names(adult)[1:3]
  source_2 <- names(adult)[6:8]
  source_3 <- setdiff(predictors, c(source_1, source_2))

  list(
    data = adult,
    outcome = "salary",
    task = "classification",
    generator = gen_missing_adult,
    source_groups = list(source1 = source_1, source2 = source_2, source3 = source_3)
  )
}

load_dataset_spec <- function(dataset, data_dir) {
  switch(
    dataset,
    bike = load_bike_spec(data_dir),
    house = load_house_spec(data_dir),
    adult = load_adult_spec(data_dir),
    stop("Unsupported dataset: ", dataset)
  )
}

new_unique_name_builder <- function() {
  used <- character()
  function(base) {
    candidate <- make.names(base)
    candidate <- make.unique(c(used, candidate), sep = "_")[length(used) + 1]
    used <<- c(used, candidate)
    candidate
  }
}

encode_predictors <- function(train, test, outcome, source_groups) {
  predictors <- setdiff(names(train), outcome)
  source_predictors <- unlist(source_groups, use.names = FALSE)
  unknown <- setdiff(source_predictors, predictors)
  uncovered <- setdiff(predictors, source_predictors)
  duplicated_predictors <- names(table(source_predictors))[table(source_predictors) > 1]

  if (length(unknown) > 0) {
    stop("Source groups reference missing predictors: ", paste(unknown, collapse = ", "))
  }
  if (length(uncovered) > 0) {
    stop("Source groups do not cover predictors: ", paste(uncovered, collapse = ", "))
  }
  if (length(duplicated_predictors) > 0) {
    stop("Predictors appear in multiple source groups: ", paste(duplicated_predictors, collapse = ", "))
  }

  next_name <- new_unique_name_builder()
  train_cols <- list()
  test_cols <- list()
  encoded_by_original <- list()

  for (name in predictors) {
    train_values <- train[[name]]
    test_values <- test[[name]]
    encoded_names <- character()

    if (is.factor(train_values) || is.character(train_values)) {
      train_chr <- as.character(train_values)
      test_chr <- as.character(test_values)
      levels_train <- sort(unique(train_chr[!is.na(train_chr)]))
      if (length(levels_train) == 0) {
        levels_train <- "__empty_level__"
      }

      for (level in levels_train) {
        col_name <- next_name(paste(name, level, sep = "__"))
        train_cols[[col_name]] <- ifelse(
          is.na(train_chr),
          NA_real_,
          as.numeric(train_chr == level)
        )
        test_cols[[col_name]] <- ifelse(
          is.na(test_chr),
          NA_real_,
          as.numeric(test_chr == level)
        )
        encoded_names <- c(encoded_names, col_name)
      }
    } else {
      col_name <- next_name(name)
      train_cols[[col_name]] <- as.numeric(train_values)
      test_cols[[col_name]] <- as.numeric(test_values)
      encoded_names <- c(encoded_names, col_name)
    }

    encoded_by_original[[name]] <- encoded_names
  }

  X_train <- data.frame(train_cols, check.names = FALSE)
  X_test <- data.frame(test_cols, check.names = FALSE)

  source_columns <- list()
  p <- integer()

  for (source_name in names(source_groups)) {
    original_cols <- source_groups[[source_name]]
    encoded_cols <- unlist(encoded_by_original[original_cols], use.names = FALSE)

    train_source_missing <- !complete.cases(train[, original_cols, drop = FALSE])
    test_source_missing <- !complete.cases(test[, original_cols, drop = FALSE])

    X_train[train_source_missing, encoded_cols] <- NA_real_
    X_test[test_source_missing, encoded_cols] <- NA_real_

    source_columns[[source_name]] <- encoded_cols
    p <- c(p, length(encoded_cols))
  }

  ordered_cols <- unlist(source_columns, use.names = FALSE)
  X_train <- X_train[, ordered_cols, drop = FALSE]
  X_test <- X_test[, ordered_cols, drop = FALSE]
  names(p) <- names(source_groups)

  list(
    X_train = as.matrix(X_train),
    X_test = as.matrix(X_test),
    y_train = train[[outcome]],
    y_test = test[[outcome]],
    p = p,
    source_columns = source_columns
  )
}

safe_divide <- function(numerator, denominator) {
  if (is.na(denominator) || denominator == 0) {
    return(NA_real_)
  }
  numerator / denominator
}

regression_metrics <- function(y_true, y_pred) {
  y_true <- as.numeric(y_true)
  y_pred <- as.numeric(y_pred)
  valid <- is.finite(y_true) & is.finite(y_pred)

  if (!any(valid)) {
    return(data.frame(
      n_valid = 0, rmse = NA_real_, mae = NA_real_, smape = NA_real_,
      mape = NA_real_, r_squared = NA_real_
    ))
  }

  y_true <- y_true[valid]
  y_pred <- y_pred[valid]
  error <- y_true - y_pred
  rmse <- sqrt(mean(error^2))
  mae <- mean(abs(error))
  smape <- mean(2 * abs(error) / (abs(y_true) + abs(y_pred)), na.rm = TRUE) * 100
  mape <- mean(abs(error / y_true), na.rm = TRUE) * 100
  ss_res <- sum(error^2)
  ss_tot <- sum((y_true - mean(y_true))^2)
  r_squared <- 1 - ss_res / ss_tot

  data.frame(
    n_valid = length(y_true),
    rmse = rmse,
    mae = mae,
    smape = smape,
    mape = mape,
    r_squared = r_squared
  )
}

classification_metrics <- function(y_true, y_pred) {
  y_true <- as.character(y_true)
  y_pred <- as.character(y_pred)
  valid <- !is.na(y_true) & !is.na(y_pred)

  if (!any(valid)) {
    return(data.frame(
      n_valid = 0, accuracy = NA_real_, balanced_accuracy = NA_real_,
      precision = NA_real_, recall = NA_real_, specificity = NA_real_,
      f1 = NA_real_, gmean = NA_real_, mcc = NA_real_, hard_label_auc = NA_real_
    ))
  }

  y_true <- y_true[valid]
  y_pred <- y_pred[valid]
  classes <- sort(unique(c(y_true, y_pred)))
  accuracy <- mean(y_true == y_pred)

  if (length(classes) != 2) {
    return(data.frame(
      n_valid = length(y_true),
      accuracy = accuracy,
      balanced_accuracy = NA_real_,
      precision = NA_real_,
      recall = NA_real_,
      specificity = NA_real_,
      f1 = NA_real_,
      gmean = NA_real_,
      mcc = NA_real_,
      hard_label_auc = NA_real_
    ))
  }

  positive <- classes[[2]]
  negative <- classes[[1]]
  tp <- sum(y_pred == positive & y_true == positive)
  tn <- sum(y_pred == negative & y_true == negative)
  fp <- sum(y_pred == positive & y_true == negative)
  fn <- sum(y_pred == negative & y_true == positive)

  precision <- safe_divide(tp, tp + fp)
  recall <- safe_divide(tp, tp + fn)
  specificity <- safe_divide(tn, tn + fp)
  balanced_accuracy <- mean(c(recall, specificity), na.rm = TRUE)
  f1 <- safe_divide(2 * precision * recall, precision + recall)
  gmean <- sqrt(recall * specificity)
  denom <- sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
  mcc <- safe_divide(tp * tn - fp * fn, denom)

  data.frame(
    n_valid = length(y_true),
    accuracy = accuracy,
    balanced_accuracy = balanced_accuracy,
    precision = precision,
    recall = recall,
    specificity = specificity,
    f1 = f1,
    gmean = gmean,
    mcc = mcc,
    hard_label_auc = balanced_accuracy
  )
}

format_source_groups <- function(source_groups) {
  paste(
    sprintf("%s=[%s]", names(source_groups), vapply(source_groups, paste, "", collapse = "|")),
    collapse = "; "
  )
}

run_one_experiment <- function(dataset, missing_rate, opts) {
  spec <- load_dataset_spec(dataset, opts$data_dir)
  generated <- spec$generator(spec$data, missing_rate)
  split <- split_train_test(
    generated,
    test_fraction = as.numeric(opts$test_fraction),
    seed = as.integer(opts$seed)
  )
  encoded <- encode_predictors(
    split$train,
    split$test,
    outcome = spec$outcome,
    source_groups = spec$source_groups
  )

  model <- bwm.train(
    p = encoded$p,
    X = encoded$X_train,
    y = encoded$y_train,
    lambda = parse_numeric_list(opts$lambda),
    maxIter = as.integer(opts$max_iter),
    maxIter.alpha = as.integer(opts$max_iter_alpha),
    maxIter.beta = as.integer(opts$max_iter_beta),
    tol = as.numeric(opts$tol),
    to.normalize = parse_bool(opts$normalize),
    verbose = FALSE
  )

  predictions <- bwm.predict(
    model = model,
    X = encoded$X_test,
    p = encoded$p,
    verbose = FALSE
  )

  metrics <- if (spec$task == "classification") {
    classification_metrics(encoded$y_test, predictions)
  } else {
    regression_metrics(encoded$y_test, predictions)
  }

  data.frame(
    dataset = dataset,
    task = spec$task,
    missing_rate = missing_rate,
    status = "ok",
    error_message = "",
    n_rows_generated = nrow(generated),
    n_train = nrow(split$train),
    n_test = nrow(split$test),
    n_predictor_columns_encoded = ncol(encoded$X_train),
    p = paste(encoded$p, collapse = "|"),
    selected_lambda = model$lambda,
    source_groups = format_source_groups(spec$source_groups),
    metrics,
    check.names = FALSE
  )
}

bind_rows_base <- function(rows) {
  all_names <- unique(unlist(lapply(rows, names), use.names = FALSE))
  normalized <- lapply(rows, function(row) {
    missing_names <- setdiff(all_names, names(row))
    for (name in missing_names) {
      row[[name]] <- NA
    }
    row[, all_names, drop = FALSE]
  })
  do.call(rbind, normalized)
}

main <- function() {
  opts <- parse_args(commandArgs(trailingOnly = TRUE))
  if (isTRUE(opts$help)) {
    usage()
    quit(status = 0)
  }

  opts$data_dir <- normalizePath(opts$data_dir, mustWork = TRUE)
  dir.create(opts$output_dir, recursive = TRUE, showWarnings = FALSE)

  source_bwm_package_files()

  datasets <- parse_dataset_list(opts$dataset)
  missing_rates <- parse_numeric_list(opts$missing_rates)
  rows <- list()

  for (dataset in datasets) {
    for (missing_rate in missing_rates) {
      message(sprintf("Running PLOS BWM: dataset=%s missing_rate=%.2f", dataset, missing_rate))
      row <- tryCatch(
        run_one_experiment(dataset, missing_rate, opts),
        error = function(e) {
          data.frame(
            dataset = dataset,
            task = NA_character_,
            missing_rate = missing_rate,
            status = "error",
            error_message = conditionMessage(e),
            check.names = FALSE
          )
        }
      )
      rows[[length(rows) + 1]] <- row
    }
  }

  results <- bind_rows_base(rows)
  timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
  output_path <- file.path(opts$output_dir, paste0("plos_bwm_main_data_metrics_", timestamp, ".csv"))
  write.csv(results, output_path, row.names = FALSE)
  message("Wrote metrics: ", output_path)
}

main()
