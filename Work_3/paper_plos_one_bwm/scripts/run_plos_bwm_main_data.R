#!/usr/bin/env Rscript

default_missing_props <- seq(0.1, 0.9, by = 0.1)

parse_args <- function(args) {
  parsed <- list(
    dataset = "all",
    missing_props = default_missing_props,
    lambda_grid = c(0, 1e-4, 1e-3, 1e-2, 1e-1),
    max_iter = 100,
    max_iter_alpha = 20,
    max_iter_beta = 20,
    tol = 1e-8,
    normalize = TRUE,
    seed = 1234,
    results_dir = NA_character_
  )

  parse_numeric_csv <- function(value) {
    as.numeric(trimws(strsplit(value, ",", fixed = TRUE)[[1]]))
  }

  i <- 1
  while (i <= length(args)) {
    key <- args[[i]]
    value <- if (i < length(args)) args[[i + 1]] else NA_character_

    if (key == "--dataset") {
      parsed$dataset <- value
      i <- i + 2
    } else if (key == "--missing-props") {
      parsed$missing_props <- parse_numeric_csv(value)
      i <- i + 2
    } else if (key == "--lambda-grid") {
      parsed$lambda_grid <- parse_numeric_csv(value)
      i <- i + 2
    } else if (key == "--max-iter") {
      parsed$max_iter <- as.integer(value)
      i <- i + 2
    } else if (key == "--max-iter-alpha") {
      parsed$max_iter_alpha <- as.integer(value)
      i <- i + 2
    } else if (key == "--max-iter-beta") {
      parsed$max_iter_beta <- as.integer(value)
      i <- i + 2
    } else if (key == "--tol") {
      parsed$tol <- as.numeric(value)
      i <- i + 2
    } else if (key == "--normalize") {
      parsed$normalize <- tolower(value) %in% c("1", "true", "yes", "y")
      i <- i + 2
    } else if (key == "--seed") {
      parsed$seed <- as.integer(value)
      i <- i + 2
    } else if (key == "--results-dir") {
      parsed$results_dir <- value
      i <- i + 2
    } else if (key %in% c("-h", "--help")) {
      cat(
        "Usage: Rscript paper_plos_one_bwm/scripts/run_plos_bwm_main_data.R [options]\n\n",
        "Options:\n",
        "  --dataset all|bike|house|adult        Dataset(s) to run. Default: all\n",
        "  --missing-props 0.1,0.2,...           Missing proportions. Default: 0.1,...,0.9\n",
        "  --lambda-grid 0,1e-4,1e-3,1e-2,1e-1  Lambda grid passed to bwm.train.\n",
        "  --max-iter N                          Outer bwm.train iterations. Default: 100\n",
        "  --max-iter-alpha N                    Alpha subproblem iterations. Default: 20\n",
        "  --max-iter-beta N                     Beta subproblem iterations. Default: 20\n",
        "  --tol FLOAT                           Convergence tolerance. Default: 1e-8\n",
        "  --normalize true|false                Use bwm.train feature normalization. Default: true\n",
        "  --seed INT                            General run seed. Default: 1234\n",
        "  --results-dir PATH                    Output directory. Default: paper_plos_one_bwm/results/run_<timestamp>\n",
        sep = ""
      )
      quit(status = 0)
    } else {
      stop(sprintf("Unknown argument: %s", key))
    }
  }

  parsed
}

script_path <- function() {
  file_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  if (length(file_arg) == 0) {
    return(normalizePath("paper_plos_one_bwm/scripts/run_plos_bwm_main_data.R", mustWork = FALSE))
  }
  normalizePath(sub("^--file=", "", file_arg[[1]]), mustWork = TRUE)
}

timestamp <- function() {
  format(Sys.time(), "%Y%m%d_%H%M%S")
}

resolve_datasets <- function(value) {
  if (tolower(value) == "all") {
    return(c("bike", "house", "adult"))
  }
  datasets <- trimws(strsplit(value, ",", fixed = TRUE)[[1]])
  valid <- c("bike", "house", "adult")
  bad <- setdiff(datasets, valid)
  if (length(bad) > 0) {
    stop(sprintf("Unsupported dataset(s): %s", paste(bad, collapse = ", ")))
  }
  datasets
}

source_bwm_package <- function(paper_dir) {
  r_files <- list.files(file.path(paper_dir, "R"), pattern = "\\.R$", full.names = TRUE)
  for (file in sort(r_files)) {
    sys.source(file, envir = globalenv())
  }
}

tr_test_st <- function(data_in) {
  smp_size <- floor(0.75 * nrow(data_in))
  set.seed(1234)
  train_ind <- sample(seq_len(nrow(data_in)), size = smp_size)
  list(train = data_in[train_ind, ], test = data_in[-train_ind, ])
}

gen_missing_bike <- function(bike, propMissing) {
  set.seed(1234)
  miss_1 <- sample(nrow(bike), propMissing * nrow(bike), replace = FALSE)
  set.seed(123)
  miss_2 <- sample(nrow(bike), propMissing * nrow(bike), replace = FALSE)

  bike_mis <- bike
  bike_mis[miss_1, names(bike_mis) %in% c("windspeed", "hum", "ToD", "weekday")] <- NA
  bike_mis[miss_2, names(bike_mis) %in% c("hr", "temp", "temp2", "weathersit")] <- NA

  miss_3 <- bike_mis[, names(bike_mis) %in% c("ToD", "hr", "weathersit", "windspeed", "temp", "temp2", "hum", "weekday")]
  miss_4 <- bike_mis[, !names(bike_mis) %in% c("ToD", "hr", "weathersit", "windspeed", "temp", "temp2", "hum", "weekday")]

  for (col in seq_len(ncol(miss_3))) {
    set.seed(12 + col)
    miss_temp <- sample(nrow(miss_3), nrow(miss_3) * 0.05, replace = FALSE)
    miss_3[miss_temp, col] <- NA
  }

  cbind(miss_3, miss_4)
}

gen_missing_house <- function(house, propMissing) {
  set.seed(1234)
  miss_1 <- sample(nrow(house), propMissing * nrow(house), replace = FALSE)
  set.seed(111)
  miss_2 <- sample(nrow(house), propMissing * nrow(house), replace = FALSE)

  house_mis <- house
  house_mis[miss_1, 1:2] <- NA
  house_mis[miss_2, 3:4] <- NA
  house_mis[2001:6000, 5:6] <- NA

  for (col in seq_len(ncol(house_mis) - 1)) {
    set.seed(12 + col)
    miss_temp <- sample(nrow(house_mis), nrow(house_mis) * 0.02, replace = FALSE)
    house_mis[miss_temp, col] <- NA
  }

  house_mis
}

gen_missing_adult <- function(adult, propMissing) {
  set.seed(1234)
  miss_1 <- sample(nrow(adult), propMissing * nrow(adult), replace = FALSE)
  set.seed(111)
  miss_2 <- sample(nrow(adult), propMissing * nrow(adult), replace = FALSE)

  adult_mis <- adult
  adult_mis[miss_1, 1:3] <- NA
  adult_mis[miss_2, 6:8] <- NA

  for (col in seq_len(ncol(adult_mis) - 1)) {
    set.seed(12 + col)
    miss_temp <- sample(nrow(adult_mis), nrow(adult_mis) * 0.02, replace = FALSE)
    adult_mis[miss_temp, col] <- NA
  }

  adult_mis
}

recodeLevels <- function(ds, mp = 0.10, ml = 5) {
  var_list <- names(ds[sapply(ds, is.factor)])
  n <- nrow(ds)
  for (name in var_list) {
    keep <- levels(ds[[name]])[table(ds[[name]]) > mp * n]
    levels(ds[[name]])[which(!levels(ds[[name]]) %in% keep)] <- "other"
  }
  for (name in var_list) {
    keep <- names(sort(table(ds[name]), decreasing = TRUE)[1:ml])
    levels(ds[[name]])[which(!levels(ds[[name]]) %in% keep)] <- "other"
  }
  ds
}

prepare_dataset <- function(dataset, data_dir, missing_props) {
  if (dataset == "bike") {
    data_all <- read.csv(file.path(data_dir, "bike_all.csv"))
    bike <- data_all[, !names(data_all) %in% c("instant", "dteday", "atemp", "yr", "workingday", "holiday")]
    bike$weekday <- as.factor(bike$weekday)
    bike$weathersit[bike$weathersit == 4] <- 3
    bike$weathersit <- as.factor(bike$weathersit)
    bike$season <- as.factor(bike$season)
    bike$ToD <- cut(bike$hr, breaks = c(-0.01, 5, 10, 16, 19, 24))
    bike$temp <- bike$temp * (39 + 8) - 8
    bike$temp2 <- bike$temp^2
    return(list(
      complete = bike,
      outcome = "cnt",
      task = "regression",
      missing = lapply(missing_props, function(prop) gen_missing_bike(bike, prop))
    ))
  }

  if (dataset == "house") {
    data_all <- read.csv(file.path(data_dir, "house.csv"))
    house <- data_all[, !names(data_all) %in% c(
      "sqft_basement", "yr_renovated", "zip", "sqft_above",
      "sqft_lot15", "grade", "waterfront", "sqft_living15"
    )]
    house <- house[house$price < quantile(house$price, 0.95), ]
    house$price <- house$price / 1000
    return(list(
      complete = house,
      outcome = "price",
      task = "regression",
      missing = lapply(missing_props, function(prop) gen_missing_house(house, prop))
    ))
  }

  if (dataset == "adult") {
    data_all <- read.csv(file.path(data_dir, "adult.csv"))
    adult <- as.data.frame(unclass(data_all), stringsAsFactors = TRUE)
    adult$salary <- ifelse(adult$salary == ">=50k", 1, 0)
    adult <- adult[, !names(adult) %in% c("fnlwt", "capital.gain", "capital.loss")]
    adult <- recodeLevels(adult)
    return(list(
      complete = adult,
      outcome = "salary",
      task = "classification",
      missing = lapply(missing_props, function(prop) gen_missing_adult(adult, prop))
    ))
  }

  stop(sprintf("Unsupported dataset: %s", dataset))
}

fit_encoder <- function(train_x) {
  lapply(train_x, function(col) {
    if (is.factor(col) || is.character(col)) {
      values <- as.factor(col)
      list(type = "factor", levels = levels(values))
    } else {
      list(type = "numeric")
    }
  })
}

transform_with_encoder <- function(data_x, encoder) {
  blocks <- list()
  names_out <- character()

  for (name in names(encoder)) {
    spec <- encoder[[name]]
    col <- data_x[[name]]

    if (spec$type == "numeric") {
      values <- as.numeric(col)
      blocks[[length(blocks) + 1]] <- matrix(values, ncol = 1)
      names_out <- c(names_out, name)
    } else {
      factor_col <- factor(col, levels = spec$levels)
      mat <- matrix(0, nrow = length(factor_col), ncol = length(spec$levels))
      for (level_idx in seq_along(spec$levels)) {
        mat[, level_idx] <- ifelse(is.na(factor_col), NA, as.integer(factor_col == spec$levels[[level_idx]]))
      }
      blocks[[length(blocks) + 1]] <- mat
      names_out <- c(names_out, paste(name, make.names(spec$levels), sep = "__"))
    }
  }

  out <- do.call(cbind, blocks)
  colnames(out) <- names_out
  out
}

source_groups <- function(dataset, feature_names) {
  if (dataset == "bike") {
    groups <- list(
      weather_time_a = c("windspeed", "hum", "ToD", "weekday"),
      weather_time_b = c("hr", "temp", "temp2", "weathersit"),
      remaining = setdiff(feature_names, c("windspeed", "hum", "ToD", "weekday", "hr", "temp", "temp2", "weathersit"))
    )
  } else if (dataset == "house") {
    groups <- list(
      house_block_1 = feature_names[1:2],
      house_block_2 = feature_names[3:4],
      house_block_3 = feature_names[5:6],
      remaining = feature_names[-c(1:6)]
    )
  } else if (dataset == "adult") {
    groups <- list(
      adult_block_1 = feature_names[1:3],
      adult_block_2 = feature_names[6:8],
      remaining = feature_names[-c(1:3, 6:8)]
    )
  } else {
    stop(sprintf("Unsupported dataset: %s", dataset))
  }

  groups[lengths(groups) > 0]
}

encoded_source_indices <- function(groups, encoded_names) {
  indices <- lapply(groups, function(cols) {
    matches <- vapply(cols, function(prefix) {
      encoded_names == prefix | startsWith(encoded_names, paste0(prefix, "__"))
    }, logical(length(encoded_names)))
    if (is.null(dim(matches))) {
      matches <- matrix(matches, ncol = 1)
    }
    which(rowSums(matches) > 0)
  })

  if (any(lengths(indices) == 0)) {
    empty <- names(indices)[lengths(indices) == 0]
    stop(sprintf("Encoded source group(s) have no columns: %s", paste(empty, collapse = ", ")))
  }

  indices
}

make_bwm_matrix <- function(train, test, outcome, groups) {
  train_x <- train[, !names(train) %in% outcome, drop = FALSE]
  test_x <- test[, !names(test) %in% outcome, drop = FALSE]
  encoder <- fit_encoder(train_x)
  x_train_encoded <- transform_with_encoder(train_x, encoder)
  x_test_encoded <- transform_with_encoder(test_x, encoder)
  index_groups <- encoded_source_indices(groups, colnames(x_train_encoded))
  ordered_indices <- unlist(index_groups, use.names = FALSE)

  list(
    X_train = x_train_encoded[, ordered_indices, drop = FALSE],
    X_test = x_test_encoded[, ordered_indices, drop = FALSE],
    y_train = train[[outcome]],
    y_test = test[[outcome]],
    p = as.integer(lengths(index_groups)),
    encoded_source_names = names(index_groups)
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
  error <- y_true - y_pred
  rmse <- sqrt(mean(error^2))
  mae <- mean(abs(error))
  mape <- mean(abs(error / y_true)) * 100
  smape <- mean(2 * abs(error) / (abs(y_true) + abs(y_pred))) * 100
  r2 <- 1 - sum(error^2) / sum((y_true - mean(y_true))^2)
  c(rmse = rmse, mae = mae, mape = mape, smape = smape, r2 = r2)
}

classification_metrics <- function(y_true, y_pred) {
  y_true <- as.integer(as.character(y_true))
  y_pred <- as.integer(as.character(y_pred))
  tp <- sum(y_pred == 1 & y_true == 1)
  tn <- sum(y_pred == 0 & y_true == 0)
  fp <- sum(y_pred == 1 & y_true == 0)
  fn <- sum(y_pred == 0 & y_true == 1)

  accuracy <- safe_divide(tp + tn, length(y_true))
  precision <- safe_divide(tp, tp + fp)
  recall <- safe_divide(tp, tp + fn)
  specificity <- safe_divide(tn, tn + fp)
  f1 <- safe_divide(2 * precision * recall, precision + recall)
  balanced_accuracy <- mean(c(recall, specificity), na.rm = TRUE)
  gmean <- sqrt(recall * specificity)
  mcc <- safe_divide(
    tp * tn - fp * fn,
    sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
  )

  c(
    accuracy = accuracy,
    balanced_accuracy = balanced_accuracy,
    recall = recall,
    specificity = specificity,
    precision = precision,
    f1 = f1,
    gmean = gmean,
    mcc = mcc,
    hard_label_auc = balanced_accuracy
  )
}

metric_rows <- function(dataset, task, missing_prop, metrics, n_train, n_test, n_invalid, extra = list()) {
  data.frame(
    paper = "PLOS_ONE_BWM",
    dataset = dataset,
    method = "bwm",
    model_family = ifelse(task == "classification", "classification", "regression"),
    missing_prop = missing_prop,
    metric = names(metrics),
    value = as.numeric(metrics),
    n_train = n_train,
    n_test = n_test,
    n_invalid_predictions = n_invalid,
    best_lambda = if (!is.null(extra$best_lambda)) extra$best_lambda else NA_real_,
    source_count = if (!is.null(extra$source_count)) extra$source_count else NA_integer_,
    stringsAsFactors = FALSE
  )
}

error_row <- function(dataset, task, missing_prop, message, n_train = NA_integer_, n_test = NA_integer_) {
  data.frame(
    paper = "PLOS_ONE_BWM",
    dataset = dataset,
    method = "bwm",
    model_family = ifelse(task == "classification", "classification", "regression"),
    missing_prop = missing_prop,
    metric = "error",
    value = NA_real_,
    n_train = n_train,
    n_test = n_test,
    n_invalid_predictions = NA_integer_,
    best_lambda = NA_real_,
    source_count = NA_integer_,
    error = message,
    stringsAsFactors = FALSE
  )
}

run_one_plos_case <- function(dataset, task, outcome, data_in, missing_prop, args) {
  split <- tr_test_st(data_in)
  train <- split$train
  test <- split$test
  feature_names <- names(train)[!names(train) %in% outcome]
  groups <- source_groups(dataset, feature_names)
  bwm_data <- make_bwm_matrix(train, test, outcome, groups)

  result <- tryCatch({
    model <- bwm.train(
      p = bwm_data$p,
      X = bwm_data$X_train,
      y = bwm_data$y_train,
      lambda = args$lambda_grid,
      maxIter = args$max_iter,
      tol = args$tol,
      maxIter.alpha = args$max_iter_alpha,
      maxIter.beta = args$max_iter_beta,
      to.normalize = args$normalize,
      verbose = FALSE
    )

    pred <- bwm.predict(model, bwm_data$X_test, bwm_data$p, verbose = FALSE)
    valid <- !is.na(pred)
    n_invalid <- sum(!valid)

    if (!any(valid)) {
      return(error_row(dataset, task, missing_prop, "bwm.predict returned no valid predictions", nrow(train), nrow(test)))
    }

    metrics <- if (task == "classification") {
      classification_metrics(bwm_data$y_test[valid], pred[valid])
    } else {
      regression_metrics(bwm_data$y_test[valid], pred[valid])
    }

    metric_rows(
      dataset = dataset,
      task = task,
      missing_prop = missing_prop,
      metrics = metrics,
      n_train = nrow(train),
      n_test = nrow(test),
      n_invalid = n_invalid,
      extra = list(best_lambda = model$lambda, source_count = length(bwm_data$p))
    )
  }, error = function(e) {
    error_row(dataset, task, missing_prop, conditionMessage(e), nrow(train), nrow(test))
  })

  result
}

run_dataset <- function(dataset, data_dir, args) {
  prepared <- prepare_dataset(dataset, data_dir, args$missing_props)
  rows <- list()

  for (i in seq_along(args$missing_props)) {
    prop <- args$missing_props[[i]]
    message(sprintf("Running PLOS BWM for %s at missing_prop=%.2f", dataset, prop))
    rows[[length(rows) + 1]] <- run_one_plos_case(
      dataset = dataset,
      task = prepared$task,
      outcome = prepared$outcome,
      data_in = prepared$missing[[i]],
      missing_prop = prop,
      args = args
    )
  }

  do.call(rbind, rows)
}

main <- function() {
  args <- parse_args(commandArgs(trailingOnly = TRUE))
  script <- script_path()
  paper_dir <- normalizePath(file.path(dirname(script), ".."), mustWork = TRUE)
  repo_root <- normalizePath(file.path(paper_dir, ".."), mustWork = TRUE)
  data_dir <- file.path(repo_root, "main_data")
  output_dir <- if (is.na(args$results_dir)) {
    file.path(paper_dir, "results", paste0("run_", timestamp()))
  } else {
    args$results_dir
  }
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

  source_bwm_package(paper_dir)
  set.seed(args$seed)

  datasets <- resolve_datasets(args$dataset)
  all_rows <- lapply(datasets, run_dataset, data_dir = data_dir, args = args)
  results <- do.call(rbind, all_rows)

  write.csv(results, file.path(output_dir, "plos_bwm_main_data_metrics.csv"), row.names = FALSE)
  write.csv(
    data.frame(
      option = names(args),
      value = vapply(args, function(x) paste(x, collapse = ","), character(1)),
      stringsAsFactors = FALSE
    ),
    file.path(output_dir, "run_config.csv"),
    row.names = FALSE
  )

  message(sprintf("Saved PLOS BWM results to: %s", normalizePath(output_dir, mustWork = FALSE)))
}

main()
