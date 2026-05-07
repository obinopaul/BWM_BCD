#!/usr/bin/env Rscript

parse_args <- function(args) {
  parsed <- list(
    informs_results_dir = NA_character_,
    plos_results_dir = NA_character_,
    output = NA_character_
  )

  i <- 1
  while (i <= length(args)) {
    key <- args[[i]]
    value <- if (i < length(args)) args[[i + 1]] else NA_character_
    if (key == "--informs-results-dir") {
      parsed$informs_results_dir <- value
      i <- i + 2
    } else if (key == "--plos-results-dir") {
      parsed$plos_results_dir <- value
      i <- i + 2
    } else if (key == "--output") {
      parsed$output <- value
      i <- i + 2
    } else if (key %in% c("-h", "--help")) {
      cat(
        "Usage: Rscript scripts/collect_main_data_comparison.R [options]\n\n",
        "Options:\n",
        "  --informs-results-dir PATH  INFORMS result directory. Default: latest run_* if present, else paper_informs_reduced/results\n",
        "  --plos-results-dir PATH     PLOS result directory. Default: latest run_* if present, else paper_plos_one_bwm/results\n",
        "  --output PATH               Output CSV. Default: comparison_results/main_data_model_metrics.csv\n",
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
    return(normalizePath("scripts/collect_main_data_comparison.R", mustWork = FALSE))
  }
  normalizePath(sub("^--file=", "", file_arg[[1]]), mustWork = TRUE)
}

latest_run_dir <- function(base_dir) {
  if (!dir.exists(base_dir)) {
    return(base_dir)
  }
  run_dirs <- list.dirs(base_dir, recursive = FALSE, full.names = TRUE)
  run_dirs <- run_dirs[grepl("^run_", basename(run_dirs))]
  if (length(run_dirs) == 0) {
    return(base_dir)
  }
  sort(run_dirs, decreasing = TRUE)[[1]]
}

bind_fill <- function(frames) {
  frames <- frames[!vapply(frames, is.null, logical(1))]
  if (length(frames) == 0) {
    return(data.frame())
  }
  cols <- unique(unlist(lapply(frames, names)))
  padded <- lapply(frames, function(frame) {
    missing <- setdiff(cols, names(frame))
    for (name in missing) {
      frame[[name]] <- NA
    }
    frame[, cols, drop = FALSE]
  })
  do.call(rbind, padded)
}

read_metadata <- function(results_dir) {
  files <- list.files(results_dir, pattern = "^metadata_.*\\.csv$", recursive = TRUE, full.names = TRUE)
  frames <- lapply(files, function(path) {
    read.csv(path, stringsAsFactors = FALSE)
  })
  bind_fill(frames)
}

lookup_metadata <- function(metadata, dataset, missing_prop, field) {
  if (nrow(metadata) == 0 || !(field %in% names(metadata))) {
    return(NA)
  }
  hit <- metadata[
    metadata$dataset == dataset & abs(as.numeric(metadata$missing_prop) - as.numeric(missing_prop)) < 1e-9,
    field,
    drop = TRUE
  ]
  if (length(hit) == 0) {
    return(NA)
  }
  hit[[1]]
}

collect_informs <- function(results_dir) {
  if (!dir.exists(results_dir)) {
    warning(sprintf("INFORMS results directory does not exist: %s", results_dir))
    return(data.frame())
  }

  metadata <- read_metadata(results_dir)
  files <- list.files(results_dir, pattern = "\\.csv$", recursive = TRUE, full.names = TRUE)
  files <- files[!grepl("(^|/)metadata_", files)]
  files <- files[!grepl("(^|/)run_config\\.csv$", files)]
  files <- files[!grepl("Scalability", basename(files), ignore.case = TRUE)]

  rows <- list()
  for (path in files) {
    stem <- sub("\\.csv$", "", basename(path))
    parts <- strsplit(stem, "_", fixed = TRUE)[[1]]
    if (length(parts) < 3) {
      next
    }

    dataset <- parts[[length(parts)]]
    model_family <- parts[[length(parts) - 1]]
    metric <- paste(parts[seq_len(length(parts) - 2)], collapse = "_")
    if (!dataset %in% c("bike", "house", "adult")) {
      next
    }

    df <- read.csv(path, check.names = FALSE, stringsAsFactors = FALSE)
    if (!("miss_prop" %in% names(df))) {
      next
    }

    method_cols <- setdiff(names(df), "miss_prop")
    for (method in method_cols) {
      values <- suppressWarnings(as.numeric(df[[method]]))
      for (i in seq_len(nrow(df))) {
        prop <- as.numeric(df$miss_prop[[i]])
        rows[[length(rows) + 1]] <- data.frame(
          paper = "INFORMS_reduced",
          dataset = dataset,
          method = method,
          model_family = model_family,
          missing_prop = prop,
          metric = metric,
          value = values[[i]],
          n_train = lookup_metadata(metadata, dataset, prop, "n_train"),
          n_test = lookup_metadata(metadata, dataset, prop, "n_test"),
          n_invalid_predictions = lookup_metadata(metadata, dataset, prop, "n_invalid_predictions"),
          best_lambda = NA_real_,
          source_count = NA_integer_,
          run_source = normalizePath(results_dir, mustWork = FALSE),
          stringsAsFactors = FALSE
        )
      }
    }
  }

  bind_fill(rows)
}

collect_plos <- function(results_dir) {
  if (!dir.exists(results_dir)) {
    warning(sprintf("PLOS results directory does not exist: %s", results_dir))
    return(data.frame())
  }

  long_files <- list.files(results_dir, pattern = "^plos_bwm_main_data_metrics\\.csv$", recursive = TRUE, full.names = TRUE)
  long_frames <- lapply(long_files, function(path) {
    df <- read.csv(path, stringsAsFactors = FALSE)
    df$run_source <- normalizePath(dirname(path), mustWork = FALSE)
    df
  })

  wide_files <- list.files(results_dir, pattern = "^plos_bwm_main_data_metrics_.*\\.csv$", recursive = TRUE, full.names = TRUE)
  wide_frames <- lapply(wide_files, function(path) {
    df <- read.csv(path, check.names = FALSE, stringsAsFactors = FALSE)
    metric_cols <- intersect(
      c(
        "rmse", "mae", "smape", "mape", "r_squared",
        "accuracy", "balanced_accuracy", "precision", "recall",
        "specificity", "f1", "gmean", "mcc", "hard_label_auc"
      ),
      names(df)
    )
    if (length(metric_cols) == 0) {
      return(NULL)
    }

    rows <- list()
    for (i in seq_len(nrow(df))) {
      for (metric in metric_cols) {
        rows[[length(rows) + 1]] <- data.frame(
          paper = "PLOS_ONE_BWM",
          dataset = df$dataset[[i]],
          method = "bwm",
          model_family = if ("task" %in% names(df)) df$task[[i]] else NA_character_,
          missing_prop = if ("missing_rate" %in% names(df)) df$missing_rate[[i]] else NA_real_,
          metric = metric,
          value = suppressWarnings(as.numeric(df[[metric]][[i]])),
          n_train = if ("n_train" %in% names(df)) df$n_train[[i]] else NA,
          n_test = if ("n_test" %in% names(df)) df$n_test[[i]] else NA,
          n_invalid_predictions = if ("n_valid" %in% names(df) && "n_test" %in% names(df)) {
            suppressWarnings(as.numeric(df$n_test[[i]]) - as.numeric(df$n_valid[[i]]))
          } else {
            NA_real_
          },
          best_lambda = if ("selected_lambda" %in% names(df)) df$selected_lambda[[i]] else NA,
          source_count = if ("p" %in% names(df)) length(strsplit(as.character(df$p[[i]]), "|", fixed = TRUE)[[1]]) else NA_integer_,
          run_source = normalizePath(dirname(path), mustWork = FALSE),
          error = if ("error_message" %in% names(df)) df$error_message[[i]] else NA_character_,
          stringsAsFactors = FALSE
        )
      }
    }
    bind_fill(rows)
  })

  bind_fill(c(long_frames, wide_frames))
}

main <- function() {
  args <- parse_args(commandArgs(trailingOnly = TRUE))
  script <- script_path()
  repo_root <- normalizePath(file.path(dirname(script), ".."), mustWork = TRUE)

  informs_dir <- if (is.na(args$informs_results_dir)) {
    latest_run_dir(file.path(repo_root, "paper_informs_reduced", "results"))
  } else {
    args$informs_results_dir
  }
  plos_dir <- if (is.na(args$plos_results_dir)) {
    latest_run_dir(file.path(repo_root, "paper_plos_one_bwm", "results"))
  } else {
    args$plos_results_dir
  }
  output <- if (is.na(args$output)) {
    file.path(repo_root, "comparison_results", "main_data_model_metrics.csv")
  } else {
    args$output
  }

  combined <- bind_fill(list(
    collect_informs(informs_dir),
    collect_plos(plos_dir)
  ))

  preferred_cols <- c(
    "paper", "dataset", "method", "model_family", "missing_prop", "metric", "value",
    "n_train", "n_test", "n_invalid_predictions", "best_lambda", "source_count",
    "run_source", "error"
  )
  ordered_cols <- c(preferred_cols[preferred_cols %in% names(combined)], setdiff(names(combined), preferred_cols))
  combined <- combined[, ordered_cols, drop = FALSE]

  dir.create(dirname(output), recursive = TRUE, showWarnings = FALSE)
  write.csv(combined, output, row.names = FALSE)
  message(sprintf("Saved combined comparison metrics to: %s", normalizePath(output, mustWork = FALSE)))
}

main()
