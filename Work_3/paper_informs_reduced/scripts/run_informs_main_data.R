#!/usr/bin/env Rscript

default_missing_props <- seq(0.1, 0.9, by = 0.1)

parse_args <- function(args) {
  parsed <- list(
    dataset = "all",
    benchmarks = "all",
    include_scalability = FALSE,
    install_python_deps = FALSE,
    results_dir = NA_character_,
    keep_runtime = FALSE
  )

  i <- 1
  while (i <= length(args)) {
    key <- args[[i]]
    value <- if (i < length(args)) args[[i + 1]] else NA_character_

    if (key == "--dataset") {
      parsed$dataset <- value
      i <- i + 2
    } else if (key == "--benchmarks") {
      parsed$benchmarks <- value
      i <- i + 2
    } else if (key == "--results-dir") {
      parsed$results_dir <- value
      i <- i + 2
    } else if (key == "--include-scalability") {
      parsed$include_scalability <- TRUE
      i <- i + 1
    } else if (key == "--install-python-deps") {
      parsed$install_python_deps <- TRUE
      i <- i + 1
    } else if (key == "--keep-runtime") {
      parsed$keep_runtime <- TRUE
      i <- i + 1
    } else if (key %in% c("-h", "--help")) {
      cat(
        "Usage: Rscript paper_informs_reduced/scripts/run_informs_main_data.R [options]\n\n",
        "Options:\n",
        "  --dataset all|bike|house|adult   Dataset(s) to run. Default: all\n",
        "  --benchmarks all                 Placeholder for compatibility; original scripts run all benchmarks.\n",
        "  --results-dir PATH               Output directory. Default: paper_informs_reduced/results/run_<timestamp>\n",
        "  --include-scalability            Also run scalability.R after the prediction scripts.\n",
        "  --install-python-deps            Let Load_packages.R install miniconda/tensorflow for GAIN.\n",
        "  --keep-runtime                   Keep the patched runtime capsule directory.\n",
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
    return(normalizePath("paper_informs_reduced/scripts/run_informs_main_data.R", mustWork = FALSE))
  }
  normalizePath(sub("^--file=", "", file_arg[[1]]), mustWork = TRUE)
}

timestamp <- function() {
  format(Sys.time(), "%Y%m%d_%H%M%S")
}

copy_and_patch_code <- function(source_code_dir, runtime_code_dir, runtime_root) {
  dir.create(runtime_code_dir, recursive = TRUE, showWarnings = FALSE)
  code_files <- list.files(source_code_dir, all.files = TRUE, no.. = TRUE, full.names = TRUE)
  for (path in code_files) {
    if (dir.exists(path)) {
      next
    }
    target <- file.path(runtime_code_dir, basename(path))
    file.copy(path, target, overwrite = TRUE)

    if (grepl("\\.(R|r)$", path) || basename(path) == "run") {
      text <- readLines(target, warn = FALSE)
      runtime_root_text <- normalizePath(runtime_root, winslash = "/", mustWork = FALSE)
      runtime_code_text <- normalizePath(runtime_code_dir, winslash = "/", mustWork = FALSE)
      text <- gsub("/root/capsule", runtime_root_text, text, fixed = TRUE)

      if (basename(path) == "Load_packages.R") {
        text <- gsub(
          "reticulate::install_miniconda\\(\\)",
          "if (isTRUE(getOption('bcd.install_python_deps', FALSE))) try(reticulate::install_miniconda(), silent = TRUE)",
          text
        )
        text <- gsub(
          "tensorflow::install_tensorflow\\(\\)",
          "if (isTRUE(getOption('bcd.install_python_deps', FALSE))) try(tensorflow::install_tensorflow(), silent = TRUE)",
          text
        )
        text <- gsub(
          "gain <- import_from_path\\('gain',path = \"[^\"]+\",   convert = TRUE\\)",
          paste0(
            "gain <- tryCatch(import_from_path('gain', path = \"",
            runtime_code_text,
            "/\", convert = TRUE), error = function(e) { warning(conditionMessage(e)); NULL })"
          ),
          text
        )
      }

      writeLines(text, target, useBytes = TRUE)
    }
  }
}

stage_runtime <- function(repo_root, paper_dir, run_id) {
  runtime_root <- file.path(paper_dir, ".runtime", paste0("capsule_", run_id))
  runtime_code_dir <- file.path(runtime_root, "code")
  runtime_data_dir <- file.path(runtime_root, "data")
  runtime_results_dir <- file.path(runtime_root, "results")

  dir.create(runtime_root, recursive = TRUE, showWarnings = FALSE)
  dir.create(runtime_data_dir, recursive = TRUE, showWarnings = FALSE)
  dir.create(runtime_results_dir, recursive = TRUE, showWarnings = FALSE)

  main_data_dir <- file.path(repo_root, "main_data")
  required_data <- c("adult.csv", "bike_all.csv", "house.csv")
  missing_data <- required_data[!file.exists(file.path(main_data_dir, required_data))]
  if (length(missing_data) > 0) {
    stop(sprintf("Missing main_data files: %s", paste(missing_data, collapse = ", ")))
  }
  file.copy(file.path(main_data_dir, required_data), runtime_data_dir, overwrite = TRUE)

  copy_and_patch_code(
    source_code_dir = file.path(paper_dir, "code"),
    runtime_code_dir = runtime_code_dir,
    runtime_root = runtime_root
  )

  list(
    root = runtime_root,
    code = runtime_code_dir,
    data = runtime_data_dir,
    results = runtime_results_dir
  )
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

write_metric_objects <- function(env, dataset, output_dir) {
  metric_names <- c(
    "rmse_reg", "rmse_tree", "mae_reg", "mae_tree", "smape_reg", "smape_tree",
    "acc_reg", "acc_tree", "auc_reg", "auc_tree", "recall_reg", "recall_tree",
    "precision_reg", "precision_tree", "f1_reg", "f1_tree"
  )

  for (name in metric_names) {
    if (exists(name, envir = env, inherits = FALSE)) {
      value <- get(name, envir = env)
      if (is.data.frame(value)) {
        write.csv(value, file.path(output_dir, sprintf("%s_%s.csv", name, dataset)), row.names = FALSE)
      }
    }
  }
}

write_metadata <- function(env, dataset, output_dir) {
  n_train <- if (exists("data_train", envir = env, inherits = FALSE)) nrow(get("data_train", envir = env)) else NA_integer_
  n_test <- if (exists("data_test", envir = env, inherits = FALSE)) nrow(get("data_test", envir = env)) else NA_integer_
  outcome <- if (exists("outcome", envir = env, inherits = FALSE)) get("outcome", envir = env) else NA_character_
  metadata <- data.frame(
    paper = "INFORMS_reduced",
    dataset = dataset,
    missing_prop = default_missing_props,
    n_train = n_train,
    n_test = n_test,
    n_invalid_predictions = NA_integer_,
    outcome = outcome,
    stringsAsFactors = FALSE
  )
  write.csv(metadata, file.path(output_dir, sprintf("metadata_%s.csv", dataset)), row.names = FALSE)
}

source_common_files <- function(env, runtime) {
  source(file.path(runtime$code, "Load_packages.R"), local = env)
  source(file.path(runtime$code, "Preliminary_functions.R"), local = env)
  source(file.path(runtime$code, "Benchmarks.R"), local = env)
  source(file.path(runtime$code, "BRM_functions.R"), local = env)
}

run_dataset <- function(dataset, runtime, output_dir) {
  env <- new.env(parent = globalenv())
  source_common_files(env, runtime)

  dataset_file <- switch(
    dataset,
    bike = "bike_.R",
    house = "house_.R",
    adult = "adult_.R"
  )

  message(sprintf("Running INFORMS reduced benchmarks for %s", dataset))
  source(file.path(runtime$code, dataset_file), local = env)
  write_metric_objects(env, dataset, output_dir)
  write_metadata(env, dataset, output_dir)
}

main <- function() {
  args <- parse_args(commandArgs(trailingOnly = TRUE))
  if (tolower(args$benchmarks) != "all") {
    warning("The original INFORMS scripts run the benchmark suite as one block; --benchmarks is currently informational.")
  }

  script <- script_path()
  paper_dir <- normalizePath(file.path(dirname(script), ".."), mustWork = TRUE)
  repo_root <- normalizePath(file.path(paper_dir, ".."), mustWork = TRUE)
  run_id <- timestamp()
  output_dir <- if (is.na(args$results_dir)) {
    file.path(paper_dir, "results", paste0("run_", run_id))
  } else {
    args$results_dir
  }
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

  options(bcd.install_python_deps = args$install_python_deps)
  runtime <- stage_runtime(repo_root = repo_root, paper_dir = paper_dir, run_id = run_id)
  on.exit({
    if (!args$keep_runtime && dir.exists(runtime$root)) {
      unlink(runtime$root, recursive = TRUE, force = TRUE)
    }
  }, add = TRUE)

  datasets <- resolve_datasets(args$dataset)
  for (dataset in datasets) {
    run_dataset(dataset = dataset, runtime = runtime, output_dir = output_dir)
  }

  if (isTRUE(args$include_scalability)) {
    env <- new.env(parent = globalenv())
    source_common_files(env, runtime)
    message("Running INFORMS scalability.R")
    source(file.path(runtime$code, "scalability.R"), local = env)
    runtime_outputs <- list.files(runtime$results, full.names = TRUE)
    if (length(runtime_outputs) > 0) {
      file.copy(runtime_outputs, output_dir, overwrite = TRUE)
    }
  }

  message(sprintf("Saved INFORMS reduced results to: %s", normalizePath(output_dir, mustWork = FALSE)))
}

main()
