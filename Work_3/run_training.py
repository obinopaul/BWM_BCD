"""
Paper note:
This is the user-facing training entrypoint for the current modular iSFS
implementation. It runs the full outer block-coordinate-descent loop described
by Section 3 of `Multi_source_BWM.pdf` together with the trust-region cost
update written in `notes.tex`.

The optimization alternates between two blocks:

    Block A:
        fix the class-cost vector C and optimize alpha, beta

    Block B:
        fix alpha, beta and update C with the trust-region step

Inside Block A, the code follows the paper's Algorithm 2 structure:

    1. with beta fixed, approximately solve the alpha subproblem
    2. with alpha fixed, approximately solve the beta subproblem

This script is intentionally more than a thin launcher. It also builds a
training report so the user can inspect:

    - outer BCD iterations
    - inner Block A solve counts
    - accepted versus rejected trust-region cost steps
    - train and test metrics
    - exact-profile results
    - optimization-group results
    - saved parameter snapshots across training

The active paper-aligned path is:

    - logistic/BCE data-fit term
    - l1-ball constraint on alpha
    - l1 penalty on beta
    - trust-region update for C

Run examples:

    python3 run_training.py --dataset real
    python3 run_training.py --dataset synthetic
    python3 run_training.py --dataset real --max-iter 12 --block-a-max-iter 40
"""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from costs import cost_weighted_bce_from_logits
from loss import sigmoid
from loss_functions import binary_cross_entropy_from_logits
from missingness_profiles import (
    build_exact_profiles,
    build_missingness_profiles,
    get_profile_code,
)
from prediction import compute_profile_logits
from regularization import beta_l1_penalty, max_alpha_l1_norm
from train_model import train_blockwise_logistic_model


DATA_DIR = Path(__file__).resolve().parent / "data"
REAL_SOURCE_FILES = {
    "source1": "source1_clinical_blockwise_missing.csv",
    "source2": "source2_imaging_blockwise_missing.csv",
    "source3": "source3_proteomics_blockwise_missing.csv",
}


def parse_cost_vector(text):
    """
    Parse a comma-separated cost vector such as "0.5,0.5".
    """
    values = [value.strip() for value in text.split(",") if value.strip()]
    if not values:
        raise ValueError("Cost vector must contain at least one numeric value.")
    return np.asarray([float(value) for value in values], dtype=float)


def load_real_dataset():
    """
    Load the real blockwise-missing multisource dataset from `data/`.
    """
    X_blocks = {}

    for source, filename in REAL_SOURCE_FILES.items():
        X = np.genfromtxt(
            DATA_DIR / filename,
            delimiter=",",
            skip_header=1,
        )
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X_blocks[source] = X

    labels = np.genfromtxt(
        DATA_DIR / "labels.csv",
        delimiter=",",
        skip_header=1,
    )
    if labels.ndim == 1:
        labels = labels.reshape(1, -1)

    y = labels[:, 1].astype(int)
    return X_blocks, y


def make_synthetic_dataset(random_state=42):
    """
    Create a small synthetic blockwise-missing dataset.
    """
    rng = np.random.default_rng(random_state)
    n_samples = 200

    X_clinical = rng.normal(size=(n_samples, 5))
    X_imaging_1 = rng.normal(size=(n_samples, 10))
    X_imaging_2 = rng.normal(size=(n_samples, 8))

    true_signal = (
        1.2 * X_clinical[:, 0]
        + 0.8 * X_imaging_1[:, 1]
        - 0.9 * X_imaging_2[:, 2]
    )
    probability = sigmoid(true_signal)
    y = (probability >= 0.5).astype(int)

    X_imaging_1[40:90, :] = np.nan
    X_imaging_2[100:150, :] = np.nan
    X_imaging_1[160:180, :] = np.nan
    X_imaging_2[160:180, :] = np.nan

    X_blocks = {
        "clinical": X_clinical,
        "imaging_1": X_imaging_1,
        "imaging_2": X_imaging_2,
    }
    return X_blocks, y


def load_dataset(dataset, random_state):
    """
    Load either the real or synthetic dataset.
    """
    if dataset == "real":
        return load_real_dataset()
    if dataset == "synthetic":
        return make_synthetic_dataset(random_state=random_state)

    raise ValueError(f"Unsupported dataset: {dataset}")


def subset_X_blocks(X_blocks, indices):
    """
    Return a row subset for every source block.
    """
    indices = np.asarray(indices, dtype=int).reshape(-1)
    return {
        source: np.asarray(X, dtype=float)[indices, :].copy()
        for source, X in X_blocks.items()
    }


def build_stratification_labels(X_blocks, y):
    """
    Build joint label/profile strata for train/test splitting.
    """
    source_order = list(X_blocks.keys())
    exact_profiles = build_exact_profiles(X_blocks, source_order=source_order)
    y = np.asarray(y, dtype=int).reshape(-1)
    labels = np.full(y.shape[0], None, dtype=object)

    for profile, indices in exact_profiles.items():
        profile_code = get_profile_code(profile, source_order)
        for index in indices:
            labels[index] = f"{int(y[index])}|{profile_code}"

    if any(label is None for label in labels):
        raise ValueError("Could not assign every sample to a split stratum.")

    return labels


def stratified_train_test_indices(strata, test_size=0.30, random_state=42):
    """
    Split indices while preserving each stratum as closely as possible.
    """
    test_size = float(test_size)
    if not (0.0 < test_size < 1.0):
        raise ValueError("test_size must lie in (0, 1).")

    strata = np.asarray(strata, dtype=object).reshape(-1)
    rng = np.random.default_rng(random_state)
    train_indices = []
    test_indices = []

    for stratum in sorted(set(strata.tolist())):
        indices = np.where(strata == stratum)[0]
        shuffled = indices.copy()
        rng.shuffle(shuffled)

        if len(shuffled) == 1:
            n_test = 0
        else:
            n_test = int(round(len(shuffled) * test_size))
            n_test = min(max(n_test, 1), len(shuffled) - 1)

        test_indices.extend(shuffled[:n_test].tolist())
        train_indices.extend(shuffled[n_test:].tolist())

    train_indices = np.asarray(sorted(train_indices), dtype=int)
    test_indices = np.asarray(sorted(test_indices), dtype=int)

    if train_indices.size == 0 or test_indices.size == 0:
        raise ValueError("Split produced an empty train or test set.")

    return train_indices, test_indices


def split_dataset(X_blocks, y, test_size=0.30, random_state=42):
    """
    Split source blocks and labels into stratified train/test partitions.
    """
    y = np.asarray(y, dtype=int).reshape(-1)
    strata = build_stratification_labels(X_blocks, y)
    train_indices, test_indices = stratified_train_test_indices(
        strata=strata,
        test_size=test_size,
        random_state=random_state,
    )

    return {
        "train_indices": train_indices,
        "test_indices": test_indices,
        "train_X_blocks": subset_X_blocks(X_blocks, train_indices),
        "test_X_blocks": subset_X_blocks(X_blocks, test_indices),
        "train_y": y[train_indices].copy(),
        "test_y": y[test_indices].copy(),
        "stratification_labels": strata,
    }


def _column_nanmeans(X):
    """
    Compute finite-value column means without warnings for all-NaN columns.
    """
    X = np.asarray(X, dtype=float)
    finite = np.isfinite(X)
    counts = finite.sum(axis=0)
    sums = np.where(finite, X, 0.0).sum(axis=0)
    means = np.zeros(X.shape[1], dtype=float)
    np.divide(sums, counts, out=means, where=counts > 0)
    return means


def fit_standardization_preprocessor(X_blocks):
    """
    Fit per-source imputation and standardization parameters on training data.
    """
    preprocessor = {}

    for source, X in X_blocks.items():
        X = np.asarray(X, dtype=float)
        missing_rows = np.all(np.isnan(X), axis=1)
        means = _column_nanmeans(X)
        filled = np.where(np.isnan(X), means, X)
        rows_for_stats = filled[~missing_rows] if np.any(~missing_rows) else filled
        stds = rows_for_stats.std(axis=0)
        stds = np.where(stds > 0.0, stds, 1.0)
        preprocessor[source] = {
            "mean": means,
            "std": stds,
            "n_rows": int(X.shape[0]),
            "n_observed_rows": int(np.sum(~missing_rows)),
        }

    return preprocessor


def transform_with_preprocessor(X_blocks, preprocessor):
    """
    Apply fitted preprocessing while preserving all-source-missing rows.
    """
    transformed = {}

    for source, X in X_blocks.items():
        if source not in preprocessor:
            raise ValueError(f"Missing preprocessing parameters for source {source}.")

        X = np.asarray(X, dtype=float)
        params = preprocessor[source]
        means = np.asarray(params["mean"], dtype=float)
        stds = np.asarray(params["std"], dtype=float)

        if X.shape[1] != means.shape[0]:
            raise ValueError(
                f"Source {source} has {X.shape[1]} features, "
                f"but preprocessing has {means.shape[0]}."
            )

        missing_rows = np.all(np.isnan(X), axis=1)
        filled = np.where(np.isnan(X), means, X)
        scaled = (filled - means) / stds
        scaled[missing_rows, :] = np.nan
        transformed[source] = scaled

    return transformed


def fit_transform_train_test(X_train_blocks, X_test_blocks):
    """
    Fit preprocessing on train data and apply it to train and test blocks.
    """
    preprocessor = fit_standardization_preprocessor(X_train_blocks)
    return (
        transform_with_preprocessor(X_train_blocks, preprocessor),
        transform_with_preprocessor(X_test_blocks, preprocessor),
        preprocessor,
    )


def summarize_split(split):
    """
    Build a compact JSON-friendly split summary.
    """
    train_indices = np.asarray(split["train_indices"], dtype=int)
    test_indices = np.asarray(split["test_indices"], dtype=int)
    strata = np.asarray(split["stratification_labels"], dtype=object)

    return {
        "enabled": True,
        "strategy": "stratified_by_label_and_exact_profile",
        "train_size": int(train_indices.size),
        "test_size": int(test_indices.size),
        "train_indices": train_indices.tolist(),
        "test_indices": test_indices.tolist(),
        "train_strata": {
            str(label): int(np.sum(strata[train_indices] == label))
            for label in sorted(set(strata[train_indices].tolist()))
        },
        "test_strata": {
            str(label): int(np.sum(strata[test_indices] == label))
            for label in sorted(set(strata[test_indices].tolist()))
        },
    }


def _binary_label_counts(y):
    """
    Count binary labels for readable split reporting.
    """
    y = np.asarray(y, dtype=int).reshape(-1)
    return {
        "class_0": int(np.sum(y == 0)),
        "class_1": int(np.sum(y == 1)),
        "total": int(y.size),
    }


def _format_label_counts(counts):
    """
    Format label counts as compact terminal text.
    """
    total = counts["total"]
    prevalence = safe_divide(counts["class_1"], total)
    return (
        f"class 0 = {counts['class_0']}, "
        f"class 1 = {counts['class_1']}, "
        f"positive rate = {prevalence:.4f}"
    )


def _profile_label_counts(X_blocks, y, source_order):
    """
    Count exact missingness profiles and labels inside one partition.
    """
    if X_blocks is None or y is None:
        return {}

    y = np.asarray(y, dtype=int).reshape(-1)
    profiles = build_exact_profiles(X_blocks, source_order=source_order)
    rows = {}

    for profile, indices in profiles.items():
        profile_code = get_profile_code(profile, source_order)
        profile_y = y[np.asarray(indices, dtype=int)]
        rows[profile_code] = {
            "sources": profile,
            "total": int(profile_y.size),
            "class_0": int(np.sum(profile_y == 0)),
            "class_1": int(np.sum(profile_y == 1)),
        }

    return rows


def _format_profile_sources(profile):
    """
    Format a source tuple for terminal output.
    """
    return "+".join(profile) if profile else "-"


def print_iteration_limits(args):
    """
    Print the nested iteration limits used by this run.
    """
    limits = {
        "Outer BCD C-update loop": args.max_iter,
        "Fixed-C Block A alpha/beta alternation": args.block_a_max_iter,
        "Alpha subproblem projected-gradient loop": args.alpha_subproblem_max_iter,
        "Beta subproblem proximal-gradient loop": args.beta_subproblem_max_iter,
    }
    production_like = all(value > 10 for value in limits.values())

    print("\nIteration limits")
    for label, value in limits.items():
        print(f"{label}: {value}")
    print(f"All listed max iteration limits > 10: {production_like}")
    if not production_like:
        print("Note: at least one iteration limit is <= 10, so this is a debug-style run.")


def print_split_details(split_summary, split, source_order):
    """
    Print a detailed train/test split report before optimization starts.
    """
    train_indices = np.asarray(split["train_indices"], dtype=int)
    test_indices = np.asarray(split["test_indices"], dtype=int)
    train_y = split["train_y"]
    test_y = split["test_y"]
    n_total = int(train_indices.size + test_indices.size)
    overlap = len(set(train_indices.tolist()).intersection(set(test_indices.tolist())))

    print("\nTrain/test split details")
    print(f"Strategy: {split_summary.get('strategy', 'unknown')}")
    print(f"Split enabled: {split_summary.get('enabled', False)}")
    if "test_fraction_requested" in split_summary:
        print(f"Requested test fraction: {split_summary['test_fraction_requested']:.4f}")
    if "random_state" in split_summary:
        print(f"Split random state: {split_summary['random_state']}")
    print(f"Total rows: {n_total}")
    print(
        f"Train rows: {train_indices.size} "
        f"({safe_divide(train_indices.size, n_total):.2%})"
    )
    print(
        f"Test rows: {test_indices.size} "
        f"({safe_divide(test_indices.size, n_total):.2%})"
    )
    print(f"Train/test index overlap: {overlap}")
    print(f"Train labels: {_format_label_counts(_binary_label_counts(train_y))}")
    if test_y is not None:
        print(f"Test labels: {_format_label_counts(_binary_label_counts(test_y))}")
    else:
        print("Test labels: n/a, in-sample evaluation mode")

    train_profile_rows = _profile_label_counts(
        split["train_X_blocks"],
        train_y,
        source_order,
    )
    test_profile_rows = _profile_label_counts(
        split["test_X_blocks"],
        test_y,
        source_order,
    )
    all_codes = sorted(set(train_profile_rows) | set(test_profile_rows))

    print("\nExact-profile split counts")
    print(
        "Code | Sources                         | "
        "Train n | Train c0/c1 | Test n | Test c0/c1"
    )
    for code in all_codes:
        train_row = train_profile_rows.get(
            code,
            {"sources": (), "total": 0, "class_0": 0, "class_1": 0},
        )
        test_row = test_profile_rows.get(
            code,
            {"sources": train_row["sources"], "total": 0, "class_0": 0, "class_1": 0},
        )
        sources = train_row["sources"] or test_row["sources"]
        print(
            f"{code:>4} | {_format_profile_sources(sources):<31} | "
            f"{train_row['total']:>7d} | "
            f"{train_row['class_0']:>3d}/{train_row['class_1']:<3d} | "
            f"{test_row['total']:>6d} | "
            f"{test_row['class_0']:>3d}/{test_row['class_1']:<3d}"
        )

    train_strata = split_summary.get("train_strata", {})
    test_strata = split_summary.get("test_strata", {})
    if train_strata or test_strata:
        print("\nLabel/profile stratification counts")
        print("Label|Profile | Train | Test")
        for stratum in sorted(set(train_strata) | set(test_strata)):
            print(
                f"{stratum:>13} | "
                f"{int(train_strata.get(stratum, 0)):>5d} | "
                f"{int(test_strata.get(stratum, 0)):>4d}"
            )


def print_preprocessing_details(preprocessing_summary):
    """
    Print how preprocessing was fit and applied.
    """
    print("\nPreprocessing details")
    print(f"Enabled: {preprocessing_summary.get('enabled', False)}")
    print(f"Method: {preprocessing_summary.get('method', 'unknown')}")

    if not preprocessing_summary.get("enabled", False):
        return

    print(
        "All-source-missing rows preserved as NaN: "
        f"{preprocessing_summary.get('preserves_all_missing_source_rows', False)}"
    )
    print("Source | Train rows | Observed rows | Missing-source rows | Features")
    for source, params in preprocessing_summary.get("sources", {}).items():
        means = np.asarray(params.get("mean", []), dtype=float)
        n_rows = int(params.get("n_rows", 0))
        n_observed = int(params.get("n_observed_rows", 0))
        print(
            f"{source:<6} | "
            f"{n_rows:>10d} | "
            f"{n_observed:>13d} | "
            f"{n_rows - n_observed:>19d} | "
            f"{means.size:>8d}"
        )


def safe_divide(numerator, denominator):
    """
    Divide safely and return 0.0 when the denominator is zero.
    """
    numerator = float(numerator)
    denominator = float(denominator)
    if denominator == 0.0:
        return 0.0
    return numerator / denominator


def compute_binary_metrics(y_true, y_pred, y_prob):
    """
    Compute binary classification metrics from labels, predictions, and scores.
    """
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=int).reshape(-1)
    y_prob = np.asarray(y_prob, dtype=float).reshape(-1)

    if not (len(y_true) == len(y_pred) == len(y_prob)):
        raise ValueError("y_true, y_pred, and y_prob must have the same length.")

    n_samples = int(len(y_true))
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    accuracy = safe_divide(tp + tn, n_samples)
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    specificity = safe_divide(tn, tn + fp)
    npv = safe_divide(tn, tn + fn)
    f1 = safe_divide(2.0 * precision * recall, precision + recall)
    balanced_accuracy = 0.5 * (recall + specificity)
    gmean = float(np.sqrt(max(recall * specificity, 0.0)))
    false_positive_rate = safe_divide(fp, fp + tn)
    false_negative_rate = safe_divide(fn, fn + tp)
    predicted_positive_rate = safe_divide(tp + fp, n_samples)
    prevalence = safe_divide(tp + fn, n_samples)

    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    matthews_corrcoef = safe_divide((tp * tn) - (fp * fn), denominator)

    if n_samples == 0:
        brier_score = 0.0
        log_loss = 0.0
        probability_min = 0.0
        probability_max = 0.0
        probability_mean = 0.0
    else:
        clipped_prob = np.clip(y_prob, 1e-12, 1.0 - 1e-12)
        brier_score = float(np.mean((y_prob - y_true) ** 2))
        log_loss = float(
            -np.mean(y_true * np.log(clipped_prob) + (1 - y_true) * np.log(1.0 - clipped_prob))
        )
        probability_min = float(np.min(y_prob))
        probability_max = float(np.max(y_prob))
        probability_mean = float(np.mean(y_prob))

    return {
        "n_samples": n_samples,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "npv": float(npv),
        "f1": float(f1),
        "balanced_accuracy": float(balanced_accuracy),
        "gmean": gmean,
        "false_positive_rate": float(false_positive_rate),
        "false_negative_rate": float(false_negative_rate),
        "predicted_positive_rate": float(predicted_positive_rate),
        "prevalence": float(prevalence),
        "matthews_corrcoef": float(matthews_corrcoef),
        "brier_score": float(brier_score),
        "log_loss": float(log_loss),
        "probability_min": probability_min,
        "probability_max": probability_max,
        "probability_mean": probability_mean,
        "positive_support": int(np.sum(y_true == 1)),
        "negative_support": int(np.sum(y_true == 0)),
    }


def compute_training_summary(model, X_blocks, y, threshold=0.5):
    """
    Compute metrics on a supplied train or test partition.
    """
    probabilities = model["predict_proba"](X_blocks)
    predictions = model["predict"](X_blocks, threshold=threshold)
    valid_mask = predictions != -1

    y_valid = np.asarray(y, dtype=int).reshape(-1)[valid_mask]
    pred_valid = predictions[valid_mask]
    prob_valid = probabilities[valid_mask]

    metrics = compute_binary_metrics(y_valid, pred_valid, prob_valid)
    metrics.update(
        {
            "threshold": float(threshold),
            "n_total_samples": int(len(y)),
            "n_valid_predictions": int(np.sum(valid_mask)),
            "n_invalid_predictions": int(len(y) - np.sum(valid_mask)),
        }
    )
    return metrics


def compute_profile_metrics(model, X_blocks, y, profiles, threshold=0.5):
    """
    Compute metrics for each exact profile or optimization group.
    """
    source_order = model["source_order"]
    cost_vector = np.asarray(model["costs"], dtype=float)
    rows = []

    for profile, indices in sorted(
        profiles.items(),
        key=lambda item: (get_profile_code(item[0], source_order), item[0]),
    ):
        indices = np.asarray(indices, dtype=int).reshape(-1)
        logits = compute_profile_logits(
            X_blocks=X_blocks,
            betas=model["betas"],
            alphas=model["alphas"],
            profile=profile,
            indices=indices,
            source_order=source_order,
        )
        probabilities = sigmoid(logits)
        predictions = (probabilities >= threshold).astype(int)
        y_group = np.asarray(y, dtype=int).reshape(-1)[indices]

        metrics = compute_binary_metrics(y_group, predictions, probabilities)
        metrics.update(
            {
                "profile_code": get_profile_code(profile, source_order),
                "profile_sources": ",".join(profile),
                "n_sources": int(len(profile)),
                "bce": float(binary_cross_entropy_from_logits(logits, y_group)),
                "cost_bce": float(
                    cost_weighted_bce_from_logits(
                        logits=logits,
                        y=y_group,
                        cost_vector=cost_vector,
                    )
                ),
            }
        )
        rows.append(metrics)

    return rows


def flatten_block_a_histories(model):
    """
    Flatten inner Block A histories for printing and CSV export.
    """
    rows = []

    initial_history = model.get("initial_block_a_history", [])
    for entry in initial_history:
        rows.append(
            {
                "phase": "initial",
                "outer_iteration": 0,
                "accepted": True,
                **entry,
            }
        )

    for solve in model.get("trial_block_a_history", []):
        for entry in solve["history"]:
            rows.append(
                {
                    "phase": "trial",
                    "outer_iteration": int(solve["outer_iteration"]),
                    "accepted": bool(solve["accepted"]),
                    **entry,
                }
            )

    return rows


def summarize_block_a_solves(model):
    """
    Build one summary row per Block A solve.
    """
    rows = []
    initial_history = model.get("initial_block_a_history", [])
    if initial_history:
        rows.append(
            {
                "phase": "initial",
                "outer_iteration": 0,
                "accepted": True,
                "iterations": int(len(initial_history)),
                "start_objective": float(initial_history[0]["objective"]),
                "end_objective": float(initial_history[-1]["objective"]),
                "objective_change": float(
                    initial_history[-1]["objective"] - initial_history[0]["objective"]
                ),
            }
        )

    for solve in model.get("trial_block_a_history", []):
        history = solve["history"]
        if not history:
            continue
        rows.append(
            {
                "phase": "trial",
                "outer_iteration": int(solve["outer_iteration"]),
                "accepted": bool(solve["accepted"]),
                "iterations": int(len(history)),
                "start_objective": float(history[0]["objective"]),
                "end_objective": float(history[-1]["objective"]),
                "objective_change": float(history[-1]["objective"] - history[0]["objective"]),
            }
        )

    return rows


def format_float(value, digits=6):
    """
    Format floats consistently for terminal output.
    """
    value = float(value)
    if np.isnan(value):
        return "nan"
    if np.isposinf(value):
        return "inf"
    if np.isneginf(value):
        return "-inf"
    return f"{value:.{digits}f}"


def format_vector(values):
    """
    Format a small numeric vector for terminal output.
    """
    array = np.asarray(values, dtype=float).reshape(-1)
    return "[" + ", ".join(format_float(value, digits=4) for value in array) + "]"


def print_bcd_explanation():
    """
    Explain the key BCD fields shown during training.
    """
    print("\nBCD field guide")
    print("Each BCD iteration uses the full dataset. It is not one iteration per sample.")
    print(
        "Accepted=False means the trial trust-region cost update for C was rejected, "
        "so the model kept the previous C, alpha, and beta."
    )
    print(
        "Group Cost-BCE is the mean cost-weighted BCE across the overlapping "
        "optimization groups used during training."
    )


def print_training_metrics(summary, title="Training performance"):
    """
    Print a detailed metrics block.
    """
    print(f"\n{title}")
    print(f"Samples: {summary['n_total_samples']}")
    print(f"Valid predictions: {summary['n_valid_predictions']}")
    print(f"Invalid predictions: {summary['n_invalid_predictions']}")
    print(f"Decision threshold: {summary['threshold']:.2f}")
    print(
        "Confusion counts: "
        f"TP={summary['tp']}, TN={summary['tn']}, "
        f"FP={summary['fp']}, FN={summary['fn']}"
    )
    print(f"Accuracy: {summary['accuracy']:.6f}")
    print(f"Precision: {summary['precision']:.6f}")
    print(f"Recall / Sensitivity: {summary['recall']:.6f}")
    print(f"Specificity: {summary['specificity']:.6f}")
    print(f"F1 score: {summary['f1']:.6f}")
    print(f"Balanced accuracy: {summary['balanced_accuracy']:.6f}")
    print(f"GMEAN: {summary['gmean']:.6f}")
    print(f"Negative predictive value: {summary['npv']:.6f}")
    print(f"False positive rate: {summary['false_positive_rate']:.6f}")
    print(f"False negative rate: {summary['false_negative_rate']:.6f}")
    print(f"Matthews correlation coefficient: {summary['matthews_corrcoef']:.6f}")
    print(f"Brier score: {summary['brier_score']:.6f}")
    print(f"Log loss: {summary['log_loss']:.6f}")
    print(
        "Probability range: "
        f"[{summary['probability_min']:.6f}, {summary['probability_max']:.6f}]"
    )


def print_parameter_summary(model):
    """
    Print the learned cost, alpha, and beta summaries.
    """
    print("\nLearned cost vector")
    print(format_vector(model["costs"]))

    print("\nAlpha summary")
    print(f"Max ||alpha_m||_1: {max_alpha_l1_norm(model['alphas']):.6f}")
    for profile, alpha_values in sorted(model["alphas"].items(), key=lambda item: item[0]):
        print(f"{profile}: {alpha_values}")

    print("\nBeta summary")
    print(
        "Weighted beta l1 penalty: "
        f"{beta_l1_penalty(model['betas'], model['lambda_beta']):.6f}"
    )
    for source, beta in model["betas"].items():
        print(
            f"{source}: shape={beta.shape}, "
            f"l2_norm={np.linalg.norm(beta):.6f}, "
            f"l1_norm={np.sum(np.abs(beta)):.6f}"
        )


def print_bcd_history(model):
    """
    Print one summary row per outer BCD iteration.
    """
    history = model.get("loss_history", [])
    accepted_steps = sum(1 for entry in history if entry.get("accepted", False))

    print("\nBCD iteration summary")
    print(f"Recorded BCD entries: {len(history)}")
    print(f"Accepted cost steps: {accepted_steps}")

    if not history:
        return

    print(
        "Iter | Accept | BlockA iters | Pred inc  | Actual inc | Ratio     | "
        "Radius        | Current obj | Trial obj"
    )
    for entry in history:
        trial_objective = entry.get("trial_reduced_objective", entry["reduced_objective"])
        print(
            f"{entry['iteration']:>4d} | "
            f"{str(entry['accepted']):>6} | "
            f"{entry.get('trial_block_a_iterations', 0):>12d} | "
            f"{format_float(entry['predicted_increase']):>9} | "
            f"{format_float(entry['actual_increase']):>10} | "
            f"{format_float(entry['ratio']):>9} | "
            f"{format_float(entry['working_radius']):>13} | "
            f"{format_float(entry['reduced_objective']):>11} | "
            f"{format_float(trial_objective):>9}"
        )
        print(
            f"     current_cost={format_vector(entry['current_cost'])} "
            f"trial_cost={format_vector(entry['trial_cost'])} "
            f"class_loss={format_vector(entry['class_loss_vector'])}"
        )


def print_block_a_summaries(model):
    """
    Print one summary row per inner Block A solve.
    """
    rows = summarize_block_a_solves(model)
    print("\nBlock A solve summary")
    print(
        "Phase   | Outer iter | Accepted | Inner iters | "
        "Start obj  | End obj    | Delta"
    )
    for row in rows:
        print(
            f"{row['phase']:<7} | "
            f"{row['outer_iteration']:>10d} | "
            f"{str(row['accepted']):>8} | "
            f"{row['iterations']:>11d} | "
            f"{format_float(row['start_objective']):>10} | "
            f"{format_float(row['end_objective']):>10} | "
            f"{format_float(row['objective_change']):>10}"
        )


def print_profile_metric_table(title, rows, max_rows=None):
    """
    Print a compact profile-level metric table.
    """
    print(f"\n{title}")
    print(
        "Code | Samples | Accuracy  | Precision | Recall    | Specificity | "
        "F1        | GMEAN      | BCE       | Cost-BCE"
    )

    display_rows = rows if max_rows is None else rows[:max_rows]
    for row in display_rows:
        print(
            f"{row['profile_code']:>4} | "
            f"{row['n_samples']:>7d} | "
            f"{format_float(row['accuracy']):>9} | "
            f"{format_float(row['precision']):>9} | "
            f"{format_float(row['recall']):>9} | "
            f"{format_float(row['specificity']):>11} | "
            f"{format_float(row['f1']):>9} | "
            f"{format_float(row['gmean']):>9} | "
            f"{format_float(row['bce']):>9} | "
            f"{format_float(row['cost_bce']):>8}"
        )


def serialize_for_json(value):
    """
    Convert numpy-heavy structures into JSON-friendly values.
    """
    if isinstance(value, dict):
        return {str(key): serialize_for_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize_for_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def stringify_for_csv(value):
    """
    Convert values into CSV-friendly strings.
    """
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (float, int, str, bool)) or value is None:
        return value
    return json.dumps(serialize_for_json(value))


def write_csv(path, rows):
    """
    Write a list of dictionaries to CSV.
    """
    path = Path(path)
    if not rows:
        with path.open("w", newline="", encoding="utf-8") as handle:
            handle.write("")
        return

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: stringify_for_csv(row.get(key)) for key in fieldnames})


def save_json(path, payload):
    """
    Write JSON with indentation.
    """
    path = Path(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(serialize_for_json(payload), handle, indent=2)


def build_output_directory(base_dir, dataset, run_name=None):
    """
    Create a timestamped output directory for one training run.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{run_name}" if run_name else ""
    output_dir = Path(base_dir) / f"{dataset}_{timestamp}{suffix}"
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def save_run_outputs(
    output_dir,
    args,
    model,
    train_summary,
    test_summary,
    train_exact_profile_rows,
    test_exact_profile_rows,
    train_optimization_group_rows,
    test_optimization_group_rows,
    split_summary,
    preprocessing_summary,
):
    """
    Save training reports, histories, and parameter snapshots to disk.
    """
    output_dir = Path(output_dir)
    block_a_history_rows = flatten_block_a_histories(model)
    block_a_summary_rows = summarize_block_a_solves(model)

    save_json(output_dir / "config.json", vars(args))
    save_json(output_dir / "split_metadata.json", split_summary)
    save_json(output_dir / "preprocessing.json", preprocessing_summary)
    save_json(output_dir / "split_summary.json", split_summary)
    save_json(output_dir / "preprocessing_summary.json", preprocessing_summary)
    save_json(output_dir / "train_summary.json", train_summary)
    if test_summary is not None:
        save_json(output_dir / "test_summary.json", test_summary)
    save_json(output_dir / "training_summary.json", train_summary)
    save_json(
        output_dir / "final_parameters.json",
        {
            "costs": model["costs"],
            "baseline_costs": model["baseline_costs"],
            "alphas": model["parameter_history"][-1]["alphas"],
            "betas": model["parameter_history"][-1]["betas"],
            "lambda_beta": model["lambda_beta"],
            "alpha_l1_radius": model["alpha_l1_radius"],
            "source_order": model["source_order"],
        },
    )
    save_json(output_dir / "parameter_history.json", model.get("parameter_history", []))

    write_csv(output_dir / "bcd_history.csv", model.get("loss_history", []))
    write_csv(output_dir / "block_a_history.csv", block_a_history_rows)
    write_csv(output_dir / "block_a_solve_summary.csv", block_a_summary_rows)
    write_csv(output_dir / "train_exact_profile_metrics.csv", train_exact_profile_rows)
    if test_summary is not None:
        write_csv(output_dir / "test_exact_profile_metrics.csv", test_exact_profile_rows)
    write_csv(
        output_dir / "train_optimization_group_metrics.csv",
        train_optimization_group_rows,
    )
    if test_summary is not None:
        write_csv(
            output_dir / "test_optimization_group_metrics.csv",
            test_optimization_group_rows,
        )
    write_csv(output_dir / "exact_profile_metrics.csv", train_exact_profile_rows)
    write_csv(output_dir / "optimization_group_metrics.csv", train_optimization_group_rows)


def build_argument_parser():
    """
    Construct the CLI parser.
    """
    parser = argparse.ArgumentParser(
        description="Train the current iSFS logistic model on real or synthetic data."
    )
    parser.add_argument(
        "--dataset",
        choices=["real", "synthetic"],
        default="real",
        help="Which dataset to train on.",
    )
    parser.add_argument("--learning-rate-beta", type=float, default=1e-2)
    parser.add_argument("--learning-rate-alpha", type=float, default=1e-2)
    parser.add_argument("--lambda-beta", type=float, default=1e-2)
    parser.add_argument("--alpha-l1-radius", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=12)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--block-a-max-iter", type=int, default=25)
    parser.add_argument("--block-a-tol", type=float, default=1e-6)
    parser.add_argument("--alpha-subproblem-max-iter", type=int, default=25)
    parser.add_argument("--alpha-subproblem-tol", type=float, default=1e-6)
    parser.add_argument("--beta-subproblem-max-iter", type=int, default=25)
    parser.add_argument("--beta-subproblem-tol", type=float, default=1e-6)
    parser.add_argument("--initial-trust-region-radius", type=float, default=0.02)
    parser.add_argument("--max-trust-region-radius", type=float, default=0.125)
    parser.add_argument("--accept-threshold", type=float, default=0.25)
    parser.add_argument("--expand-threshold", type=float, default=0.75)
    parser.add_argument("--shrink-factor", type=float, default=0.25)
    parser.add_argument("--expand-factor", type=float, default=2.0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--split-random-state", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument("--decision-threshold", type=float, default=0.5)
    parser.add_argument(
        "--initial-cost-vector",
        type=str,
        default="0.5,0.5",
        help='Comma-separated initial cost vector, e.g. "0.5,0.5".',
    )
    parser.add_argument(
        "--baseline-cost-vector",
        type=str,
        default="0.5,0.5",
        help='Comma-separated baseline cost vector, e.g. "0.5,0.5".',
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="training_outputs",
        help="Directory where per-run reports will be saved.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional suffix added to the saved run directory name.",
    )
    parser.add_argument(
        "--no-save-results",
        action="store_true",
        help="Disable writing CSV/JSON reports to disk.",
    )
    parser.add_argument(
        "--in-sample-evaluation",
        "--no-test-split",
        dest="in_sample_evaluation",
        action="store_true",
        help="Train and evaluate on all rows, matching the old behavior.",
    )
    parser.add_argument(
        "--no-standardize",
        "--no-preprocess",
        dest="no_standardize",
        action="store_true",
        help="Disable train-only feature standardization.",
    )
    return parser


def print_configuration(args, X_blocks, y, initial_cost_vector, baseline_cost_vector):
    """
    Print the training configuration before fitting.
    """
    print("Training configuration")
    print(f"Dataset: {args.dataset}")
    print(f"Sources: {list(X_blocks.keys())}")
    print(f"Samples: {len(y)}")
    print(
        "Label counts: "
        f"class 0 = {int(np.sum(y == 0))}, class 1 = {int(np.sum(y == 1))}"
    )
    print_iteration_limits(args)
    print(
        "Learning rates: "
        f"alpha={args.learning_rate_alpha}, beta={args.learning_rate_beta}"
    )
    print(f"lambda_beta: {args.lambda_beta}")
    print(f"alpha_l1_radius: {args.alpha_l1_radius}")
    print(f"Decision threshold: {args.decision_threshold}")
    if args.in_sample_evaluation:
        print("Evaluation split: in-sample all rows")
    else:
        print(f"Evaluation split: stratified holdout (test_size={args.test_size})")
    print(f"Feature standardization: {not args.no_standardize}")
    print(f"Initial cost vector: {initial_cost_vector}")
    print(f"Baseline cost vector: {baseline_cost_vector}")


def main():
    parser = build_argument_parser()
    args = parser.parse_args()

    X_blocks, y = load_dataset(args.dataset, random_state=args.random_state)
    initial_cost_vector = parse_cost_vector(args.initial_cost_vector)
    baseline_cost_vector = parse_cost_vector(args.baseline_cost_vector)

    print_configuration(args, X_blocks, y, initial_cost_vector, baseline_cost_vector)
    print_bcd_explanation()

    if args.in_sample_evaluation:
        all_indices = np.arange(len(y), dtype=int)
        split = {
            "train_indices": all_indices,
            "test_indices": np.asarray([], dtype=int),
            "train_X_blocks": subset_X_blocks(X_blocks, all_indices),
            "test_X_blocks": None,
            "train_y": y.copy(),
            "test_y": None,
            "stratification_labels": build_stratification_labels(X_blocks, y),
        }
        split_summary = {
            "enabled": False,
            "strategy": "in_sample_all_rows",
            "train_size": int(len(y)),
            "test_size": 0,
            "train_indices": all_indices.tolist(),
            "test_indices": [],
        }
    else:
        split = split_dataset(
            X_blocks=X_blocks,
            y=y,
            test_size=args.test_size,
            random_state=args.split_random_state,
        )
        split_summary = summarize_split(split)
        split_summary["test_fraction_requested"] = float(args.test_size)
        split_summary["random_state"] = int(args.split_random_state)

    print_split_details(
        split_summary=split_summary,
        split=split,
        source_order=list(X_blocks.keys()),
    )

    if args.no_standardize:
        train_X_blocks = split["train_X_blocks"]
        test_X_blocks = split["test_X_blocks"]
        preprocessing_summary = {
            "enabled": False,
            "method": "raw_features_partial_nans_zero_filled_inside_model",
        }
    elif split["test_X_blocks"] is None:
        preprocessor = fit_standardization_preprocessor(split["train_X_blocks"])
        train_X_blocks = transform_with_preprocessor(split["train_X_blocks"], preprocessor)
        test_X_blocks = None
        preprocessing_summary = {
            "enabled": True,
            "method": "train_mean_impute_partial_nans_then_standardize",
            "preserves_all_missing_source_rows": True,
            "sources": preprocessor,
        }
    else:
        train_X_blocks, test_X_blocks, preprocessor = fit_transform_train_test(
            split["train_X_blocks"],
            split["test_X_blocks"],
        )
        preprocessing_summary = {
            "enabled": True,
            "method": "train_mean_impute_partial_nans_then_standardize",
            "preserves_all_missing_source_rows": True,
            "sources": preprocessor,
        }

    print_preprocessing_details(preprocessing_summary)

    train_y = split["train_y"]
    test_y = split["test_y"]

    model = train_blockwise_logistic_model(
        X_blocks=train_X_blocks,
        y=train_y,
        learning_rate_beta=args.learning_rate_beta,
        learning_rate_alpha=args.learning_rate_alpha,
        lambda_beta=args.lambda_beta,
        alpha_l1_radius=args.alpha_l1_radius,
        alpha_subproblem_max_iter=args.alpha_subproblem_max_iter,
        alpha_subproblem_tol=args.alpha_subproblem_tol,
        beta_subproblem_max_iter=args.beta_subproblem_max_iter,
        beta_subproblem_tol=args.beta_subproblem_tol,
        max_iter=args.max_iter,
        tol=args.tol,
        random_state=args.random_state,
        block_a_max_iter=args.block_a_max_iter,
        block_a_tol=args.block_a_tol,
        initial_cost_vector=initial_cost_vector,
        baseline_cost_vector=baseline_cost_vector,
        initial_trust_region_radius=args.initial_trust_region_radius,
        max_trust_region_radius=args.max_trust_region_radius,
        accept_threshold=args.accept_threshold,
        expand_threshold=args.expand_threshold,
        shrink_factor=args.shrink_factor,
        expand_factor=args.expand_factor,
        verbose=True,
    )

    train_summary = compute_training_summary(
        model=model,
        X_blocks=train_X_blocks,
        y=train_y,
        threshold=args.decision_threshold,
    )
    test_summary = None
    train_exact_profile_rows = compute_profile_metrics(
        model=model,
        X_blocks=train_X_blocks,
        y=train_y,
        profiles=model["exact_profiles"],
        threshold=args.decision_threshold,
    )
    train_optimization_group_rows = compute_profile_metrics(
        model=model,
        X_blocks=train_X_blocks,
        y=train_y,
        profiles=model["optimization_groups"],
        threshold=args.decision_threshold,
    )
    test_exact_profile_rows = []
    test_optimization_group_rows = []
    if test_X_blocks is not None and test_y is not None:
        test_summary = compute_training_summary(
            model=model,
            X_blocks=test_X_blocks,
            y=test_y,
            threshold=args.decision_threshold,
        )
        test_exact_profile_rows = compute_profile_metrics(
            model=model,
            X_blocks=test_X_blocks,
            y=test_y,
            profiles=build_exact_profiles(test_X_blocks, source_order=model["source_order"]),
            threshold=args.decision_threshold,
        )
        test_optimization_group_rows = compute_profile_metrics(
            model=model,
            X_blocks=test_X_blocks,
            y=test_y,
            profiles=build_missingness_profiles(test_X_blocks, source_order=model["source_order"]),
            threshold=args.decision_threshold,
        )

    print_training_metrics(train_summary, title="Train performance")
    if test_summary is not None:
        print_training_metrics(test_summary, title="Test performance")
    print_parameter_summary(model)
    print_bcd_history(model)
    print_block_a_summaries(model)
    print_profile_metric_table("Exact-profile train metrics", train_exact_profile_rows)
    print_profile_metric_table(
        "Optimization-group train metrics",
        train_optimization_group_rows,
    )
    if test_summary is not None:
        print_profile_metric_table("Exact-profile test metrics", test_exact_profile_rows)
        print_profile_metric_table(
            "Optimization-group test metrics",
            test_optimization_group_rows,
        )

    if not args.no_save_results:
        output_dir = build_output_directory(
            base_dir=args.output_dir,
            dataset=args.dataset,
            run_name=args.run_name,
        )
        save_run_outputs(
            output_dir=output_dir,
            args=args,
            model=model,
            train_summary=train_summary,
            test_summary=test_summary,
            train_exact_profile_rows=train_exact_profile_rows,
            test_exact_profile_rows=test_exact_profile_rows,
            train_optimization_group_rows=train_optimization_group_rows,
            test_optimization_group_rows=test_optimization_group_rows,
            split_summary=split_summary,
            preprocessing_summary=preprocessing_summary,
        )
        print(f"\nSaved reports to: {output_dir}")


if __name__ == "__main__":
    main()
