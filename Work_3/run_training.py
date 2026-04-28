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
    - final training metrics
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
    python3 run_training.py --dataset real --max-iter 10 --block-a-max-iter 40
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
from missingness_profiles import get_profile_code
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
    Compute training metrics on the same dataset used for fitting.
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


def print_training_metrics(summary):
    """
    Print a detailed training-metrics block.
    """
    print("\nTraining performance")
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
        "F1        | BCE       | Cost-BCE"
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
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
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
    training_summary,
    exact_profile_rows,
    optimization_group_rows,
):
    """
    Save training reports, histories, and parameter snapshots to disk.
    """
    output_dir = Path(output_dir)
    block_a_history_rows = flatten_block_a_histories(model)
    block_a_summary_rows = summarize_block_a_solves(model)

    save_json(output_dir / "config.json", vars(args))
    save_json(output_dir / "training_summary.json", training_summary)
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
    write_csv(output_dir / "exact_profile_metrics.csv", exact_profile_rows)
    write_csv(output_dir / "optimization_group_metrics.csv", optimization_group_rows)


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
    parser.add_argument("--max-iter", type=int, default=6)
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
    print(f"Outer max_iter: {args.max_iter}")
    print(f"Block A max_iter: {args.block_a_max_iter}")
    print(
        "Alpha/Beta subproblem max_iter: "
        f"{args.alpha_subproblem_max_iter}/{args.beta_subproblem_max_iter}"
    )
    print(
        "Learning rates: "
        f"alpha={args.learning_rate_alpha}, beta={args.learning_rate_beta}"
    )
    print(f"lambda_beta: {args.lambda_beta}")
    print(f"alpha_l1_radius: {args.alpha_l1_radius}")
    print(f"Decision threshold: {args.decision_threshold}")
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

    model = train_blockwise_logistic_model(
        X_blocks=X_blocks,
        y=y,
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

    training_summary = compute_training_summary(
        model=model,
        X_blocks=X_blocks,
        y=y,
        threshold=args.decision_threshold,
    )
    exact_profile_rows = compute_profile_metrics(
        model=model,
        X_blocks=X_blocks,
        y=y,
        profiles=model["exact_profiles"],
        threshold=args.decision_threshold,
    )
    optimization_group_rows = compute_profile_metrics(
        model=model,
        X_blocks=X_blocks,
        y=y,
        profiles=model["optimization_groups"],
        threshold=args.decision_threshold,
    )

    print_training_metrics(training_summary)
    print_parameter_summary(model)
    print_bcd_history(model)
    print_block_a_summaries(model)
    print_profile_metric_table("Exact-profile training metrics", exact_profile_rows)
    print_profile_metric_table(
        "Optimization-group training metrics",
        optimization_group_rows,
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
            training_summary=training_summary,
            exact_profile_rows=exact_profile_rows,
            optimization_group_rows=optimization_group_rows,
        )
        print(f"\nSaved reports to: {output_dir}")


if __name__ == "__main__":
    main()
