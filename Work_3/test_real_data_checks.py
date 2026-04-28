"""
Paper note:
This file is the real-data audit script for the current modular implementation.
It does not create toy data inside the script. Instead, it reads the actual CSV
files in `data/` and checks whether the code follows the paper and the
cost-sensitive trust-region notes on the dataset you are using now.

The checks are organized around the active building blocks:

    1. Missingness profiles:
       verify that Section 3 profile construction is driven only by source
       availability in the real CSV rows.

    2. Logistic loss:
       verify that the BCE objective and its gradient are numerically stable on
       real labels and real source-derived logits.

    3. Cost block:
       verify that the class-loss vector

           L(alpha, beta) = [L_1(alpha, beta), ..., L_K(alpha, beta)]^T

       is computed on the real data, and that the trust-region cost update
       satisfies the same feasibility conditions written in `notes.tex`:

           (1 / 2) * ||d||_2^2 <= Delta_k,
           C + d >= 0,
           (1 / 2) * ||C + d - C^0||_2^2 <= Delta_max.

    4. Prediction:
       verify that exact-profile inference runs on the real data and returns
       finite probabilities for samples with at least one observed source.

    5. Regularization:
       verify that the current paper-aligned regularization path keeps alpha
       inside its l1-ball and measures the beta l1 penalty on real data.

Use:

    --target profiles   to inspect exact and overlapping iSFS groups
    --target loss       to inspect BCE and its finite-difference gradient check
    --target costs      to inspect L(alpha, beta) and one trust-region cost step
    --target prediction to inspect exact-profile inference on real data
    --target regularization to inspect alpha/beta regularization on real data
    --target all           to print every real-data check in sequence
"""

import argparse
from pathlib import Path

import numpy as np

from loss_functions import (
    binary_cross_entropy_from_logits,
    binary_cross_entropy_gradient_from_logits,
)
from alpha_update import initialize_alphas
from alpha_update import update_alphas
from beta_update import initialize_betas
from beta_update import update_betas
from costs import compute_class_loss_vector, solve_cost_trust_region_subproblem
from missingness_profiles import (
    build_exact_profiles,
    build_missingness_profiles,
    print_profiles,
)
from prediction import predict_class, predict_proba
from regularization import beta_l1_penalty, max_alpha_l1_norm


DATA_DIR = Path(__file__).resolve().parent / "data"
SOURCE_ORDER = ["source1", "source2", "source3"]
SOURCE_FILES = {
    "source1": "source1_clinical_blockwise_missing.csv",
    "source2": "source2_imaging_blockwise_missing.csv",
    "source3": "source3_proteomics_blockwise_missing.csv",
}


def load_real_blockwise_sources():
    """
    Load the real blockwise-missing source matrices from the data folder.
    """
    X_blocks = {}

    for source, filename in SOURCE_FILES.items():
        X = np.genfromtxt(
            DATA_DIR / filename,
            delimiter=",",
            skip_header=1,
        )

        if X.ndim == 1:
            X = X.reshape(1, -1)

        X_blocks[source] = X

    return X_blocks


def load_real_labels():
    """
    Load the real binary labels from labels.csv.
    """
    labels = np.genfromtxt(
        DATA_DIR / "labels.csv",
        delimiter=",",
        skip_header=1,
    )

    if labels.ndim == 1:
        labels = labels.reshape(1, -1)

    return labels[:, 1].astype(int)


def _format_profile_counts(profiles):
    """
    Convert tuple-keyed profiles into paper-style binary-code counts.
    """
    return {
        "".join("1" if source in profile else "0" for source in SOURCE_ORDER): len(indices)
        for profile, indices in sorted(
            profiles.items(),
            key=lambda item: (len(item[0]), item[0]),
        )
    }


def run_profiles_check():
    """
    Print exact participant profiles and overlapping iSFS groups on real data.
    """
    X_blocks = load_real_blockwise_sources()
    exact_profiles = build_exact_profiles(X_blocks, source_order=SOURCE_ORDER)
    optimization_groups = build_missingness_profiles(
        X_blocks,
        source_order=SOURCE_ORDER,
    )

    print("Real-data profile check")
    print(f"Samples: {next(iter(X_blocks.values())).shape[0]}")
    print()
    print_profiles(
        exact_profiles,
        source_order=SOURCE_ORDER,
        title="Exact Missingness Profiles",
    )
    print_profiles(
        optimization_groups,
        source_order=SOURCE_ORDER,
        title="iSFS Optimization Groups",
    )
    print()
    print("Exact profile counts:", _format_profile_counts(exact_profiles))
    print("Optimization group counts:", _format_profile_counts(optimization_groups))


def build_real_loss_logits():
    """
    Create deterministic logits from the real clinical source for BCE checks.

    This is not the full iSFS model. It is only a real-data loss sanity check.
    """
    X = load_real_blockwise_sources()["source1"]

    column_means = np.nanmean(X, axis=0)
    column_means = np.where(np.isfinite(column_means), column_means, 0.0)
    X_filled = np.where(np.isnan(X), column_means, X)

    column_std = X_filled.std(axis=0)
    column_std = np.where(column_std > 0, column_std, 1.0)
    X_scaled = (X_filled - X_filled.mean(axis=0)) / column_std

    weights = np.zeros(X_scaled.shape[1], dtype=float)
    template = np.array([0.80, -0.60, 0.40, -0.20, 0.10], dtype=float)
    weights[: min(len(template), X_scaled.shape[1])] = template[: X_scaled.shape[1]]

    return X_scaled @ weights


def run_loss_check():
    """
    Print BCE and a gradient finite-difference check on the real data.
    """
    y = load_real_labels()
    logits = build_real_loss_logits()

    loss = binary_cross_entropy_from_logits(logits, y)
    grad = binary_cross_entropy_gradient_from_logits(logits, y)

    check_indices = np.linspace(
        0,
        len(logits) - 1,
        num=min(25, len(logits)),
        dtype=int,
    )
    finite_diff_errors = []
    eps = 1e-6

    for idx in check_indices:
        plus = logits.copy()
        minus = logits.copy()
        plus[idx] += eps
        minus[idx] -= eps

        finite_diff = (
            binary_cross_entropy_from_logits(plus, y)
            - binary_cross_entropy_from_logits(minus, y)
        ) / (2 * eps)

        finite_diff_errors.append(abs(finite_diff - grad[idx]))

    print("Real-data loss check")
    print(f"Samples: {len(y)}")
    print(f"Label counts: class 0 = {(y == 0).sum()}, class 1 = {(y == 1).sum()}")
    print("Logits source: standardized source1 clinical blockwise-missing data")
    print(f"BCE: {loss:.6f}")
    print(f"Mean absolute gradient: {np.mean(np.abs(grad)):.8f}")
    print(
        "Max finite-difference error on sampled logits: "
        f"{max(finite_diff_errors):.3e}"
    )


def run_cost_check():
    """
    Print the class-loss vector and one trust-region cost update on real data.
    """
    X_blocks = load_real_blockwise_sources()
    y = load_real_labels()
    optimization_groups = build_missingness_profiles(
        X_blocks,
        source_order=SOURCE_ORDER,
    )

    betas = initialize_betas(X_blocks, random_state=42)
    alphas = initialize_alphas(optimization_groups)

    class_loss_vector = compute_class_loss_vector(
        X_blocks=X_blocks,
        y=y,
        betas=betas,
        alphas=alphas,
        optimization_groups=optimization_groups,
        n_classes=2,
    )

    baseline_cost = np.array([0.5, 0.5], dtype=float)
    step_info = solve_cost_trust_region_subproblem(
        current_cost=baseline_cost,
        baseline_cost=baseline_cost,
        class_loss_vector=class_loss_vector,
        working_radius=0.125,
        max_radius=0.125,
    )

    displacement = step_info["trial_cost"] - baseline_cost

    print("Real-data cost check")
    print(f"Class-loss vector L(alpha, beta): {class_loss_vector}")
    print(f"Baseline cost C0: {baseline_cost}")
    print(f"Trial cost from trust region: {step_info['trial_cost']}")
    print(f"Predicted increase L^T d: {step_info['predicted_increase']:.6f}")
    print(f"Step norm ||d||_2: {step_info['step_norm']:.6f}")
    print(
        "Global-ball check 0.5||C-C0||^2: "
        f"{0.5 * np.dot(displacement, displacement):.6f}"
    )
    print(f"Nonnegativity satisfied: {bool(np.all(step_info['trial_cost'] >= 0.0))}")


def run_prediction_check():
    """
    Print a real-data sanity check for the exact-profile prediction path.
    """
    X_blocks = load_real_blockwise_sources()
    optimization_groups = build_missingness_profiles(
        X_blocks,
        source_order=SOURCE_ORDER,
    )

    betas = initialize_betas(X_blocks, random_state=42)
    alphas = initialize_alphas(optimization_groups)

    probabilities = predict_proba(
        X_blocks=X_blocks,
        betas=betas,
        alphas=alphas,
        source_order=SOURCE_ORDER,
    )
    predictions = predict_class(
        X_blocks=X_blocks,
        betas=betas,
        alphas=alphas,
        source_order=SOURCE_ORDER,
    )

    valid_mask = ~np.isnan(probabilities)

    print("Real-data prediction check")
    print(f"Samples: {len(probabilities)}")
    print(f"Valid probability count: {int(np.sum(valid_mask))}")
    print(
        "Probability range on valid samples: "
        f"[{np.min(probabilities[valid_mask]):.6f}, "
        f"{np.max(probabilities[valid_mask]):.6f}]"
    )
    print(
        "Predicted class counts on valid samples: "
        f"class 0 = {int(np.sum(predictions[valid_mask] == 0))}, "
        f"class 1 = {int(np.sum(predictions[valid_mask] == 1))}"
    )
    print(
        "All valid probabilities finite: "
        f"{bool(np.all(np.isfinite(probabilities[valid_mask])))}"
    )


def run_regularization_check():
    """
    Print a real-data sanity check for the active l1 regularization path.
    """
    X_blocks = load_real_blockwise_sources()
    y = load_real_labels()
    optimization_groups = build_missingness_profiles(
        X_blocks,
        source_order=SOURCE_ORDER,
    )

    betas = initialize_betas(X_blocks, random_state=42)
    alphas = initialize_alphas(optimization_groups)
    cost_vector = np.array([0.5, 0.5], dtype=float)
    lambda_beta = 1e-2
    alpha_l1_radius = 1.0

    alphas = update_alphas(
        X_blocks=X_blocks,
        y=y,
        betas=betas,
        alphas=alphas,
        profiles=optimization_groups,
        cost_vector=cost_vector,
        learning_rate_alpha=1e-2,
        alpha_l1_radius=alpha_l1_radius,
    )
    betas = update_betas(
        X_blocks=X_blocks,
        y=y,
        betas=betas,
        alphas=alphas,
        profiles=optimization_groups,
        cost_vector=cost_vector,
        learning_rate_beta=1e-2,
        lambda_beta=lambda_beta,
    )

    print("Real-data regularization check")
    print(f"Alpha max l1 norm after one update: {max_alpha_l1_norm(alphas):.6f}")
    print(f"Beta l1 penalty after one update: {beta_l1_penalty(betas, lambda_beta):.6f}")
    print(
        "All alpha profiles satisfy ||alpha_m||_1 <= 1: "
        f"{bool(max_alpha_l1_norm(alphas) <= alpha_l1_radius + 1e-9)}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run real-data checks for iSFS profiles, loss, costs, prediction, and regularization."
    )
    parser.add_argument(
        "--target",
        choices=["profiles", "loss", "costs", "prediction", "regularization", "all"],
        default="all",
        help="Which real-data check to run.",
    )
    args = parser.parse_args()

    if args.target in {"profiles", "all"}:
        run_profiles_check()
        if args.target == "all":
            print("\n" + "=" * 72 + "\n")

    if args.target in {"loss", "all"}:
        run_loss_check()
        if args.target == "all":
            print("\n" + "=" * 72 + "\n")

    if args.target in {"costs", "all"}:
        run_cost_check()
        if args.target == "all":
            print("\n" + "=" * 72 + "\n")

    if args.target in {"prediction", "all"}:
        run_prediction_check()
        if args.target == "all":
            print("\n" + "=" * 72 + "\n")

    if args.target in {"regularization", "all"}:
        run_regularization_check()


if __name__ == "__main__":
    main()
