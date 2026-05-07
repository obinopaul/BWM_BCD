"""
Paper note:
This file implements the outer block-coordinate-descent loop for the
cost-sensitive extension of the incomplete-source feature-selection model.

Reference structure:

    - Section 3 of `Multi_source_BWM.pdf` supplies the alternating logic for the
      iSFS variables alpha and beta.
    - `notes.tex` supplies the trust-region derivation for the class-cost vector
      C.

Notation carried from the paper:

    - pf: the set of optimization groups built from missingness profiles.
    - G_m: the sample indices belonging to optimization group m.
    - n_m = |G_m|.
    - X_m^i: source-i feature block restricted to the samples in G_m.
    - beta^i: shared coefficient vector for source i.
    - alpha_m^i: group-specific source weight for source i inside group m.

For every optimization group m, the blockwise predictor is

    z_m = sum_{i in m} alpha_m^i X_m^i beta^i.

The original paper writes its objective with least squares. This repository
keeps the same group structure but replaces the data-fit term with BCE from
logits:

    ell(z_{m,j}, y_{m,j})
    = log(1 + exp(z_{m,j})) - y_{m,j} z_{m,j}.

When class costs C are introduced, the group loss becomes

    f_m^C(alpha, beta, C)
    = (1 / n_m) * sum_{j in G_m} C_{y_{m,j}} * ell(z_{m,j}, y_{m,j}).

The fixed-cost Block A objective is therefore

    Phi(alpha, beta ; C)
    = (1 / |pf|) * sum_{m in pf} f_m^C(alpha, beta, C).

Block A follows the same alternating spirit as the paper:

    alpha-step:
        fix beta and C, then update alpha

    beta-step:
        fix alpha and C, then update beta

The C-block uses the classwise loss coefficients from `notes.tex`:

    L_k(alpha, beta)
    = (1 / |pf|) * sum_{m in pf} (1 / n_m)
      * sum_{j in G_m : y_{m,j} = k} ell(z_{m,j}, y_{m,j}).

In vector form,

    L(alpha, beta) = [L_1(alpha, beta), ..., L_K(alpha, beta)]^T,

and the cost-only objective is

    Phi_C(C) = L(alpha, beta)^T C.

Because this is linear in C when alpha and beta are fixed, the cost update is
posed as a trust-region subproblem around a baseline cost vector C^0:

    maximize_d L_t^T d

subject to

    (1 / 2) * ||d||_2^2 <= Delta_t,
    C^(t) + d >= 0,
    (1 / 2) * ||C^(t) + d - C^0||_2^2 <= Delta_max.

The reduced objective for the cost block is

    F(C) = min_{alpha, beta} Phi(alpha, beta ; C).

This matters because the trust-region acceptance test should not compare only
the linear model prediction L_t^T d. It should compare the predicted increase
against the actual change in the reduced objective after Block A is re-solved at
the trial cost:

    predicted increase = L_t^T d_t

    actual increase
    = F(C^(t) + d_t) - F(C^(t)).

The trust-region ratio is therefore

    rho_t
    = [F(C^(t) + d_t) - F(C^(t))] / [L_t^T d_t].

Implementation map:

    1. Build exact profiles and overlapping optimization groups from the source
       availability pattern.
    2. Initialize beta, alpha, and the current cost vector C.
    3. Solve the alpha subproblem with beta fixed.
    4. Solve the beta subproblem with alpha fixed.
    5. Compute the class-loss vector L(alpha, beta).
    6. Solve the linear trust-region C-subproblem to get a trial cost.
    7. Re-solve Block A at the trial cost to evaluate F(C_trial).
    8. Accept or reject the cost step using the trust-region ratio, then update
       the working trust-region radius.

For the active path, we follow the paper's experimental choice and use the
l1-norm for both regularizers:

    R_alpha(alpha_m) = ||alpha_m||_1
    R_beta(beta) = sum_i ||beta^i||_1.

So the fixed-cost Block A problem solved in code is

    minimize_{alpha, beta}
        (1 / |pf|) * sum_{m in pf} f_m^C(alpha, beta, C)
        + lambda_beta * sum_i ||beta^i||_1

    subject to
        ||alpha_m||_1 <= r_alpha
        for every m in pf.

The alpha-step is therefore projected gradient on the l1-ball, while the
beta-step is proximal gradient with soft-thresholding.

Audit note:
Algorithm 2 in the paper solves the alpha subproblem and the beta subproblem in
alternation. The active implementation therefore uses inner solvers for both
subproblems inside Block A, rather than alternating only single alpha and beta
steps.
"""

import copy

import numpy as np

from alpha_update import initialize_alphas, solve_alphas_with_fixed_betas
from beta_update import initialize_betas, solve_betas_with_fixed_alphas
from costs import (
    compute_class_loss_vector,
    compute_trust_region_ratio,
    cost_weighted_bce_from_logits,
    infer_n_classes,
    solve_cost_trust_region_subproblem,
    update_trust_region_radius,
    validate_cost_vector,
)
from loss import binary_cross_entropy
from missingness_profiles import (
    build_exact_profiles,
    build_missingness_profiles,
    print_profiles,
)
from prediction import compute_all_logits, predict_class, predict_proba
from regularization import (
    beta_l1_penalty,
    max_alpha_constraint_violation,
    max_alpha_l1_norm,
)


def _profile_key(profile):
    """
    Convert a tuple profile into a stable printable key.
    """
    return "|".join(profile)


def _serialize_alphas(alphas):
    """
    Convert alpha dictionaries into JSON-friendly nested dictionaries.
    """
    return {
        _profile_key(profile): {
            source: float(value)
            for source, value in alpha_values.items()
        }
        for profile, alpha_values in alphas.items()
    }


def _serialize_betas(betas):
    """
    Convert beta arrays into JSON-friendly lists.
    """
    return {
        source: np.asarray(beta, dtype=float).reshape(-1).tolist()
        for source, beta in betas.items()
    }


def _build_parameter_snapshot(
    stage,
    outer_iteration,
    cost_vector,
    alphas,
    betas,
    reduced_objective,
    objective_parts,
    accepted=None,
):
    """
    Capture a model-state snapshot for reporting and later inspection.
    """
    snapshot = {
        "stage": stage,
        "outer_iteration": int(outer_iteration),
        "accepted": accepted,
        "costs": np.asarray(cost_vector, dtype=float).reshape(-1).tolist(),
        "reduced_objective": float(reduced_objective),
        "alphas": _serialize_alphas(alphas),
        "betas": _serialize_betas(betas),
    }

    for key, value in objective_parts.items():
        snapshot[key] = float(value)

    return snapshot


def _step_diagnostics(step_info):
    """
    Keep trust-region C-step diagnostics in saved BCD history rows.
    """
    return {
        "cost_step_norm": float(step_info["step_norm"]),
        "cost_step_active_constraint": step_info["active_constraint"],
        "cost_step_working_limit": float(step_info["working_step_limit"]),
        "cost_step_global_limit": float(step_info["global_step_limit"]),
        "cost_step_nonnegativity_limit": float(
            step_info["nonnegativity_step_limit"]
        ),
        "cost_baseline_distance": float(step_info["baseline_distance"]),
        "cost_baseline_radius": float(step_info["baseline_radius"]),
        "cost_baseline_margin": float(step_info["baseline_margin"]),
    }


def _format_step_limit(value):
    """
    Format finite and infinite step limits for terminal diagnostics.
    """
    value = float(value)
    if np.isposinf(value):
        return "inf"
    return f"{value:.6f}"


def _print_cost_step_diagnostics(step_info, indent="    "):
    """
    Print the active trust-region constraint for the C update.
    """
    print(
        f"{indent}Cost step: norm={step_info['step_norm']:.6f}, "
        f"active_constraint={step_info['active_constraint']}"
    )
    print(
        f"{indent}Step limits: "
        f"working={_format_step_limit(step_info['working_step_limit'])}, "
        f"global={_format_step_limit(step_info['global_step_limit'])}, "
        f"nonnegativity={_format_step_limit(step_info['nonnegativity_step_limit'])}"
    )
    print(
        f"{indent}Baseline ball: "
        f"distance={step_info['baseline_distance']:.6f}, "
        f"radius={step_info['baseline_radius']:.6f}, "
        f"remaining_margin={step_info['baseline_margin']:.6f}"
    )


def compute_block_a_objective(
    X_blocks,
    y,
    betas,
    alphas,
    cost_vector,
    optimization_groups,
    exact_profiles,
    lambda_beta,
    alpha_l1_radius,
):
    """
    Compute the fixed-C Block A objective and useful diagnostics.

    The objective minimized by Block A is

        Phi(alpha, beta ; C)
        = (1 / |pf|) * sum_{m in pf} f_m^C(alpha, beta, C)
          + lambda_beta * sum_i ||beta^i||_1

    subject to ||alpha_m||_1 <= alpha_l1_radius for every profile m.
    """
    logits_all, valid_mask = compute_all_logits(
        X_blocks=X_blocks,
        betas=betas,
        alphas=alphas,
        exact_profiles=exact_profiles,
    )

    y_valid = y[valid_mask]
    logits_valid = logits_all[valid_mask]

    sample_cost_bce = cost_weighted_bce_from_logits(
        logits_valid,
        y_valid,
        cost_vector,
    )
    sample_bce = binary_cross_entropy(logits_valid, y_valid)

    group_cost_losses = []
    group_bce_losses = []

    for profile, indices in optimization_groups.items():
        logits_group = np.zeros(len(indices), dtype=float)

        for source in profile:
            X_source = np.nan_to_num(X_blocks[source][indices, :], nan=0.0)
            logits_group += alphas[profile][source] * (X_source @ betas[source])

        y_group = y[indices]
        group_cost_losses.append(
            cost_weighted_bce_from_logits(
                logits_group,
                y_group,
                cost_vector,
            )
        )
        group_bce_losses.append(binary_cross_entropy(logits_group, y_group))

    group_cost_objective = float(np.mean(group_cost_losses))
    beta_penalty = beta_l1_penalty(betas, lambda_beta)
    objective = group_cost_objective + beta_penalty

    parts = {
        "group_cost_bce": group_cost_objective,
        "sample_cost_bce": float(sample_cost_bce),
        "group_bce": float(np.mean(group_bce_losses)),
        "sample_bce": float(sample_bce),
        "beta_l1_penalty": float(beta_penalty),
        "alpha_max_l1_norm": float(max_alpha_l1_norm(alphas)),
        "alpha_constraint_violation": float(
            max_alpha_constraint_violation(alphas, radius=alpha_l1_radius)
        ),
    }

    return objective, parts


def solve_block_a_with_fixed_cost(
    X_blocks,
    y,
    betas,
    alphas,
    cost_vector,
    optimization_groups,
    exact_profiles,
    learning_rate_alpha,
    learning_rate_beta,
    lambda_beta,
    alpha_l1_radius,
    alpha_subproblem_max_iter,
    alpha_subproblem_tol,
    beta_subproblem_max_iter,
    beta_subproblem_tol,
    max_iter,
    tol,
    verbose=False,
    prefix="Block A",
    print_every=1,
):
    """
    Solve the (alpha, beta) block approximately for a fixed class-cost vector C.

    This function follows the paper's alternating logic more closely:

        1. with beta fixed, approximately solve the constrained alpha subproblem
        2. with alpha fixed, approximately solve the regularized beta subproblem

    until the fixed-C Block A objective stabilizes.
    """
    history = []
    previous_objective = np.inf
    current_objective = np.inf
    current_parts = None

    for iteration in range(1, max_iter + 1):
        alphas = solve_alphas_with_fixed_betas(
            X_blocks=X_blocks,
            y=y,
            betas=betas,
            alphas=alphas,
            profiles=optimization_groups,
            cost_vector=cost_vector,
            learning_rate_alpha=learning_rate_alpha,
            alpha_l1_radius=alpha_l1_radius,
            max_iter=alpha_subproblem_max_iter,
            tol=alpha_subproblem_tol,
        )

        betas = solve_betas_with_fixed_alphas(
            X_blocks=X_blocks,
            y=y,
            betas=betas,
            alphas=alphas,
            profiles=optimization_groups,
            cost_vector=cost_vector,
            learning_rate_beta=learning_rate_beta,
            lambda_beta=lambda_beta,
            max_iter=beta_subproblem_max_iter,
            tol=beta_subproblem_tol,
        )

        current_objective, current_parts = compute_block_a_objective(
            X_blocks=X_blocks,
            y=y,
            betas=betas,
            alphas=alphas,
            cost_vector=cost_vector,
            optimization_groups=optimization_groups,
            exact_profiles=exact_profiles,
            lambda_beta=lambda_beta,
            alpha_l1_radius=alpha_l1_radius,
        )

        entry = {"iteration": iteration, "objective": current_objective, **current_parts}
        history.append(entry)

        if verbose and (
            iteration == 1
            or iteration % max(1, int(print_every)) == 0
            or iteration == max_iter
        ):
            print(
                f"{prefix} iter {iteration:04d} | "
                f"Obj: {current_objective:.6f} | "
                f"Group Cost-BCE: {current_parts['group_cost_bce']:.6f} | "
                f"Beta L1: {current_parts['beta_l1_penalty']:.6f} | "
                f"Sample Cost-BCE: {current_parts['sample_cost_bce']:.6f} | "
                f"Alpha Max L1: {current_parts['alpha_max_l1_norm']:.6f}"
            )

        objective_change = abs(previous_objective - current_objective)
        if objective_change < tol:
            break

        previous_objective = current_objective

    return betas, alphas, current_objective, current_parts, history


def _initialize_cost_vectors(y, initial_cost_vector, baseline_cost_vector):
    """
    Initialize the current and baseline cost vectors.
    """
    n_classes = infer_n_classes(y)

    if initial_cost_vector is None:
        initial_cost_vector = np.ones(n_classes, dtype=float) / n_classes
    else:
        initial_cost_vector = validate_cost_vector(initial_cost_vector, n_classes)

    if baseline_cost_vector is None:
        baseline_cost_vector = initial_cost_vector.copy()
    else:
        baseline_cost_vector = validate_cost_vector(baseline_cost_vector, n_classes)

    return initial_cost_vector.copy(), baseline_cost_vector.copy(), n_classes


def train_blockwise_logistic_model(
    X_blocks,
    y,
    learning_rate_beta=1e-2,
    learning_rate_alpha=1e-2,
    lambda_beta=1e-2,
    alpha_l1_radius=1.0,
    alpha_subproblem_max_iter=25,
    alpha_subproblem_tol=1e-6,
    beta_subproblem_max_iter=25,
    beta_subproblem_tol=1e-6,
    max_iter=25,
    tol=1e-6,
    random_state=42,
    block_a_max_iter=100,
    block_a_tol=1e-6,
    initial_cost_vector=None,
    baseline_cost_vector=None,
    initial_trust_region_radius=0.02,
    max_trust_region_radius=0.125,
    accept_threshold=0.25,
    expand_threshold=0.75,
    shrink_factor=0.25,
    expand_factor=2.0,
    verbose=True,
):
    """
    Train the cost-sensitive blockwise logistic model by BCD.

    Outer loop:

        Block A:  fix C and minimize over (alpha, beta)
        Block B:  fix (alpha, beta) and update C by trust region

    The returned model contains the final alpha, beta, and cost vector, plus the
    trust-region history for later inspection.
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    current_cost, baseline_cost, n_classes = _initialize_cost_vectors(
        y=y,
        initial_cost_vector=initial_cost_vector,
        baseline_cost_vector=baseline_cost_vector,
    )

    exact_profiles = build_exact_profiles(X_blocks)
    optimization_groups = build_missingness_profiles(X_blocks)

    if verbose:
        source_order = list(X_blocks.keys())
        print_profiles(
            exact_profiles,
            source_order=source_order,
            title="Exact Missingness Profiles",
        )
        print_profiles(
            optimization_groups,
            source_order=source_order,
            title="iSFS Optimization Groups",
        )
        print("\nInitial cost vector:", current_cost)
        print("Baseline cost vector:", baseline_cost)
        print(f"Beta l1 penalty weight: {lambda_beta}")
        print(f"Alpha l1 radius: {alpha_l1_radius}")

    betas = initialize_betas(X_blocks=X_blocks, random_state=random_state)
    alphas = initialize_alphas(optimization_groups)

    betas, alphas, current_reduced_objective, current_parts, initial_block_history = solve_block_a_with_fixed_cost(
        X_blocks=X_blocks,
        y=y,
        betas=betas,
        alphas=alphas,
        cost_vector=current_cost,
        optimization_groups=optimization_groups,
        exact_profiles=exact_profiles,
        learning_rate_alpha=learning_rate_alpha,
        learning_rate_beta=learning_rate_beta,
        lambda_beta=lambda_beta,
        alpha_l1_radius=alpha_l1_radius,
        alpha_subproblem_max_iter=alpha_subproblem_max_iter,
        alpha_subproblem_tol=alpha_subproblem_tol,
        beta_subproblem_max_iter=beta_subproblem_max_iter,
        beta_subproblem_tol=beta_subproblem_tol,
        max_iter=block_a_max_iter,
        tol=block_a_tol,
        verbose=verbose,
        prefix="Initial Block A",
    )

    working_radius = min(initial_trust_region_radius, max_trust_region_radius)
    block_a_history = [initial_block_history]
    trial_block_a_history = []
    bcd_history = []
    parameter_history = [
        _build_parameter_snapshot(
            stage="initial_block_a",
            outer_iteration=0,
            cost_vector=current_cost,
            alphas=alphas,
            betas=betas,
            reduced_objective=current_reduced_objective,
            objective_parts=current_parts,
            accepted=True,
        )
    ]

    for iteration in range(1, max_iter + 1):
        class_loss_vector = compute_class_loss_vector(
            X_blocks=X_blocks,
            y=y,
            betas=betas,
            alphas=alphas,
            optimization_groups=optimization_groups,
            n_classes=n_classes,
        )

        step_info = solve_cost_trust_region_subproblem(
            current_cost=current_cost,
            baseline_cost=baseline_cost,
            class_loss_vector=class_loss_vector,
            working_radius=working_radius,
            max_radius=max_trust_region_radius,
        )

        predicted_increase = step_info["predicted_increase"]
        trial_cost = step_info["trial_cost"]

        if predicted_increase <= 0.0 or step_info["step_norm"] < tol:
            if verbose:
                print(
                    f"BCD iter {iteration:03d} | "
                    f"Accepted: False | "
                    f"Termination: no_cost_step | "
                    f"Pred: {predicted_increase:.6f} | "
                    f"Step norm: {step_info['step_norm']:.6f}"
                )
                print(
                    "    No trust-region cost step was taken. "
                    "The current C, alpha, and beta were kept."
                )
                _print_cost_step_diagnostics(step_info)
            bcd_history.append(
                {
                    "iteration": iteration,
                    "accepted": False,
                    "termination": "no_cost_step",
                    "current_cost": current_cost.copy(),
                    "trial_cost": trial_cost.copy(),
                    "class_loss_vector": class_loss_vector.copy(),
                    "predicted_increase": float(predicted_increase),
                    "actual_increase": 0.0,
                    "ratio": -np.inf,
                    "working_radius": float(working_radius),
                    "reduced_objective": float(current_reduced_objective),
                    "trial_block_a_iterations": 0,
                    "trial_block_a_start_objective": float(current_reduced_objective),
                    "trial_block_a_end_objective": float(current_reduced_objective),
                    "trial_block_a_objective_change": 0.0,
                    **_step_diagnostics(step_info),
                    **current_parts,
                }
            )
            parameter_history.append(
                _build_parameter_snapshot(
                    stage="bcd_no_cost_step_final_state",
                    outer_iteration=iteration,
                    cost_vector=current_cost,
                    alphas=alphas,
                    betas=betas,
                    reduced_objective=current_reduced_objective,
                    objective_parts=current_parts,
                    accepted=False,
                )
            )
            break

        trial_betas = copy.deepcopy(betas)
        trial_alphas = copy.deepcopy(alphas)

        trial_betas, trial_alphas, trial_reduced_objective, trial_parts, trial_block_history = solve_block_a_with_fixed_cost(
            X_blocks=X_blocks,
            y=y,
            betas=trial_betas,
            alphas=trial_alphas,
            cost_vector=trial_cost,
            optimization_groups=optimization_groups,
            exact_profiles=exact_profiles,
            learning_rate_alpha=learning_rate_alpha,
            learning_rate_beta=learning_rate_beta,
            lambda_beta=lambda_beta,
            alpha_l1_radius=alpha_l1_radius,
            alpha_subproblem_max_iter=alpha_subproblem_max_iter,
            alpha_subproblem_tol=alpha_subproblem_tol,
            beta_subproblem_max_iter=beta_subproblem_max_iter,
            beta_subproblem_tol=beta_subproblem_tol,
            max_iter=block_a_max_iter,
            tol=block_a_tol,
            verbose=False,
            prefix=f"Trial Block A {iteration}",
        )
        trial_block_a_history.append(
            {
                "outer_iteration": iteration,
                "accepted": None,
                "trial_cost": trial_cost.copy(),
                "history": trial_block_history,
            }
        )

        actual_increase = float(trial_reduced_objective - current_reduced_objective)
        ratio = compute_trust_region_ratio(
            actual_increase=actual_increase,
            predicted_increase=predicted_increase,
        )
        accepted = ratio >= accept_threshold

        bcd_entry = {
            "iteration": iteration,
            "accepted": accepted,
            "current_cost": current_cost.copy(),
            "trial_cost": trial_cost.copy(),
            "step": step_info["step"].copy(),
            "step_norm": float(step_info["step_norm"]),
            "class_loss_vector": class_loss_vector.copy(),
            "predicted_increase": float(predicted_increase),
            "actual_increase": float(actual_increase),
            "ratio": float(ratio),
            "working_radius": float(working_radius),
            "reduced_objective": float(current_reduced_objective),
            "trial_reduced_objective": float(trial_reduced_objective),
            "group_cost_bce": float(current_parts["group_cost_bce"]),
            "sample_cost_bce": float(current_parts["sample_cost_bce"]),
            "group_bce": float(current_parts["group_bce"]),
            "sample_bce": float(current_parts["sample_bce"]),
            "beta_l1_penalty": float(current_parts["beta_l1_penalty"]),
            "alpha_max_l1_norm": float(current_parts["alpha_max_l1_norm"]),
            "alpha_constraint_violation": float(current_parts["alpha_constraint_violation"]),
            "trial_group_cost_bce": float(trial_parts["group_cost_bce"]),
            "trial_sample_cost_bce": float(trial_parts["sample_cost_bce"]),
            "trial_group_bce": float(trial_parts["group_bce"]),
            "trial_sample_bce": float(trial_parts["sample_bce"]),
            "trial_beta_l1_penalty": float(trial_parts["beta_l1_penalty"]),
            "trial_alpha_max_l1_norm": float(trial_parts["alpha_max_l1_norm"]),
            "trial_alpha_constraint_violation": float(trial_parts["alpha_constraint_violation"]),
            "trial_block_a_iterations": int(len(trial_block_history)),
            "trial_block_a_start_objective": float(trial_block_history[0]["objective"]),
            "trial_block_a_end_objective": float(trial_block_history[-1]["objective"]),
            "trial_block_a_objective_change": float(
                trial_block_history[-1]["objective"] - trial_block_history[0]["objective"]
            ),
            **_step_diagnostics(step_info),
        }
        bcd_history.append(bcd_entry)
        parameter_history.append(
            _build_parameter_snapshot(
                stage="bcd_trial_state",
                outer_iteration=iteration,
                cost_vector=trial_cost,
                alphas=trial_alphas,
                betas=trial_betas,
                reduced_objective=trial_reduced_objective,
                objective_parts=trial_parts,
                accepted=accepted,
            )
        )

        old_radius = working_radius
        new_radius = update_trust_region_radius(
            ratio=ratio,
            current_radius=working_radius,
            max_radius=max_trust_region_radius,
            accept_threshold=accept_threshold,
            expand_threshold=expand_threshold,
            shrink_factor=shrink_factor,
            expand_factor=expand_factor,
        )

        if accepted:
            current_cost = trial_cost
            betas = trial_betas
            alphas = trial_alphas
            current_reduced_objective = trial_reduced_objective
            current_parts = trial_parts
            block_a_history.append(trial_block_history)
            trial_block_a_history[-1]["accepted"] = True
        else:
            trial_block_a_history[-1]["accepted"] = False

        parameter_history.append(
            _build_parameter_snapshot(
                stage="bcd_final_state",
                outer_iteration=iteration,
                cost_vector=current_cost,
                alphas=alphas,
                betas=betas,
                reduced_objective=current_reduced_objective,
                objective_parts=current_parts,
                accepted=accepted,
            )
        )

        if verbose:
            print(
                f"BCD iter {iteration:03d} | "
                f"Accepted: {accepted} | "
                f"Trial Block A Iters: {len(trial_block_history)} | "
                f"Pred: {predicted_increase:.6f} | "
                f"Actual: {actual_increase:.6f} | "
                f"Ratio: {ratio:.6f} | "
                f"Radius: {old_radius:.6f} -> {new_radius:.6f}"
            )
            print(
                f"    Trial Cost: {trial_cost} | "
                f"Chosen Cost: {current_cost} | "
                f"Obj: {current_reduced_objective:.6f} | "
                f"Group Cost-BCE: {current_parts['group_cost_bce']:.6f} | "
                f"Beta L1: {current_parts['beta_l1_penalty']:.6f} | "
                f"Alpha Max L1: {current_parts['alpha_max_l1_norm']:.6f}"
            )
            _print_cost_step_diagnostics(step_info)
            if not accepted:
                print(
                    "    Accepted=False means the trial trust-region update for C "
                    "was rejected. The model kept the previous C, alpha, and beta, "
                    "and only the trust-region radius was updated."
                )
            else:
                print(
                    "    Accepted=True means the trial trust-region update for C "
                    "was accepted, so the model moved to the trial C, alpha, and beta."
                )

        working_radius = new_radius

        if accepted and abs(actual_increase) < tol and step_info["step_norm"] < tol:
            break

    source_order = list(X_blocks.keys())

    model = {
        "betas": betas,
        "alphas": alphas,
        "costs": current_cost,
        "baseline_costs": baseline_cost,
        "lambda_beta": float(lambda_beta),
        "alpha_l1_radius": float(alpha_l1_radius),
        "source_order": source_order,
        "profiles": optimization_groups,
        "exact_profiles": exact_profiles,
        "optimization_groups": optimization_groups,
        "loss_history": bcd_history,
        "initial_block_a_history": initial_block_history,
        "block_a_history": block_a_history,
        "trial_block_a_history": trial_block_a_history,
        "parameter_history": parameter_history,
        "predict_proba": lambda X_new: predict_proba(
            X_blocks=X_new,
            betas=betas,
            alphas=alphas,
            source_order=source_order,
        ),
        "predict": lambda X_new, threshold=0.5: predict_class(
            X_blocks=X_new,
            betas=betas,
            alphas=alphas,
            threshold=threshold,
            source_order=source_order,
        ),
    }

    return model
