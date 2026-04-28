"""
Paper note:
This file implements the class-cost block C for the cost-sensitive extension of
the incomplete-source feature-selection model in Section 3 of
`Multi_source_BWM.pdf`, using the trust-region derivation written in
`notes.tex`.

Section 3 of the paper defines one optimization problem for each source
combination m. Let G_m be the set of samples whose observed sources contain the
combination m, and let n_m = |G_m|. For source i inside combination m, the
group-restricted feature matrix is X_m^i, the shared source coefficient is
beta^i, and the group-specific source weight is alpha_m^i. The paper prediction
for group m is

    z_m = sum_{i in m} alpha_m^i X_m^i beta^i.

The original paper writes the data-fit term with least squares. In that form,
the group loss is

    f_m^LS(alpha, beta)
    = (1 / (2 n_m)) * || z_m - y_m ||_2^2.

This repository keeps the same iSFS structure, but replaces squared loss with
binary cross-entropy written directly from logits. For one sample j in group m,

    ell(z_{m,j}, y_{m,j})
    = log(1 + exp(z_{m,j})) - y_{m,j} z_{m,j}.

To introduce class costs, let C in R^K be the nonnegative class-cost vector.
If sample j has class label y_{m,j}, then its weighted contribution is

    C_{y_{m,j}} * ell(z_{m,j}, y_{m,j}).

So the cost-sensitive group objective used in code is

    f_m^C(alpha, beta, C)
    = (1 / n_m) * sum_{j in G_m} C_{y_{m,j}} * ell(z_{m,j}, y_{m,j}).

Two details matter here:

    1. The factor (1 / n_m) remains because we still average over the samples in
       group m.
    2. The factor (1 / 2) from least squares disappears because it belonged to
       the squared-loss form, not to BCE.

With alpha and beta fixed, `notes.tex` isolates the C-dependent part by grouping
the loss by class:

    L_k(alpha, beta)
    = (1 / |pf|) * sum_{m in pf} (1 / n_m)
      * sum_{j in G_m : y_{m,j} = k} ell(z_{m,j}, y_{m,j}),

where pf is the set of optimization groups returned by
`build_missingness_profiles`. In vector form,

    L(alpha, beta) = [L_1(alpha, beta), ..., L_K(alpha, beta)]^T.

Then the entire C-block objective becomes

    Phi_C(C) = L(alpha, beta)^T C.

This is linear in C, so without constraints it would be unbounded above. The
trust-region notes therefore constrain C to stay inside a ball around a baseline
cost vector C^0:

    (1 / 2) * || C - C^0 ||_2^2 <= Delta_max,
    C >= 0.

At trust-region iteration k, with current cost C^(k), we write the candidate
step as d = C - C^(k). The linear TR-L subproblem is

    maximize_d L_k^T d

subject to

    (1 / 2) * || d ||_2^2 <= Delta_k,
    C^(k) + d >= 0,
    (1 / 2) * || C^(k) + d - C^0 ||_2^2 <= Delta_max.

Because the local model is linear, the unconstrained Cauchy direction is simply
the normalized class-loss vector:

    d_k^lin = sqrt(2 * Delta_k) * L_k / ||L_k||_2.

The final feasible step may be shorter if the global baseline ball or the
nonnegativity constraint becomes active first.

Implementation map:

    - `pointwise_bce_from_logits` computes ell(z_j, y_j).
    - `compute_class_loss_vector` computes L(alpha, beta).
    - `compute_cost_block_objective` evaluates Phi_C(C) = L^T C.
    - `solve_cost_trust_region_subproblem` builds the TR-L step for C.
    - `compute_trust_region_ratio` and `update_trust_region_radius` support the
      outer block-coordinate-descent loop in `train_model.py`.
"""

import numpy as np

from loss_functions import sigmoid
from prediction import compute_profile_logits


def infer_n_classes(y):
    """
    Infer the number of classes from integer labels.
    """
    y = np.asarray(y).reshape(-1)

    if y.size == 0:
        raise ValueError("y must contain at least one sample.")

    if not np.all(np.isfinite(y)):
        raise ValueError("y must contain only finite values.")

    y_int = y.astype(int)
    if not np.array_equal(y, y_int):
        raise ValueError("y must contain integer class labels.")

    if np.min(y_int) < 0:
        raise ValueError("y must contain nonnegative class labels.")

    return int(np.max(y_int)) + 1


def validate_cost_vector(cost_vector, n_classes):
    """
    Validate that the class-cost vector is nonnegative and has length K.
    """
    cost_vector = np.asarray(cost_vector, dtype=float).reshape(-1)

    if cost_vector.shape[0] != n_classes:
        raise ValueError(
            "cost_vector must have one entry per class. "
            f"Expected {n_classes}, got {cost_vector.shape[0]}."
        )

    if not np.all(np.isfinite(cost_vector)):
        raise ValueError("cost_vector must contain only finite values.")

    if np.any(cost_vector < 0.0):
        raise ValueError("cost_vector must be componentwise nonnegative.")

    return cost_vector


def build_sample_cost_weights(y, cost_vector):
    """
    Map each label y_j to its class cost C_{y_j}.
    """
    y = np.asarray(y).reshape(-1)
    y_int = y.astype(int)
    cost_vector = np.asarray(cost_vector, dtype=float).reshape(-1)

    if not np.array_equal(y, y_int):
        raise ValueError("y must contain integer class labels.")

    if np.min(y_int) < 0:
        raise ValueError("y must contain nonnegative class labels.")

    if np.max(y_int) >= cost_vector.shape[0]:
        raise ValueError(
            "cost_vector does not contain enough class entries for the labels in y."
        )

    if not np.all(np.isfinite(cost_vector)):
        raise ValueError("cost_vector must contain only finite values.")

    if np.any(cost_vector < 0.0):
        raise ValueError("cost_vector must be componentwise nonnegative.")

    return cost_vector[y_int]


def pointwise_bce_from_logits(logits, y):
    """
    Return the unreduced per-sample BCE values from logits.
    """
    logits = np.asarray(logits, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)

    if logits.shape[0] != y.shape[0]:
        raise ValueError("logits and y must have the same number of samples.")

    return np.logaddexp(0.0, logits) - y * logits


def cost_weighted_bce_from_logits(logits, y, cost_vector):
    """
    Compute mean class-cost-weighted BCE from logits.

    For labels y_j and class costs C_{y_j}, this is

        (1 / n) * sum_j C_{y_j} * ell(z_j, y_j).
    """
    pointwise_loss = pointwise_bce_from_logits(logits, y)
    sample_weights = build_sample_cost_weights(y, cost_vector)
    return float(np.mean(sample_weights * pointwise_loss))


def cost_weighted_bce_gradient_from_logits(logits, y, cost_vector):
    """
    Compute the gradient of mean class-cost-weighted BCE with respect to logits.

        d / dz_j [ (1 / n) * C_{y_j} * ell(z_j, y_j) ]
        = (1 / n) * C_{y_j} * (sigma(z_j) - y_j).
    """
    logits = np.asarray(logits, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)

    if logits.shape[0] != y.shape[0]:
        raise ValueError("logits and y must have the same number of samples.")

    sample_weights = build_sample_cost_weights(y, cost_vector)
    return sample_weights * (sigmoid(logits) - y) / y.shape[0]


def compute_class_loss_vector(
    X_blocks,
    y,
    betas,
    alphas,
    optimization_groups,
    n_classes=None,
):
    """
    Compute the classwise loss vector L(alpha, beta) used in the C-block.

    This implements

        L_k(alpha, beta)
        = (1 / |pf|) * sum_{m in pf} (1 / n_m)
          * sum_{j in G_m : y_{m,j} = k} ell(z_{m,j}, y_{m,j}).
    """
    y = np.asarray(y).reshape(-1)
    y_int = y.astype(int)

    if n_classes is None:
        n_classes = infer_n_classes(y_int)

    loss_vector = np.zeros(n_classes, dtype=float)
    n_profiles = len(optimization_groups)

    for profile, indices in optimization_groups.items():
        logits_m = compute_profile_logits(
            X_blocks=X_blocks,
            betas=betas,
            alphas=alphas,
            profile=profile,
            indices=indices,
        )
        y_m = y_int[indices]
        pointwise_loss = pointwise_bce_from_logits(logits_m, y_m)

        for class_index in range(n_classes):
            class_mask = y_m == class_index
            if np.any(class_mask):
                loss_vector[class_index] += np.sum(pointwise_loss[class_mask]) / len(indices)

    return loss_vector / n_profiles


def compute_cost_block_objective(class_loss_vector, cost_vector):
    """
    Compute Phi_C(C) = L^T C for fixed alpha and beta.
    """
    class_loss_vector = np.asarray(class_loss_vector, dtype=float).reshape(-1)
    cost_vector = validate_cost_vector(cost_vector, class_loss_vector.shape[0])
    return float(np.dot(class_loss_vector, cost_vector))


def _remaining_step_to_global_ball(current_cost, baseline_cost, direction, max_radius):
    """
    Compute the largest feasible step length along a unit direction before the
    baseline trust-region ball is violated.

    The baseline feasible set is

        (1 / 2) * || C - C^0 ||_2^2 <= Delta_max,

    which is the Euclidean ball of radius sqrt(2 * Delta_max) around C^0.
    """
    radius = np.sqrt(2.0 * max_radius)
    offset = current_cost - baseline_cost
    offset_norm_sq = float(np.dot(offset, offset))

    if offset_norm_sq >= radius ** 2:
        return 0.0

    directional_offset = float(np.dot(offset, direction))
    discriminant = directional_offset ** 2 + radius ** 2 - offset_norm_sq
    discriminant = max(discriminant, 0.0)

    return float(-directional_offset + np.sqrt(discriminant))


def _nonnegativity_step_limit(current_cost, direction):
    """
    Compute the maximum step length before some cost component would turn negative.
    """
    negative_components = direction < 0.0

    if not np.any(negative_components):
        return np.inf

    return float(np.min(current_cost[negative_components] / (-direction[negative_components])))


def solve_cost_trust_region_subproblem(
    current_cost,
    baseline_cost,
    class_loss_vector,
    working_radius,
    max_radius,
):
    """
    Solve the linear trust-region subproblem for the C-block.

    With fixed alpha and beta, the local model is

        m_k(d) = Phi_C(C^(k)) + L_k^T d.

    This routine returns the linear Cauchy step along the current loss vector,
    clipped by the working radius, the baseline ball around C^0, and the
    nonnegativity bound.
    """
    class_loss_vector = np.asarray(class_loss_vector, dtype=float).reshape(-1)
    n_classes = class_loss_vector.shape[0]
    current_cost = validate_cost_vector(current_cost, n_classes)
    baseline_cost = validate_cost_vector(baseline_cost, n_classes)

    if working_radius < 0.0 or max_radius < 0.0:
        raise ValueError("Trust-region radii must be nonnegative.")

    loss_norm = float(np.linalg.norm(class_loss_vector))
    if loss_norm == 0.0:
        zero_step = np.zeros_like(current_cost)
        return {
            "trial_cost": current_cost.copy(),
            "step": zero_step,
            "step_norm": 0.0,
            "predicted_increase": 0.0,
            "working_step_limit": 0.0,
            "global_step_limit": 0.0,
            "nonnegativity_step_limit": np.inf,
        }

    direction = class_loss_vector / loss_norm
    working_step_limit = np.sqrt(2.0 * working_radius)
    global_step_limit = _remaining_step_to_global_ball(
        current_cost=current_cost,
        baseline_cost=baseline_cost,
        direction=direction,
        max_radius=max_radius,
    )
    nonnegativity_step_limit = _nonnegativity_step_limit(
        current_cost=current_cost,
        direction=direction,
    )

    step_length = min(
        working_step_limit,
        global_step_limit,
        nonnegativity_step_limit,
    )
    step_length = max(step_length, 0.0)

    step = step_length * direction
    trial_cost = current_cost + step

    tiny_negative = (trial_cost < 0.0) & (trial_cost > -1e-12)
    trial_cost[tiny_negative] = 0.0

    predicted_increase = float(np.dot(class_loss_vector, step))

    return {
        "trial_cost": trial_cost,
        "step": step,
        "step_norm": float(np.linalg.norm(step)),
        "predicted_increase": predicted_increase,
        "working_step_limit": float(working_step_limit),
        "global_step_limit": float(global_step_limit),
        "nonnegativity_step_limit": float(nonnegativity_step_limit),
    }


def compute_trust_region_ratio(actual_increase, predicted_increase):
    """
    Compute the trust-region ratio r_k = Ared_k / Pred_k.
    """
    actual_increase = float(actual_increase)
    predicted_increase = float(predicted_increase)

    if predicted_increase <= 0.0:
        return -np.inf

    return actual_increase / predicted_increase


def update_trust_region_radius(
    ratio,
    current_radius,
    max_radius,
    accept_threshold=0.25,
    expand_threshold=0.75,
    shrink_factor=0.25,
    expand_factor=2.0,
):
    """
    Update the working trust-region radius from the ratio r_k.
    """
    ratio = float(ratio)
    current_radius = float(current_radius)
    max_radius = float(max_radius)

    if ratio < accept_threshold:
        return current_radius * shrink_factor

    if ratio >= expand_threshold:
        return min(current_radius * expand_factor, max_radius)

    return current_radius
