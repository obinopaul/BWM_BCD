"""
Paper note:
This file implements the alpha-block update for the iSFS model when the shared
source coefficients beta and the class-cost vector C are fixed.

Section 3.2.1 of `Multi_source_BWM.pdf` solves for alpha_m with beta fixed. In
the paper's least-squares version, one group m solves

    minimize_{alpha_m}
        (1 / (2 n_m)) * || sum_{i in m} alpha_m^i X_m^i beta^i - y_m ||_2^2
    subject to
        R_alpha(alpha_m) <= 1.

Here:

    - m is one source combination such as {source1, source3},
    - G_m is the set of samples whose observed sources contain m,
    - n_m = |G_m|,
    - X_m^i is the source-i feature matrix restricted to the samples in G_m,
    - beta^i is the shared coefficient vector for source i,
    - alpha_m^i is the source weight for source i inside group m.

The paper leaves R_alpha generic, but in the ADNI experiments it states that an
l1-norm regularization is used for both R_alpha and R_beta. The active
implementation follows that choice and sets

    R_alpha(alpha_m) = ||alpha_m||_1.

This repository keeps the same blockwise decomposition but replaces the squared
loss by cost-sensitive BCE from logits. Define

    z_m = sum_{i in m} alpha_m^i X_m^i beta^i

and

    ell(z_{m,j}, y_{m,j})
    = log(1 + exp(z_{m,j})) - y_{m,j} z_{m,j}.

Then the fixed-(beta, C) alpha objective for group m becomes

    f_m^C(alpha_m)
    = (1 / n_m) * sum_{j in G_m} C_{y_{m,j}} * ell(z_{m,j}, y_{m,j}).

The averaging factor (1 / n_m) stays because we still average over the samples
in group m. The factor (1 / 2) from least squares disappears because it was
specific to the quadratic loss.

To derive the alpha gradient, define the source-specific score vector

    s_m^i = X_m^i beta^i.

Since

    z_m = sum_{i in m} alpha_m^i s_m^i,

the chain rule gives

    d f_m^C / d alpha_m^i
    = (d z_m / d alpha_m^i)^T * (d f_m^C / d z_m)
    = (s_m^i)^T g_m,

where the logits gradient is

    g_m
    = d f_m^C / d z_m
    = (1 / n_m) * [ C_{y_m} odot (sigma(z_m) - y_m) ].

This means each alpha_m^i is updated by measuring how strongly the score vector
for source i aligns with the current weighted residual direction g_m.

Because R_alpha is implemented as the l1-ball constraint

    ||alpha_m||_1 <= r_alpha,

the actual update in code is a projected-gradient step:

    alpha_temp = alpha_m - eta_alpha * nabla f_m^C(alpha_m)
    alpha_next = Pi_{||.||_1 <= r_alpha}(alpha_temp).

Algorithmic note:
the paper's Algorithm 2 says that when beta is fixed, each alpha_m is computed
by solving the constrained subproblem (16), not by taking only one gradient
step. To stay aligned with that structure, this file exposes both:

    - `update_alphas`: one projected-gradient step for alpha
    - `solve_alphas_with_fixed_betas`: repeated projected-gradient steps that
      approximately solve each alpha_m subproblem before beta is updated

Implementation map:

    1. `compute_profile_logits` builds z_m for one group m.
    2. `cost_weighted_bce_gradient_from_logits` computes g_m.
    3. For each source i in group m, `X_m^i beta^i` forms s_m^i.
    4. The scalar gradient `(s_m^i)^T g_m` is used in a gradient step.
    5. The whole alpha_m vector is projected back onto the l1-ball.
    6. `solve_alphas_with_fixed_betas` repeats this until the alpha subproblem
       stabilizes.
"""

import numpy as np

from costs import cost_weighted_bce_from_logits, cost_weighted_bce_gradient_from_logits
from prediction import compute_profile_logits
from regularization import project_onto_l1_ball


def initialize_alphas(profiles):
    """
    Initialize alpha for each optimization group.

    Each group starts with equal weight across its currently observed sources.
    """
    alphas = {}

    for profile in profiles:
        n_sources = len(profile)
        equal_weight = 1.0 / n_sources

        alphas[profile] = {
            source: equal_weight
            for source in profile
        }

    return alphas


def compute_alpha_profile_objective_and_gradient(
    X_blocks,
    y,
    betas,
    alpha_vector,
    profile,
    indices,
    cost_vector,
):
    """
    Compute the fixed-beta alpha-subproblem objective and gradient for one group.
    """
    y_m = np.asarray(y, dtype=float).reshape(-1)[indices]
    alpha_vector = np.asarray(alpha_vector, dtype=float).reshape(-1)
    indices = np.asarray(indices, dtype=int).reshape(-1)
    sources = list(profile)

    if alpha_vector.shape[0] != len(sources):
        raise ValueError(
            "alpha_vector must have one coefficient per source in profile. "
            f"Expected {len(sources)}, got {alpha_vector.shape[0]}."
        )

    source_scores = np.zeros((indices.shape[0], len(sources)), dtype=float)

    for source_index, source in enumerate(sources):
        X_source = np.asarray(X_blocks[source], dtype=float)[indices, :]
        X_source = np.nan_to_num(X_source, nan=0.0)
        source_scores[:, source_index] = X_source @ np.asarray(betas[source], dtype=float)

    logits = source_scores @ alpha_vector
    objective = cost_weighted_bce_from_logits(
        logits=logits,
        y=y_m,
        cost_vector=cost_vector,
    )
    grad_logits = cost_weighted_bce_gradient_from_logits(
        logits=logits,
        y=y_m,
        cost_vector=cost_vector,
    )
    gradient = source_scores.T @ grad_logits

    return float(objective), np.asarray(gradient, dtype=float)


def update_alphas(
    X_blocks,
    y,
    betas,
    alphas,
    profiles,
    cost_vector,
    learning_rate_alpha,
    alpha_l1_radius=1.0,
):
    """
    Update alpha while beta and C are fixed.

    For each optimization group m, this uses

        g_m
        = (1 / n_m) * [ C_{y_m} odot (sigma(z_m) - y_m) ]

    together with

        d f_m^C / d alpha_m^i
        = (X_m^i beta^i)^T g_m.

    The returned alpha values are updated by one projected-gradient step onto
    the l1-ball ||alpha_m||_1 <= alpha_l1_radius.
    """
    y = np.asarray(y, dtype=float).reshape(-1)

    for profile, indices in profiles.items():
        sources = list(profile)
        alpha_vector = np.asarray(
            [alphas[profile][source] for source in sources],
            dtype=float,
        )
        _, grad_alpha_vector = compute_alpha_profile_objective_and_gradient(
            X_blocks=X_blocks,
            y=y,
            betas=betas,
            alpha_vector=alpha_vector,
            profile=profile,
            indices=indices,
            cost_vector=cost_vector,
        )
        alpha_trial = alpha_vector - learning_rate_alpha * grad_alpha_vector
        alpha_projected = project_onto_l1_ball(
            alpha_trial,
            radius=alpha_l1_radius,
        )

        for source, value in zip(sources, alpha_projected):
            alphas[profile][source] = float(value)

    return alphas


def solve_alphas_with_fixed_betas(
    X_blocks,
    y,
    betas,
    alphas,
    profiles,
    cost_vector,
    learning_rate_alpha,
    alpha_l1_radius=1.0,
    max_iter=50,
    tol=1e-6,
):
    """
    Approximately solve each alpha_m subproblem with beta fixed.
    """
    y = np.asarray(y, dtype=float).reshape(-1)

    for profile, indices in profiles.items():
        sources = list(profile)
        alpha_vector = np.asarray(
            [alphas[profile][source] for source in sources],
            dtype=float,
        )
        previous_objective = np.inf

        for _ in range(max_iter):
            objective, gradient = compute_alpha_profile_objective_and_gradient(
                X_blocks=X_blocks,
                y=y,
                betas=betas,
                alpha_vector=alpha_vector,
                profile=profile,
                indices=indices,
                cost_vector=cost_vector,
            )
            alpha_trial = alpha_vector - learning_rate_alpha * gradient
            alpha_next = project_onto_l1_ball(
                alpha_trial,
                radius=alpha_l1_radius,
            )

            if (
                np.linalg.norm(alpha_next - alpha_vector) < tol
                or abs(previous_objective - objective) < tol
            ):
                alpha_vector = alpha_next
                break

            alpha_vector = alpha_next
            previous_objective = objective

        for source, value in zip(sources, alpha_vector):
            alphas[profile][source] = float(value)

    return alphas
