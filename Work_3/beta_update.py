"""
Paper note:
This file implements the beta-block update for the iSFS model when alpha and
the class-cost vector C are fixed.

Section 3.2.2 of `Multi_source_BWM.pdf` solves for the shared source vectors
beta with alpha fixed. In the paper's least-squares form, the beta-block
objective is

    g(beta)
    = (1 / |pf|) * sum_{m in pf}
      (1 / (2 n_m)) * || sum_{i in m} alpha_m^i X_m^i beta^i - y_m ||_2^2.

Here pf is the set of optimization groups constructed from the missingness
profiles. Each beta^i is shared across every optimization group that contains
source i, so the beta update is different from the alpha update: alpha_m is
group-specific, while beta^i must aggregate information across groups.

The paper leaves R_beta generic, but in the ADNI experiments it says the
l1-norm penalty is used for both R_alpha and R_beta. The active implementation
therefore sets

    R_beta(beta) = sum_i ||beta^i||_1.

For the logistic extension used in this repository, we keep the same averaging
structure but replace squared loss with class-cost-weighted BCE. Define

    z_m = sum_{i in m} alpha_m^i X_m^i beta^i

and

    ell(z_{m,j}, y_{m,j})
    = log(1 + exp(z_{m,j})) - y_{m,j} z_{m,j}.

Then the cost-sensitive group loss is

    f_m^C(beta)
    = (1 / n_m) * sum_{j in G_m} C_{y_{m,j}} * ell(z_{m,j}, y_{m,j}),

and the fixed-alpha beta-block objective is

    g_C(beta)
    = (1 / |pf|) * sum_{m in pf} f_m^C(beta)
      + lambda_beta * sum_i ||beta^i||_1.

Again, the factor (1 / n_m) stays because each group loss is averaged over its
samples, while the factor (1 / 2) disappears because we no longer use squared
loss.

For one group m, the gradient with respect to logits is

    g_m
    = d f_m^C / d z_m
    = (1 / n_m) * [ C_{y_m} odot (sigma(z_m) - y_m) ].

Now fix one source i. Since

    z_m = ... + alpha_m^i X_m^i beta^i + ...,

the chain rule gives

    d z_m / d beta^i = alpha_m^i X_m^i.

Therefore the contribution of group m to the gradient of beta^i is

    d f_m^C / d beta^i
    = (alpha_m^i X_m^i)^T g_m.

Because beta^i is shared across groups, the total gradient is the average over
every group containing source i:

    nabla_{beta^i} g_C(beta)
    = (1 / |pf|) * sum_{m : i in m}
      (alpha_m^i X_m^i)^T g_m.

The l1 penalty is then applied by the proximal operator

    beta_temp^i = beta^i - eta_beta * nabla_{beta^i} g_C_without_penalty(beta)
    beta_next^i = prox_{eta_beta * lambda_beta ||.||_1}(beta_temp^i)
                = sign(beta_temp^i)
                  odot max(|beta_temp^i| - eta_beta * lambda_beta, 0).

Algorithmic note:
the paper's Algorithm 2 updates beta by solving the regularized subproblem
(17) with alpha fixed, not by taking only one proximal-gradient step. To stay
close to that structure, this file exposes both:

    - `update_betas`: one proximal-gradient step
    - `solve_betas_with_fixed_alphas`: repeated proximal-gradient steps that
      approximately solve the beta subproblem before alpha is revisited

Implementation map:

    1. `compute_profile_logits` builds z_m.
    2. `cost_weighted_bce_gradient_from_logits` computes g_m.
    3. `(alpha_m^i X_m^i)^T g_m` is accumulated for each shared beta^i.
    4. The average group gradient is used to form a proximal-gradient step.
    5. Soft-thresholding applies the l1 regularization to beta.
    6. `solve_betas_with_fixed_alphas` repeats this until the beta subproblem
       stabilizes.
"""

import numpy as np

from costs import cost_weighted_bce_from_logits, cost_weighted_bce_gradient_from_logits
from prediction import compute_profile_logits
from regularization import beta_l1_penalty, beta_l1_prox


def initialize_betas(X_blocks, random_state=42):
    """
    Initialize one shared beta_i for each source.
    """
    rng = np.random.default_rng(random_state)
    betas = {}

    for source, X in X_blocks.items():
        betas[source] = rng.normal(
            loc=0.0,
            scale=0.01,
            size=X.shape[1],
        )

    return betas


def compute_beta_smooth_objective_and_gradients(
    X_blocks,
    y,
    betas,
    alphas,
    profiles,
    cost_vector,
):
    """
    Compute the smooth beta objective and its gradient dictionary.
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    beta_gradients = {
        source: np.zeros_like(beta, dtype=float)
        for source, beta in betas.items()
    }
    group_losses = []
    n_profiles = len(profiles)

    for profile, indices in profiles.items():
        y_m = y[indices]

        logits_m = compute_profile_logits(
            X_blocks=X_blocks,
            betas=betas,
            alphas=alphas,
            profile=profile,
            indices=indices,
        )
        group_losses.append(
            cost_weighted_bce_from_logits(
                logits=logits_m,
                y=y_m,
                cost_vector=cost_vector,
            )
        )

        grad_logits = cost_weighted_bce_gradient_from_logits(
            logits=logits_m,
            y=y_m,
            cost_vector=cost_vector,
        )

        for source in profile:
            X_source = np.asarray(X_blocks[source], dtype=float)[indices, :]
            X_source = np.nan_to_num(X_source, nan=0.0)
            alpha_mi = alphas[profile][source]
            grad_beta_i = alpha_mi * (X_source.T @ grad_logits)
            beta_gradients[source] += grad_beta_i / n_profiles

    smooth_objective = float(np.mean(group_losses))
    return smooth_objective, beta_gradients


def update_betas(
    X_blocks,
    y,
    betas,
    alphas,
    profiles,
    cost_vector,
    learning_rate_beta,
    lambda_beta,
):
    """
    Update beta while alpha and C are fixed.

    This uses the group-averaged gradient

        nabla_{beta^i} g_C(beta)
        = (1 / |pf|) * sum_{m : i in m}
          (alpha_m^i X_m^i)^T g_m,

    where

        g_m
        = (1 / n_m) * [ C_{y_m} odot (sigma(z_m) - y_m) ].

    The returned beta values are updated by one proximal-gradient step for the
    l1-regularized beta objective.
    """
    _, beta_gradients = compute_beta_smooth_objective_and_gradients(
        X_blocks=X_blocks,
        y=y,
        betas=betas,
        alphas=alphas,
        profiles=profiles,
        cost_vector=cost_vector,
    )

    for source in betas:
        beta_trial = betas[source] - learning_rate_beta * beta_gradients[source]
        betas[source] = beta_l1_prox(
            beta_trial,
            step_size=learning_rate_beta,
            lambda_beta=lambda_beta,
        )

    return betas


def solve_betas_with_fixed_alphas(
    X_blocks,
    y,
    betas,
    alphas,
    profiles,
    cost_vector,
    learning_rate_beta,
    lambda_beta,
    max_iter=50,
    tol=1e-6,
):
    """
    Approximately solve the beta subproblem with alpha fixed.
    """
    previous_objective = np.inf

    for _ in range(max_iter):
        smooth_objective, beta_gradients = compute_beta_smooth_objective_and_gradients(
            X_blocks=X_blocks,
            y=y,
            betas=betas,
            alphas=alphas,
            profiles=profiles,
            cost_vector=cost_vector,
        )
        current_objective = smooth_objective + beta_l1_penalty(betas, lambda_beta)

        step_norm_sq = 0.0
        for source in betas:
            beta_trial = betas[source] - learning_rate_beta * beta_gradients[source]
            beta_next = beta_l1_prox(
                beta_trial,
                step_size=learning_rate_beta,
                lambda_beta=lambda_beta,
            )
            step_norm_sq += float(np.sum((beta_next - betas[source]) ** 2))
            betas[source] = beta_next

        if np.sqrt(step_norm_sq) < tol or abs(previous_objective - current_objective) < tol:
            break

        previous_objective = current_objective

    return betas
