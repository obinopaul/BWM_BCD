"""
Paper note:
Section 3.1 of `Multi_source_BWM.pdf` defines the incomplete-source objective

    minimize_{alpha, beta}
        (1 / |pf|) * sum_{m in pf} f(X_m, beta, alpha_m, y_m)
        + lambda * R_beta(beta)

    subject to
        R_alpha(alpha_m) <= 1
        for every m in pf.

The paper deliberately leaves the forms of R_alpha and R_beta generic. Section
3.2 then says the optimization machinery only needs two ingredients:

    1. a projection or constrained solver for the alpha-subproblem
    2. a proximal operator for the beta-subproblem

Remark 4 states that different choices are possible, and the experimental
section says the authors employ the l1-norm penalty for both R_alpha and
R_beta. The active implementation in this repository follows that paper-aligned
choice:

    R_alpha(alpha_m) = ||alpha_m||_1
    R_beta(beta) = sum_i ||beta^i||_1.

With this choice, the alpha-subproblem for one group m becomes a constrained
logistic problem

    minimize_{alpha_m} f_m^C(alpha_m)
    subject to ||alpha_m||_1 <= r_alpha,

where r_alpha defaults to 1.0. A first-order projected-gradient step is

    alpha_temp = alpha_m - eta_alpha * nabla f_m^C(alpha_m)
    alpha_next = Pi_{||.||_1 <= r_alpha}(alpha_temp),

where Pi denotes Euclidean projection onto the l1-ball.

For beta, the fixed-alpha subproblem is

    minimize_beta g_C(beta) + lambda_beta * sum_i ||beta^i||_1.

Using proximal gradient, one source block beta^i is updated by

    v_i = beta^i - eta_beta * nabla_{beta^i} g_C(beta)
    beta_next^i = prox_{eta_beta * lambda_beta * ||.||_1}(v_i)

and the l1 proximal operator is the soft-threshold map

    prox_{tau ||.||_1}(v)
    = sign(v) odot max(|v| - tau, 0).

This file implements those penalty values and operators:

    - l1 penalty for beta
    - soft-threshold proximal map for beta
    - l1-ball projection for alpha
    - helpers for measuring alpha feasibility
"""

import numpy as np


def beta_l1_penalty(betas, lambda_beta):
    """
    Compute lambda_beta * sum_i ||beta^i||_1.
    """
    lambda_beta = float(lambda_beta)
    if lambda_beta < 0.0:
        raise ValueError("lambda_beta must be nonnegative.")

    penalty = 0.0
    for beta in betas.values():
        penalty += np.sum(np.abs(np.asarray(beta, dtype=float)))

    return lambda_beta * float(penalty)


def soft_threshold(v, threshold):
    """
    Apply the l1 proximal operator componentwise.
    """
    v = np.asarray(v, dtype=float)
    threshold = float(threshold)

    if threshold < 0.0:
        raise ValueError("threshold must be nonnegative.")

    return np.sign(v) * np.maximum(np.abs(v) - threshold, 0.0)


def beta_l1_prox(beta, step_size, lambda_beta):
    """
    Proximal map for step_size * lambda_beta * ||beta||_1.
    """
    step_size = float(step_size)
    lambda_beta = float(lambda_beta)

    if step_size < 0.0:
        raise ValueError("step_size must be nonnegative.")

    if lambda_beta < 0.0:
        raise ValueError("lambda_beta must be nonnegative.")

    return soft_threshold(beta, step_size * lambda_beta)


def alpha_l1_norm(alpha):
    """
    Compute ||alpha||_1 for either an array or a profile dict.
    """
    if isinstance(alpha, dict):
        values = np.asarray(list(alpha.values()), dtype=float)
    else:
        values = np.asarray(alpha, dtype=float).reshape(-1)

    return float(np.sum(np.abs(values)))


def project_onto_l1_ball(v, radius=1.0):
    """
    Project a vector onto the Euclidean l1-ball {x : ||x||_1 <= radius}.

    This is the standard Duchi et al. projection implemented by projecting the
    absolute values onto the simplex and then restoring the original signs.
    """
    v = np.asarray(v, dtype=float).reshape(-1)
    radius = float(radius)

    if radius < 0.0:
        raise ValueError("radius must be nonnegative.")

    if radius == 0.0:
        return np.zeros_like(v)

    if np.sum(np.abs(v)) <= radius:
        return v.copy()

    u = np.sort(np.abs(v))[::-1]
    cssv = np.cumsum(u)
    rho_candidates = np.where(u - (cssv - radius) / np.arange(1, len(u) + 1) > 0)[0]

    if rho_candidates.size == 0:
        return np.zeros_like(v)

    rho = rho_candidates[-1]
    theta = (cssv[rho] - radius) / float(rho + 1)
    return np.sign(v) * np.maximum(np.abs(v) - theta, 0.0)


def project_alpha_profile_to_l1_ball(alphas, profile, radius=1.0):
    """
    Project one profile's alpha vector onto the l1-ball of given radius.
    """
    sources = list(profile)
    values = np.asarray([alphas[profile][source] for source in sources], dtype=float)
    projected = project_onto_l1_ball(values, radius=radius)

    for source, value in zip(sources, projected):
        alphas[profile][source] = float(value)

    return alphas


def max_alpha_l1_norm(alphas):
    """
    Return max_m ||alpha_m||_1 across all profiles.
    """
    if not alphas:
        return 0.0

    return max(alpha_l1_norm(alpha_profile) for alpha_profile in alphas.values())


def max_alpha_constraint_violation(alphas, radius=1.0):
    """
    Return max_m max(||alpha_m||_1 - radius, 0).
    """
    radius = float(radius)
    if radius < 0.0:
        raise ValueError("radius must be nonnegative.")

    return max(0.0, max_alpha_l1_norm(alphas) - radius)
