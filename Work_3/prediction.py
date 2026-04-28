"""
Paper note:
This file implements inference for the incomplete-source feature-selection
model from Section 3 of `Multi_source_BWM.pdf`.

The paper defines one source-combination-specific linear predictor for each
optimization group m:

    z_m = sum_{i in m} alpha_m^i X_m^i beta^i.

Here:

    - X_m^i is the source-i feature block restricted to the samples whose
      observed sources contain combination m,
    - beta^i is the shared coefficient vector for source i,
    - alpha_m^i is the group-specific weight for source i inside combination m.

During training, the section-3 optimization groups overlap. For example, a
sample with exact observed-source pattern 111 contributes to the groups 111,
110, 101, 011, 100, 010, and 001. That overlap is correct for the alternating
updates of alpha and beta.

Prediction is different. At inference time, each sample belongs to exactly one
exact observed-source pattern p_j. If sample j has available sources

    A_j = { i : source i is observed for sample j },

then the prediction for that sample must use only the alpha weights associated
with that exact profile:

    z_j = sum_{i in A_j} alpha_{A_j}^i x_j^i beta^i.

So the prediction code must use exact disjoint profiles, not the overlapping
optimization groups used during training.

For binary classification, logits are converted to probabilities by the
Bernoulli link

    P(y_j = 1 | x_j) = sigma(z_j) = 1 / (1 + exp(-z_j)).

Class prediction then applies the threshold rule

    y_hat_j = 1  if sigma(z_j) >= tau,
            = 0  otherwise.

One important modeling point: the class-cost vector C does not enter the
prediction equation directly. Costs affect training because they change the
learned alpha and beta values, but once parameters are fixed the predictor is
still the paper's linear score followed by the logistic link.

Implementation map:

    1. `build_exact_profiles(..., allow_all_missing=True)` reconstructs the one
       exact observed-source pattern for each input sample.
    2. `compute_profile_logits` evaluates
           z_m = sum_{i in m} alpha_m^i X_m^i beta^i
       for all samples in one exact profile.
    3. `compute_all_logits` stitches those profilewise logits back into the
       original sample order and marks samples with no observed source as
       invalid.
    4. `predict_proba` applies sigma(z).
    5. `predict_class` applies the threshold tau.

Robustness detail:
the profile keys stored in `alphas` depend on the training source order. The
helper below does not infer a new order from the prediction data. Instead it
reuses the canonical training order, taken from the trained beta dictionary
unless you pass `source_order` explicitly. This keeps the profile tuples used at
prediction time consistent with the tuples used when alpha was trained, even if
the input `X_blocks` dictionary is created in a different key order.
"""

import numpy as np

from loss import sigmoid
from missingness_profiles import build_exact_profiles


def _get_prediction_source_order(X_blocks, betas, source_order=None):
    """
    Reuse the canonical training source order for prediction.
    """
    if not X_blocks:
        raise ValueError("X_blocks must contain at least one source.")

    if source_order is None:
        source_order = list(betas.keys())
    else:
        source_order = list(source_order)

    missing_sources = [source for source in source_order if source not in X_blocks]
    if missing_sources:
        raise ValueError(
            "X_blocks is missing trained sources required for prediction: "
            f"{missing_sources}"
        )

    unknown_beta_sources = [source for source in source_order if source not in betas]
    if unknown_beta_sources:
        raise ValueError(
            "betas does not contain all sources required by source_order: "
            f"{unknown_beta_sources}"
        )

    extra_sources = [source for source in X_blocks if source not in source_order]
    if extra_sources:
        raise ValueError(
            "source_order must contain every source in X_blocks exactly once. "
            f"Unexpected sources: {extra_sources}"
        )

    return source_order


def _resolve_profile_key(profile, alphas, source_order):
    """
    Match an input-data profile to the canonical alpha key.
    """
    if profile in alphas:
        return profile

    profile_set = set(profile)
    canonical_profile = tuple(
        source for source in source_order if source in profile_set
    )

    if canonical_profile in alphas and len(canonical_profile) == len(profile):
        return canonical_profile

    matches = [
        alpha_profile
        for alpha_profile in alphas
        if len(alpha_profile) == len(profile) and set(alpha_profile) == profile_set
    ]
    if len(matches) == 1:
        return matches[0]

    raise ValueError(
        "No alpha weights are available for exact profile "
        f"{profile}. Expected one of {list(alphas.keys())}."
    )


def compute_profile_logits(
    X_blocks,
    betas,
    alphas,
    profile,
    indices,
    source_order=None,
):
    """
    Compute logits for one exact observed-source profile.

    For profile m, this evaluates

        z_m = sum_{i in m} alpha_m^i X_m^i beta^i

    on the samples whose indices belong to that exact profile.
    """
    source_order = _get_prediction_source_order(
        X_blocks,
        betas,
        source_order=source_order,
    )
    profile = _resolve_profile_key(profile, alphas, source_order=source_order)
    indices = np.asarray(indices, dtype=int).reshape(-1)

    logits = np.zeros(indices.shape[0], dtype=float)

    for source in profile:
        X_source = np.asarray(X_blocks[source], dtype=float)[indices, :]
        X_source = np.nan_to_num(X_source, nan=0.0)

        beta_i = np.asarray(betas[source], dtype=float).reshape(-1)
        if X_source.shape[1] != beta_i.shape[0]:
            raise ValueError(
                f"Source {source} has {X_source.shape[1]} features, "
                f"but beta has length {beta_i.shape[0]}."
            )

        alpha_mi = float(alphas[profile][source])
        logits += alpha_mi * (X_source @ beta_i)

    return logits


def compute_all_logits(
    X_blocks,
    betas,
    alphas,
    exact_profiles=None,
    source_order=None,
):
    """
    Compute logits for all samples using exact disjoint profiles.
    """
    source_order = _get_prediction_source_order(
        X_blocks,
        betas,
        source_order=source_order,
    )

    validated_profiles = build_exact_profiles(
        X_blocks,
        source_order=source_order,
        allow_all_missing=True,
    )
    if exact_profiles is None:
        exact_profiles = validated_profiles

    n_samples = np.asarray(X_blocks[source_order[0]], dtype=float).shape[0]
    logits_all = np.zeros(n_samples, dtype=float)
    valid_mask = np.zeros(n_samples, dtype=bool)

    for profile, indices in exact_profiles.items():
        canonical_profile = _resolve_profile_key(
            profile,
            alphas,
            source_order=source_order,
        )
        indices = np.asarray(indices, dtype=int).reshape(-1)

        logits_all[indices] = compute_profile_logits(
            X_blocks=X_blocks,
            betas=betas,
            alphas=alphas,
            profile=canonical_profile,
            indices=indices,
            source_order=source_order,
        )
        valid_mask[indices] = True

    return logits_all, valid_mask


def predict_proba(
    X_blocks,
    betas,
    alphas,
    exact_profiles=None,
    source_order=None,
):
    """
    Predict probabilities P(y = 1 | X) for binary classification.
    """
    logits_all, valid_mask = compute_all_logits(
        X_blocks=X_blocks,
        betas=betas,
        alphas=alphas,
        exact_profiles=exact_profiles,
        source_order=source_order,
    )

    probabilities = np.asarray(sigmoid(logits_all), dtype=float)
    probabilities[~valid_mask] = np.nan
    return probabilities


def predict_class(
    X_blocks,
    betas,
    alphas,
    exact_profiles=None,
    threshold=0.5,
    source_order=None,
):
    """
    Predict binary class labels using a probability threshold.
    """
    if not (0.0 <= threshold <= 1.0):
        raise ValueError("threshold must lie in [0, 1].")

    probabilities = predict_proba(
        X_blocks=X_blocks,
        betas=betas,
        alphas=alphas,
        exact_profiles=exact_profiles,
        source_order=source_order,
    )

    predictions = np.where(probabilities >= threshold, 1, 0).astype(int)
    predictions[np.isnan(probabilities)] = -1
    return predictions
