"""
Paper note:
Section 3 writes the model loss as a convex function L applied to
the profile-specific logits sum_i alpha_m^i X_m^i beta^i.
For our binary classification setting, that convex loss is binary
cross-entropy, also called logistic loss.
The paper is the source of truth for the modeling choice, while this file
only implements the reusable loss math.
In code we validate binary labels, compute stable BCE from logits,
and expose the gradient used by the alpha and beta updates.
"""

import numpy as np


def _validate_binary_targets(y):
    """
    Validate that y is a flat binary label vector in {0, 1}.
    """
    y = np.asarray(y, dtype=float).reshape(-1)

    if not np.all(np.isfinite(y)):
        raise ValueError("y must contain only finite values.")

    unique = np.unique(y)
    if not np.all(np.isin(unique, [0.0, 1.0])):
        raise ValueError("y must contain only binary labels 0 or 1.")

    return y


def _validate_logits_and_targets(logits, y):
    """
    Validate logits/target shapes and return flattened arrays.
    """
    logits = np.asarray(logits, dtype=float).reshape(-1)
    y = _validate_binary_targets(y)

    if logits.shape[0] != y.shape[0]:
        raise ValueError("logits and y must have the same number of samples.")

    if not np.all(np.isfinite(logits)):
        raise ValueError("logits must contain only finite values.")

    return logits, y


def sigmoid(z):
    """
    Compute sigmoid(z) = 1 / (1 + exp(-z)) safely.
    """
    z = np.asarray(z, dtype=float)

    return np.where(
        z >= 0,
        1.0 / (1.0 + np.exp(-z)),
        np.exp(z) / (1.0 + np.exp(z)),
    )


def binary_cross_entropy_from_logits(logits, y):
    """
    Compute mean binary cross-entropy directly from logits.

    Using the identity

        BCE(z, y) = log(1 + exp(z)) - y * z

    avoids the unstable sigmoid/log/clipping route.
    """
    logits, y = _validate_logits_and_targets(logits, y)

    loss = np.logaddexp(0.0, logits) - y * logits
    return float(np.mean(loss))


def binary_cross_entropy_gradient_from_logits(logits, y):
    """
    Compute the gradient of mean BCE with respect to logits.
    """
    logits, y = _validate_logits_and_targets(logits, y)
    return (sigmoid(logits) - y) / y.shape[0]


def binary_cross_entropy(logits, y):
    """
    Backward-compatible alias used by the training modules.
    """
    return binary_cross_entropy_from_logits(logits, y)


def bce_logit_gradient(logits, y):
    """
    Backward-compatible alias used by the training modules.
    """
    return binary_cross_entropy_gradient_from_logits(logits, y)
