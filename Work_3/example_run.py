"""
Paper note:
This file is only a small runnable example for the modular implementation.
It generates synthetic blockwise-missing data, starts from a baseline class-cost
vector, solves Block A at fixed cost, then lets the outer BCD loop update the
cost vector by trust region while re-solving alpha and beta. In the current
paper-aligned path, alpha uses an l1-ball constraint and beta uses an l1
penalty.
The equations themselves live in the core modules:
`missingness_profiles.py`, `alpha_update.py`, `beta_update.py`, `costs.py`,
and `train_model.py`.
"""

import numpy as np

from train_model import train_blockwise_logistic_model
from loss import sigmoid


def make_test_data():
    """
    Create a small synthetic blockwise missing multi-source dataset.
    """
    np.random.seed(42)

    n_samples = 200

    X_clinical = np.random.normal(size=(n_samples, 5))
    X_imaging_1 = np.random.normal(size=(n_samples, 10))
    X_imaging_2 = np.random.normal(size=(n_samples, 8))

    true_signal = (
        1.2 * X_clinical[:, 0]
        + 0.8 * X_imaging_1[:, 1]
        - 0.9 * X_imaging_2[:, 2]
    )

    probability = sigmoid(true_signal)

    y = (probability >= 0.5).astype(int)

    # Introduce blockwise missingness.
    # These are not random individual cells.
    # Whole source blocks are missing for some samples.
    X_imaging_1[40:90, :] = np.nan
    X_imaging_2[100:150, :] = np.nan

    X_imaging_1[160:180, :] = np.nan
    X_imaging_2[160:180, :] = np.nan

    X_blocks = {
        "clinical": X_clinical,
        "imaging_1": X_imaging_1,
        "imaging_2": X_imaging_2
    }

    return X_blocks, y


if __name__ == "__main__":

    X_blocks, y = make_test_data()

    model = train_blockwise_logistic_model(
        X_blocks=X_blocks,
        y=y,
        learning_rate_beta=1e-2,
        learning_rate_alpha=1e-2,
        lambda_beta=1e-2,
        alpha_l1_radius=1.0,
        max_iter=12,
        tol=1e-7,
        random_state=42,
        block_a_max_iter=50,
        block_a_tol=1e-7,
        initial_cost_vector=np.array([0.5, 0.5]),
        baseline_cost_vector=np.array([0.5, 0.5]),
        initial_trust_region_radius=0.02,
        max_trust_region_radius=0.125,
        verbose=True
    )

    predictions = model["predict"](X_blocks, threshold=0.5)

    valid_mask = predictions != -1

    accuracy = np.mean(predictions[valid_mask] == y[valid_mask])

    print("\nFinal accuracy:", accuracy)

    print("\nLearned alpha values:")
    for profile, alpha_values in model["alphas"].items():
        print(profile, alpha_values)

    print("\nFinal cost vector:")
    print(model["costs"])

    print("\nAlpha max l1 norm:")
    print(max(np.sum(np.abs(list(alpha_values.values()))) for alpha_values in model["alphas"].values()))

    print("\nLearned beta dimensions:")
    for source, beta in model["betas"].items():
        print(source, beta.shape)
