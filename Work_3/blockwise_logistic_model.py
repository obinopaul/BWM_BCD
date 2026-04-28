"""
blockwise_logistic_model.py

Paper note:
This file is an older all-in-one educational wrapper that combines profile
construction, prediction, loss tracking, and parameter updates inside one class.
It is not the active modular training path used by `example_run.py`.
The current source of truth in this repo is the split implementation:
`missingness_profiles.py`, `alpha_update.py`, `beta_update.py`,
`prediction.py`, `loss_functions.py`, and `train_model.py`.
So if you are checking the model against Section 3 of the paper, read the
modular files first. This class should be treated as a legacy wrapper until
it is rewritten to match the simplified modular version exactly.

Data format expected
--------------------
X_blocks:
    A dictionary where each key is a source name and each value is a 2D array.

    Example:
        X_blocks = {
            "clinical": X_clinical,
            "imaging_1": X_imaging_1,
            "imaging_2": X_imaging_2,
        }

    Each X_blocks[source] must have shape:

        (n_samples, n_features_for_that_source)

    Missing blocks should be represented by np.nan rows.
    For example, if patient j has no imaging_1 data, then:

        X_imaging_1[j, :] = np.nan

y:
    Binary label array of shape (n_samples,).
    Labels must be 0 or 1.

Important
---------
This is a clean educational implementation, not yet an industrial optimizer.
It is written to help you understand the calculation from scratch.
"""

import numpy as np
from loss_functions import sigmoid, binary_cross_entropy_from_logits


class BlockwiseLogisticModel:
    def __init__(
        self,
        lambda_beta=1e-3,
        lambda_alpha=1e-3,
        learning_rate_beta=1e-2,
        learning_rate_alpha=1e-2,
        max_iter=500,
        tol=1e-6,
        use_cost_penalty=False,
        lambda_cost=0.0,
        random_state=42,
        verbose=True,
    ):
        """
        Parameters
        ----------
        lambda_beta : float
            Regularization strength for beta.

        lambda_alpha : float
            Regularization strength for alpha.

        learning_rate_beta : float
            Gradient descent learning rate for beta update.

        learning_rate_alpha : float
            Gradient descent learning rate for alpha update.

        max_iter : int
            Number of outer iterations k.

        tol : float
            Convergence tolerance.

        use_cost_penalty : bool
            Whether to add a cost-sensitive penalty.

        lambda_cost : float
            Strength of the cost-sensitive penalty.

        random_state : int
            Random seed.

        verbose : bool
            Whether to print training progress.
        """
        self.lambda_beta = lambda_beta
        self.lambda_alpha = lambda_alpha
        self.learning_rate_beta = learning_rate_beta
        self.learning_rate_alpha = learning_rate_alpha
        self.max_iter = max_iter
        self.tol = tol
        self.use_cost_penalty = use_cost_penalty
        self.lambda_cost = lambda_cost
        self.random_state = random_state
        self.verbose = verbose

        self.sources_ = None
        self.betas_ = None
        self.alphas_ = None
        self.profiles_ = None
        self.loss_history_ = []

    # ---------------------------------------------------------------------
    # Missingness profile construction
    # ---------------------------------------------------------------------

    def _get_observed_mask(self, X):
        """
        A block/source is observed for a sample if the row is not all NaN.
        """
        return ~np.all(np.isnan(X), axis=1)

    def _build_profiles(self, X_blocks):
        """
        Build missingness profiles.

        A profile is a tuple of available source names.

        Example:
            ("clinical", "imaging_1")
            ("clinical", "imaging_2")
            ("clinical", "imaging_1", "imaging_2")
        """
        sources = list(X_blocks.keys())
        n_samples = next(iter(X_blocks.values())).shape[0]

        profiles = {}

        for j in range(n_samples):
            available_sources = []

            for source in sources:
                row = X_blocks[source][j, :]
                if not np.all(np.isnan(row)):
                    available_sources.append(source)

            if len(available_sources) == 0:
                # Skip samples with no source available
                continue

            profile = tuple(available_sources)

            if profile not in profiles:
                profiles[profile] = []

            profiles[profile].append(j)

        profiles = {
            profile: np.array(indices, dtype=int)
            for profile, indices in profiles.items()
        }

        return profiles

    # ---------------------------------------------------------------------
    # Parameter initialization
    # ---------------------------------------------------------------------

    def _initialize_parameters(self, X_blocks):
        """
        Initialize beta for each source and alpha for each profile.
        """
        rng = np.random.default_rng(self.random_state)

        self.sources_ = list(X_blocks.keys())

        self.betas_ = {}
        for source in self.sources_:
            n_features = X_blocks[source].shape[1]
            self.betas_[source] = rng.normal(
                loc=0.0,
                scale=0.01,
                size=n_features
            )

        self.alphas_ = {}
        for profile in self.profiles_:
            n_sources_in_profile = len(profile)

            # Start with equal source contribution within each profile
            alpha_values = np.ones(n_sources_in_profile) / n_sources_in_profile

            self.alphas_[profile] = {
                source: alpha_values[i]
                for i, source in enumerate(profile)
            }

    # ---------------------------------------------------------------------
    # Forward pass
    # ---------------------------------------------------------------------

    def _compute_logits_for_profile(self, X_blocks, profile, indices):
        """
        Compute logits for samples belonging to one missingness profile.

        z = sum_i alpha_m,i * X_i * beta_i
        """
        n = len(indices)
        logits = np.zeros(n)

        for source in profile:
            X_source = X_blocks[source][indices, :]

            # Replace NaN with 0 for safety.
            # For this profile, source should be observed, so this should not matter.
            X_source = np.nan_to_num(X_source, nan=0.0)

            beta_i = self.betas_[source]
            alpha_m_i = self.alphas_[profile][source]

            logits += alpha_m_i * (X_source @ beta_i)

        return logits

    def _compute_all_logits(self, X_blocks):
        """
        Compute logits for all samples that have at least one observed source.
        """
        n_samples = next(iter(X_blocks.values())).shape[0]
        logits_all = np.zeros(n_samples)
        valid_mask = np.zeros(n_samples, dtype=bool)

        for profile, indices in self.profiles_.items():
            logits_all[indices] = self._compute_logits_for_profile(
                X_blocks,
                profile,
                indices
            )
            valid_mask[indices] = True

        return logits_all, valid_mask

    # ---------------------------------------------------------------------
    # Regularization and cost penalty
    # ---------------------------------------------------------------------

    def _beta_l2_penalty(self):
        """
        Simple L2 penalty on beta.

        You can later replace this with:
            - L1 penalty
            - group lasso penalty
            - sparse group lasso penalty
            - nonconvex block penalty
        """
        penalty = 0.0

        for source in self.sources_:
            penalty += np.sum(self.betas_[source] ** 2)

        return 0.5 * self.lambda_beta * penalty

    def _alpha_l2_penalty(self):
        """
        Simple L2 penalty on alpha values.
        """
        penalty = 0.0

        for profile in self.alphas_:
            for source in self.alphas_[profile]:
                penalty += self.alphas_[profile][source] ** 2

        return 0.5 * self.lambda_alpha * penalty

    def _cost_sensitive_penalty(self, probabilities, y):
        """
        Placeholder for your cost-sensitive penalty.

        For now, this is a simple example:

            penalty = mean(abs(p - y))

        But you can replace it with your actual cost-sensitive formula,
        such as a cost matrix, class-dependent cost, false-negative cost,
        false-positive cost, trust-region cost, or blockwise cost vector.

        Parameters
        ----------
        probabilities : np.ndarray
            Predicted probabilities.

        y : np.ndarray
            Binary labels.

        Returns
        -------
        float
            Cost penalty value.
        """
        if not self.use_cost_penalty:
            return 0.0

        return self.lambda_cost * np.mean(np.abs(probabilities - y))

    def _total_loss(self, X_blocks, y):
        """
        Compute total objective:

            BCE + beta regularization + alpha regularization + cost penalty
        """
        logits_all, valid_mask = self._compute_all_logits(X_blocks)

        y_valid = y[valid_mask]
        logits_valid = logits_all[valid_mask]

        bce = binary_cross_entropy_from_logits(logits_valid, y_valid)

        p_valid = sigmoid(logits_valid)

        beta_penalty = self._beta_l2_penalty()
        alpha_penalty = self._alpha_l2_penalty()
        cost_penalty = self._cost_sensitive_penalty(p_valid, y_valid)

        total = bce + beta_penalty + alpha_penalty + cost_penalty

        return total, bce, beta_penalty, alpha_penalty, cost_penalty

    # ---------------------------------------------------------------------
    # Gradients
    # ---------------------------------------------------------------------

    def _update_alpha(self, X_blocks, y):
        """
        Update alpha while beta is fixed.

        This corresponds to the alpha-step in the alternating optimization.

        In the paper's logic:
            beta is fixed,
            alpha_m is learned separately for each missingness profile m.
        """
        for profile, indices in self.profiles_.items():
            y_m = y[indices]

            logits_m = self._compute_logits_for_profile(
                X_blocks,
                profile,
                indices
            )

            p_m = sigmoid(logits_m)

            # Gradient of BCE wrt logits
            error = (p_m - y_m) / len(indices)

            for source in profile:
                X_source = np.nan_to_num(X_blocks[source][indices, :], nan=0.0)
                beta_i = self.betas_[source]

                # score_i = X_i beta_i
                source_score = X_source @ beta_i

                # dL/d alpha_m_i
                grad_alpha = np.sum(error * source_score)

                # L2 regularization gradient
                grad_alpha += self.lambda_alpha * self.alphas_[profile][source]

                # Optional cost penalty gradient approximation
                # This is kept simple for now.
                if self.use_cost_penalty:
                    grad_alpha += self.lambda_cost * grad_alpha

                self.alphas_[profile][source] -= (
                    self.learning_rate_alpha * grad_alpha
                )

            # Optional: project alpha to be nonnegative and sum to 1.
            # This makes alpha behave like source-combination weights.
            self._project_alpha_simplex(profile)

    def _project_alpha_simplex(self, profile):
        """
        Project alpha_m onto the probability simplex:

            alpha_m_i >= 0
            sum_i alpha_m_i = 1

        This is useful if you want alpha values to act like source weights.
        """
        sources = list(profile)
        values = np.array([self.alphas_[profile][s] for s in sources])

        projected = self._simplex_projection(values)

        for s, value in zip(sources, projected):
            self.alphas_[profile][s] = value

    @staticmethod
    def _simplex_projection(v):
        """
        Euclidean projection onto simplex:

            {x : x >= 0, sum(x) = 1}

        Reference implementation based on sorting.
        """
        v = np.asarray(v, dtype=float)

        if v.size == 1:
            return np.array([1.0])

        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        rho_candidates = u * np.arange(1, len(v) + 1) > (cssv - 1)

        if not np.any(rho_candidates):
            return np.ones_like(v) / len(v)

        rho = np.where(rho_candidates)[0][-1]
        theta = (cssv[rho] - 1.0) / (rho + 1.0)

        w = np.maximum(v - theta, 0.0)
        return w

    def _update_beta(self, X_blocks, y):
        """
        Update beta while alpha is fixed.

        This corresponds to the beta-step in the alternating optimization.

        The beta_i values are shared across all missingness profiles.
        """
        beta_grads = {
            source: np.zeros_like(self.betas_[source])
            for source in self.sources_
        }

        for profile, indices in self.profiles_.items():
            y_m = y[indices]

            logits_m = self._compute_logits_for_profile(
                X_blocks,
                profile,
                indices
            )

            p_m = sigmoid(logits_m)

            # Gradient of BCE wrt logits
            error = (p_m - y_m) / len(indices)

            for source in profile:
                X_source = np.nan_to_num(X_blocks[source][indices, :], nan=0.0)
                alpha_m_i = self.alphas_[profile][source]

                # dL/d beta_i = X_i.T @ error * alpha_m_i
                grad_beta = alpha_m_i * (X_source.T @ error)

                beta_grads[source] += grad_beta

        # Add beta L2 regularization and update
        for source in self.sources_:
            beta_grads[source] += self.lambda_beta * self.betas_[source]

            if self.use_cost_penalty:
                beta_grads[source] += self.lambda_cost * beta_grads[source]

            self.betas_[source] -= self.learning_rate_beta * beta_grads[source]

    # ---------------------------------------------------------------------
    # Main training loop
    # ---------------------------------------------------------------------

    def fit(self, X_blocks, y):
        """
        Fit the blockwise logistic model.

        Parameters
        ----------
        X_blocks : dict[str, np.ndarray]
            Dictionary of source matrices.

        y : np.ndarray
            Binary labels.

        Returns
        -------
        self
        """
        y = np.asarray(y, dtype=float).reshape(-1)

        # Basic checks
        n_samples = y.shape[0]

        for source, X in X_blocks.items():
            if X.shape[0] != n_samples:
                raise ValueError(
                    f"Source {source} has {X.shape[0]} samples, "
                    f"but y has {n_samples} samples."
                )

        self.profiles_ = self._build_profiles(X_blocks)
        self._initialize_parameters(X_blocks)

        previous_loss = np.inf

        for k in range(1, self.max_iter + 1):
            # Step 1: update alpha while beta is fixed
            self._update_alpha(X_blocks, y)

            # Step 2: update beta while alpha is fixed
            self._update_beta(X_blocks, y)

            # Step 3: compute loss
            total, bce, beta_penalty, alpha_penalty, cost_penalty = self._total_loss(
                X_blocks,
                y
            )

            self.loss_history_.append(total)

            # Step 4: print progress
            if self.verbose and (k == 1 or k % 25 == 0):
                print(
                    f"Iteration {k:04d} | "
                    f"Total Loss: {total:.6f} | "
                    f"BCE: {bce:.6f} | "
                    f"Beta Penalty: {beta_penalty:.6f} | "
                    f"Alpha Penalty: {alpha_penalty:.6f} | "
                    f"Cost Penalty: {cost_penalty:.6f}"
                )

            # Step 5: convergence check
            loss_change = abs(previous_loss - total)

            if loss_change < self.tol:
                if self.verbose:
                    print(
                        f"Converged at iteration {k}. "
                        f"Loss change = {loss_change:.8f}"
                    )
                break

            previous_loss = total

        return self

    # ---------------------------------------------------------------------
    # Prediction
    # ---------------------------------------------------------------------

    def predict_proba(self, X_blocks):
        """
        Predict probability P(y=1 | X).
        """
        logits_all, valid_mask = self._compute_all_logits(X_blocks)
        probabilities = sigmoid(logits_all)

        # Samples with no available source get NaN probability
        probabilities[~valid_mask] = np.nan

        return probabilities

    def predict(self, X_blocks, threshold=0.5):
        """
        Predict binary class labels.
        """
        probabilities = self.predict_proba(X_blocks)

        predictions = np.where(probabilities >= threshold, 1, 0)
        predictions[np.isnan(probabilities)] = -1

        return predictions


# -------------------------------------------------------------------------
# Example usage
# -------------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(42)

    n_samples = 200

    # Example with three data sources
    X_clinical = np.random.normal(size=(n_samples, 5))
    X_imaging_1 = np.random.normal(size=(n_samples, 10))
    X_imaging_2 = np.random.normal(size=(n_samples, 8))

    # Create binary labels
    true_signal = (
        X_clinical[:, 0]
        + 0.5 * X_imaging_1[:, 1]
        - 0.7 * X_imaging_2[:, 2]
    )

    true_prob = sigmoid(true_signal)
    y = (true_prob > 0.5).astype(int)

    # Introduce blockwise missingness
    # Some samples missing imaging_1
    X_imaging_1[40:90, :] = np.nan

    # Some samples missing imaging_2
    X_imaging_2[100:150, :] = np.nan

    # Some samples missing both imaging sources, but still have clinical
    X_imaging_1[160:180, :] = np.nan
    X_imaging_2[160:180, :] = np.nan

    X_blocks = {
        "clinical": X_clinical,
        "imaging_1": X_imaging_1,
        "imaging_2": X_imaging_2,
    }

    model = BlockwiseLogisticModel(
        lambda_beta=1e-3,
        lambda_alpha=1e-3,
        learning_rate_beta=1e-2,
        learning_rate_alpha=1e-2,
        max_iter=300,
        tol=1e-7,
        use_cost_penalty=False,
        lambda_cost=0.0,
        verbose=True,
    )

    model.fit(X_blocks, y)

    probabilities = model.predict_proba(X_blocks)
    predictions = model.predict(X_blocks)

    accuracy = np.mean(predictions[predictions != -1] == y[predictions != -1])

    print("\nFinal accuracy:", accuracy)

    print("\nLearned alpha values by missingness profile:")
    for profile, alpha_values in model.alphas_.items():
        print(profile, alpha_values)

    print("\nLearned beta shapes:")
    for source, beta in model.betas_.items():
        print(source, beta.shape)
