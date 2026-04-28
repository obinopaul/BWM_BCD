"""
Paper note:
These unit tests verify the paper-to-code translation for the active modular
implementation.

The tests cover three layers:

    1. BCE from logits:
       the repository uses binary cross-entropy as the classification loss, so
       its value and gradient must be correct and numerically stable.

    2. Section 3 missingness logic:
       the paper builds exact source-availability profiles and overlapping iSFS
       optimization groups from the observed-source pattern, not from labels.

    3. Cost-sensitive trust-region block:
       once alpha and beta are fixed, the code computes the class-loss vector

           L(alpha, beta) = [L_1(alpha, beta), ..., L_K(alpha, beta)]^T

       and uses it to form the trust-region step for the class-cost vector C.

These tests intentionally use the real CSV files in `data/` whenever the goal
is to verify actual profile counts or real-data trust-region feasibility, so
the checks match the dataset that the current implementation is built around.
"""

import unittest
from pathlib import Path

import numpy as np

from loss_functions import (
    binary_cross_entropy_from_logits,
    binary_cross_entropy_gradient_from_logits,
)
from alpha_update import (
    compute_alpha_profile_objective_and_gradient,
    initialize_alphas,
)
from beta_update import (
    compute_beta_smooth_objective_and_gradients,
    initialize_betas,
)
from costs import (
    compute_class_loss_vector,
    cost_weighted_bce_from_logits,
    cost_weighted_bce_gradient_from_logits,
    solve_cost_trust_region_subproblem,
)
from missingness_profiles import build_exact_profiles, build_missingness_profiles
from prediction import predict_class, predict_proba
from regularization import (
    beta_l1_penalty,
    beta_l1_prox,
    max_alpha_l1_norm,
    project_onto_l1_ball,
)


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def load_blockwise_sources():
    return {
        "source1": np.genfromtxt(
            DATA_DIR / "source1_clinical_blockwise_missing.csv",
            delimiter=",",
            skip_header=1,
        ),
        "source2": np.genfromtxt(
            DATA_DIR / "source2_imaging_blockwise_missing.csv",
            delimiter=",",
            skip_header=1,
        ),
        "source3": np.genfromtxt(
            DATA_DIR / "source3_proteomics_blockwise_missing.csv",
            delimiter=",",
            skip_header=1,
        ),
    }


class LossFunctionTests(unittest.TestCase):
    def test_bce_gradient_matches_finite_difference(self):
        rng = np.random.default_rng(0)
        logits = rng.normal(size=8)
        y = rng.integers(0, 2, size=8)

        analytic = binary_cross_entropy_gradient_from_logits(logits, y)
        finite_diff = np.zeros_like(logits)
        eps = 1e-6

        for idx in range(len(logits)):
            plus = logits.copy()
            minus = logits.copy()
            plus[idx] += eps
            minus[idx] -= eps
            finite_diff[idx] = (
                binary_cross_entropy_from_logits(plus, y)
                - binary_cross_entropy_from_logits(minus, y)
            ) / (2 * eps)

        self.assertTrue(np.allclose(analytic, finite_diff, atol=1e-7))

    def test_bce_rejects_nonbinary_targets(self):
        with self.assertRaises(ValueError):
            binary_cross_entropy_from_logits([0.1, -0.2], [0, 2])

    def test_bce_is_finite_for_extreme_logits(self):
        loss = binary_cross_entropy_from_logits(
            np.array([-1000.0, 1000.0]),
            np.array([0, 1]),
        )
        self.assertTrue(np.isfinite(loss))
        self.assertLess(loss, 1e-8)


class MissingnessProfileTests(unittest.TestCase):
    def test_exact_profiles_match_current_dataset(self):
        exact_profiles = build_exact_profiles(load_blockwise_sources())
        actual = {
            "".join("1" if source in profile else "0" for source in ["source1", "source2", "source3"]): len(indices)
            for profile, indices in exact_profiles.items()
        }
        expected = {
            "001": 27,
            "010": 27,
            "011": 61,
            "100": 41,
            "101": 56,
            "110": 75,
            "111": 213,
        }
        self.assertEqual(actual, expected)

    def test_optimization_groups_match_current_dataset(self):
        groups = build_missingness_profiles(load_blockwise_sources())
        actual = {
            "".join("1" if source in profile else "0" for source in ["source1", "source2", "source3"]): len(indices)
            for profile, indices in groups.items()
        }
        expected = {
            "001": 357,
            "010": 376,
            "011": 274,
            "100": 385,
            "101": 269,
            "110": 288,
            "111": 213,
        }
        self.assertEqual(actual, expected)

    def test_partial_nan_row_is_still_observed(self):
        X_blocks = {
            "source1": np.array([[1.0, np.nan], [np.nan, np.nan]]),
            "source2": np.array([[np.nan, np.nan], [2.0, 3.0]]),
        }

        exact_profiles = build_exact_profiles(X_blocks)
        actual = {profile: indices.tolist() for profile, indices in exact_profiles.items()}

        self.assertEqual(actual, {("source1",): [0], ("source2",): [1]})

    def test_inconsistent_sample_counts_raise(self):
        X_blocks = {
            "source1": np.zeros((3, 2)),
            "source2": np.zeros((4, 2)),
        }

        with self.assertRaises(ValueError):
            build_exact_profiles(X_blocks)


class CostBlockTests(unittest.TestCase):
    def test_cost_weighted_bce_gradient_matches_finite_difference(self):
        logits = np.array([-0.3, 0.2, 0.9, -1.1])
        y = np.array([0, 1, 1, 0])
        cost_vector = np.array([0.5, 1.5])

        analytic = cost_weighted_bce_gradient_from_logits(logits, y, cost_vector)
        finite_diff = np.zeros_like(logits)
        eps = 1e-6

        for idx in range(len(logits)):
            plus = logits.copy()
            minus = logits.copy()
            plus[idx] += eps
            minus[idx] -= eps
            finite_diff[idx] = (
                cost_weighted_bce_from_logits(plus, y, cost_vector)
                - cost_weighted_bce_from_logits(minus, y, cost_vector)
            ) / (2 * eps)

        self.assertTrue(np.allclose(analytic, finite_diff, atol=1e-7))

    def test_class_loss_vector_and_trust_region_step_on_real_data(self):
        X_blocks = load_blockwise_sources()
        labels = np.genfromtxt(
            DATA_DIR / "labels.csv",
            delimiter=",",
            skip_header=1,
        )[:, 1].astype(int)

        optimization_groups = build_missingness_profiles(X_blocks)
        alphas = initialize_alphas(optimization_groups)
        betas = initialize_betas(X_blocks, random_state=42)

        class_loss_vector = compute_class_loss_vector(
            X_blocks=X_blocks,
            y=labels,
            betas=betas,
            alphas=alphas,
            optimization_groups=optimization_groups,
            n_classes=2,
        )

        self.assertEqual(class_loss_vector.shape, (2,))
        self.assertTrue(np.all(class_loss_vector >= 0.0))

        baseline_cost = np.array([0.5, 0.5], dtype=float)
        step_info = solve_cost_trust_region_subproblem(
            current_cost=baseline_cost,
            baseline_cost=baseline_cost,
            class_loss_vector=class_loss_vector,
            working_radius=0.125,
            max_radius=0.125,
        )

        displacement = step_info["trial_cost"] - baseline_cost

        self.assertTrue(np.all(step_info["trial_cost"] >= 0.0))
        self.assertLessEqual(
            0.5 * float(np.dot(displacement, displacement)),
            0.125 + 1e-9,
        )
        self.assertGreaterEqual(step_info["predicted_increase"], 0.0)

    def test_all_missing_sample_raises(self):
        X_blocks = {
            "source1": np.array([[np.nan, np.nan]]),
            "source2": np.array([[np.nan, np.nan]]),
        }

        with self.assertRaises(ValueError):
            build_exact_profiles(X_blocks)


class PredictionTests(unittest.TestCase):
    def test_prediction_handles_all_missing_sample(self):
        X_blocks = {
            "source1": np.array([[1.0], [2.0], [np.nan]]),
            "source2": np.array([[np.nan], [3.0], [np.nan]]),
        }
        betas = {
            "source1": np.array([1.0]),
            "source2": np.array([2.0]),
        }
        alphas = {
            ("source1",): {"source1": 1.0},
            ("source2",): {"source2": 1.0},
            ("source1", "source2"): {"source1": 0.25, "source2": 0.75},
        }

        probabilities = predict_proba(X_blocks, betas, alphas)
        predictions = predict_class(X_blocks, betas, alphas, threshold=0.5)

        expected_logits = np.array([
            1.0,
            0.25 * 2.0 + 0.75 * (3.0 * 2.0),
        ])
        expected_probabilities = 1.0 / (1.0 + np.exp(-expected_logits))

        self.assertTrue(np.allclose(probabilities[:2], expected_probabilities))
        self.assertTrue(np.isnan(probabilities[2]))
        self.assertEqual(predictions[2], -1)

    def test_prediction_is_robust_to_reordered_source_dict(self):
        X_blocks = {
            "source2": np.array([[np.nan], [4.0]]),
            "source1": np.array([[2.0], [3.0]]),
        }
        betas = {
            "source1": np.array([1.5]),
            "source2": np.array([0.5]),
        }
        alphas = {
            ("source1",): {"source1": 1.0},
            ("source2",): {"source2": 1.0},
            ("source1", "source2"): {"source1": 0.6, "source2": 0.4},
        }

        probabilities = predict_proba(X_blocks, betas, alphas)

        expected_logits = np.array([
            2.0 * 1.5,
            0.6 * (3.0 * 1.5) + 0.4 * (4.0 * 0.5),
        ])
        expected_probabilities = 1.0 / (1.0 + np.exp(-expected_logits))

        self.assertTrue(np.allclose(probabilities, expected_probabilities))


class RegularizationTests(unittest.TestCase):
    def test_l1_ball_projection_enforces_radius(self):
        v = np.array([1.2, -0.7, 0.3])
        projected = project_onto_l1_ball(v, radius=1.0)

        self.assertLessEqual(np.sum(np.abs(projected)), 1.0 + 1e-9)

    def test_beta_l1_prox_soft_thresholds(self):
        beta = np.array([2.0, -0.4, 0.1])
        updated = beta_l1_prox(beta, step_size=1.0, lambda_beta=0.5)

        expected = np.array([1.5, 0.0, 0.0])
        self.assertTrue(np.allclose(updated, expected))

    def test_alpha_update_keeps_real_data_profiles_in_l1_ball(self):
        X_blocks = load_blockwise_sources()
        labels = np.genfromtxt(
            DATA_DIR / "labels.csv",
            delimiter=",",
            skip_header=1,
        )[:, 1].astype(int)

        optimization_groups = build_missingness_profiles(X_blocks)
        alphas = initialize_alphas(optimization_groups)
        betas = initialize_betas(X_blocks, random_state=42)

        from alpha_update import update_alphas

        alphas = update_alphas(
            X_blocks=X_blocks,
            y=labels,
            betas=betas,
            alphas=alphas,
            profiles=optimization_groups,
            cost_vector=np.array([0.5, 0.5]),
            learning_rate_alpha=1e-2,
            alpha_l1_radius=1.0,
        )

        self.assertLessEqual(max_alpha_l1_norm(alphas), 1.0 + 1e-9)

    def test_beta_l1_penalty_is_nonnegative(self):
        betas = {
            "source1": np.array([1.0, -2.0]),
            "source2": np.array([0.5]),
        }
        penalty = beta_l1_penalty(betas, lambda_beta=0.1)
        self.assertAlmostEqual(penalty, 0.35)


class AlphaBetaAuditTests(unittest.TestCase):
    def test_alpha_profile_gradient_matches_finite_difference(self):
        X_blocks = {
            "source1": np.array([[1.0, 0.5], [0.3, -0.2], [0.7, 0.1]]),
            "source2": np.array([[0.2], [1.1], [-0.4]]),
        }
        y = np.array([0, 1, 0], dtype=float)
        betas = {
            "source1": np.array([0.6, -0.4]),
            "source2": np.array([0.3]),
        }
        alpha = np.array([0.4, -0.2], dtype=float)
        profile = ("source1", "source2")
        indices = np.array([0, 1, 2])
        cost_vector = np.array([0.5, 1.5], dtype=float)

        _, analytic = compute_alpha_profile_objective_and_gradient(
            X_blocks=X_blocks,
            y=y,
            betas=betas,
            alpha_vector=alpha,
            profile=profile,
            indices=indices,
            cost_vector=cost_vector,
        )

        eps = 1e-6
        finite_diff = np.zeros_like(alpha)
        for idx in range(alpha.shape[0]):
            alpha_plus = alpha.copy()
            alpha_minus = alpha.copy()
            alpha_plus[idx] += eps
            alpha_minus[idx] -= eps

            objective_plus, _ = compute_alpha_profile_objective_and_gradient(
                X_blocks=X_blocks,
                y=y,
                betas=betas,
                alpha_vector=alpha_plus,
                profile=profile,
                indices=indices,
                cost_vector=cost_vector,
            )
            objective_minus, _ = compute_alpha_profile_objective_and_gradient(
                X_blocks=X_blocks,
                y=y,
                betas=betas,
                alpha_vector=alpha_minus,
                profile=profile,
                indices=indices,
                cost_vector=cost_vector,
            )
            finite_diff[idx] = (objective_plus - objective_minus) / (2 * eps)

        self.assertTrue(np.allclose(analytic, finite_diff, atol=1e-7))

    def test_beta_smooth_gradient_matches_finite_difference(self):
        X_blocks = {
            "source1": np.array([[0.4, -1.0], [0.2, 0.5], [1.2, 0.3]]),
            "source2": np.array([[1.0], [0.7], [-0.5]]),
        }
        y = np.array([1, 0, 1], dtype=float)
        betas = {
            "source1": np.array([0.2, -0.1]),
            "source2": np.array([0.5]),
        }
        alphas = {
            ("source1",): {"source1": 1.0},
            ("source2",): {"source2": 1.0},
            ("source1", "source2"): {"source1": 0.3, "source2": 0.7},
        }
        profiles = {
            ("source1",): np.array([0]),
            ("source2",): np.array([1]),
            ("source1", "source2"): np.array([2]),
        }
        cost_vector = np.array([1.0, 1.2], dtype=float)

        _, gradients = compute_beta_smooth_objective_and_gradients(
            X_blocks=X_blocks,
            y=y,
            betas=betas,
            alphas=alphas,
            profiles=profiles,
            cost_vector=cost_vector,
        )

        eps = 1e-6
        for source in betas:
            finite_diff = np.zeros_like(betas[source])
            for idx in range(betas[source].shape[0]):
                betas_plus = {key: value.copy() for key, value in betas.items()}
                betas_minus = {key: value.copy() for key, value in betas.items()}
                betas_plus[source][idx] += eps
                betas_minus[source][idx] -= eps

                objective_plus, _ = compute_beta_smooth_objective_and_gradients(
                    X_blocks=X_blocks,
                    y=y,
                    betas=betas_plus,
                    alphas=alphas,
                    profiles=profiles,
                    cost_vector=cost_vector,
                )
                objective_minus, _ = compute_beta_smooth_objective_and_gradients(
                    X_blocks=X_blocks,
                    y=y,
                    betas=betas_minus,
                    alphas=alphas,
                    profiles=profiles,
                    cost_vector=cost_vector,
                )
                finite_diff[idx] = (objective_plus - objective_minus) / (2 * eps)

            self.assertTrue(np.allclose(gradients[source], finite_diff, atol=1e-7))


if __name__ == "__main__":
    unittest.main()
