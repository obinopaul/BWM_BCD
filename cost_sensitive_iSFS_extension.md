# Cost-Sensitive iSFS via Block Coordinate Descent

> A cost-sensitive extension of the **incomplete Source-Feature Selection
> (iSFS)** model of Xiang, Yuan, Fan, Wang, Thompson & Ye (KDD 2013). We
> introduce a learnable per-class cost vector `C` and derive the BCD updates
> for jointly optimizing `(α, β, C)` using only the paper's notation.
>
> Sources used throughout:
>
> - **Xiang et al. (2013).** *Multi-Source Learning with Block-wise Missing
>   Data for Alzheimer's Disease Prediction.* KDD '13. (`Multi_source_BWM.pdf`)
>   The iSFS objective is eq (14); its data-fitting term is eq (15); the
>   alternating block updates for `α` and `β` are eq (16) and eqs (17)–(18).
>   Algorithm 2 in §3.2 lays out the original BCD scheme.
> - **Aggarwal,** *Linear Algebra and Optimization for Machine Learning,*
>   §4.10 (`1660642748-214-217.pdf`) and §8.3 (`1660642748-356-362.pdf`).
>   The block coordinate descent framework, the multi-convex monotonicity
>   theorem, and the K-means / ALS examples we inherit.
> - **Bertsimas, Pawlowski, Zhuo (JMLR 2018),** *From Predictive Methods to
>   Missing Data Imputation* (`17-073.pdf`). Uses BCD vs CD as the inner
>   solver for non-convex imputation.
> - **`BCD_explained.md`** (this folder). Working derivation of the BCD
>   framework with the K-means and ALS case studies.
> - **Namkoong & Duchi (2017),** *Variance-based regularization with convex
>   objectives,* NeurIPS — distributionally robust optimization with the
>   chi-squared ball, the closed-form max via KKT.
> - **Hu, Niu, Sato, Sugiyama (2018),** *Does distributionally robust
>   supervised learning give robust classifiers?,* ICML — connects DRO
>   directly to cost-sensitive learning.

---

## Table of contents

1. [Recap of the iSFS notation we inherit from the paper](#1-recap-of-the-isfs-notation-we-inherit-from-the-paper)
2. [Why a cost-sensitive extension — and why fixed `C` is not enough](#2-why-a-cost-sensitive-extension--and-why-fixed-c-is-not-enough)
3. [The cost-sensitive iSFS objective](#3-the-cost-sensitive-isfs-objective)
4. [The constraint set `𝒞_ρ`](#4-the-constraint-set-c_)
5. [Step 1 — fix `C`, solve `(α, β)`](#5-step-1--fix-c-solve-alpha-beta)
6. [Step 2 — fix `(α, β)`, solve `C`](#6-step-2--fix-alpha-beta-solve-c)
7. [Algorithm 3 — full pseudocode and numerical walkthrough](#7-algorithm-3--full-pseudocode-and-numerical-walkthrough)
8. [Connection to BCD theory](#8-connection-to-bcd-theory)
9. [Quick reference card](#9-quick-reference-card)

---

## 1. Recap of the iSFS notation we inherit from the paper

We use the **exact symbols** from Xiang et al. (2013) §3.1. No new variables until §3.

**Variables:**

- `S` — number of data sources (e.g., PET, MRI, CSF; so `S = 3`).
- `m` — a profile index. A profile is the binary indicator of which sources are present, encoded as a decimal integer.
- `pf` — vector of profiles for all participants. `|pf|` is the number of distinct profiles.
- `n_m` — number of samples in profile `m`.
- `n` — number of rows of `X` in a generic context (the paper uses both `n` and `n_m`).
- `X^i ∈ ℝ^{n × p_i}` — data matrix for source `i`.
- `X^i_m` — data matrix `X^i` restricted to samples in profile `m`.
- `β^i ∈ ℝ^{p_i}` — feature coefficient block for source `i`. **Shared across profiles.**
- `β = (β^1, …, β^S) ∈ ℝ^{p_1 + … + p_S}` — full feature coefficient vector.
- `α^i_m` — scalar weight on source `i` for profile `m`. **Profile-specific.**
- `α_m = (α^1_m, …, α^S_m) ∈ ℝ^S` — full source-weight vector for profile `m`.
- `y_m` — label vector for samples in profile `m`.
- `L(·, ·)` — convex loss function (least-squares, logistic, etc.). The paper says: *"L can be any convex loss function such as the least squares loss function or the logistic loss function and `n` is number of rows of `X`."*
- `R_α, R_β` — regularizers on `α` and `β`.
- `λ` — regularization strength on `R_β`.

**Equation (14)** — the iSFS objective:

$$
\min_{\boldsymbol{\alpha},\,\boldsymbol{\beta}}\;\;\frac{1}{|\mathbf{pf}|}\sum_{m \in \mathbf{pf}} f(\mathbf{X}_m,\boldsymbol{\beta},\boldsymbol{\alpha}_m,\mathbf{y}_m) \;+\; \lambda\,\mathbf{R}_{\boldsymbol{\beta}}(\boldsymbol{\beta})
$$

$$
\text{s.t.}\quad \mathbf{R}_{\boldsymbol{\alpha}}(\boldsymbol{\alpha}_m) \leq 1\;\;\forall\,m \in \mathbf{pf}
$$

**Equation (15)** — the per-profile data-fitting term:

$$
f(\mathbf{X},\boldsymbol{\beta},\boldsymbol{\alpha},\mathbf{y}) \;=\; \frac{1}{n}\,\mathbf{L}\!\left(\sum_{i=1}^{S}\alpha^{i}\mathbf{X}^{i}\boldsymbol{\beta}^{i},\;\mathbf{y}\right)
$$

**Equation (16)** — the `α` subproblem (when `β` is fixed). Decouples by profile:

$$
\min_{\boldsymbol{\alpha}}\;\;\Big\|\sum_{i=1}^{S}\alpha^{i}\mathbf{X}^{i}\boldsymbol{\beta}^{i} - \mathbf{y}\Big\|_{2}^{2}
\quad\text{s.t.}\quad \mathbf{R}_{\boldsymbol{\alpha}}(\boldsymbol{\alpha}) \leq 1
$$

**Equation (17)** — the `β` subproblem (when `α` is fixed):

$$
\min_{\boldsymbol{\beta}}\;\;g(\boldsymbol{\beta}) + \lambda\,\mathbf{R}_{\boldsymbol{\beta}}(\boldsymbol{\beta}),
\qquad
g(\boldsymbol{\beta}) \;=\; \frac{1}{|\mathbf{pf}|}\sum_{m \in \mathbf{pf}}\frac{1}{2 n_m}\Big\|\sum_{i=1}^{S}\alpha^{i}_{m}\mathbf{X}^{i}_{m}\boldsymbol{\beta}^{i} - \mathbf{y}_m\Big\|_{2}^{2}
$$

**Equation (18)** — the gradient of `g` for the `β` step (used by FISTA):

$$
\nabla g(\boldsymbol{\beta}^{i}) \;=\; \frac{1}{|\mathbf{pf}|}\sum_{m \in \mathbf{pf}}\frac{1}{n_m}\,\mathbf{I}\!\left(m\,\&\,2^{S-i}\neq 0\right)\,(\alpha^{i}_{m}\mathbf{X}^{i}_{m})^{T}\!\Big(\textstyle\sum_{j=1}^{S}\alpha^{j}_{m}\mathbf{X}^{j}_{m}\boldsymbol{\beta}^{j} - \mathbf{y}_m\Big)
$$

The bitwise indicator `I(m & 2^{S−i} ≠ 0)` equals `1` iff source `i` is present in profile `m`. This is a literal bit-test on the profile integer.

**Algorithm 2 in the paper** is the BCD that alternates eq (16) and eq (17) until the objective stops decreasing. It is exactly the multi-convex BCD pattern from Aggarwal §4.10.2 — convex in each block given the other fixed, monotonic decrease across the alternation.

That is the entire toolkit we inherit. Everything below adds **one new variable `C`** and derives the corresponding new block update.

---

## 2. Why a cost-sensitive extension — and why fixed `C` is not enough

### 2.1 Motivation

In Alzheimer's disease prediction (and most clinical settings), the minority class (e.g., progressive MCI converters) is rare. Standard iSFS minimizes average loss and quietly becomes majority-biased. The classical fix is **cost-sensitive learning**: assign higher misclassification penalty to minority-class errors. From Elkan (2001):

$$
\text{Total Cost} \;=\; C(0,1)\cdot\text{FN} \;+\; C(1,0)\cdot\text{FP},\qquad C(0,1) \gg C(1,0).
$$

For our setting, define a **per-class cost vector** `C = (C_1, …, C_K)` where `K` is the number of classes (`K = 2` for binary). Each sample's loss contribution is multiplied by `C_{y_j}` — the cost weight that corresponds to its class label.

### 2.2 Option A — fixed `C` (classical Elkan-style)

**The fixed-`C` rule.** The standard "inverse class frequency" choice:

$$
C_k^{\,0} \;=\; \frac{N}{K \cdot n_k},\qquad k = 1, \ldots, K
$$

where:
- `N = Σ_k n_k` is the total sample count.
- `n_k` is the number of samples in class `k`.

This makes minority-class samples count more in the loss. For our running example with `n_- = 12, n_+ = 4, N = 16, K = 2`:

$$
C_-^{\,0} = \frac{16}{2\cdot 12} \approx 0.667,\qquad C_+^{\,0} = \frac{16}{2\cdot 4} = 2.0.
$$

The minority class gets `n_-/n_+ = 3×` the per-sample weight of the majority class.

**The cost-sensitive version of `L`.** Promote the paper's loss function `L` to its class-weighted form:

$$
\mathbf{L}^{C}(\hat{\mathbf{y}},\mathbf{y}) \;=\; \frac{1}{2}\sum_{j=1}^{n} C_{y_{j}}\,(\hat{y}_{j} - y_{j})^{2}
$$

Plug this into the iSFS objective in place of `L`. With `C` fixed at `C^0`, the entire optimization remains in `(α, β)`:

$$
\min_{\boldsymbol{\alpha},\,\boldsymbol{\beta}}\;\;\frac{1}{|\mathbf{pf}|}\sum_{m \in \mathbf{pf}} f^{C^{0}}(\mathbf{X}_m,\boldsymbol{\beta},\boldsymbol{\alpha}_m,\mathbf{y}_m) \;+\; \lambda\,\mathbf{R}_{\boldsymbol{\beta}}(\boldsymbol{\beta})
$$

where

$$
f^{C}(\mathbf{X},\boldsymbol{\beta},\boldsymbol{\alpha},\mathbf{y}) \;=\; \frac{1}{n}\,\mathbf{L}^{C}\!\left(\sum_{i=1}^{S}\alpha^{i}\mathbf{X}^{i}\boldsymbol{\beta}^{i},\;\mathbf{y}\right).
$$

**Algorithm 2 from the paper runs unchanged** — only the inside of `L` carries the `C^0` weights.

### 2.3 Why BCD on `C` is impossible in Option A

Define the per-class loss contribution at the current `(α, β)`:

$$
L_k(\boldsymbol{\alpha},\boldsymbol{\beta}) \;=\; \frac{1}{|\mathbf{pf}|}\sum_{m \in \mathbf{pf}}\frac{1}{2 n_m}\!\!\sum_{\substack{j: y_{m,j} = k}}\!(\hat{y}_{m,j} - y_{m,j})^{2}
$$

The data-fitting part of the objective as a function of `C` alone is

$$
\Phi_C(C) \;=\; \sum_{k=1}^{K} C_k\,L_k(\boldsymbol{\alpha},\boldsymbol{\beta}).
$$

This is **linear in `C`** with positive coefficients. Now check what happens if you try to "solve for `C`" with no constraint:

| try | result |
|---|---|
| `min_C Φ_C(C)`, `C ∈ ℝ^K` | `−∞` (unbounded below) |
| `min_C Φ_C(C)`, `C_k ≥ 0` | `C* = 0` — trivial, model doesn't learn anything |
| `max_C Φ_C(C)`, `C_k ≥ 0` | `+∞` (unbounded above) |
| `max_C Φ_C(C)`, single fixed value | not an optimization, just a constant |

In every case, the "fix `α, β`, solve for `C`" step is **degenerate**. There is no well-defined sub-problem in `C` alone unless `C` is constrained to a non-trivial bounded set.

This is exactly why classical cost-sensitive learning **never solves for `C`**:

- **Elkan (2001),** *The Foundations of Cost-Sensitive Learning,* IJCAI — `C` is set from domain knowledge, never optimized.
- **Domingos (1999),** *MetaCost: A general method for making classifiers cost-sensitive,* KDD — `C` is fixed by the user.
- **King & Zeng (2001),** *Logistic regression in rare events data,* Political Analysis — uses inverse-frequency weights; `C` is computed from class counts and never updated.
- **`class_weight='balanced'`** in scikit-learn — implements `C_k = N/(K · n_k)` exactly, then it is fixed.

**Conclusion for Option A:** Two-block BCD only — `(α, β)`. No third block. There is no "fix `α, β`, solve for `C`" step in this framework.

To get a real `C` step, `C` must live in a bounded set. That is what §3 below builds.

---

## 3. The cost-sensitive iSFS objective

**One new variable** beyond what the paper uses:

- `C ∈ ℝ^K` — per-class cost vector. `K` parameters total (just 2 for binary classification).
- `C^0 ∈ ℝ^K` — baseline cost vector. We use the rescaled inverse-frequency baseline (defined in §4).
- `ρ > 0` — trust-region radius around `C^0` (hyperparameter).
- `𝒞_ρ` — feasible set for `C` (defined in §4).

**Cost-sensitive loss function** — extends the paper's `L` by attaching the class-cost weight to each sample's residual:

$$
\boxed{\;\mathbf{L}^{C}(\hat{\mathbf{y}},\mathbf{y}) \;=\; \frac{1}{2}\sum_{j=1}^{n} C_{y_{j}}\,(\hat{y}_{j} - y_{j})^{2}\;}
$$

Read this in three pieces:
- `(ŷ_j − y_j)²` — squared residual for sample `j` (same as paper's least-squares).
- `C_{y_j}` — look up the cost weight for sample `j`'s class label `y_j`. This is K-way indexing, **not** a per-sample free parameter. `C` still has only `K` total parameters.
- `Σ_j` — sum over all samples.

When `C_k = 1` for all `k`, `L^C` reduces to `½‖ŷ − y‖²`, which is exactly the paper's least-squares `L`.

**Cost-sensitive data-fitting term** — extends eq (15) by replacing `L` with `L^C`:

$$
\boxed{\;f^{C}(\mathbf{X},\boldsymbol{\beta},\boldsymbol{\alpha},\mathbf{y}) \;=\; \frac{1}{n}\,\mathbf{L}^{C}\!\left(\sum_{i=1}^{S}\alpha^{i}\mathbf{X}^{i}\boldsymbol{\beta}^{i},\;\mathbf{y}\right)\;}
$$

**Cost-sensitive iSFS objective** — saddle-point extension of eq (14):

$$
\boxed{\;
\min_{\boldsymbol{\alpha},\,\boldsymbol{\beta}}\;\max_{C \in \mathcal{C}_\rho}\;\;\frac{1}{|\mathbf{pf}|}\sum_{m \in \mathbf{pf}} f^{C}(\mathbf{X}_m,\boldsymbol{\beta},\boldsymbol{\alpha}_m,\mathbf{y}_m) \;+\; \lambda\,\mathbf{R}_{\boldsymbol{\beta}}(\boldsymbol{\beta})
\;}
$$

$$
\text{s.t.}\quad \mathbf{R}_{\boldsymbol{\alpha}}(\boldsymbol{\alpha}_m) \leq 1\;\;\forall\,m \in \mathbf{pf}
$$

**Three things to read off:**

1. **`min` over `(α, β)`, `max` over `C`.** This is the saddle-point structure used in distributionally robust optimization (DRO). The model is trained against the worst-case re-weighting of the classes.
2. **`C` lives in `𝒞_ρ`.** No regularizer term `μ R_C(C)` in the objective. `C` is constrained, not penalized.
3. **The only changes from eq (14):** new variable `C`; `f → f^C`; new constraint `C ∈ 𝒞_ρ`. Everything else is identical.

**Why a max over `C`, not a min.** A pure-min framing in `C` gives the wrong sign — classes with higher loss would get *lower* cost weight, the opposite of cost-sensitive intent. The max framing is the principled way to learn `C` jointly: the `max` is "the adversary picks the worst-case re-weighting for the current model", and minimizing against that adversary forces the model to do well across **all** re-weightings inside the ball, including the cost-sensitive ones.

---

## 4. The constraint set `𝒞_ρ`

`𝒞_ρ` is built from **two stacked constraints**. Each one alone is broken; the combination is what works.

### 4.1 The two pieces

**Piece 1 — simplex (budget):**

$$
C_k \geq 0,\qquad \sum_{k=1}^{K} C_k = K
$$

Pins the total cost weight to `K`, so the average per class is exactly `1`. Without this, `C` could be sent to `0` or `+∞`.

**Piece 2 — ball (trust region around `C^0`):**

$$
\tfrac{1}{2}\,\|C - C^{0}\|_{2}^{2}\;\leq\;\rho
$$

Bounds how far `C` is allowed to move away from the baseline. Without this, even on the simplex, the max of a linear function would land at a degenerate corner.

**Combined:**

$$
\boxed{\;\mathcal{C}_\rho \;=\; \Big\{\,C \in \mathbb{R}^{K} \;:\; \underbrace{C_k \geq 0,\;\;\textstyle\sum_{k=1}^{K} C_k = K}_{\text{simplex}},\;\;\underbrace{\tfrac{1}{2}\|C - C^{0}\|_{2}^{2} \leq \rho}_{\text{ball}}\,\Big\}\;}
$$

**Why both pieces are needed:**

| only this | what goes wrong |
|---|---|
| simplex alone | `max_C Σ_k C_k L_k` puts **all** mass on the class with the highest `L_k` (a corner of the simplex). Degenerate. |
| ball alone | no upper bound on `C_k`. Max is unbounded. |
| **both** | max is **bounded, interior, continuous in `(α, β)`**, with a closed-form solution via Cauchy–Schwarz. |

### 4.2 The baseline `C^0` (rescaled inverse frequency)

To make `C^0` live inside `𝒞_ρ`, rescale the inverse-frequency rule so it sums to `K`:

$$
C_k^{\,0} \;=\; \frac{K \cdot (1/n_k)}{\displaystyle\sum_{j=1}^{K}(1/n_j)}
$$

For binary `n_- = 12, n_+ = 4, K = 2`:

$$
C_-^{\,0} = \frac{2\cdot(1/12)}{(1/12)+(1/4)} = 0.5,\qquad C_+^{\,0} = \frac{2\cdot(1/4)}{(1/12)+(1/4)} = 1.5
$$

Same `n_-/n_+ = 3×` ratio for the minority, just rescaled to sum to `K = 2`. By construction, `C^0 ∈ 𝒞_ρ` for any `ρ ≥ 0` (it sits at the centroid of the ball).

### 4.3 What `ρ` means in plain terms

`ρ` is the **squared-distance budget** that `C` is allowed to spend moving away from `C^0`. It is a **trust-region radius**, not a regularizer.

| `ρ` value | behavior of `C` |
|---|---|
| `ρ → 0` | `C` is frozen at `C^0` — recovers Option A (classical fixed cost-sensitive) |
| `ρ` small | `C` wiggles a little around `C^0` — "almost-classical" |
| `ρ` large | `C` can roam most of the simplex — aggressive adaptation |
| `ρ → ∞` | constraint becomes vacuous — max collapses to a corner of the simplex (degenerate) |

`ρ` smoothly interpolates between *fixed `C^0`* and *unconstrained*. Tune by cross-validation.

### 4.4 `K = 2` visualization (binary case)

With `K = 2`, the simplex `C_- + C_+ = 2` is a line segment. Substitute `C_+ = 2 − C_-` into the ball constraint:

$$
\tfrac{1}{2}\!\left[(C_- - C_-^{0})^{2} + (C_+ - C_+^{0})^{2}\right] \;=\; (C_- - C_-^{0})^{2} \;\leq\; \rho
$$

(The two squared deviations are equal because `C_- − C_-^0 = −(C_+ − C_+^0)` from the simplex constraint.) So `𝒞_ρ` collapses to a one-dimensional interval:

$$
C_- \in \big[\,C_-^{0} - \sqrt{\rho},\;\;C_-^{0} + \sqrt{\rho}\,\big],\qquad C_+ = 2 - C_-.
$$

For `C^0 = (0.5, 1.5)`:

| `ρ` | `C_-` lives in | `C_+` lives in | meaning |
|-----|-----|-----|-----|
| `0.04` | `[0.30, 0.70]` | `[1.30, 1.70]` | tight wiggle: `C` stays within ±0.2 of baseline |
| `0.25` | `[0.00, 1.00]` | `[1.00, 2.00]` | loose: `C_-` can drop to 0 |
| `1.00` | `[0.00, 1.50]` | `[0.50, 2.00]` | very loose: ball edge hits the simplex corners |

You pick `ρ` to control how much room `C` gets around the inverse-frequency anchor.

### 4.5 Where this comes from — research

The chi-squared / L2 ball constraint on a per-class re-weighting vector is the **distributionally robust optimization (DRO)** framework:

- **Namkoong & Duchi (2017),** *Variance-based regularization with convex objectives,* NeurIPS — defines DRO with a chi-squared divergence ball, proves the closed-form max via KKT, and shows it is equivalent to empirical risk minimization plus a variance penalty. **The ball lives in the constraint set, not in the objective.**
- **Duchi & Namkoong (2021),** *Learning models with uniform performance via DRO,* Annals of Statistics — extends to general divergence balls (chi-squared, KL, Wasserstein) and proves uniform convergence under DRO.
- **Hu, Niu, Sato, Sugiyama (2018),** *Does distributionally robust supervised learning give robust classifiers?,* ICML — connects DRO **directly** to cost-sensitive learning. They show that DRO over a chi-squared ball around the empirical class distribution is a principled way to **learn** cost weights instead of fixing them by hand.

**The intuition from these papers, in one sentence:**

> Train `(α, β)` to minimize loss under the **worst-case re-weighting** `C` of the classes that lies within a ball of radius `√(2ρ)` around the baseline `C^0`.

That is exactly what the saddle-point `min_{α,β} max_{C ∈ 𝒞_ρ}` says.

---

## 5. Step 1 — fix `C`, solve `(α, β)`

Hold `C` fixed inside `f^C`. The cost-sensitive iSFS problem reduces to a `C`-weighted version of the paper's eq (14):

$$
\min_{\boldsymbol{\alpha},\,\boldsymbol{\beta}}\;\;\frac{1}{|\mathbf{pf}|}\sum_{m \in \mathbf{pf}} f^{C}(\mathbf{X}_m,\boldsymbol{\beta},\boldsymbol{\alpha}_m,\mathbf{y}_m) \;+\; \lambda\,\mathbf{R}_{\boldsymbol{\beta}}(\boldsymbol{\beta})
$$

For least-squares loss the inner expansion is:

$$
\mathbf{L}^{C}\!\left(\sum_{i}\alpha^{i}_{m}\mathbf{X}^{i}_{m}\boldsymbol{\beta}^{i},\;\mathbf{y}_m\right) \;=\; \frac{1}{2}\sum_{j=1}^{n_m} C_{y_{m,j}}\!\left(\hat{y}_{m,j} - y_{m,j}\right)^{2}
$$

**One auxiliary definition:**

$$
\hat{y}_{m,j}(\boldsymbol{\alpha}_m,\boldsymbol{\beta}) \;=\; \sum_{i=1}^{S}\alpha^{i}_{m}\,\mathbf{x}^{i}_{m,j}\!\cdot\boldsymbol{\beta}^{i}
$$

Variables:
- `j ∈ {1, …, n_m}` — sample index inside profile `m`.
- `y_{m,j}` — the class label of sample `j` in profile `m`.
- `x^i_{m,j}` — the `j`-th row of `X^i_m`, i.e., the source-`i` feature vector of sample `j` in profile `m`.
- `ŷ_{m,j}` — the model prediction for sample `j` in profile `m`.

`ŷ_{m,j}` is just the per-sample form of `Σ_i α^i_m X^i_m β^i` from eq (15). It is linear in `α_m` (with `β` fixed) and linear in `β` (with `α_m` fixed).

The inner block is itself a 2-block alternation, exactly Algorithm 2 of the paper, with one change: every squared residual carries `C_{y_{m,j}}`.

### 5.1 Inner step 1a — `α` update (per profile)

For each profile `m`, the cost-sensitive version of paper eq (16):

$$
\boxed{\;\min_{\boldsymbol{\alpha}_m}\;\;\frac{1}{2}\sum_{j=1}^{n_m} C_{y_{m,j}}\!\left(\sum_{i=1}^{S}\alpha^{i}_{m}\,\mathbf{x}^{i}_{m,j}\!\cdot\boldsymbol{\beta}^{i} \;-\; y_{m,j}\right)^{\!2}\quad\text{s.t.}\quad \mathbf{R}_{\boldsymbol{\alpha}}(\boldsymbol{\alpha}_m) \leq 1\;}
$$

**Sanity check.** When `C_k = 1` for all `k`, every `C_{y_{m,j}}` becomes `1`, the inner sum collapses to `½ ‖Σ_i α^i_m X^i_m β^i − y_m‖²`, and we recover paper eq (16) verbatim. ✓

**Gradient (used by FISTA / proximal gradient).** Take the partial with respect to `α^i_m`:

$$
\frac{\partial}{\partial \alpha^{i}_{m}}\!\left[\tfrac{1}{2}\sum_{j=1}^{n_m} C_{y_{m,j}}(\hat{y}_{m,j} - y_{m,j})^{2}\right] \;=\; \sum_{j=1}^{n_m} C_{y_{m,j}}\,(\hat{y}_{m,j} - y_{m,j})\,\big(\mathbf{x}^{i}_{m,j}\!\cdot\boldsymbol{\beta}^{i}\big)
$$

Read this in three pieces:
- `(ŷ_{m,j} − y_{m,j})` — residual for sample `j`.
- `C_{y_{m,j}}` — class-cost weight for sample `j` (this is the cost-sensitive entry point).
- `(x^i_{m,j} · β^i)` — partial of `ŷ_{m,j}` with respect to `α^i_m`.

When `C ≡ 1` the `C` factors disappear and this is the gradient of paper eq (16). ✓

The unit-ball constraint `R_α(α_m) ≤ 1` is handled by projection inside FISTA, exactly as in the paper's discussion of eq (16).

### 5.2 Inner step 1b — `β` update (global)

Stack profiles. The cost-sensitive version of paper's `g` (eq 17):

$$
g^{C}(\boldsymbol{\beta}) \;=\; \frac{1}{|\mathbf{pf}|}\sum_{m \in \mathbf{pf}}\frac{1}{2 n_m}\sum_{j=1}^{n_m} C_{y_{m,j}}\!\left(\hat{y}_{m,j} - y_{m,j}\right)^{2}
$$

The β subproblem:

$$
\boxed{\;\min_{\boldsymbol{\beta}}\;\;g^{C}(\boldsymbol{\beta}) \;+\; \lambda\,\mathbf{R}_{\boldsymbol{\beta}}(\boldsymbol{\beta})\;}
$$

**Gradient with respect to `β^i`** — cost-sensitive version of paper eq (18):

$$
\boxed{\;\nabla g^{C}(\boldsymbol{\beta}^{i}) \;=\; \frac{1}{|\mathbf{pf}|}\sum_{m \in \mathbf{pf}}\frac{1}{n_m}\,\mathbf{I}\!\left(m\,\&\,2^{S-i}\neq 0\right)\sum_{j=1}^{n_m} C_{y_{m,j}}\,(\hat{y}_{m,j} - y_{m,j})\,\alpha^{i}_{m}\,\mathbf{x}^{i}_{m,j}\;}
$$

Variables in this expression:
- `I(m & 2^{S−i} ≠ 0)` — bitwise indicator from eq (18) of the paper. Equals `1` iff source `i` is present in profile `m`. **Identical to the paper.**
- `α^i_m x^i_{m,j}` — the source-`i` contribution from sample `j` in profile `m`.
- `(ŷ_{m,j} − y_{m,j})` — residual.
- `C_{y_{m,j}}` — class-cost weight on the residual.

Compare to the paper's eq (18):

$$
\nabla g(\boldsymbol{\beta}^{i}) \;=\; \frac{1}{|\mathbf{pf}|}\sum_{m \in \mathbf{pf}}\frac{1}{n_m}\,\mathbf{I}\!\left(m\,\&\,2^{S-i}\neq 0\right)\,(\alpha^{i}_{m}\mathbf{X}^{i}_{m})^{T}\!\Big(\textstyle\sum_{j}\alpha^{j}_{m}\mathbf{X}^{j}_{m}\boldsymbol{\beta}^{j} - \mathbf{y}_m\Big)
$$

The only difference: each per-sample term carries an extra `C_{y_{m,j}}` factor. With `C ≡ 1` the two gradients are identical. ✓

The β subproblem is solved by accelerated gradient (FISTA), exactly as in Algorithm 2 of the paper. **No new machinery — only the gradient picks up the `C_{y_{m,j}}` weight.**

---

## 6. Step 2 — fix `(α, β)`, solve `C`

Hold `(α, β)` fixed. The `λ R_β(β)` term has no `C` in it; drop it. The data-fitting term, with `(α, β)` fixed:

$$
\frac{1}{|\mathbf{pf}|}\sum_{m \in \mathbf{pf}}\frac{1}{n_m}\,\mathbf{L}^{C}(\cdots) \;=\; \frac{1}{|\mathbf{pf}|}\sum_{m \in \mathbf{pf}}\frac{1}{2 n_m}\sum_{j=1}^{n_m} C_{y_{m,j}}\,(\hat{y}_{m,j} - y_{m,j})^{2}
$$

### 6.1 Reduce to a linear function of `C`

Group every per-sample term by the class label `y_{m,j}`. Define for each `k ∈ {1, …, K}`:

$$
\boxed{\;L_k(\boldsymbol{\alpha},\boldsymbol{\beta}) \;=\; \frac{1}{|\mathbf{pf}|}\sum_{m \in \mathbf{pf}}\frac{1}{2 n_m}\!\!\sum_{\substack{j=1,\ldots,n_m \\ y_{m,j} = k}}\!\!(\hat{y}_{m,j} - y_{m,j})^{2}\;}
$$

`L_k` is the **per-class loss coefficient**. With `(α, β)` fixed, it is just a positive number — the total residual-squared contribution from samples of class `k`, weighted by the same `(1/|pf|)(1/n_m)` factors that appear in the iSFS objective.

The data-fitting term is then **linear in `C`**:

$$
\frac{1}{|\mathbf{pf}|}\sum_{m \in \mathbf{pf}}\frac{1}{n_m}\,\mathbf{L}^{C}(\cdots) \;=\; \sum_{k=1}^{K} C_k\,L_k(\boldsymbol{\alpha},\boldsymbol{\beta})
$$

So the `C` subproblem is:

$$
\boxed{\;\max_{C \in \mathcal{C}_\rho}\;\;\sum_{k=1}^{K} C_k\,L_k\;}
$$

This is **a linear function maximized over a convex compact set** — closed form via KKT.

### 6.2 Closed-form derivation in 5 lines

**Setup.** Re-parameterize `C` by its deviation from the baseline:

$$
d \;=\; C - C^{0} \;\in\; \mathbb{R}^{K}
$$

The simplex constraint `Σ_k C_k = K` and the fact `Σ_k C^0_k = K` together force `Σ_k d_k = 0`. So `d` lives in the hyperplane `H = {d : Σ_k d_k = 0}`.

**Re-write the objective as a function of `d`.** Using `Σ_k d_k = 0`:

$$
\sum_{k} C_k L_k \;=\; \underbrace{\sum_{k} C_k^{0} L_k}_{\text{constant}} \;+\; \sum_{k} d_k L_k \;=\; \text{const} \;+\; \sum_{k} d_k\,(L_k - \overline{L})
$$

where `L̄ = (1/K) Σ_k L_k`. The shift by `L̄` is free because `Σ_k d_k = 0`.

**Define the centered loss vector:**

$$
\widetilde{L} \;=\; L - \overline{L}\,\mathbf{1} \;\in\; H
$$

so the C subproblem becomes:

$$
\max_{d}\;\;d^{T}\widetilde{L}\quad\text{s.t.}\quad d \in H,\;\;\tfrac{1}{2}\|d\|_{2}^{2} \leq \rho,\;\;d_k \geq -C^{0}_{k}.
$$

**Apply Cauchy–Schwarz** on the hyperplane `H` (both `d` and `L̃` live there). The maximizer of an inner product on a sphere is the unit vector aligned with the target:

$$
d^{*} \;=\; \sqrt{2\rho}\,\frac{\widetilde{L}}{\|\widetilde{L}\|_{2}}
$$

**Convert back to `C`** via `C = C^0 + d`:

$$
\boxed{\;C_k^{*} \;=\; C_k^{0} \;+\; \sqrt{2\rho}\;\frac{L_k - \overline{L}}{\|L - \overline{L}\,\mathbf{1}\|_{2}}\;}
$$

That is the entire `C` block update. No iteration, no learning rate, no inner loop. Three quantities to compute:
1. `L_k` for each class — one pass over the residuals.
2. `L̄ = (1/K) Σ_k L_k` — average.
3. `‖L − L̄·1‖₂` — norm of the centered vector.

Plug into the boxed formula. `O(K)` time, where `K = 2` for binary.

### 6.3 Three sanity checks

**Check 1 — budget preserved (`Σ C_k = K`).**

$$
\sum_{k} C_k^{*} \;=\; \sum_{k} C_k^{0} \;+\; \sqrt{2\rho}\,\frac{\sum_{k}(L_k - \overline{L})}{\|\widetilde{L}\|_{2}} \;=\; K \;+\; \sqrt{2\rho}\cdot\frac{0}{\|\widetilde{L}\|_{2}} \;=\; K \quad ✓
$$

(`Σ_k (L_k − L̄) = 0` by construction.)

**Check 2 — ball constraint active (`½‖C − C^0‖² = ρ`).**

$$
\tfrac{1}{2}\|C^{*} - C^{0}\|^{2} \;=\; \tfrac{1}{2}\,2\rho\,\frac{\|\widetilde{L}\|^{2}}{\|\widetilde{L}\|^{2}} \;=\; \rho \quad ✓
$$

The maximizer always sits **on the boundary** of the ball — the trust-region budget is fully spent.

**Check 3 — direction is cost-sensitive (`C_k > C_k^0` iff `L_k > L̄`).**

$$
C_k^{*} - C_k^{0} \;=\; \sqrt{2\rho}\,\frac{L_k - \overline{L}}{\|\widetilde{L}\|_{2}}
$$

The sign of `C_k^* − C_k^0` is the sign of `L_k − L̄`:
- Class with above-average loss → upweighted.
- Class with below-average loss → downweighted.

✓ This is the cost-sensitive direction.

### 6.4 `ρ → 0` recovers Option A

As `ρ → 0`:

$$
C_k^{*} \;=\; C_k^{0} \;+\; \underbrace{\sqrt{2\rho}}_{\to 0}\cdot(\ldots) \;\longrightarrow\; C_k^{0}
$$

The C update freezes at the inverse-frequency baseline. **Option A is the `ρ → 0` limit of Option B.** The DRO formulation interpolates smoothly between *fixed cost-sensitive* and *unconstrained adaptive*.

### 6.5 Non-negativity projection

The closed form above assumes the unconstrained-by-non-negativity solution stays non-negative. When `ρ` is small or `L̃` is mild this is automatic. If `ρ` is large enough that some `C_k^*` would go negative:

1. Identify the active set `A = {k : C_k^* < 0}`.
2. Peg those entries to `C_k = 0`.
3. Re-distribute the resulting deficit `K − Σ_{k ∉ A} C_k^*` proportionally onto the still-positive entries.

For binary `K = 2` this terminates in one extra step. In practice, `ρ` is chosen small enough that this projection never triggers.

---

## 7. Algorithm 3 — full pseudocode and numerical walkthrough

### 7.1 Pseudocode (matches the style of Algorithm 2 in the paper)

```
─────────────────────────────────────────────────────────────────
ALGORITHM 3:  Cost-Sensitive iSFS via Block Coordinate Descent
─────────────────────────────────────────────────────────────────
Input:   data {X_m, y_m for m ∈ pf}, baseline C^0, ball radius ρ,
         β-regularization parameter λ, α-regularization R_α
Output:  α (per-profile weights), β (shared coefficients), C (class costs)

1.  Initialize β^(0) by fitting each source individually on its
    available data (paper's Algorithm 2 init).
2.  Initialize C^(0) ← C^0 (rescaled inverse-frequency baseline).

3.  for outer iteration t = 1, 2, … do

      ── Block A — fix C^(t-1), solve (α, β) ──
4.    Run paper's Algorithm 2 with L^C in place of L:
        Repeat until inner objective stops decreasing:
          for each profile m:    α_m  ← solve eq (16) with C_{y_{m,j}} weights
          β                      ← FISTA on g^C(β) + λ R_β(β),
                                   gradient ∇g^C(β^i) per §5.2

      ── Block B — fix (α, β), solve C ──
5.    For each class k = 1, …, K compute the per-class loss coefficient
            L_k = (1/|pf|) Σ_m (1/(2 n_m)) Σ_{j: y_{m,j}=k} (ŷ_{m,j} − y_{m,j})²
6.    L̄  ← (1/K) Σ_k L_k
7.    L̃  ← L − L̄·1
8.    C^(t) ← C^0 + √(2ρ) · L̃ / ‖L̃‖₂                  (Cauchy–Schwarz closed form)
9.    If any C_k^(t) < 0, run the active-set projection from §6.5.

10.   if  |C^(t) − C^(t-1)|_∞ < ε  AND  |β^(t) − β^(t-1)|_2 < ε  then
11.       return α^(t), β^(t), C^(t)
─────────────────────────────────────────────────────────────────
```

Two outer blocks; three quantities to compute per outer iteration; no inner gradient loop on the C step.

### 7.2 Numerical walkthrough — synthetic 3-profile dataset

**Setup.** Two sources of two features each (`β ∈ ℝ^4`). Three profiles. Class-imbalanced binary labels in `{−1, +1}`. Random seed `np.random.seed(11)`, `np.random.default_rng(11)`. The dataset:

| profile | mask | `n_m` | labels `y_m` |
|---|---|---|---|
| 1 | source 1 only | 5 | `(−1, −1, −1, −1, +1)` |
| 2 | source 2 only | 5 | `(+1, −1, −1, −1, −1)` |
| 3 | both sources | 6 | `(−1, −1, −1, −1, +1, +1)` |

Class counts: `n_- = 12, n_+ = 4, N = 16, K = 2`. Inverse-frequency baseline:

$$
C^{0} \;=\; (0.5,\;1.5)
$$

Choose `ρ = 0.04` (so `√(2ρ) = 0.2828` — modest trust region; `C` can move at most `±0.2` from baseline in each component).

**Initialize `β^(0)`** by per-source ridge regression on each source's available data:

$$
\boldsymbol{\beta}^{(0)} \;\approx\; (-0.6234,\;-0.5804,\;\;0.4264,\;\;0.7807)
$$

**Outer iteration 1.**

*Block A* — run inner BCD with `C = (0.5, 1.5)`. After 15 inner sweeps:

$$
\boldsymbol{\beta}^{(1)} \;\approx\; (-0.299,\;-0.507,\;\;0.423,\;\;0.722)
$$

*Block B* — compute per-class loss and apply the closed form:

$$
L \;\approx\; (0.2123,\;0.0091),\qquad \overline{L} \;\approx\; 0.1107,\qquad \widetilde{L} \;\approx\; (0.1016,\;-0.1016),\qquad \|\widetilde{L}\|_{2} \;\approx\; 0.1437
$$

$$
C^{(1)} \;=\; (0.5,\;1.5) \;+\; 0.2828 \cdot \frac{(0.1016,\;-0.1016)}{0.1437} \;\approx\; (0.5 + 0.2,\;\;1.5 - 0.2) \;=\; (0.7,\;\;1.3)
$$

Sanity check: `Σ C_k = 2.0 = K` ✓; `½‖C − C^0‖² = 0.04 = ρ` ✓.

**Outer iteration 2.**

*Block A* — re-run inner BCD with `C = (0.7, 1.3)`. After 15 inner sweeps:

$$
\boldsymbol{\beta}^{(2)} \;\approx\; (-0.426,\;-0.508,\;\;0.410,\;\;0.711)
$$

*Block B* — recompute the per-class losses:

$$
L \;\approx\; (0.1931,\;0.0142),\qquad \overline{L} \;\approx\; 0.1037,\qquad \widetilde{L} \;\approx\; (0.0895,\;-0.0895),\qquad \|\widetilde{L}\|_{2} \;\approx\; 0.1265
$$

$$
C^{(2)} \;=\; (0.5,\;1.5) \;+\; 0.2828 \cdot \frac{(0.0895,\;-0.0895)}{0.1265} \;=\; (0.5 + 0.2,\;\;1.5 - 0.2) \;=\; (0.7,\;\;1.3)
$$

`C` is unchanged from iteration 1. (For `K = 2`, the centered vector `L̃` always has the form `(a, −a)`, so its **direction** is `(1, −1)/√2` whenever `L_- > L_+`. The C-step closed form depends only on this direction, so once the model is on the "majority is harder" side, `C` is determined.)

**Trajectory table:**

| outer iter `t` | `L` | `‖L̃‖₂` | `C` | `β` | `Φ` |
|---|---|---|---|---|---|
| 0 (init) | — | — | `(0.500, 1.500)` | `(−0.623, −0.580, 0.426, 0.781)` | — |
| 1 | `(0.2123, 0.0091)` | `0.1437` | `(0.700, 1.300)` | `(−0.299, −0.507, 0.423, 0.722)` | `0.18662` |
| 2 | `(0.1931, 0.0142)` | `0.1265` | `(0.700, 1.300)` | `(−0.426, −0.508, 0.410, 0.711)` | `0.18155` |
| 3 | `(0.1931, 0.0142)` | `0.1265` | `(0.700, 1.300)` | `(−0.426, −0.508, 0.410, 0.711)` | `0.18155` |

Converges in **2 outer iterations** to:

$$
\boldsymbol{\beta}^{*} \approx (-0.426,\;-0.508,\;\;0.410,\;\;0.711),\qquad C^{*} = (0.700,\;\;1.300),\qquad \Phi^{*} \approx 0.18155
$$

The minority-class weight settles at `1.3` (down from baseline `1.5`) because in this dataset the **majority class has the larger total loss contribution** `L_- > L_+` (more samples). The DRO step adapts the cost vector along the data-driven direction, while the inverse-frequency baseline `C^0 = (0.5, 1.5)` keeps the minority weight above the majority weight overall.

### 7.3 Reproducible Python script

Save as `cost_sens_isfs.py` and run with `python3 cost_sens_isfs.py`.

```python
import numpy as np

# ============== synthetic data ==============
np.random.seed(11)
beta_true = np.array([1.5, -0.8, 0.6, 1.2])
profiles = [
    {'mask': [True, False], 'n': 5},
    {'mask': [False, True], 'n': 5},
    {'mask': [True, True],  'n': 6},
]

def make_profile(n, src1, src2, rng):
    X1 = rng.normal(0, 1, size=(n, 2)) if src1 else None
    X2 = rng.normal(0, 1, size=(n, 2)) if src2 else None
    z = np.zeros(n)
    if src1: z += X1 @ beta_true[:2]
    if src2: z += X2 @ beta_true[2:]
    z += rng.normal(0, 0.5, size=n)
    y = np.where(z > np.quantile(z, 0.75), 1.0, -1.0)
    return X1, X2, y

rng = np.random.default_rng(11)
data = []
for p in profiles:
    X1, X2, y = make_profile(p['n'], p['mask'][0], p['mask'][1], rng)
    data.append({'X1': X1, 'X2': X2, 'y': y, 'mask': p['mask']})

all_y = np.concatenate([d['y'] for d in data])
n_per_class = np.array([(all_y == -1).sum(), (all_y == 1).sum()])
N, K = int(n_per_class.sum()), 2
inv_freq = 1.0 / n_per_class
C0 = inv_freq / inv_freq.sum() * K          # = (0.5, 1.5)

def class_idx(y_arr): return ((y_arr + 1) // 2).astype(int)

def predict(d, alpha_m, beta):
    yhat = np.zeros(len(d['y']))
    if d['mask'][0]: yhat += alpha_m[0] * (d['X1'] @ beta[:2])
    if d['mask'][1]: yhat += alpha_m[1] * (d['X2'] @ beta[2:])
    return yhat

def init_beta():
    beta = np.zeros(4)
    for src, slc in [(0, slice(0, 2)), (1, slice(2, 4))]:
        Xs, ys = [], []
        for d in data:
            if d['mask'][src]:
                Xs.append(d['X1'] if src == 0 else d['X2'])
                ys.append(d['y'])
        Xa = np.vstack(Xs); ya = np.concatenate(ys)
        beta[slc] = np.linalg.solve(Xa.T @ Xa + 0.1*np.eye(2), Xa.T @ ya)
    return beta

# ============== Step 1: inner BCD on (α, β) given C ==============
def inner_bcd(beta, C, rho_beta=0.05, n_inner=15):
    alpha_list = [np.array([0.7, 0.7]) for _ in data]
    n_pf, n_beta = len(data), beta.shape[0]
    for _ in range(n_inner):
        # α update per profile (weighted normal equations + projection to unit ball)
        for m, d in enumerate(data):
            cols, idxs = [], []
            if d['mask'][0]:
                cols.append(d['X1'] @ beta[:2]); idxs.append(0)
            if d['mask'][1]:
                cols.append(d['X2'] @ beta[2:]); idxs.append(1)
            Phi = np.vstack(cols).T
            w = C[class_idx(d['y'])]
            A = Phi.T @ np.diag(w) @ Phi + 1e-6 * np.eye(Phi.shape[1])
            b = Phi.T @ (w * d['y'])
            a = np.linalg.solve(A, b)
            nrm = np.linalg.norm(a)
            if nrm > 1: a /= nrm
            new_alpha = np.zeros(2)
            for kl, kg in enumerate(idxs): new_alpha[kg] = a[kl]
            alpha_list[m] = new_alpha
        # β update (weighted normal equations across profiles)
        H = rho_beta * np.eye(n_beta); bvec = np.zeros(n_beta)
        for m, d in enumerate(data):
            n_m = len(d['y'])
            w = C[class_idx(d['y'])]
            AX = np.zeros((n_m, n_beta))
            if d['mask'][0]: AX[:, :2] = alpha_list[m][0] * d['X1']
            if d['mask'][1]: AX[:, 2:] = alpha_list[m][1] * d['X2']
            H += (1/(n_pf*n_m)) * AX.T @ np.diag(w) @ AX
            bvec += (1/(n_pf*n_m)) * AX.T @ (w * d['y'])
        beta = np.linalg.solve(H + 1e-9*np.eye(n_beta), bvec)
    return alpha_list, beta

# ============== Step 2: closed-form C update ==============
def update_C(L, C0, rho):
    L_bar = L.mean()
    L_tilde = L - L_bar
    norm = np.linalg.norm(L_tilde)
    if norm < 1e-12:
        return C0.copy()
    C = C0 + np.sqrt(2 * rho) * L_tilde / norm
    if (C < 0).any():                               # active-set projection
        C = np.maximum(C, 0.0)
        deficit = K - C.sum()
        positive = C > 0
        C[positive] += deficit / positive.sum()
    return C

def per_class_L(alpha_list, beta):
    L = np.zeros(2); n_pf = len(data)
    for m, d in enumerate(data):
        n_m = len(d['y'])
        sq = 0.5 * (predict(d, alpha_list[m], beta) - d['y']) ** 2
        cls = class_idx(d['y'])
        for k in (0, 1):
            L[k] += (1/(n_pf * n_m)) * sq[cls == k].sum()
    return L

def saddle_phi(alpha_list, beta, C, rho_beta=0.05):
    n_pf, val = len(data), 0.0
    for m, d in enumerate(data):
        n_m = len(d['y'])
        w = C[class_idx(d['y'])]
        sq = 0.5 * (predict(d, alpha_list[m], beta) - d['y'])**2
        val += (1/(n_pf*n_m)) * (w * sq).sum()
    val += 0.5 * rho_beta * np.sum(beta**2)
    return val

# ============== outer BCD loop ==============
beta = init_beta()
C = C0.copy()
rho = 0.04
print(f"init: β = {beta.round(3)}, C = {C}")
for t in range(5):
    alpha_list, beta = inner_bcd(beta, C)
    L = per_class_L(alpha_list, beta)
    C = update_C(L, C0, rho)
    phi = saddle_phi(alpha_list, beta, C)
    print(f"  iter {t+1}: L = {L.round(4)}, C = {C.round(4)}, "
          f"β = {beta.round(3)}, Φ = {phi:.5f}")
```

Run that and you should see:

```
init: β = [-0.623 -0.58   0.426  0.781], C = [0.5 1.5]
  iter 1: L = [0.2123 0.0091], C = [0.7 1.3], β = [-0.299 -0.507  0.423  0.722], Φ = 0.18662
  iter 2: L = [0.1931 0.0142], C = [0.7 1.3], β = [-0.426 -0.508  0.41   0.711], Φ = 0.18155
  iter 3: L = [0.1931 0.0142], C = [0.7 1.3], β = [-0.426 -0.508  0.41   0.711], Φ = 0.18155
  iter 4: L = [0.1931 0.0142], C = [0.7 1.3], β = [-0.426 -0.508  0.41   0.711], Φ = 0.18155
  iter 5: L = [0.1931 0.0142], C = [0.7 1.3], β = [-0.426 -0.508  0.41   0.711], Φ = 0.18155
```

That is the cost-sensitive iSFS BCD in ~80 lines of code.

---

## 8. Connection to BCD theory

### 8.1 Multi-convex structure

Cost-sensitive iSFS sits inside the **multi-convex** family from `BCD_explained.md` §3 / Aggarwal §4.10.2:

| block fixed | block updated | sub-problem | reason it is convex (or concave) |
|---|---|---|---|
| `β, C` | `α_m` per profile | per-profile weighted least-squares | positive-semidefinite quadratic in `α_m` |
| `α, C` | `β` | weighted regularized least-squares | positive-definite quadratic in `β` |
| `α, β` | `C ∈ 𝒞_ρ` | linear-in-`C` over a convex compact set | linear function on a convex polytope |

Each block sub-problem has a **closed-form** solve. No learning rate, no line search inside a block.

### 8.2 What converges and what doesn't

Two facts coexist:

1. **Inner BCD on `(α, β)` with `C` fixed converges monotonically.** This is the standard multi-convex BCD theorem from Aggarwal §4.10.2: each step decreases the inner objective, so the inner loop is monotone.

2. **The outer trajectory is *not* a strict monotonic decrease in `Φ`** — the C-block is a `max`, so it can increase `Φ` between iterations even though the (α, β) block decreases it. The right invariant for the outer loop is the **saddle-point gap**:

$$
\text{gap}(\alpha, \beta, C) \;=\; \max_{C' \in \mathcal{C}_\rho}\Phi(\alpha,\beta,C') \;-\; \min_{\alpha',\beta'}\Phi(\alpha',\beta',C)
$$

Under standard conditions (strong convexity in `(α, β)` and the bounded-set constraint on `C`), saddle-point alternation reduces the gap to zero. In our numerical experiment the gap effectively closes in two outer iterations because for `K = 2` the C direction is determined by the sign of `L_- − L_+`, which is fixed once the model is in the "majority is harder" or "minority is harder" regime.

### 8.3 Connection to the textbook BCD case studies

The cost-sensitive iSFS BCD is structurally identical to the case studies in `BCD_explained.md`:

- **K-means as BCD** (Aggarwal §4.10.3): two blocks, centroids and assignments, each with a closed-form optimum (mean and nearest-prototype). Cost-sensitive iSFS does the same with `(α, β)` and `C`, using weighted normal equations and a Cauchy-Schwarz projection.
- **ALS as BCD** (Aggarwal §8.3.2.3): two blocks `(U, V)`, each row solved as an independent ridge regression. Cost-sensitive iSFS's `β` update is structurally the same — one global weighted ridge solve.
- **`opt.impute`** (Bertsimas et al. JMLR 2018): two blocks, auxiliary state `U` and imputations `(W, V)`, alternated. Cost-sensitive iSFS adds a third "auxiliary state" (`C`) and treats it the same way.

The trade-off Aggarwal flags (§4.10.2) applies directly:

> "Each step in block coordinate descent is more expensive, but fewer steps are required."

For our cost-sensitive iSFS at moderate `n` and small `K`, the closed-form block solves are by far the cheapest option.

### 8.4 Practical recommendations

- **Set `C^0` to the rescaled inverse-frequency baseline** — `C^0_k = K(1/n_k) / Σ_j(1/n_j)`. This is the cost-sensitive prior.
- **Pick `ρ` by cross-validation** — sweep e.g. `ρ ∈ {0.01, 0.04, 0.1, 0.25, 0.5, 1.0}` and choose the one with best validation performance. `ρ → 0` recovers Option A.
- **Pick `λ` (the `β` ridge / sparsity reg) by cross-validation** — same as in the original iSFS Algorithm 2.
- **Run inner BCD to a loose tolerance.** Aggarwal's monotonic-decrease guarantee makes the inner loop forgiving — 10–20 inner sweeps per outer iteration is enough.
- **Stop on `(C, β)` infinity-norm change**, not on `Φ` — the saddle-point trajectory of `Φ` may not be strictly monotonic, but the iterates themselves stabilize quickly.

### 8.5 What is non-trivial here

The non-trivial pieces of this extension are:

1. **Recognizing that pure-min in `C` has the wrong sign.** This is the subtlety that motivates the saddle-point framing.
2. **Recognizing that a regularizer `R_C(C)` is conceptually wrong.** `C` is not a model parameter; it lives in a constraint set, not in the objective.
3. **Picking a constraint set `𝒞_ρ` that admits a closed-form max** of a linear function. The chi-squared / L2 ball intersected with the simplex gives the cleanest Cauchy–Schwarz solution.
4. **Anchoring `C^0` to inverse frequency** so the saddle-point smoothly contains classical fixed-cost cost-sensitive learning as the `ρ → 0` limit.
5. **Recognizing that the `(α, β)` block is itself an inner BCD.** The two-level nesting lets us inherit the original Algorithm 2 wholesale, with only the `C_{y_{m,j}}` weight inserted into the gradient.

---

## 9. Quick reference card

| symbol | meaning | where it lives |
|---|---|---|
| `α^i_m ∈ ℝ` | source-`i` weight for profile `m` | inner block, `R_α(α_m) ≤ 1` |
| `β^i ∈ ℝ^{p_i}` | shared feature coefficients for source `i` | inner block, `R_β` regularized |
| `C ∈ ℝ^K` | per-class cost vector | outer block, `C ∈ 𝒞_ρ` |
| `C^0 ∈ ℝ^K` | rescaled inverse-frequency baseline | user-chosen prior |
| `ρ` | trust-region radius around `C^0` | hyperparameter (cross-validated) |
| `λ` | strength of `R_β` | hyperparameter (cross-validated) |
| `L_k` | per-class loss coefficient `(1/|pf|) Σ_m (1/(2n_m)) Σ_{j: y=k} (ŷ−y)²` | recomputed every C step |
| `L̄ = (1/K) Σ_k L_k` | average per-class loss | recomputed every C step |
| `L̃ = L − L̄ · 1` | centered loss vector (lives in the simplex tangent) | recomputed every C step |
| `ŷ_{m,j}` | per-sample model prediction `Σ_i α^i_m x^i_{m,j}·β^i` | computed inside both blocks |
| `C_{y_{m,j}}` | class-cost lookup for sample `j` in profile `m` | scalar from the `K`-vector `C` |

| update | formula | meaning |
|---|---|---|
| `α_m` | argmin of `½ Σ_j C_{y_{m,j}}(ŷ−y)²` s.t. `R_α ≤ 1` | weighted per-profile constrained LS |
| `β` | argmin of `g^C(β) + λ R_β(β)` via FISTA, gradient `∇g^C(β^i)` | weighted regularized LS |
| `C` | `C_k* = C_k^0 + √(2ρ)·(L_k − L̄)/‖L̃‖₂` | Cauchy–Schwarz projection onto `𝒞_ρ` |

| theoretical fact | source |
|---|---|
| BCD with closed-form block updates monotonically decreases the inner objective on multi-convex problems | Aggarwal §4.10.2; `BCD_explained.md` §3 |
| The chi-squared ball DRO with linear inner objective has a closed-form Cauchy–Schwarz max | Namkoong & Duchi (2017) |
| DRO with chi-squared ball is principled cost-sensitive learning | Hu, Niu, Sato, Sugiyama (2018) |
| The original iSFS Algorithm 2 (2-block BCD on `α, β`) is the inner loop here | Xiang et al. (2013), §3.2 |
| `K`-means and ALS are the canonical 2-block BCD examples this method generalizes | Aggarwal §4.10.3 and §8.3.2.3; `BCD_explained.md` §4–5 |

---

*Cost-sensitive iSFS via DRO is a strict generalization of the original iSFS
model: setting `ρ → 0` pins `C` to the inverse-frequency baseline and recovers
classical fixed-cost cost-sensitive learning, while finite `ρ` lets the
optimization adapt `C` from the data via the closed-form Cauchy–Schwarz
projection step that fits inside the same BCD machinery the rest of the
model already uses.*
