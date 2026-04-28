# Trust-Region Correction for Cost-Sensitive iSFS

This note rewrites the cost-sensitive part of the current iSFS draft so it matches a pure trust-region view.

The main correction is simple:

- `C` should be treated as a cost vector, not as a simplex-constrained probability vector.
- The `C`-block should be written as a trust-region step around the current cost iterate.
- The simplex-plus-DRO derivation can be removed if the goal is a trust-region cost update only.

## 1. iSFS equations we inherit from Xiang et al. (2013)

For block-wise missing data, the original iSFS model is

$$
\min_{\alpha,\beta}\;\frac{1}{|\mathrm{pf}|}\sum_{m\in\mathrm{pf}} f(X_m,\beta,\alpha_m,y_m) + \lambda R_\beta(\beta)
$$

subject to

$$
R_\alpha(\alpha_m) \le 1,\qquad \forall m \in \mathrm{pf}.
$$

With least-squares loss,

$$
f(X_m,\beta,\alpha_m,y_m)
=
\frac{1}{2n_m}\left\|\sum_{i=1}^S \alpha_m^i X_m^i \beta^i - y_m\right\|_2^2.
$$

The original alternating scheme is:

$$
\alpha_m^{(t+1)}
=
\arg\min_{\alpha_m}\;
\frac{1}{2n_m}\left\|\sum_{i=1}^S \alpha_m^i X_m^i \beta^{i,(t)} - y_m\right\|_2^2
\quad
\text{s.t. } R_\alpha(\alpha_m)\le 1,
$$

and

$$
\beta^{(t+1)}
=
\arg\min_{\beta}\;g(\beta)+\lambda R_\beta(\beta),
$$

where

$$
g(\beta)
=
\frac{1}{|\mathrm{pf}|}\sum_{m\in\mathrm{pf}}
\frac{1}{2n_m}
\left\|
\sum_{i=1}^S \alpha_m^i X_m^i \beta^i - y_m
\right\|_2^2.
$$

## 2. Trust-region equations from the trust-region survey

The survey in `s10107-015-0893-2.pdf` gives the general trust-region pattern

$$
\min_d\; m_k(d)
\qquad
\text{s.t.}\quad \|d\|_{W_k} \le \Delta_k.
$$

The standard quadratic trust-region subproblem is

$$
\min_d\; m_k(d)
=
f(x_k) + g_k^\top d + \frac{1}{2}d^\top B_k d
\qquad
\text{s.t.}\quad \|d\|_2 \le \Delta_k.
$$

The predicted and actual reductions are

$$
\mathrm{Pred}_k = m_k(0) - m_k(s_k),
\qquad
\mathrm{Ared}_k = f(x_k) - f(x_k+s_k),
$$

and the standard acceptance ratio is

$$
r_k = \frac{\mathrm{Ared}_k}{\mathrm{Pred}_k}.
$$

The survey also gives the nonlinear least-squares trust-region subproblem

$$
\min_d\; \|F_k + J_k d\|_2^2
\qquad
\text{s.t.}\quad \|d\|_2 \le \Delta_k,
$$

which is the most relevant family here because the iSFS fitting term is also least-squares.

Other trust-region variants in the survey that are relevant but optional for this project are:

- Standard quadratic trust region:
  $$
  \min_d\; g_k^\top d + \frac{1}{2}d^\top B_k d
  \quad
  \text{s.t.}\quad \|d\|_2 \le \Delta_k.
  $$
- Subspace trust region:
  $$
  \min_d\; m_k(d)
  \quad
  \text{s.t.}\quad \|d\|_{W_k}\le \Delta_k,\;\; d\in S_k.
  $$
- Cubic regularization / ARC:
  $$
  m_k(d)
  =
  f(x_k)+g_k^\top d+\frac{1}{2}d^\top B_k d+\frac{\sigma_k}{3}\|d\|_2^3.
  $$

For the current paper, the cleanest choice is still the basic Euclidean trust-region model.

## 3. Why the simplex should be removed

If the research direction is "trust-region cost-sensitive iSFS" rather than "DRO reweighting over a simplex", then the following should be removed from the main draft:

- the simplex constraint `\sum_k C_k = K`,
- the saddle-point `\max_{C\in\mathcal C_\rho}` formulation,
- the Cauchy-Schwarz derivation that depends on the simplex hyperplane,
- the interpretation of `C` as a reweighting distribution.

Why:

- A class-cost vector is not naturally a probability vector.
- Trust-region methods need a center and a radius; they do not need a simplex.
- The trust-region variable is the local step `d_C`, not a global one-shot adversarial optimizer.

The ball is enough:

$$
\|d_C\|_2 \le \Delta_t.
$$

If you want a hard anchor to the initial baseline `C^0`, you can still add a ball safeguard later, but it is not a simplex:

$$
\|C^{(t+1)} - C^0\|_2 \le R_{\max}.
$$

## 4. Corrected cost-sensitive iSFS objective

Define the weighted residual for sample `j` in profile `m`:

$$
r_{m,j}(\alpha_m,\beta)
=
\sum_{i=1}^S \alpha_m^i \, x_{m,j}^{i\top}\beta^i - y_{m,j}.
$$

Then the cost-sensitive iSFS objective is

$$
\Phi(\alpha,\beta,C)
=
\frac{1}{|\mathrm{pf}|}\sum_{m\in\mathrm{pf}}
\frac{1}{2n_m}\sum_{j=1}^{n_m}
C_{y_{m,j}}\,
r_{m,j}(\alpha_m,\beta)^2
\;+\;
\lambda R_\beta(\beta),
$$

subject to

$$
R_\alpha(\alpha_m)\le 1,\qquad \forall m\in\mathrm{pf},
\qquad
C \ge 0.
$$

This is the point where cost-sensitivity enters iSFS. No simplex is needed.

## 5. Block A stays the same: fix `C`, solve `(\alpha,\beta)`

With `C` fixed, the profile-wise `\alpha` update becomes

$$
\alpha_m^{(t+1)}
=
\arg\min_{\alpha_m}\;
\frac{1}{2n_m}
\sum_{j=1}^{n_m}
C_{y_{m,j}}^{(t)}
\left(
\sum_{i=1}^S \alpha_m^i x_{m,j}^{i\top}\beta^{i,(t)}
- y_{m,j}
\right)^2
$$

subject to `R_\alpha(\alpha_m)\le 1`.

The `\beta` update becomes

$$
\beta^{(t+1)}
=
\arg\min_{\beta}\; g^{C^{(t)}}(\beta)+\lambda R_\beta(\beta),
$$

where

$$
g^{C^{(t)}}(\beta)
=
\frac{1}{|\mathrm{pf}|}\sum_{m\in\mathrm{pf}}
\frac{1}{2n_m}
\sum_{j=1}^{n_m}
C_{y_{m,j}}^{(t)}
\left(
\sum_{i=1}^S \alpha_m^{i,(t+1)} x_{m,j}^{i\top}\beta^i
- y_{m,j}
\right)^2.
$$

So the original iSFS optimization machinery is preserved. Only the residuals are now class-weighted by the current cost vector.

## 6. Trust-region `C`-step

### 6.1 Per-class loss coefficients

After updating `(\alpha,\beta)`, compute the per-class loss coefficients

$$
L_k^{(t+1)}
=
\frac{1}{|\mathrm{pf}|}\sum_{m\in\mathrm{pf}}
\frac{1}{2n_m}
\sum_{\{j:\,y_{m,j}=k\}}
r_{m,j}\!\left(\alpha_m^{(t+1)},\beta^{(t+1)}\right)^2,
\qquad
k=1,\dots,K.
$$

These tell us which classes are currently harder for the model.

### 6.2 Centered trust-region direction

Using `L_k` directly creates an arbitrary common upward drift in every component of `C`. That changes overall scale, but not the relative cost sensitivity between classes.

To make the trust-region step act only on the differential class difficulty, define

$$
\bar L^{(t+1)} = \frac{1}{K}\sum_{k=1}^K L_k^{(t+1)},
\qquad
\tilde L_k^{(t+1)} = L_k^{(t+1)} - \bar L^{(t+1)}.
$$

Thus the trust-region direction is

$$
g_C^{(t+1)} = \tilde L^{(t+1)}.
$$

If class `k` has above-average loss, then `\tilde L_k^{(t+1)} > 0` and the trust-region update pushes `C_k` upward.

### 6.3 Pure trust-region subproblem for `C`

Write the local cost step as

$$
C^{(t+1)} = C^{(t)} + d_C^{(t)}.
$$

The pure trust-region `C`-subproblem is

$$
\min_{d_C}\; m_t(d_C)
=
-\big(g_C^{(t+1)}\big)^\top d_C
$$

subject to

$$
\|d_C\|_2 \le \Delta_t,
\qquad
C^{(t)} + d_C \ge 0.
$$

This is the trust-region version of "move the cost vector toward the hardest classes, but only inside a ball."

If no nonnegativity bound is active, the closed-form step is

$$
d_C^{(t)}
=
\Delta_t
\frac{g_C^{(t+1)}}{\|g_C^{(t+1)}\|_2},
\qquad
g_C^{(t+1)}\neq 0.
$$

Hence

$$
C^{(t+1)}
=
\Pi_{\mathbb{R}_+^K}
\left(
C^{(t)}
\;+\;
\Delta_t\frac{g_C^{(t+1)}}{\|g_C^{(t+1)}\|_2}
\right),
\qquad
C^{(0)} = C^0.
$$

Here `\Pi_{\mathbb R_+^K}` is projection onto the nonnegative orthant.

### 6.4 Optional quadratic trust-region model

If you want a more classical second-order trust-region subproblem, replace the linear model by

$$
m_t^{\mathrm{quad}}(d_C)
=
-\big(g_C^{(t+1)}\big)^\top d_C
\;+\;
\frac{1}{2}d_C^\top B_t d_C
$$

subject to

$$
\|d_C\|_2 \le \Delta_t,
\qquad
C^{(t)} + d_C \ge 0.
$$

The simplest choice is `B_t = \mu_t I`, which gives

$$
m_t^{\mathrm{quad}}(d_C)
=
-\big(g_C^{(t+1)}\big)^\top d_C
\;+\;
\frac{\mu_t}{2}\|d_C\|_2^2.
$$

Ignoring the nonnegativity bound, the step is

$$
d_C^{(t)}
=
\min\!\left\{\frac{1}{\mu_t},\;\frac{\Delta_t}{\|g_C^{(t+1)}\|_2}\right\}
g_C^{(t+1)}.
$$

This is the version to use if you later want a more explicit Hessian-style trust-region derivation in the paper.

## 7. Full alternating scheme

The corrected trust-region cost-sensitive iSFS algorithm is:

1. Initialize `\beta^{(0)}` as in iSFS Algorithm 2.
2. Initialize `C^{(0)} = C^0`.
3. Choose a trust-region radius `\Delta_0 > 0`.
4. For `t = 0,1,2,\dots`:

$$
\alpha_m^{(t+1)}
\leftarrow
\arg\min_{\alpha_m}
\frac{1}{2n_m}
\sum_{j=1}^{n_m}
C_{y_{m,j}}^{(t)} r_{m,j}(\alpha_m,\beta^{(t)})^2
\quad
\text{s.t. } R_\alpha(\alpha_m)\le 1,
$$

$$
\beta^{(t+1)}
\leftarrow
\arg\min_{\beta}\;
g^{C^{(t)}}(\beta)+\lambda R_\beta(\beta),
$$

$$
L_k^{(t+1)}
\leftarrow
\frac{1}{|\mathrm{pf}|}\sum_{m\in\mathrm{pf}}
\frac{1}{2n_m}\sum_{\{j:\,y_{m,j}=k\}}
r_{m,j}\!\left(\alpha_m^{(t+1)},\beta^{(t+1)}\right)^2,
$$

$$
g_C^{(t+1)}
\leftarrow
L^{(t+1)} - \bar L^{(t+1)}\mathbf 1,
$$

$$
d_C^{(t)}
\leftarrow
\arg\min_{\|d_C\|_2\le \Delta_t,\;C^{(t)}+d_C\ge 0}
\left[-\big(g_C^{(t+1)}\big)^\top d_C\right],
$$

$$
C^{(t+1)}
\leftarrow
C^{(t)} + d_C^{(t)}.
$$

If you want a strict trust-region radius update, define the `C`-merit

$$
\psi_t(C) = -\big(g_C^{(t+1)}\big)^\top C,
$$

then use

$$
\mathrm{Pred}_t = m_t(0) - m_t(d_C^{(t)}),
\qquad
\mathrm{Ared}_t = \psi_t(C^{(t)}) - \psi_t(C^{(t)} + d_C^{(t)}),
\qquad
r_t = \frac{\mathrm{Ared}_t}{\mathrm{Pred}_t}.
$$

For the linear `C`-model above, `r_t = 1` whenever projection does not change the step, so a fixed trust-region radius is acceptable in the first write-up.

## 8. Ready-to-paste replacement for the main paper

If you want a compact equation to insert into the main LaTeX draft, use this:

$$
\Phi(\alpha,\beta,C)
=
\frac{1}{|\mathrm{pf}|}\sum_{m\in\mathrm{pf}}
\frac{1}{2n_m}\sum_{j=1}^{n_m}
C_{y_{m,j}}
\left(
\sum_{i=1}^S \alpha_m^i x_{m,j}^{i\top}\beta^i - y_{m,j}
\right)^2
\;+\;\lambda R_\beta(\beta),
$$

subject to

$$
R_\alpha(\alpha_m)\le 1,\qquad C\ge 0,
$$

with the trust-region `C`-update

$$
d_C^{(t)}
=
\arg\min_{\|d_C\|_2\le \Delta_t,\;C^{(t)}+d_C\ge 0}
\left[-\big(g_C^{(t+1)}\big)^\top d_C\right],
\qquad
g_C^{(t+1)} = L^{(t+1)} - \bar L^{(t+1)}\mathbf 1,
$$

and

$$
C^{(t+1)} = C^{(t)} + d_C^{(t)}.
$$

## 9. What this changes in the current draft

Keep:

- the original iSFS equations (14) to (18),
- the weighted `\alpha` and `\beta` updates,
- the use of a baseline `C^0`,
- the idea that `C` should move toward harder classes.

Replace:

- `\min_{\alpha,\beta}\max_C` by a three-block alternating scheme,
- the simplex-plus-ball feasible set by a pure trust-region ball in `d_C`,
- the closed-form simplex/DRO projection by a trust-region step,
- the DRO language by trust-region language.

That gives you a cleaner story:

- iSFS handles block-wise missingness,
- cost-sensitive weighting enters the residual loss,
- trust-region optimization updates the class-cost vector locally around the current iterate.
