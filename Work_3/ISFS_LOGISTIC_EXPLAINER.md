# iSFS Logistic / Cost-Sensitive Explainer

## Purpose of This Document

This document explains, in one place, the theory behind the current
implementation of the incomplete-source feature-selection (iSFS) model in this
repository.

The goal is to connect three things clearly:

1. the original iSFS equations in Section 3 of `Multi_source_BWM.pdf`
2. the logistic / binary-cross-entropy (BCE) version used in this repository
3. the code modules that implement each part of the method

This is the active modeling path in the repository:

- missingness profiles and overlapping training groups
- logistic loss for binary classification
- profile-specific source weights `alpha`
- shared source coefficients `beta`
- class-cost vector `C` updated by a trust-region step

Important scope note:

- The original paper derives the optimization mainly with least squares.
- Remark 3 in the paper states that the method can be extended to logistic loss.
- This repository follows that logistic extension.
- The trust-region cost block comes from `notes.tex`, but the loss inside the
  cost block is adapted here from squared loss to BCE so that it matches the
  current classifier.

## 1. Where iSFS Starts in the Paper

Equation (13) in the paper still belongs to the complete-data model from
Section 2. The incomplete-data iSFS model starts in Section 3, and the main
paper equations for the current work are:

- Equation (14): the iSFS optimization problem
- Equation (15): the profile-level loss building block
- Equation (16): the `alpha` subproblem
- Equation (17): the `beta` subproblem
- Equation (18): the gradient of the least-squares `beta` block

So, for the current blockwise-missing setting, Equations (14) to (18) are the
main source of truth.

## 2. Original iSFS Formulation in the Paper

The paper defines the incomplete-source feature-selection model as

$$
\min_{\boldsymbol{\alpha}, \boldsymbol{\beta}}
\frac{1}{|\mathbf{pf}|}
\sum_{m \in \mathbf{pf}}
f(\mathbf{X}_m, \boldsymbol{\beta}, \boldsymbol{\alpha}_m, \mathbf{y}_m)
+ \lambda R_{\beta}(\boldsymbol{\beta})
$$

subject to

$$
R_{\alpha}(\boldsymbol{\alpha}_m) \le 1,
\qquad \forall m \in \mathbf{pf}.
$$

This is Equation (14) in the paper.

The paper then defines the profile-level loss block as

$$
f(\mathbf{X}, \boldsymbol{\beta}, \boldsymbol{\alpha}, \mathbf{y})
=
\frac{1}{n}
L\!\left(
\sum_{i=1}^{S} \alpha^i \mathbf{X}^i \boldsymbol{\beta}^i,
\mathbf{y}
\right),
$$

which is Equation (15).

Here:

- $S$ is the number of sources
- $\mathbf{pf}$ is the set of profile-based training groups
- $\mathbf{X}_m^i$ is the source-$i$ design matrix restricted to group $m$
- $\boldsymbol{\beta}^i$ is the shared coefficient vector for source $i$
- $\alpha_m^i$ is the source weight for source $i$ inside group $m$
- $L(\cdot,\cdot)$ is any convex loss
- $R_{\alpha}$ and $R_{\beta}$ are regularization functions

The profile-specific score is therefore

$$
\mathbf{z}_m
=
\sum_{i=1}^{S} \alpha_m^i \mathbf{X}_m^i \boldsymbol{\beta}^i.
$$

This is the central linear model used throughout the iSFS formulation.

## 3. Paper-to-Code Change: Least Squares to Logistic Loss

In Section 3.2, the paper says it focuses on the least-squares loss for the
derivation, but Remark 3 explicitly states that the same model can be extended
to logistic loss.

That is exactly what this repository does.

### 3.1 Least-squares form in the paper

For the least-squares case, the profile-level data-fit term is effectively

$$
\frac{1}{2n_m}
\left\|
\sum_{i=1}^{S} \alpha_m^i \mathbf{X}_m^i \boldsymbol{\beta}^i
- \mathbf{y}_m
\right\|_2^2.
$$

This is why the paper's `beta` block becomes quadratic.

### 3.2 Logistic / BCE form used in this repository

For binary classification, we instead use the BCE loss written directly from
logits:

$$
\ell(z, y)
=
\log(1 + e^z) - yz,
\qquad y \in \{0,1\}.
$$

For profile $m$, the logit vector is

$$
\mathbf{z}_m
=
\sum_{i=1}^{S} \alpha_m^i \mathbf{X}_m^i \boldsymbol{\beta}^i.
$$

So the logistic profile loss is

$$
f_m^{\mathrm{log}}(\boldsymbol{\alpha}_m, \boldsymbol{\beta})
=
\frac{1}{n_m}
\sum_{j \in G_m}
\ell(z_{m,j}, y_{m,j}).
$$

Two details matter here:

1. the factor $\frac{1}{n_m}$ stays, because we still average over samples in
   group $m$
2. the factor $\frac{1}{2}$ disappears, because it belonged to squared loss,
   not to BCE

This is the loss used in [loss_functions.py](loss_functions.py).

## 4. Regularization Choice Used Here

The paper leaves $R_{\alpha}$ and $R_{\beta}$ generic in Equation (14), then
states in Remark 4 and in the experiments section that different regularizers
are possible, and that the $\ell_1$ penalty is used in the experiments.

The current repository follows that paper-aligned choice:

$$
R_{\alpha}(\boldsymbol{\alpha}_m)
=
\|\boldsymbol{\alpha}_m\|_1
$$

and

$$
R_{\beta}(\boldsymbol{\beta})
=
\sum_{i=1}^{S} \|\boldsymbol{\beta}^i\|_1.
$$

In code, this becomes:

- an $\ell_1$-ball constraint for `alpha`
- an $\ell_1$ penalty plus proximal step for `beta`

This is implemented in [regularization.py](regularization.py).

## 5. Missingness Profiles and Block Construction

This is the first major step in the incomplete-data model.

### 5.1 Availability indicator for each participant

Suppose there are $S$ sources. For participant $j$, define the binary source
availability vector

$$
\mathbf{I}_j = [I_{j1}, I_{j2}, \dots, I_{jS}],
$$

where

$$
I_{ji} =
\begin{cases}
1, & \text{if source } i \text{ is observed for participant } j \\
0, & \text{if source } i \text{ is missing for participant } j.
\end{cases}
$$

The paper then converts this binary vector into a single integer profile:

$$
p_j
=
\sum_{i=1}^{S} I_{ji} 2^{S-i}.
$$

For $S=3$:

- $[1,1,1] \mapsto 7$
- $[1,1,0] \mapsto 6$
- $[1,0,1] \mapsto 5$
- $[0,1,1] \mapsto 3$
- $[1,0,0] \mapsto 4$
- $[0,1,0] \mapsto 2$
- $[0,0,1] \mapsto 1$

In code, we usually display the same idea as a binary code such as `111`,
`110`, `101`, and so on.

### 5.2 Exact participant profile

Each participant has exactly one exact observed-source profile:

$$
A_j = \{ i : I_{ji} = 1 \}.
$$

This exact profile is what prediction uses later.

### 5.3 Overlapping optimization group $G_m$

The paper does not train directly on exact profiles only. Instead, for each
source combination $m$, it builds an overlapping training group:

$$
G_m
=
\{ j : (p_j \,\&\, m) = m \},
$$

where `&` is the bitwise AND operation.

This means:

- sample $j$ belongs to $G_m$ if sample $j$ contains every source required by
  combination $m$
- a sample has one exact profile, but it can belong to several training groups

For example, if a participant has exact profile `111`, then that participant
belongs to all of the following training groups:

- `111`
- `110`
- `101`
- `011`
- `100`
- `010`
- `001`

This is why the training groups overlap.

### 5.4 How the block matrices are formed

Once $G_m$ is known, the block matrices are formed by row restriction:

$$
\mathbf{X}_m^i = \mathbf{X}^i[G_m, :],
\qquad
\mathbf{y}_m = \mathbf{y}[G_m].
$$

So, for each combination $m$, we gather the rows whose available sources contain
$m$, and then use those rows to define the subproblem for that block.

### 5.5 Small illustration table

Assume there are 3 sources, denoted $S_1, S_2, S_3$.

| Sample | $S_1$ | $S_2$ | $S_3$ | Availability vector | Profile code | Exact profile | Training groups containing the sample |
|---|---:|---:|---:|---|---|---|---|
| 1 | 1 | 1 | 1 | $[1,1,1]$ | `111` | `111` | `001,010,011,100,101,110,111` |
| 2 | 1 | 1 | 0 | $[1,1,0]$ | `110` | `110` | `100,010,110` |
| 3 | 1 | 0 | 1 | $[1,0,1]$ | `101` | `101` | `100,001,101` |
| 4 | 0 | 1 | 1 | $[0,1,1]$ | `011` | `011` | `010,001,011` |
| 5 | 1 | 0 | 0 | $[1,0,0]$ | `100` | `100` | `100` |
| 6 | 0 | 1 | 0 | $[0,1,0]$ | `010` | `010` | `010` |
| 7 | 0 | 0 | 1 | $[0,0,1]$ | `001` | `001` | `001` |
| 8 | 1 | 1 | 1 | $[1,1,1]$ | `111` | `111` | `001,010,011,100,101,110,111` |

Now take $m = 110$. Then

$$
G_{110}
=
\{ j : (p_j \,\&\, 110) = 110 \}.
$$

In this small table, that includes samples whose exact profile is `110` or
`111`. So the rows in $G_{110}$ are the rows used to build:

$$
\mathbf{X}_{110}^{(1)},
\qquad
\mathbf{X}_{110}^{(2)},
\qquad
\mathbf{y}_{110}.
$$

This is implemented in [missingness_profiles.py](missingness_profiles.py).

## 6. Alpha Update When Beta and Cost Are Fixed

The paper's Equation (16) says that when $\boldsymbol{\beta}$ is fixed, each
profile-specific weight vector $\boldsymbol{\alpha}_m$ is obtained from

$$
\min_{\boldsymbol{\alpha}}
\left\|
\sum_{i=1}^{S} \alpha^i \mathbf{X}^i \boldsymbol{\beta}^i
- \mathbf{y}
\right\|_2^2
$$

subject to

$$
R_{\alpha}(\boldsymbol{\alpha}) \le 1.
$$

### 6.1 Logistic / cost-sensitive alpha objective used here

In the current repository, with BCE and class costs, the profile-$m$ alpha
subproblem becomes

$$
f_m^C(\boldsymbol{\alpha}_m)
=
\frac{1}{n_m}
\sum_{j \in G_m}
C_{y_{m,j}}
\ell(z_{m,j}, y_{m,j}),
$$

where

$$
z_{m,j}
=
\sum_{i=1}^{S}
\alpha_m^i
\big(\mathbf{x}_{m,j}^i\big)^\top
\boldsymbol{\beta}^i.
$$

### 6.2 Gradient with respect to alpha

Define the source-specific score vector

$$
\mathbf{s}_m^i = \mathbf{X}_m^i \boldsymbol{\beta}^i.
$$

Then

$$
\mathbf{z}_m
=
\sum_{i=1}^{S} \alpha_m^i \mathbf{s}_m^i.
$$

The gradient of the BCE block with respect to the logits is

$$
\mathbf{g}_m
=
\frac{1}{n_m}
\left[
C_{\mathbf{y}_m}
\odot
\big(\sigma(\mathbf{z}_m) - \mathbf{y}_m\big)
\right].
$$

Therefore, by the chain rule,

$$
\frac{\partial f_m^C}{\partial \alpha_m^i}
=
\big(\mathbf{s}_m^i\big)^\top \mathbf{g}_m
=
\big(\mathbf{X}_m^i \boldsymbol{\beta}^i\big)^\top \mathbf{g}_m.
$$

### 6.3 Constrained update used in code

Because the active choice is

$$
R_{\alpha}(\boldsymbol{\alpha}_m)
=
\|\boldsymbol{\alpha}_m\|_1,
$$

the code uses projected gradient:

$$
\boldsymbol{\alpha}_m^{\mathrm{temp}}
=
\boldsymbol{\alpha}_m
- \eta_{\alpha} \nabla f_m^C(\boldsymbol{\alpha}_m),
$$

$$
\boldsymbol{\alpha}_m^{\mathrm{next}}
=
\Pi_{\|\cdot\|_1 \le r_{\alpha}}
\big(
\boldsymbol{\alpha}_m^{\mathrm{temp}}
\big).
$$

This is implemented in [alpha_update.py](alpha_update.py).

## 7. Beta Update When Alpha and Cost Are Fixed

The paper's Equation (17) says that when $\boldsymbol{\alpha}$ is fixed, the
`beta` block is

$$
\min_{\boldsymbol{\beta}}
g(\boldsymbol{\beta}) + \lambda R_{\beta}(\boldsymbol{\beta}).
$$

For least squares, the paper writes

$$
g(\boldsymbol{\beta})
=
\frac{1}{|\mathbf{pf}|}
\sum_{m \in \mathbf{pf}}
\frac{1}{2n_m}
\left\|
\sum_{i=1}^{S}
(\alpha_m^i \mathbf{X}_m^i)\boldsymbol{\beta}^i
- \mathbf{y}_m
\right\|_2^2.
$$

It then gives the least-squares gradient in Equation (18):

$$
\nabla g(\boldsymbol{\beta}^i)
=
\frac{1}{|\mathbf{pf}|}
\sum_{m \in \mathbf{pf}}
\frac{1}{n_m}
I(m \,\&\, 2^{S-i} \ne 0)
(\alpha_m^i \mathbf{X}_m^i)^T
\left(
\sum_{i=1}^{S}
\alpha_m^i \mathbf{X}_m^i \boldsymbol{\beta}^i
- \mathbf{y}_m
\right).
$$

### 7.1 Logistic / cost-sensitive beta objective used here

In the current repository, this becomes

$$
g_C^{\mathrm{log}}(\boldsymbol{\beta})
=
\frac{1}{|\mathbf{pf}|}
\sum_{m \in \mathbf{pf}}
\frac{1}{n_m}
\sum_{j \in G_m}
C_{y_{m,j}}
\ell(z_{m,j}, y_{m,j}).
$$

### 7.2 Logistic gradient with respect to beta

The same logits gradient is

$$
\mathbf{g}_m
=
\frac{1}{n_m}
\left[
C_{\mathbf{y}_m}
\odot
\big(\sigma(\mathbf{z}_m) - \mathbf{y}_m\big)
\right].
$$

Then the gradient with respect to source-$i$ coefficients is

$$
\nabla_{\boldsymbol{\beta}^i} g_C^{\mathrm{log}}(\boldsymbol{\beta})
=
\frac{1}{|\mathbf{pf}|}
\sum_{m : i \in m}
(\alpha_m^i \mathbf{X}_m^i)^T \mathbf{g}_m.
$$

This is the logistic analogue of Equation (18).

### 7.3 Proximal update used in code

Because

$$
R_{\beta}(\boldsymbol{\beta})
=
\sum_{i=1}^{S} \|\boldsymbol{\beta}^i\|_1,
$$

the code uses a proximal-gradient step:

$$
\mathbf{v}_i
=
\boldsymbol{\beta}^i
- \eta_{\beta}
\nabla_{\boldsymbol{\beta}^i} g_C^{\mathrm{log}}(\boldsymbol{\beta}),
$$

$$
\boldsymbol{\beta}^{i,\mathrm{next}}
=
\operatorname{prox}_{\eta_{\beta}\lambda \|\cdot\|_1}(\mathbf{v}_i)
=
\operatorname{sign}(\mathbf{v}_i)
\odot
\max(|\mathbf{v}_i| - \eta_{\beta}\lambda, 0).
$$

This is implemented in [beta_update.py](beta_update.py).

## 8. Cost Vector and Trust-Region Update

This is the main extension beyond the original paper path.

### 8.1 Cost-sensitive BCE block

For group $m$, the current repository uses

$$
f_m^C(\mathbf{X}_m, \boldsymbol{\beta}, \boldsymbol{\alpha}_m, \mathbf{y}_m)
=
\frac{1}{n_m}
\sum_{j \in G_m}
C_{y_{m,j}}
\ell(z_{m,j}, y_{m,j}).
$$

So the full fixed-cost Block A objective is

$$
\Phi(\boldsymbol{\alpha}, \boldsymbol{\beta}; C)
=
\frac{1}{|\mathbf{pf}|}
\sum_{m \in \mathbf{pf}}
f_m^C(\mathbf{X}_m, \boldsymbol{\beta}, \boldsymbol{\alpha}_m, \mathbf{y}_m)
+
\lambda_{\beta}
\sum_{i=1}^{S} \|\boldsymbol{\beta}^i\|_1
$$

subject to

$$
\|\boldsymbol{\alpha}_m\|_1 \le r_{\alpha}.
$$

### 8.2 Per-class loss vector

With $\boldsymbol{\alpha}$ and $\boldsymbol{\beta}$ fixed, define

$$
L_k(\boldsymbol{\alpha}, \boldsymbol{\beta})
=
\frac{1}{|\mathbf{pf}|}
\sum_{m \in \mathbf{pf}}
\frac{1}{n_m}
\sum_{j \in G_m : y_{m,j}=k}
\ell(z_{m,j}, y_{m,j}),
$$

for each class $k$.

Then define the class-loss vector

$$
L(\boldsymbol{\alpha}, \boldsymbol{\beta})
=
\big(
L_1(\boldsymbol{\alpha}, \boldsymbol{\beta}),
\dots,
L_K(\boldsymbol{\alpha}, \boldsymbol{\beta})
\big)^T.
$$

The cost-only objective is then

$$
\Phi_C(C)
=
L(\boldsymbol{\alpha}, \boldsymbol{\beta})^T C.
$$

### 8.3 Trust-region subproblem for the cost vector

Let the current cost vector at outer iteration $t$ be $C^{(t)}$, and let
$C^0$ be the baseline cost vector. The trust-region step is written in terms of
$d_t$ where

$$
C^{\mathrm{trial}} = C^{(t)} + d_t.
$$

The trust-region problem solved for the `C` block is

$$
\max_{d}
\;
L_t^T d
$$

subject to

$$
\frac{1}{2}\|d\|_2^2 \le \Delta_t,
\qquad
C^{(t)} + d \ge 0,
\qquad
\frac{1}{2}\|C^{(t)} + d - C^0\|_2^2 \le \Delta_{\max}.
$$

This means:

- the step must stay inside the current working trust region
- the updated cost vector must stay nonnegative
- the updated cost vector must remain inside the larger baseline-centered ball

### 8.4 Predicted increase, actual increase, and acceptance ratio

Define the reduced objective

$$
F(C)
=
\min_{\boldsymbol{\alpha}, \boldsymbol{\beta}}
\Phi(\boldsymbol{\alpha}, \boldsymbol{\beta}; C).
$$

Then the trust-region model predicts

$$
\mathrm{Pred}_t = L_t^T d_t,
$$

while the actual change is

$$
\mathrm{Ared}_t
=
F(C^{(t)} + d_t) - F(C^{(t)}).
$$

The trust-region ratio is

$$
\rho_t
=
\frac{\mathrm{Ared}_t}{\mathrm{Pred}_t}.
$$

If $\rho_t$ is large enough, the trial step is accepted. If not, the step is
rejected and the model keeps the previous $C$, $\alpha$, and $\beta$.

This is implemented in [costs.py](costs.py) and orchestrated in
[train_model.py](train_model.py).

## 9. Prediction Rule

Training and prediction do not use the same grouping rule.

### 9.1 During training

Training uses the overlapping optimization groups $G_m$.

### 9.2 During prediction

Prediction uses the exact observed-source pattern of the new sample.

For a new sample $j$, define its exact observed-source set

$$
A_j = \{ i : \text{source } i \text{ is observed for sample } j \}.
$$

Then prediction uses the exact-profile-specific alpha weights:

$$
z_j
=
\sum_{i \in A_j}
\alpha_{A_j}^i
\big(\mathbf{x}_j^i\big)^T
\boldsymbol{\beta}^i.
$$

The probability of class 1 is

$$
\Pr(y_j = 1 \mid x_j)
=
\sigma(z_j)
=
\frac{1}{1 + e^{-z_j}}.
$$

The predicted class is

$$
\hat{y}_j
=
\mathbf{1}\{\sigma(z_j) \ge \tau\},
$$

where $\tau = 0.5$ by default.

Important point:

- the learned cost vector $C$ does not appear directly in the prediction
  formula
- $C$ affects prediction indirectly by changing the learned $\alpha$ and
  $\beta$ during training

This is implemented in [prediction.py](prediction.py).

## 10. Full Training Loop Used in This Repository

The current training loop is:

### Step 0. Build profiles and training blocks

From the source availability pattern:

1. compute the exact profile of each sample
2. build the overlapping optimization groups $G_m$
3. build the block matrices $\mathbf{X}_m^i$ and labels $\mathbf{y}_m$

### Step 1. Initialize parameters

The paper's Algorithm 2 says:

- initialize each $\boldsymbol{\beta}^i$ by fitting each source individually on
  the available data

The current code uses a practical initialization:

- initialize each $\boldsymbol{\beta}^i$ with small random values
- initialize each $\boldsymbol{\alpha}_m$ with equal weight across the sources
  present in profile $m$
- initialize the cost vector with

$$
C^{(0)} = [0.5, 0.5]^T
$$

and set the baseline cost vector to

$$
C^0 = [0.5, 0.5]^T
$$

### Step 2. Solve Block A with cost fixed

With $C^{(t)}$ fixed:

1. solve the `alpha` subproblem with `beta` fixed
2. solve the `beta` subproblem with `alpha` fixed
3. repeat until the fixed-cost Block A objective stabilizes

Symbolically,

$$
(\boldsymbol{\alpha}^{(t+1)}, \boldsymbol{\beta}^{(t+1)})
\approx
\arg\min_{\boldsymbol{\alpha}, \boldsymbol{\beta}}
\Phi(\boldsymbol{\alpha}, \boldsymbol{\beta}; C^{(t)}).
$$

### Step 3. Compute the class-loss vector

With the updated $\boldsymbol{\alpha}^{(t+1)}$ and
$\boldsymbol{\beta}^{(t+1)}$, compute

$$
L^{(t+1)}
=
L(\boldsymbol{\alpha}^{(t+1)}, \boldsymbol{\beta}^{(t+1)}).
$$

### Step 4. Solve the trust-region cost step

Using $L^{(t+1)}$, solve for a trial cost step:

$$
d_t
\approx
\arg\max_d L^{(t+1)T} d
$$

subject to the trust-region and nonnegativity constraints.

This produces

$$
C^{\mathrm{trial}} = C^{(t)} + d_t.
$$

### Step 5. Re-solve Block A at the trial cost

Now fix $C^{\mathrm{trial}}$ and solve again for the corresponding
$\alpha,\beta$ block:

$$
(\boldsymbol{\alpha}^{\mathrm{trial}}, \boldsymbol{\beta}^{\mathrm{trial}})
\approx
\arg\min_{\boldsymbol{\alpha}, \boldsymbol{\beta}}
\Phi(\boldsymbol{\alpha}, \boldsymbol{\beta}; C^{\mathrm{trial}}).
$$

### Step 6. Accept or reject the cost step

Compute

$$
\mathrm{Pred}_t = L_t^T d_t,
\qquad
\mathrm{Ared}_t = F(C^{\mathrm{trial}}) - F(C^{(t)}),
\qquad
\rho_t = \frac{\mathrm{Ared}_t}{\mathrm{Pred}_t}.
$$

Then:

- if $\rho_t$ is large enough, accept the step and keep
  $(C^{\mathrm{trial}}, \boldsymbol{\alpha}^{\mathrm{trial}}, \boldsymbol{\beta}^{\mathrm{trial}})$
- otherwise reject the step and keep the previous
  $(C^{(t)}, \boldsymbol{\alpha}^{(t+1)}, \boldsymbol{\beta}^{(t+1)})$

### Step 7. Update the trust-region radius

Depending on $\rho_t$:

- shrink the radius if the step was poor
- keep the radius if the step was reasonable
- enlarge the radius if the step was very reliable

### Step 8. Repeat until convergence

Repeat Steps 2 to 7 until:

- the outer BCD iteration limit is reached, or
- the cost step becomes too small, or
- the reduced objective stabilizes

This full loop is implemented in [train_model.py](train_model.py) and exposed by
[run_training.py](run_training.py) and [run_training.sh](run_training.sh).

## 11. What Is Actually Minimized and What Is Maximized

One subtle but important point is that the current training loop is not a pure
"minimize everything" procedure.

For fixed cost $C$, Block A solves

$$
\min_{\boldsymbol{\alpha}, \boldsymbol{\beta}}
\Phi(\boldsymbol{\alpha}, \boldsymbol{\beta}; C).
$$

But for fixed $\boldsymbol{\alpha}, \boldsymbol{\beta}$, the cost block solves

$$
\max_C \Phi_C(C).
$$

So the outer scheme is a blockwise alternating procedure with:

- minimization over $(\boldsymbol{\alpha}, \boldsymbol{\beta})$
- maximization over $C$ inside the trust-region constraints

This is why, during training output:

- `Accepted=True` means the proposed trust-region move in $C$ was kept
- `Accepted=False` means the proposed move in $C$ was rejected

## 12. Code Map

The current code modules line up with the mathematics as follows.

| Theoretical part | Code file |
|---|---|
| Stable logistic / BCE loss | [loss_functions.py](loss_functions.py) |
| Profile construction and overlapping groups | [missingness_profiles.py](missingness_profiles.py) |
| `alpha` subproblem | [alpha_update.py](alpha_update.py) |
| `beta` subproblem | [beta_update.py](beta_update.py) |
| `\ell_1` regularization operators | [regularization.py](regularization.py) |
| Cost-sensitive BCE and trust-region cost step | [costs.py](costs.py) |
| Exact-profile prediction | [prediction.py](prediction.py) |
| Full outer BCD training loop | [train_model.py](train_model.py) |
| User-facing runner and saved reports | [run_training.py](run_training.py) |

## 13. Short Summary

The iSFS model in the paper is built around:

- one shared coefficient vector $\boldsymbol{\beta}^i$ per source
- one profile-specific weight vector $\boldsymbol{\alpha}_m$ per source
  combination
- a profile-based decomposition of blockwise-missing data

This repository keeps that structure, but replaces the least-squares data-fit
term with BCE from logits for binary classification, and adds a trust-region
update for the class-cost vector $C$.

So the full current pipeline is:

1. compute exact profiles and overlapping training groups
2. train the logistic iSFS model by alternating `alpha` and `beta`
3. update the class-cost vector `C` by trust region
4. predict with the exact-profile logistic model

That is the theoretical path implemented by the current codebase.
