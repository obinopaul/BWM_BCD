# Block Coordinate Descent (BCD) — A Working Explainer

> A field guide to **Block Coordinate Descent**, written for someone who wants
> the equations *and* the actual numbers. Every formula is followed by a small
> hand-traced example and a runnable Python snippet so you can re-execute it
> yourself.
>
> Sources: Aggarwal, *Linear Algebra and Optimization for Machine Learning*,
> §4.10 (`1660642748-214-217.pdf`) and §8.3 (`1660642748-356-362.pdf`); and
> Bertsimas, Pawlowski & Zhuo, "From Predictive Methods to Missing Data
> Imputation," JMLR 2018 (`17-073.pdf`).

---

## Table of contents

1. [The big idea — in plain English](#1-the-big-idea--in-plain-english)
2. [Coordinate Descent — the simpler cousin](#2-coordinate-descent--the-simpler-cousin)
3. [From CD to BCD — the formal definition](#3-from-cd-to-bcd--the-formal-definition)
4. [Case study A — K-means *is* BCD](#4-case-study-a--k-means-is-bcd)
5. [Case study B — Alternating Least Squares for movie recommendations](#5-case-study-b--alternating-least-squares-for-movie-recommendations)
6. [Case study C — A real research paper using BCD](#6-case-study-c--a-real-research-paper-using-bcd)
7. [When to reach for BCD — a decision checklist](#7-when-to-reach-for-bcd--a-decision-checklist)

---

## 1. The big idea — in plain English

Imagine you are standing on a hilly landscape and you want to walk to the
lowest point. The catch: you are allowed to walk in **only one compass
direction at a time**.

- **Gradient descent** — you look at the slope under your feet (a vector that
  points in *every* direction at once) and take a small step in the steepest
  downhill direction.
- **Coordinate descent (CD)** — you look only along the *north-south* axis,
  walk to the lowest point on that line, then turn 90° and walk to the lowest
  point along the *east-west* axis. Repeat.
- **Block coordinate descent (BCD)** — you split the directions into *groups*
  ("blocks"). For example, "horizontal axes" and "vertical axes". You freeze
  every group except one and *fully solve* the problem inside the unfrozen
  group, then switch.

The reason this is useful in optimization is not aesthetic — it is that **the
sub-problem inside one block is often dramatically easier** than the full
problem. Three things commonly happen when you fix all but one block:

1. **The non-convex problem becomes convex.** This is the case in matrix
   factorization $D \approx UV^T$: optimizing $U$ and $V$ jointly is non-convex,
   but if you freeze $V$, the resulting problem in $U$ alone is a plain
   least-squares — which has a closed-form solution.
2. **The constraints decouple.** In $k$-means, the assignment constraint
   $\sum_j y_{ij} = 1$ couples all the assignment variables together. But
   if you freeze the centroids, each point chooses its own nearest centroid
   independently — the constraint trivializes.
3. **Each block has a closed-form (or near-closed-form) optimum.** No more
   line searches or learning rates inside a block: you just compute the answer.

That is the entire intuition. Aggarwal puts it crisply (§4.10.2):
> "Although each step in block coordinate descent is more expensive, fewer
> steps are required."

The rest of this document earns that one sentence with three worked examples.

---

## 2. Coordinate Descent — the simpler cousin

Before BCD it helps to nail down its single-variable cousin, since the
mechanics are nearly identical. We follow the derivation in §4.10.1 of
Aggarwal's book.

### 2.1 The setup

We have a least-squares regression problem:

$$
\min_{w \in \mathbb{R}^d}\;\; J(w)\;=\;\tfrac{1}{2}\|y - Xw\|^2
$$

where $X$ is an $n \times d$ data matrix, $y$ is an $n$-vector of targets, and
$w$ is the unknown weight vector. Let $d_i$ be the $i$-th **column** of $X$
(the values of feature $i$ across all rows), and let $r = y - Xw$ be the
current **residual**.

### 2.2 The derivation

Take the partial derivative of $J$ with respect to a single weight $w_i$ while
treating all other $w_j$ ($j \neq i$) as constants:

$$
\frac{\partial J}{\partial w_i}
\;=\; -\,d_i^{T}(y - Xw)
\;=\; -\,d_i^{T} r
$$

Setting this to zero — the **exact** minimizer of $J$ as a function of $w_i$
alone — gives the rearranged form Aggarwal derives in eq. (4.74):

$$
\boxed{\;w_i \;\leftarrow\; w_i \;+\; \frac{d_i^{T} r}{\|d_i\|^{2}}\;}
$$

A subtle but beautiful point in the book: even though the right-hand side has
a $w_i$ in it, after the update one can show that the result equals the
*exact* minimizer of $J$ over $w_i$ — not a step in that direction, but the
optimum itself. That's what makes coordinate descent so efficient: each
single-coordinate update is a one-shot solve, not a gradient step.

If columns are standardized so that $\|d_i\|^2 = 1$, the update collapses
further to $w_i \leftarrow w_i + d_i^{T} r$ (eq. 4.75 in the book).

### 2.3 A worked numerical example

Take a tiny dataset where the answer is known so you can sanity-check:

$$
X = \begin{bmatrix} 1 & \phantom{-}1 \\ 2 & -1 \\ 3 & \phantom{-}1 \\ 4 & -1 \end{bmatrix},
\qquad y = \begin{bmatrix} 3 \\ 0 \\ 5 \\ 2 \end{bmatrix}
$$

Closed-form OLS gives $w^* = (X^T X)^{-1} X^T y = \begin{bmatrix} 1 \\ 2 \end{bmatrix}$
exactly. Let's see if CD finds it.

**Initial state:** $w = (0, 0)$, residual $r = y - Xw = (3, 0, 5, 2)$.

**Sweep 1, update $w_1$:**

| quantity | value |
|---|---|
| $d_1 = X[:,1]$ | $(1, 2, 3, 4)$ |
| $\|d_1\|^2$ | $1 + 4 + 9 + 16 = 30$ |
| $d_1^{T} r$ | $1\cdot 3 + 2\cdot 0 + 3\cdot 5 + 4\cdot 2 = 26$ |
| $\Delta w_1 = d_1^{T} r / \|d_1\|^2$ | $26 / 30 = 0.8667$ |
| new $w_1$ | $0 + 0.8667 = 0.8667$ |
| new residual $r$ | $r - 0.8667 \cdot d_1 = (2.133,\, -1.733,\, 2.400,\, -1.467)$ |

**Sweep 1, update $w_2$:**

| quantity | value |
|---|---|
| $d_2 = X[:,2]$ | $(1, -1, 1, -1)$ |
| $\|d_2\|^2$ | $1 + 1 + 1 + 1 = 4$ |
| $d_2^{T} r$ | $2.133 - (-1.733) + 2.400 - (-1.467) = 7.733$ |
| $\Delta w_2$ | $7.733 / 4 = 1.9333$ |
| new $w_2$ | $0 + 1.9333 = 1.9333$ |
| new residual $r$ | $(0.200,\, 0.200,\, 0.467,\, 0.467)$ |

After **one full sweep** of CD: $w = (0.8667,\, 1.9333)$.

**Sweep 2** (just the final values):

| | $w_1$ | $w_2$ | cost $\tfrac{1}{2}\|r\|^2$ |
|---|---|---|---|
| start | 0.8667 | 1.9333 | 0.2578 |
| after $w_1$ update | 0.9956 | 1.9333 | — |
| after $w_2$ update | 0.9956 | 1.9978 | 0.000286 |

After **two sweeps** we're at $(0.9956, 1.9978)$ — already within 0.5% of the
true answer $(1, 2)$. After **three sweeps**: $(0.99985,\, 1.99993)$. CD has
converged.

### 2.4 The Python check

```python
import numpy as np

X = np.array([[1, 1], [2, -1], [3, 1], [4, -1]], dtype=float)
y = np.array([3, 0, 5, 2], dtype=float)

w = np.zeros(2)
for sweep in range(3):
    for i in range(2):
        r = y - X @ w
        d_i = X[:, i]
        w[i] += (d_i @ r) / (d_i @ d_i)
    print(f"sweep {sweep+1}: w = {w},  cost = {0.5*np.sum((y - X @ w)**2):.6f}")

# Output:
# sweep 1: w = [0.8667 1.9333],  cost = 0.257778
# sweep 2: w = [0.9956 1.9978],  cost = 0.000286
# sweep 3: w = [0.9999 1.9999],  cost = 0.000000
```

Run that snippet and you should see the numbers in the table above. **That is
coordinate descent in 8 lines of code.**

---

## 3. From CD to BCD — the formal definition

Now we generalize. Partition the variables into $B$ disjoint **blocks**:

$$
x \;=\; (x_1,\, x_2,\, \ldots,\, x_B), \qquad x_b \in \mathbb{R}^{n_b}
$$

Block coordinate descent cycles through blocks, fully optimizing one block
while holding the others fixed:

$$
\boxed{\;
x_b^{(k+1)} \;=\; \arg\min_{\,x_b}\;
f\!\left(x_1^{(k+1)},\, \ldots,\, x_{b-1}^{(k+1)},\; x_b,\; x_{b+1}^{(k)},\, \ldots,\, x_B^{(k)}\right)
\;}
$$

Three things are notable:

1. **The inner $\arg\min$ is exact.** We do not take a gradient step toward the
   block minimum — we *go to* the block minimum. This is what makes BCD
   different from "block gradient descent."
2. **CD is the special case** where $B = d$ and each $x_b$ is a single scalar.
3. **Convergence guarantees exist** when the problem is convex in each block
   given the others fixed (the "multi-convex" setting). In that case the
   objective decreases monotonically. Outside multi-convexity (the setting of
   the Bertsimas paper in §6 below), BCD can stall at a non-global stationary
   point — Stephen Wright's 2015 *Mathematical Programming* survey on
   coordinate descent is the standard reference for the convergence theory.

The trade-off Aggarwal flags is the one-sentence summary of the whole method:
> Each step is **more expensive** than CD because you solve a multivariate
> sub-problem; but you typically need **fewer total steps**.

In the next two sections we see this play out on K-means and matrix
factorization.

---

## 4. Case study A — K-means *is* BCD

Most people learn $k$-means as a *heuristic*: "guess centroids, assign points
to the nearest one, recompute centroids, repeat." Aggarwal shows in §4.10.3
that this is not a heuristic at all — **it is exactly block coordinate descent
applied to a mixed-integer optimization problem**.

### 4.1 The optimization problem

Given $n$ data points $X_1, \ldots, X_n \in \mathbb{R}^d$ and $k$
prototypes $z_1, \ldots, z_k \in \mathbb{R}^d$, with binary assignment
indicators $y_{ij} \in \{0, 1\}$ saying whether point $i$ is in cluster $j$:

$$
\begin{aligned}
\text{minimize} \quad & \sum_{j=1}^{k} \sum_{i=1}^{n} y_{ij}\,\|X_i - z_j\|^2 \\
\text{subject to} \quad & \sum_{j=1}^{k} y_{ij} = 1 \quad \forall i\\
& y_{ij} \in \{0, 1\}
\end{aligned}
$$

This is a **mixed-integer nonlinear program**. Mixed-integer programs are
generally NP-hard. But it has gorgeous block structure.

### 4.2 The two blocks

- **Block Z** = the $k \cdot d$ continuous prototype values $\{z_1, \ldots, z_k\}$.
- **Block Y** = the $n \cdot k$ binary assignment values $\{y_{ij}\}$.

**Subproblem 1: fix Z, optimize Y.** The objective decomposes across points:
each point independently picks the cluster $j$ that minimizes
$\|X_i - z_j\|^2$. The constraint $\sum_j y_{ij} = 1$ trivializes — one-hot
on the nearest centroid, done.

**Subproblem 2: fix Y, optimize Z.** The objective decomposes across clusters.
The portion of the cost contributed by cluster $j$ is

$$
O_j \;=\; \sum_{i=1}^n y_{ij}\,\|X_i - z_j\|^2
$$

Take the gradient with respect to $z_j$ and set it to zero (eq. 4.76 in the book):

$$
\frac{\partial O_j}{\partial z_j} \;=\; 2 \sum_{i=1}^n y_{ij}\,(X_i - z_j) \;=\; 0
$$

Points with $y_{ij} = 0$ drop out, so this collapses to:

$$
z_j \;=\; \frac{1}{|\{i : y_{ij} = 1\}|}\;\sum_{i:\,y_{ij}=1} X_i
$$

i.e. the **mean** of the points currently assigned to cluster $j$. The famous
"centroid update" is just the closed-form block minimizer.

So the standard $k$-means algorithm — *assign, recompute, assign, recompute* —
is literally BCD with the assignment block and the centroid block alternating.

### 4.3 A worked numerical example

Six points in 2D and $k = 2$ clusters:

```
P1 = (1, 1)    P2 = (2, 1)    P3 = (1, 2)
P4 = (8, 8)    P5 = (9, 8)    P6 = (8, 9)
```

Two clear clusters (lower-left and upper-right). We will start with **bad**
initial centroids to make the iteration interesting:

$$z_1^{(0)} = (3, 7), \qquad z_2^{(0)} = (6, 3)$$

**Iteration 1 — assign points (block Y):** compute Euclidean distance from
each point to each centroid:

| point | $d(\cdot, z_1)$ | $d(\cdot, z_2)$ | nearest |
|---|---|---|---|
| P1 (1,1) | $\sqrt{4+36} = 6.32$ | $\sqrt{25+4} = 5.39$ | **2** |
| P2 (2,1) | $\sqrt{1+36} = 6.08$ | $\sqrt{16+4} = 4.47$ | **2** |
| P3 (1,2) | $\sqrt{4+25} = 5.39$ | $\sqrt{25+1} = 5.10$ | **2** |
| P4 (8,8) | $\sqrt{25+1} = 5.10$ | $\sqrt{4+25} = 5.39$ | **1** |
| P5 (9,8) | $\sqrt{36+1} = 6.08$ | $\sqrt{9+25} = 5.83$ | **2** |
| P6 (8,9) | $\sqrt{25+4} = 5.39$ | $\sqrt{4+36} = 6.32$ | **1** |

Cluster 1 = {P4, P6}; Cluster 2 = {P1, P2, P3, P5}. Notice how the bad
centroid placement put **P5 in the wrong cluster** — that is going to be the
interesting bit on iteration 2.

**Iteration 1 — recompute centroids (block Z):**

$$z_1^{(1)} = \tfrac{1}{2}\big((8,8) + (8,9)\big) = (8.0,\, 8.5)$$
$$z_2^{(1)} = \tfrac{1}{4}\big((1,1) + (2,1) + (1,2) + (9,8)\big) = (3.25,\, 3.0)$$

The centroids moved a lot. Now repeat.

**Iteration 2 — assign points:**

| point | $d(\cdot, z_1)$ | $d(\cdot, z_2)$ | nearest |
|---|---|---|---|
| P1 (1,1) | 10.26 | 3.01 | **2** |
| P2 (2,1) | 9.61 | 2.36 | **2** |
| P3 (1,2) | 9.55 | 2.46 | **2** |
| P4 (8,8) | 0.50 | 6.90 | **1** |
| P5 (9,8) | 1.12 | 7.62 | **1** ← *moved!* |
| P6 (8,9) | 0.50 | 7.65 | **1** |

**P5 has switched clusters.** Cluster 1 is now {P4, P5, P6}; cluster 2 is now
{P1, P2, P3}.

**Iteration 2 — recompute centroids:**

$$z_1^{(2)} = \tfrac{1}{3}\big((8,8)+(9,8)+(8,9)\big) = (8.333,\, 8.333)$$
$$z_2^{(2)} = \tfrac{1}{3}\big((1,1)+(2,1)+(1,2)\big) = (1.333,\, 1.333)$$

**Iteration 3 — assign points:** same assignments as iteration 2. **No change
→ converged.** The total cost has dropped at every step (this is guaranteed
by BCD because each block subproblem is solved to optimality).

### 4.4 The Python check

```python
import numpy as np

P = np.array([[1,1],[2,1],[1,2],[8,8],[9,8],[8,9]], dtype=float)
z = np.array([[3.0, 7.0], [6.0, 3.0]])

for it in range(10):
    # Block Y: assign each point to nearest centroid
    d = np.linalg.norm(P[:, None, :] - z[None, :, :], axis=2)
    assign = d.argmin(axis=1)
    # Block Z: recompute centroids as cluster means
    z_new = np.array([P[assign == j].mean(axis=0) for j in range(2)])
    if np.allclose(z, z_new):
        print(f"converged at iter {it+1}: z = {z_new.tolist()}")
        break
    z = z_new
# converged at iter 3: z = [[8.333..., 8.333...], [1.333..., 1.333...]]
```

The take-away: **K-means is not a quirky standalone algorithm.** It is BCD
on a mixed-integer program, and the reason it works is that both blocks
have closed-form minimizers (a sort and a mean). Knowing this lets you
recognize when other clustering / assignment problems can be solved the same
way.

---

## 5. Case study B — Alternating Least Squares for movie recommendations

This is the centerpiece. We will take the actual movie-ratings matrix from
**Figure 8.2** of Aggarwal's book and run BCD on it by hand.

### 5.1 The problem

Given an $n \times d$ matrix $D$ of user × item ratings, factorize it as
$D \approx U V^T$ where $U \in \mathbb{R}^{n \times k}$ ("user factors") and
$V \in \mathbb{R}^{d \times k}$ ("item factors"). The trick: most entries of
$D$ are *missing*, so we only sum the loss over the observed set
$S = \{(i,j) : x_{ij} \text{ is observed}\}$. With Tikhonov regularization the
objective is:

$$
\boxed{\;
\min_{U,V}\;\; J(U,V) \;=\;
\tfrac{1}{2}\!\!\sum_{(i,j) \in S}\!\Bigg(x_{ij} - \sum_{s=1}^{k} u_{is}\, v_{js}\Bigg)^{\!2}
\;+\;\tfrac{\lambda}{2}\big(\|U\|_F^2 + \|V\|_F^2\big)
\;}
$$

This is **non-convex in $(U, V)$ jointly** (you can verify by noting that
swapping the sign of any column of $U$ and the corresponding column of $V$
leaves $UV^T$ unchanged). But it is **bi-convex**: convex in $U$ alone, and
convex in $V$ alone.

That bi-convexity is exactly the BCD invitation. Two blocks: $U$ and $V$.

### 5.2 The block updates

**Subproblem: fix $V$, solve for each row $u_i$ of $U$.** Let $S_i = \{j : (i,j) \in S\}$
be the items user $i$ has rated, and let $V_{S_i}$ be the $|S_i| \times k$
sub-matrix of $V$ formed by stacking the rows of items in $S_i$. Let $x_{S_i}$
be the $|S_i|$-vector of user $i$'s observed ratings. Then user $i$'s factor
row solves a **ridge regression**:

$$
\min_{u_i}\;\; \tfrac{1}{2}\,\|x_{S_i} - V_{S_i}\, u_i^{T}\|^2 \;+\; \tfrac{\lambda}{2}\,\|u_i\|^2
$$

with closed-form solution

$$
\boxed{\;u_i \;=\; \big(V_{S_i}^{T}\, V_{S_i} \;+\; \lambda I_k\big)^{-1}\, V_{S_i}^{T}\, x_{S_i}\;}
$$

**Subproblem: fix $U$, solve for each row $v_j$ of $V$.** Symmetric — one
ridge regression per item.

That's the entire algorithm:

```
Initialize U and V randomly.
Repeat:
    For each user i:    u_i = (V_Si^T V_Si + λI)^-1 V_Si^T x_Si
    For each item j:    v_j = (U_Sj^T U_Sj + λI)^-1 U_Sj^T x_Sj
Until converged.
```

Aggarwal explicitly calls this "Alternating Least Squares" and identifies it
as block coordinate descent in §8.3.2.3. Notice that **each user's update is
independent** — you can solve all $n$ user updates in parallel, and similarly
for the items. This is one of the practical reasons ALS is preferred over
stochastic gradient descent in industrial recommender systems.

### 5.3 The data — Aggarwal's Figure 8.2

|       | GLAD | GODF | BENHUR | GOOD | SCAR | SPART |
|-------|:----:|:----:|:------:|:----:|:----:|:-----:|
| TOM   |  1   |  ·   |   ·    |  5   |  ·   |   2   |
| JIM   |  ·   |  5   |   ·    |  ·   |  4   |   ·   |
| JOE   |  5   |  3   |   ·    |  1   |  ·   |   ·   |
| ANN   |  ·   |  ·   |   3    |  ·   |  ·   |   4   |
| JILL  |  ·   |  ·   |   ·    |  3   |  5   |   ·   |
| SUE   |  5   |  ·   |   4    |  ·   |  ·   |   ·   |

(Movies: GLAD=Gladiator, GODF=Godfather, BENHUR=Ben-Hur, GOOD=Goodfellas,
SCAR=Scarface, SPART=Spartacus. Dots are missing.)

Six users, six movies, **only 16 of the 36 entries are observed** — about
44% known. Ratings are 1–5.

### 5.4 The factorization setup

Let's use rank $k = 2$ (two latent factors) and regularization $\lambda = 0.1$.
We need an initial $U$ and $V$. We'll fix the random seed so you can reproduce
the numbers exactly. Initialize from $\mathrm{Uniform}(0.1, 0.9)$ with
`np.random.seed(42)`:

$$
U^{(0)} \approx \begin{bmatrix}
0.4000 & 0.8610 \\
0.6860 & 0.5790 \\
0.2250 & 0.2250 \\
0.1460 & 0.7930 \\
0.5810 & 0.6660 \\
0.1160 & 0.8760
\end{bmatrix},
\qquad
V^{(0)} \approx \begin{bmatrix}
0.7660 & 0.2699 \\
0.2455 & 0.2467 \\
0.3434 & 0.5198 \\
0.4456 & 0.3330 \\
0.5895 & 0.2116 \\
0.3337 & 0.3931
\end{bmatrix}
$$

(These are the seed-42 values rounded to 4 decimal places. The Python script
in §5.7 uses full precision; expect agreement to about ±0.001.)

The starting reconstruction error $\tfrac{1}{2}\sum_{(i,j) \in S}(x_{ij} - \hat{x}_{ij})^2$
is **84.74** — terrible, as expected from a random init.

### 5.5 One BCD sweep, hand-computed for JOE

To keep this tractable on paper, we will walk through **just one row** of the
$U$-update — for user JOE — and trust the code for the rest. JOE has rated
three movies: GLAD (5), GODF (3), GOOD (1).

**Step 1: pull out the relevant rows of $V$.**
JOE rated movies GLAD (column 1), GODF (column 2), and GOOD (column 4), so:

$$
V_{S_{\text{JOE}}} \;=\;
\begin{bmatrix}
v_{\text{GLAD}}^{T} \\ v_{\text{GODF}}^{T} \\ v_{\text{GOOD}}^{T}
\end{bmatrix}
\;=\;
\begin{bmatrix}
0.7660 & 0.2699 \\
0.2455 & 0.2467 \\
0.4456 & 0.3330
\end{bmatrix},
\qquad
x_{S_{\text{JOE}}} = \begin{bmatrix} 5 \\ 3 \\ 1 \end{bmatrix}
$$

**Step 2: form the $k \times k$ ridge matrix $A = V_{S_{\text{JOE}}}^{T} V_{S_{\text{JOE}}} + \lambda I$.**
Compute the entries one at a time:

$$
(V^{T} V)_{11} = 0.7660^2 + 0.2455^2 + 0.4456^2 \approx 0.5868 + 0.0603 + 0.1986 = 0.8456
$$
$$
(V^{T} V)_{22} = 0.2699^2 + 0.2467^2 + 0.3330^2 \approx 0.0729 + 0.0609 + 0.1109 = 0.2446
$$
$$
(V^{T} V)_{12} = (0.7660)(0.2699) + (0.2455)(0.2467) + (0.4456)(0.3330) \approx 0.2068 + 0.0606 + 0.1484 = 0.4157
$$

So $V^{T} V \approx \begin{bmatrix} 0.8456 & 0.4157 \\ 0.4157 & 0.2446 \end{bmatrix}$.
Add $\lambda I = 0.1 \cdot I_2$:

$$
A \;\approx\; \begin{bmatrix} 0.9456 & 0.4157 \\ 0.4157 & 0.3446 \end{bmatrix}
$$

**Step 3: form the right-hand side $b = V_{S_{\text{JOE}}}^{T}\, x_{S_{\text{JOE}}}$.**

$$
b_1 = (0.7660)(5) + (0.2455)(3) + (0.4456)(1) = 3.8300 + 0.7365 + 0.4456 \approx 5.0121
$$
$$
b_2 = (0.2699)(5) + (0.2467)(3) + (0.3330)(1) = 1.3495 + 0.7401 + 0.3330 \approx 2.4226
$$

So $b \approx (5.0121,\; 2.4226)$.

**Step 4: solve the $2 \times 2$ system $A\, u_{\text{JOE}} = b$.** The
determinant of $A$ is $0.9456 \cdot 0.3446 - 0.4157^2 \approx 0.3259 - 0.1728 = 0.1530$.
By Cramer's rule:

$$
u_{\text{JOE}, 1} = \frac{(0.3446)(5.0121) - (0.4157)(2.4226)}{0.1530}
                  \approx \frac{1.7272 - 1.0074}{0.1530}
                  \approx 4.705
$$
$$
u_{\text{JOE}, 2} = \frac{(0.9456)(2.4226) - (0.4157)(5.0121)}{0.1530}
                  \approx \frac{2.2908 - 2.0835}{0.1530}
                  \approx 1.354
$$

So after the first $U$-update, JOE's latent factor row is

$$u_{\text{JOE}} \approx (4.705,\; 1.354)$$

That single row is what one block sub-problem looks like. The exact same
recipe runs for the other five users — independently, in parallel — and then
the same thing flips for $V$. The full $U$ after this single sweep:

$$
U^{(1)} \approx
\begin{bmatrix}
1.405 & 4.650 \\
5.021 & 5.590 \\
4.705 & 1.355 \\
3.527 & 3.887 \\
5.890 & 1.754 \\
4.731 & 3.625
\end{bmatrix}
$$

Notice the dramatic change from $U^{(0)}$. The starting random values were
absorbing none of the structure; in one block solve they have already snapped
into a much better configuration. This is BCD's "expensive per step, fewer
steps" advantage.

After also solving the $V$-update inside the same sweep, the reconstruction
error has dropped from **84.74 to 0.57** in a single BCD sweep.

### 5.6 Convergence and final predictions

Run 20 sweeps (each is one $U$-update followed by one $V$-update). The error
trajectory:

| sweep | reconstruction error |
|---|---|
| 0 (init) | 84.74 |
| 1 | 0.5727 |
| 5 | ~0.05 |
| 20 | 0.018 |

The **final** factor matrices:

$$
U^{(20)} \approx
\begin{bmatrix}
\phantom{-}0.865 & \phantom{-}2.493 \\
\phantom{-}2.189 & \phantom{-}2.017 \\
\phantom{-}2.757 & \phantom{-}0.219 \\
\phantom{-}2.290 & \phantom{-}0.116 \\
\phantom{-}2.862 & \phantom{-}1.251 \\
\phantom{-}2.687 & -0.693
\end{bmatrix}
\quad
V^{(20)} \approx
\begin{bmatrix}
\phantom{-}1.805 & -0.219 \\
\phantom{-}0.986 & \phantom{-}1.374 \\
\phantom{-}1.342 & -0.473 \\
\phantom{-}0.218 & \phantom{-}1.899 \\
\phantom{-}1.600 & \phantom{-}0.267 \\
\phantom{-}1.705 & \phantom{-}0.209
\end{bmatrix}
$$

The full **predicted** ratings matrix $\hat{D} = U V^T$:

|       |  GLAD | GODF  | BENHUR | GOOD  | SCAR  | SPART |
|-------|------:|------:|-------:|------:|------:|------:|
| TOM   | **1.02** | 4.28  | -0.02  | **4.92** | 2.05  | **2.00** |
| JIM   |  3.51 | **4.93** |  1.98 |  4.31 | **4.04** |  4.15 |
| JOE   | **4.93** | **3.02** |  3.60 | **1.02** |  4.47 |  4.75 |
| ANN   |  4.11 |  2.42 | **3.02** |  0.72 |  3.69 | **3.93** |
| JILL  |  4.89 |  4.54 |  3.25 | **3.00** | **4.91** |  5.14 |
| SUE   | **5.00** |  1.70 | **3.94** | -0.73 |  4.12 |  4.44 |

**Bold** entries are the originally observed ones — they all reproduce
within ±0.1 of their true values. The non-bold entries are the **predictions
ALS makes for unrated movies**. For example:

- **TOM has not rated Godfather → predicted 4.28.** ALS thinks Tom would
  probably enjoy it. (Looking at the data: Tom liked Goodfellas and Spartacus
  / disliked Gladiator; the model has discovered that he favors mob/period
  drama over Roman epics.)
- **TOM has not rated Scarface → predicted 2.05.** Mild dislike — consistent
  with his low rating for Gladiator.
- **TOM has not rated Ben-Hur → predicted -0.02.** Negative is, of course,
  outside the 1–5 rating scale; this is the model honestly reporting that
  *it has very little signal* for Tom on Ben-Hur. Only two other users (Ann
  and Sue) have rated Ben-Hur, and Tom's profile doesn't strongly resemble
  either of them. Negative predictions like this are a known artifact of
  unconstrained ALS on sparse data and are usually clipped to the rating
  range $[1, 5]$ at serving time.

That negative prediction is actually pedagogically valuable: it shows that
**BCD optimizes the in-sample objective (reconstruction error on the 16
observed entries)**, not the out-of-sample plausibility of the 20 missing
entries. Good in-sample fit ≠ good predictions everywhere. Section 6 below
shows how a real research paper handles this exact issue.

### 5.7 The Python check

```python
import numpy as np

# 6x6 movie ratings matrix from Aggarwal Figure 8.2
nan = np.nan
D = np.array([
    [1,    nan, nan, 5,   nan, 2  ],  # TOM
    [nan,  5,   nan, nan, 4,   nan],  # JIM
    [5,    3,   nan, 1,   nan, nan],  # JOE
    [nan,  nan, 3,   nan, nan, 4  ],  # ANN
    [nan,  nan, nan, 3,   5,   nan],  # JILL
    [5,    nan, 4,   nan, nan, nan],  # SUE
])

n, d, k, lam = 6, 6, 2, 0.1
np.random.seed(42)
U = np.random.uniform(0.1, 0.9, size=(n, k))
V = np.random.uniform(0.1, 0.9, size=(d, k))

def error(D, U, V):
    pred, mask = U @ V.T, ~np.isnan(D)
    return 0.5 * np.sum((D[mask] - pred[mask])**2)

print(f"init error = {error(D, U, V):.4f}")

for sweep in range(20):
    # Block 1: update each row of U holding V fixed
    for i in range(n):
        obs = ~np.isnan(D[i])
        Vo, xo = V[obs], D[i, obs]
        U[i] = np.linalg.solve(Vo.T @ Vo + lam * np.eye(k), Vo.T @ xo)
    # Block 2: update each row of V holding U fixed
    for j in range(d):
        obs = ~np.isnan(D[:, j])
        Uo, xo = U[obs], D[obs, j]
        V[j] = np.linalg.solve(Uo.T @ Uo + lam * np.eye(k), Uo.T @ xo)
    if sweep in (0, 4, 19):
        print(f"after sweep {sweep+1}: error = {error(D, U, V):.4f}")

print("\nPredicted ratings (U @ V.T):")
print(np.round(U @ V.T, 2))
```

Run that and you should see:
```
init error = 84.7432
after sweep 1: error = 0.5727
after sweep 5: error = 0.0469
after sweep 20: error = 0.0181
```

**That is BCD on real data in 25 lines of code.**

---

## 6. Case study C — A real research paper using BCD

The two examples above are from a textbook. To show that BCD is not just
pedagogy, here is how a 2018 JMLR paper uses it as its workhorse algorithm.

### 6.1 The paper

Bertsimas, Pawlowski & Zhuo (2018), *"From Predictive Methods to Missing
Data Imputation: An Optimization Approach"*, **JMLR 18:1–39**
(`17-073.pdf` in this folder).

### 6.2 The problem

Given a data matrix $X$ with some entries missing, **fill in the missing
values** so that downstream machine-learning models trained on the imputed
data perform well. The authors pose this as an optimization problem (their
eq. 1):

$$
\min_{U, W, V}\;\; c(U, W, V; X)
\quad
\text{s.t.}\;\; w_{id} = x_{id}\;\; \forall (i,d) \in \mathcal{N}_0,
\;\; v_{id} = x_{id}\;\; \forall (i,d) \in \mathcal{N}_1,
\;\; (U, W, V) \in \mathcal{U}
$$

where $W$ holds the imputed continuous values, $V$ holds the imputed
categorical values, and $U$ is **auxiliary** model state — it could be a
$K$-NN assignment matrix $Z$, SVM coefficients $(\beta, \theta)$, or a
decision-tree leaf-membership matrix $T$, depending on which predictive
model defines the cost $c(\cdot)$.

The constraint $w_{id} = x_{id}$ on observed entries pins those values down;
only the missing entries are free variables.

This problem is **non-convex** for all three model choices (KNN, SVM, trees).
A direct mixed-integer formulation would be slow to solve. So the authors
use BCD.

### 6.3 The algorithm — Algorithm 1, *opt.impute*

The blocks are:
- **Block 1**: the auxiliary model state $U$ (e.g., the binary KNN assignment
  matrix $Z$, or the SVM weights, or the tree structure).
- **Block 2**: the imputation matrices $(W, V)$.

The full BCD loop (their Algorithm 1, p. 8):

```
Initialize W, V from a warm start (e.g., mean impute).
Repeat:
    Step 1: Fix (W, V). Update U to optimize the model under
            current imputation. (E.g., compute K nearest neighbors
            of each row, or fit an SVM, or fit a tree.)

    Step 2a (BCD variant):
            Fix U. Update all of (W, V) jointly to minimize
            c(U, W, V; X) subject to the observed-entry constraints.

    or

    Step 2b (CD variant):
            Fix U. Update one missing entry w_{jr} or v_{jr}
            at a time.
Until cost decrease < δ_0.
```

**This is exactly the same BCD pattern as ALS** — alternate between two
groups, fully solve inside each. The difference is that the inner solves are
no longer pure ridge regressions, because the cost function depends on which
predictive model you chose.

### 6.4 The KNN closed-form update — eq. (10)

For the KNN imputation model, the inner update for a single missing
continuous entry $w_{id}$ (the CD variant) has a beautifully clean closed
form (their eq. 10, p. 11):

$$
\boxed{\;
w_{id} \;=\;
\frac{\sum_{j=1}^{n} z_{ij}\, w_{jd} \;+\; \sum_{j \in I} z_{ji}\, w_{jd}}
     {K \;+\; \sum_{j \in I} z_{ji}}
\;}
$$

In English: **the imputed value for cell $(i, d)$ is a weighted average over
two groups of users** — the $K$ nearest neighbors of user $i$ (the
$z_{ij}$ term in the numerator), *plus* every other user $j$ that has user
$i$ as one of *their* neighbors (the $z_{ji}$ term). The denominator is the
total weight.

Compare this to the ALS user-update in §5.2: in both cases, BCD has reduced
a hairy non-convex global problem to a one-shot weighted average / linear
solve.

### 6.5 The empirical lesson — when BCD beats CD

The Bertsimas paper benchmarks BCD against CD on **84 real datasets** from
the UCI Machine Learning Repository. The headline result on the choice of
descent method (p. 30–31, paraphrased):

> "In general, the BCD algorithm for opt.knn did not significantly improve
> upon imputation accuracy compared to the CD algorithm, but only improved
> upon speed. ... when $n \leq 10{,}000$ we recommend running both methods
> and selecting the imputation which yields the lowest objective value;
> when $n$ is larger, run only coordinate descent because BCD slows down."

This is the practical lesson hiding inside the textbook math: **BCD is faster
per iteration when each block has cheap structure, but at very large $n$ the
linear systems inside each block become expensive and CD's per-entry updates
catch up.** Their Table 9 (p. 32) shows the crossover concretely: on the
245k-row `skin-segmentation` dataset, the BCD variant fails to converge in 4
hours while CD finishes in ~25 minutes.

The opt.impute method in this paper improves out-of-sample classification
accuracy by **1.7% on average at 50% missing data** over the best
benchmark imputation method, across 5 classification tasks — the kind of
improvement that BCD's clean optimization framing enables.

---

## 7. When to reach for BCD — a decision checklist

Use BCD when **two or more** of these are true:

1. **The variables naturally split into groups.** Examples: $(U, V)$ in
   matrix factorization; $(z_j, y_{ij})$ in $k$-means; $(\beta, w)$ in
   semi-supervised learning where $\beta$ is model parameters and $w$ is
   imputed labels.
2. **Each group's sub-problem has a closed form (or a fast solver).**
   Ridge regression, sorting, taking a mean, solving a small linear system —
   these are all good signs.
3. **The constraints decouple across groups.** If the constraint $\sum_j y_{ij} = 1$
   only involves one group's variables, BCD will trivialize it inside the
   block update.
4. **You can parallelize the per-block solves.** ALS user-updates are
   independent — you can fan them out across cores or machines, which
   stochastic gradient descent cannot do as cleanly.
5. **The full problem is non-convex but bi-convex (or multi-convex).** BCD
   gives you a guaranteed monotonic decrease in cost on multi-convex
   problems, even though convergence to a *global* minimum is not guaranteed.

**Avoid BCD when:**

- The Hessian is dense and has no block structure, so the inner sub-problem
  is no easier than the full problem. (In that case stick with gradient
  descent or quasi-Newton.)
- The blocks are tightly coupled — e.g. a single shared scalar appears in
  every constraint. (BCD will zig-zag and converge slowly.)
- The data is so large that even one block solve is too expensive. (Use
  stochastic gradient descent or coordinate descent — sample one entry, not
  one block, per step.)

---

## Quick reference card

| concept | formula | one-line meaning |
|---|---|---|
| coordinate descent update (least-squares) | $w_i \leftarrow w_i + (d_i^T r) / \|d_i\|^2$ | exact 1-D minimizer along axis $i$ |
| BCD update | $x_b^{(k+1)} = \arg\min_{x_b} f(\ldots, x_b, \ldots)$ | exact minimizer inside one block |
| K-means centroid step | $z_j = \mathrm{mean}\{X_i : y_{ij} = 1\}$ | closed-form Z-update with Y fixed |
| K-means assignment step | $y_{ij} = \mathbf{1}[j = \arg\min_{j'} \|X_i - z_{j'}\|]$ | closed-form Y-update with Z fixed |
| ALS user-update | $u_i = (V_{S_i}^T V_{S_i} + \lambda I)^{-1} V_{S_i}^T x_{S_i}$ | ridge regression per user |
| ALS item-update | $v_j = (U_{S_j}^T U_{S_j} + \lambda I)^{-1} U_{S_j}^T x_{S_j}$ | ridge regression per item |
| opt.impute KNN entry update | $w_{id} = (\sum z_{ij} w_{jd} + \sum z_{ji} w_{jd}) / (K + \sum z_{ji})$ | weighted average of neighbors |

---

*If you got this far: you can now look at any "alternate between two
groups of variables" algorithm in the wild — EM, ALS, k-means, mean-shift,
many GANs, half of the deep-learning lore — and recognize the BCD skeleton
underneath.*
