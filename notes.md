# Mathematical Notes: Generic Covariance Matrix Extensions

## 1. Current Method: Gohberg-Semencul for Symmetric Toeplitz

Let $C$ be an $N\times N$ symmetric positive definite Toeplitz matrix with first column
$\mathbf{c} = [c_0, c_1, \ldots, c_{N-1}]^\top$. Define $\mathbf{e}_1 = [1, 0, \ldots, 0]^\top$.

Solve the Toeplitz system

$$C \mathbf{x} = \mathbf{e}_1 \implies \mathbf{x} = C^{-1}\mathbf{e}_1,$$

and set $\mathbf{y} = J\mathbf{x}$ where $J$ is the exchange matrix, i.e.\ $y_i = x_{N-1-i}$.

**Gohberg-Semencul theorem.** Let $L(\mathbf{v})$ denote the lower-triangular Toeplitz matrix
with first column $\mathbf{v}$. Then

$$C^{-1} = \frac{1}{x_0}\Bigl[L(\mathbf{x})L(\mathbf{x})^\top - L(J\mathbf{y})L(J\mathbf{y})^\top\Bigr].$$

For any vector $\mathbf{v}$:

$$C^{-1}\mathbf{v} = \frac{1}{x_0}\Bigl[L(\mathbf{x})\bigl(L(\mathbf{x})^\top \mathbf{v}\bigr) - L(J\mathbf{y})\bigl(L(J\mathbf{y})^\top \mathbf{v}\bigr)\Bigr].$$

Each $L(\cdot)^\top \mathbf{v}$ is a correlation (upper-triangular Toeplitz-vector product) computed
via FFT in $O(N\log N)$. Each $L(\cdot)\mathbf{w}$ is a convolution, also $O(N\log N)$.
**Total cost per apply: $O(N\log N)$.**

### Displacement identity underlying GS

Define the lower-shift matrix $(Z_1)_{ij} = \delta_{i,j+1}$. For any Toeplitz matrix $C$:

$$Z_1 C - C Z_1^\top = \mathbf{g}_1 \mathbf{h}_1^\top - \mathbf{g}_2 \mathbf{h}_2^\top$$

where $\mathbf{g}_1, \mathbf{h}_1, \mathbf{g}_2, \mathbf{h}_2$ depend only on the first row/column of $C$.
For *symmetric* Toeplitz, this collapses to rank 1. This rank-1 displacement means $C^{-1}$
requires only 2 vectors to represent.

---

## 2. Displacement Rank Theory

**Definition.** For square matrices $A$, $B$, the $(A,B)$-displacement of $C$ is

$$\nabla_{A,B}(C) := AC - CB.$$

The **displacement rank** of $C$ w.r.t.\ $(A,B)$ is $\rho = \operatorname{rank}\bigl(\nabla_{A,B}(C)\bigr)$.
Write $\nabla_{A,B}(C) = GH^\top$ with $G,H \in \mathbb{R}^{N\times\rho}$.

**Inversion theorem (Gohberg-Kailath-Van Loan).** If $C$ is invertible and $AC - CB = GH^\top$, then

$$\nabla_{B,A}(C^{-1}) = -(C^{-1}G)(H^\top C^{-1})^\top,$$

so $C^{-1}$ has displacement rank $\leq \rho$ w.r.t.\ $(B,A)$, representable by $\rho$ vector pairs.

For symmetric Toeplitz: $\rho = 1$, giving exactly the GS formula.
For Toeplitz + rank-$r$ perturbation: $\rho \leq 1+r$, so $C^{-1}$ requires at most $1+r$ vector pairs.
The Woodbury formula (Section 3) provides a cleaner implementation of this.

---

## 3. Woodbury Extension: Toeplitz + Low-Rank

Let

$$C = T + UDU^\top,$$

where:
- $T$ is $N\times N$ symmetric Toeplitz with GS representation $(\mathbf{x}, \mathbf{y})$
- $U \in \mathbb{R}^{N\times r}$ with $r \ll N$
- $D \in \mathbb{R}^{r\times r}$ positive definite diagonal

**Woodbury identity:**

$$C^{-1} = T^{-1} - T^{-1}U\underbrace{\left(D^{-1} + U^\top T^{-1}U\right)^{-1}}_{M^{-1}\,\in\,\mathbb{R}^{r\times r}}U^\top T^{-1}.$$

### Precomputation (once per run)

1. Compute $W = T^{-1}U \in \mathbb{R}^{N\times r}$: apply `ToeplitzSolver` to each column of $U$.
   **Cost: $O(rN\log N)$.**

2. Compute $M = D^{-1} + U^\top W \in \mathbb{R}^{r\times r}$:
   - $U^\top W$: $O(r^2 N)$ matrix product.
   - Add $D^{-1}$: $O(r)$.

3. Factorize $M$ (Cholesky, since $M \succ 0$): $O(r^3)$, negligible for $r \ll N$.

### Application to vector $\mathbf{v}$

$$C^{-1}\mathbf{v} = T^{-1}\mathbf{v} - W\,M^{-1}(U^\top T^{-1}\mathbf{v})$$

Step-by-step:
1. $\mathbf{w}_1 = T^{-1}\mathbf{v}$: one GS apply. **$O(N\log N)$.**
2. $\boldsymbol{\alpha} = U^\top \mathbf{w}_1 \in \mathbb{R}^r$: **$O(rN)$.**
3. $\boldsymbol{\beta} = M^{-1}\boldsymbol{\alpha}$: triangular solve, **$O(r^2)$.**
4. $C^{-1}\mathbf{v} = \mathbf{w}_1 - W\boldsymbol{\beta}$: **$O(rN)$.**

**Total per apply: $O(N\log N) + O(rN)$.** For $r \sim 10$–$50$, the $O(rN)$ term is negligible.

### Constructing $U$ for a time-domain glitch

A glitch occupying time indices $[t_s, t_e]$ contributes excess covariance in the rows/columns
corresponding to those indices. Model the glitch template as a rank-$r$ matrix:

$$C_{\text{glitch}} = \sum_{k=1}^{r} \sigma_k^2\, \mathbf{u}_k \mathbf{u}_k^\top,$$

where $\mathbf{u}_k$ are windowed (e.g.\ Tukey-windowed) basis vectors supported on $[t_s, t_e]$,
and $\sigma_k^2$ are the corresponding variances (fit from data). Then $U = [\mathbf{u}_1,\ldots,\mathbf{u}_r]$
and $D = \operatorname{diag}(\sigma_1^2,\ldots,\sigma_r^2)$.

### Constructing $U$ for a spectral line at $f_0$

A monochromatic line at frequency $f_0$ contributes a rank-2 perturbation:

$$U = \frac{1}{\sqrt{N}}\begin{bmatrix}\cos(2\pi f_0 t_0) & \sin(2\pi f_0 t_0)\\ \vdots & \vdots \\ \cos(2\pi f_0 t_{N-1}) & \sin(2\pi f_0 t_{N-1})\end{bmatrix}, \quad D = \sigma_{\text{line}}^2 I_2.$$

---

## 4. Block-Diagonal Toeplitz: Piecewise Stationary Noise

Partition $\{0,\ldots,N-1\}$ into $K$ non-overlapping segments $\mathcal{S}_1,\ldots,\mathcal{S}_K$ of lengths
$N_1,\ldots,N_K$ with $\sum_k N_k = N$. If the noise PSD within segment $k$ is approximately
stationary, the covariance is

$$C = \operatorname{diag}(T_1, T_2, \ldots, T_K),$$

where $T_k$ is $N_k\times N_k$ symmetric Toeplitz with ACF $\mathbf{r}_k$.

Then $C^{-1} = \operatorname{diag}(T_1^{-1}, \ldots, T_K^{-1})$, and for $\mathbf{v} = [\mathbf{v}_1;\ldots;\mathbf{v}_K]$:

$$C^{-1}\mathbf{v} = [T_1^{-1}\mathbf{v}_1;\, T_2^{-1}\mathbf{v}_2;\, \ldots;\, T_K^{-1}\mathbf{v}_K].$$

Each $T_k^{-1}\mathbf{v}_k$ is a GS apply with vectors $(\mathbf{x}_k, \mathbf{y}_k)$ computed from $\mathbf{r}_k$.

**Precomputation cost:** $O\!\left(\sum_k N_k \log N_k\right) \leq O(N\log N)$.

**Per-apply cost:** $O\!\left(\sum_k N_k \log N_k\right) = O(N\log N)$.

**Note on boundary effects:** This approximation ignores correlations between adjacent segments.
It is exact when segments are separated by gaps longer than the noise correlation length $\xi$
(where $\xi \sim 1/f_{\min}$ for a highpass-filtered detector).

---

## 5. Preconditioned Conjugate Gradient (PCG)

For a generic SPD matrix $C$ with available matvec $\mathbf{v} \mapsto C\mathbf{v}$, solve
$C\mathbf{x} = \mathbf{b}$ via PCG with preconditioner $M \approx C$.

**Choice of $M$:** Take $M = T$ (the Toeplitz matrix from the stationary approximation to $C$,
e.g.\ using the time-averaged PSD). Then $M^{-1}\mathbf{v}$ is a GS apply.

**PCG algorithm:** Initialize $\mathbf{x}_0 = \mathbf{0}$, $\mathbf{r}_0 = \mathbf{b}$,
$\mathbf{z}_0 = M^{-1}\mathbf{r}_0$, $\mathbf{p}_0 = \mathbf{z}_0$. Iterate:

$$\alpha_k = \frac{\mathbf{r}_k^\top \mathbf{z}_k}{\mathbf{p}_k^\top C\mathbf{p}_k}, \quad
\mathbf{x}_{k+1} = \mathbf{x}_k + \alpha_k \mathbf{p}_k, \quad
\mathbf{r}_{k+1} = \mathbf{r}_k - \alpha_k C\mathbf{p}_k,$$

$$\mathbf{z}_{k+1} = M^{-1}\mathbf{r}_{k+1}, \quad
\beta_k = \frac{\mathbf{r}_{k+1}^\top \mathbf{z}_{k+1}}{\mathbf{r}_k^\top \mathbf{z}_k}, \quad
\mathbf{p}_{k+1} = \mathbf{z}_{k+1} + \beta_k \mathbf{p}_k.$$

Each iteration: one $C$-matvec + one $M^{-1}$-apply $= O(N\log N) + O(N\log N)$.

**Convergence rate:** PCG converges in at most $k^*$ iterations satisfying

$$\frac{\|\mathbf{x}_{k^*} - \mathbf{x}\|_{C}}{\|\mathbf{x}_0 - \mathbf{x}\|_{C}} \leq 2\left(\frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}\right)^{k^*},$$

where $\kappa = \lambda_{\max}(M^{-1}C)/\lambda_{\min}(M^{-1}C)$ is the condition number of the
preconditioned system. If $M \approx C$ (good preconditioner), $\kappa \approx 1$ and
$k^* \sim O(1)$–$O(10)$.

---

## 6. Summary Data with Generic $C$

The summary data (Eq. 22 of the paper) are:

$$A_0(b) = \sum_{j\in b} \bigl(C^{-1}\mathbf{d}\bigr)_j\, h^0_j, \qquad
A_1(b) = \sum_{j\in b} \bigl(C^{-1}\mathbf{d}\bigr)_j\, h^0_j\,(t_j - t_c),$$

$$B_0(b_1, b_2) = \sum_{i\in b_1}\sum_{j\in b_2} h^0_i\,(C^{-1})_{ij}\,h^0_j,\qquad
B_1(b_1, b_2) = \sum_{i\in b_1}\sum_{j\in b_2} h^0_i\,(C^{-1})_{ij}\,h^0_j\,(t_j-t_c),$$

$$B_2(b_1, b_2) = \sum_{i\in b_1}\sum_{j\in b_2} (h^0_i)^*\,(C^{-1})_{ij}\,h^0_j,\qquad
B_3(b_1, b_2) = \sum_{i\in b_1}\sum_{j\in b_2} (h^0_i)^*\,(C^{-1})_{ij}\,h^0_j\,(t_j-t_c).$$

Defining the $N\times N_{\rm bins}$ matrix $H_0$ whose $b$-th column equals the fiducial waveform
$\mathbf{h}^0$ masked to bin $b$ (zero outside $b$), the $B$-terms become:

$$[B_0]_{b_1 b_2} = \bigl[H_0^\top C^{-1} H_0\bigr]_{b_1 b_2}, \qquad
[B_2]_{b_1 b_2} = \bigl[(H_0^*)^\top C^{-1} H_0\bigr]_{b_1 b_2}.$$

So all $B$-terms require $C^{-1}H_0$ and $C^{-1}H_0^*$, i.e.\ applying $C^{-1}$ to each of the
$N_{\rm bins}$ columns of $H_0$ (and $H_0^*$). All $A$-terms require $C^{-1}\mathbf{d}$ (one vector).

**These are the only places $C^{-1}$ appears in the summary data computation.**
Any backend that implements `apply_C_inv(V)` for a matrix $V$ (column-wise) is drop-in compatible.

---

## 7. Computational Complexity Summary

| Backend | Precompute | Per-apply (1 column) | Conditions for use |
|---|---|---|---|
| `ToeplitzSolver` | $O(N\log N)$ | $O(N\log N)$ | Stationary noise |
| `WoodburySolver` | $O(rN\log N)$ | $O(N\log N)+O(rN)$ | Low-rank perturbation, $r\ll N$ |
| `BlockDiagonalToeplitzSolver` | $O(N\log N)$ | $O(N\log N)$ | Piecewise stationary |
| `PCGSolver` | $O(N\log N)$ for $M$ | $O(kN\log N)$, $k$ iterations | Generic; good precond.\ available |
| `CholeskySolver` | $O(N^3)$ | $O(N^2)$ | Testing only; $N\lesssim 10^4$ |

For GW signals with $N\sim 10^5$–$10^6$, only the first four backends are computationally viable.
