#!/usr/bin/env python
# coding: utf-8
"""
Covariance solver backends for C^{-1} v computation.

All backends expose a single method: apply_C_inv(v) which computes C^{-1} v
for a 1D vector v or applies column-wise for a 2D matrix v.

See notes.md for mathematical derivations.
"""
import numpy as np
from scipy.linalg import matmul_toeplitz, cho_factor, cho_solve, solve_toeplitz
from scipy.sparse.linalg import cg, LinearOperator


class CovarianceSolver:
    """Abstract base: compute C^{-1} v for arbitrary v."""

    def apply_C_inv(self, v: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        v : ndarray, shape (N,) or (N, K)

        Returns
        -------
        ndarray, same shape as v: C^{-1} v (or C^{-1} applied column-wise)
        """
        raise NotImplementedError


class ToeplitzSolver(CovarianceSolver):
    """
    Gohberg-Semencul representation of C^{-1} for symmetric Toeplitz C.

    C^{-1} v = (1/x[0]) [ L(x) L(x)^T v - L(Jy) L(Jy)^T v ]

    where L(w) is the lower-triangular Toeplitz matrix with first column w,
    and J is the exchange (reversal) matrix.  Each L(·)^T v and L(·) w is
    a Toeplitz matrix-vector product computed via FFT in O(N log N).

    Parameters
    ----------
    x, y : 1D arrays
        Gohberg-Semencul vectors.  Obtain via
        ACF_noise_and_covariance_matrix_data/Compute_x_and_y.py or
        Compute_generic_solver.make_solver_from_psd(..., backend='toeplitz').
    """

    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = x
        self.y = y
        self._N = len(x)
        self._x0_inv = 1.0 / x[0]
        # Pre-build the constant row/column arrays used in matmul_toeplitz calls
        self._x_first = np.concatenate(([x[0]], np.zeros(self._N - 1)))
        self._y_shifted = np.concatenate(([0.0], y[:-1]))
        self._zeros = np.zeros(self._N)

    def apply_C_inv(self, v: np.ndarray) -> np.ndarray:
        x, y = self.x, self.y
        x0_inv = self._x0_inv
        xf = self._x_first
        ys = self._y_shifted
        zs = self._zeros

        # term1 = L(x) [ L(x)^T v ] = matmul_toeplitz((x, xf), matmul_toeplitz((xf, x), v))
        # term2 = L(Jy_shifted) [ L(Jy_shifted)^T v ]
        term1 = matmul_toeplitz((x, xf), matmul_toeplitz((xf, x), v))
        term2 = matmul_toeplitz((ys, zs), matmul_toeplitz((zs, ys), v))
        return x0_inv * (term1 - term2)


class WoodburySolver(CovarianceSolver):
    """
    C^{-1} for C = T + U D U^T, where T is Toeplitz and U D U^T is rank-r.

    Uses the Woodbury identity:
        C^{-1} v = T^{-1} v - W M^{-1} (U^T T^{-1} v)
    where W = T^{-1} U  and  M = D^{-1} + U^T W.

    Precomputes W and the Cholesky factor of M at construction time.
    Per-apply cost: O(N log N) + O(r N).

    Parameters
    ----------
    toeplitz_solver : ToeplitzSolver
    U : ndarray, shape (N, r)
    D : ndarray, shape (r,) — diagonal entries of D (all positive)
    """

    def __init__(self, toeplitz_solver: ToeplitzSolver, U: np.ndarray, D: np.ndarray):
        self.ts = toeplitz_solver
        self.U = U
        r = U.shape[1]

        # W = T^{-1} U  [N x r]
        self.W = self.ts.apply_C_inv(U)

        # M = D^{-1} + U^T W  [r x r]
        M = np.diag(1.0 / D) + U.T @ self.W
        self._M_cho = cho_factor(M)

    def apply_C_inv(self, v: np.ndarray) -> np.ndarray:
        w1 = self.ts.apply_C_inv(v)                    # T^{-1} v, O(N log N)
        alpha = self.U.T @ w1                           # U^T w1,  O(r N)
        beta = cho_solve(self._M_cho, alpha)            # M^{-1} alpha, O(r^2)
        return w1 - self.W @ beta                       # O(r N)


class BlockDiagonalToeplitzSolver(CovarianceSolver):
    """
    C^{-1} for block-diagonal C = diag(T_1, ..., T_K) where each T_k is Toeplitz.

    Parameters
    ----------
    solvers : list of ToeplitzSolver, length K
    segment_slices : list of slice, length K
        Each slice selects the rows/columns of a block from the full vector.
        E.g. [slice(0, N1), slice(N1, N1+N2), ...]
    """

    def __init__(self, solvers: list, segment_slices: list):
        self.solvers = solvers
        self.slices = segment_slices

    def apply_C_inv(self, v: np.ndarray) -> np.ndarray:
        out = np.empty_like(v)
        for solver, sl in zip(self.solvers, self.slices):
            out[sl] = solver.apply_C_inv(v[sl])
        return out


class CholeskySolver(CovarianceSolver):
    """
    C^{-1} via precomputed Cholesky factorization.

    Only feasible for N ≲ 10^4; intended for correctness testing.

    Parameters
    ----------
    C : ndarray, shape (N, N) — symmetric positive definite
    """

    def __init__(self, C: np.ndarray):
        self._cho = cho_factor(C)

    def apply_C_inv(self, v: np.ndarray) -> np.ndarray:
        return cho_solve(self._cho, v)


class PCGSolver(CovarianceSolver):
    """
    C^{-1} v via Preconditioned Conjugate Gradient.

    Parameters
    ----------
    C_matvec : callable (N,) -> (N,)
        Function computing C @ v.
    preconditioner : CovarianceSolver
        Approximate inverse of C; typically a ToeplitzSolver built from the
        stationary approximation.
    N : int
        Size of the system.
    tol : float
    maxiter : int
    """

    def __init__(self, C_matvec, preconditioner: CovarianceSolver, N: int,
                 tol: float = 1e-8, maxiter: int = 200):
        self._C_op = LinearOperator((N, N), matvec=C_matvec, dtype=np.float64)
        self._M_op = LinearOperator((N, N), matvec=preconditioner.apply_C_inv, dtype=np.float64)
        self._tol = tol
        self._maxiter = maxiter
        self._N = N

    def apply_C_inv(self, v: np.ndarray) -> np.ndarray:
        if v.ndim == 1:
            x, info = cg(self._C_op, v, M=self._M_op, tol=self._tol, maxiter=self._maxiter)
            if info != 0:
                raise RuntimeError(f"PCG did not converge (info={info})")
            return x
        # 2D: apply column-wise
        out = np.empty_like(v)
        for j in range(v.shape[1]):
            out[:, j] = self.apply_C_inv(v[:, j])
        return out


def make_solver(backend: str, **kwargs) -> CovarianceSolver:
    """
    Factory function for CovarianceSolver instances.

    Parameters
    ----------
    backend : str
        One of 'toeplitz', 'woodbury', 'block_diagonal', 'cholesky', 'pcg'.

    Keyword arguments (backend-specific):
    toeplitz:
        x, y : GS vectors
    woodbury:
        x, y : GS vectors for the Toeplitz part
        U    : (N, r) perturbation basis
        D    : (r,) perturbation diagonal (positive)
    block_diagonal:
        acfs            : list of 1D ACF arrays, one per segment
        segment_slices  : list of slice objects
    cholesky:
        C : (N, N) SPD matrix
    pcg:
        C_matvec      : callable
        preconditioner : CovarianceSolver (already built)
        N             : int
        tol, maxiter  : optional
    """
    if backend == 'toeplitz':
        return ToeplitzSolver(kwargs['x'], kwargs['y'])

    elif backend == 'woodbury':
        ts = ToeplitzSolver(kwargs['x'], kwargs['y'])
        return WoodburySolver(ts, kwargs['U'], kwargs['D'])

    elif backend == 'block_diagonal':
        solvers = []
        for acf in kwargs['acfs']:
            e1 = np.zeros(len(acf))
            e1[0] = 1.0
            x = solve_toeplitz((acf, acf), e1)
            y = x[::-1]
            solvers.append(ToeplitzSolver(x, y))
        return BlockDiagonalToeplitzSolver(solvers, kwargs['segment_slices'])

    elif backend == 'cholesky':
        return CholeskySolver(kwargs['C'])

    elif backend == 'pcg':
        return PCGSolver(
            kwargs['C_matvec'],
            kwargs['preconditioner'],
            kwargs['N'],
            tol=kwargs.get('tol', 1e-8),
            maxiter=kwargs.get('maxiter', 200),
        )

    else:
        raise ValueError(f"Unknown backend: {backend!r}")
