#!/usr/bin/env python
# coding: utf-8
"""
Generalization of Compute_x_and_y.py: produce CovarianceSolver objects
from noise PSD data, supporting stationary, Toeplitz+low-rank, and
piecewise-stationary noise models.
"""
import numpy as np
from scipy.linalg import solve_toeplitz
from covariance_solver import ToeplitzSolver, WoodburySolver, BlockDiagonalToeplitzSolver, make_solver


def acf_from_psd(psd_data, fs):
    """Convert one-sided PSD to autocorrelation function (ACF)."""
    delta_t = 1.0 / fs
    return 0.5 * np.real(np.fft.irfft(psd_data)) / delta_t


def toeplitz_solver_from_acf(acf):
    """Build a ToeplitzSolver from a 1D ACF array."""
    e1 = np.zeros(len(acf))
    e1[0] = 1.0
    x = solve_toeplitz((acf, acf), e1)
    y = x[::-1]
    return ToeplitzSolver(x, y)


def make_solver_from_psd(psd_data, fs, duration, backend='toeplitz'):
    """
    Build a CovarianceSolver from a PSD array.

    Parameters
    ----------
    psd_data : 1D array
        One-sided PSD evaluated at rfft frequencies.
    fs : float
        Sampling frequency in Hz.
    duration : float
        Duration of the data segment in seconds.
    backend : str
        'toeplitz' (default): GS representation — exact for stationary noise.
        'cholesky': dense Cholesky — small N only, for testing.

    Returns
    -------
    CovarianceSolver
    """
    N = int(duration * fs)
    acf = acf_from_psd(psd_data, fs)[:N]

    if backend == 'toeplitz':
        return toeplitz_solver_from_acf(acf)
    elif backend == 'cholesky':
        from scipy.linalg import toeplitz
        C = toeplitz(acf)
        return make_solver('cholesky', C=C)
    else:
        raise ValueError(f"Unsupported backend for PSD-based solver: {backend!r}")


def make_solver_from_matrix(C, toeplitz_acf=None, backend='cholesky',
                             pcg_tol=1e-8, pcg_maxiter=200):
    """
    Build a CovarianceSolver from an explicit dense covariance matrix.

    Parameters
    ----------
    C : ndarray, shape (N, N)
        Full symmetric positive definite covariance matrix.
    toeplitz_acf : 1D array or None
        ACF of the nearest-stationary Toeplitz approximation to C, used as
        PCG preconditioner when backend='pcg'.
    backend : str
        'cholesky': exact factorization (only feasible for N ≲ 10^4).
        'pcg': iterative with Toeplitz preconditioner (requires toeplitz_acf).

    Returns
    -------
    CovarianceSolver
    """
    N = C.shape[0]
    if backend == 'cholesky':
        return make_solver('cholesky', C=C)
    elif backend == 'pcg':
        if toeplitz_acf is None:
            raise ValueError("toeplitz_acf required for PCG backend")
        precond = toeplitz_solver_from_acf(toeplitz_acf[:N])
        return make_solver('pcg', C_matvec=lambda v: C @ v, preconditioner=precond,
                           N=N, tol=pcg_tol, maxiter=pcg_maxiter)
    else:
        raise ValueError(f"Unknown backend: {backend!r}")


def make_woodbury_solver(toeplitz_acf, U, D):
    """
    Build a WoodburySolver for C = T + U diag(D) U^T.

    Parameters
    ----------
    toeplitz_acf : 1D array
        ACF of the stationary Toeplitz part T.
    U : ndarray, shape (N, r)
        Low-rank perturbation basis.
    D : ndarray, shape (r,)
        Diagonal of the perturbation amplitude matrix (positive).

    Returns
    -------
    WoodburySolver
    """
    ts = toeplitz_solver_from_acf(toeplitz_acf[:U.shape[0]])
    return WoodburySolver(ts, U, D)


def make_woodbury_solver_glitch(toeplitz_acf, t_array, glitch_start, glitch_end,
                                 glitch_sigma, rank=1):
    """
    Convenience wrapper: build WoodburySolver for a Tukey-windowed glitch.

    The glitch covariance is modeled as rank-`rank` using Tukey-windowed
    basis vectors localised to [glitch_start, glitch_end].

    Parameters
    ----------
    toeplitz_acf : 1D array
    t_array : 1D array of time stamps
    glitch_start, glitch_end : float
        Time range of the glitch (same units as t_array).
    glitch_sigma : float or array of length `rank`
        RMS amplitude(s) of the glitch basis components.
    rank : int

    Returns
    -------
    WoodburySolver
    """
    from scipy.signal import tukey

    mask = (t_array >= glitch_start) & (t_array <= glitch_end)
    idx = np.where(mask)[0]
    n_glitch = len(idx)
    N = len(t_array)

    glitch_sigma = np.broadcast_to(glitch_sigma, (rank,)).copy().astype(float)

    U = np.zeros((N, rank))
    win = tukey(n_glitch, alpha=0.5)
    for k in range(rank):
        u = np.zeros(N)
        # Shift the window slightly for each basis component (crude orthogonalisation)
        shifted = np.roll(win, k * (n_glitch // (rank + 1)))
        u[idx] = shifted
        norm = np.linalg.norm(u)
        if norm > 0:
            u /= norm
        U[:, k] = u

    D = glitch_sigma ** 2
    return make_woodbury_solver(toeplitz_acf, U, D)


def make_solver_piecewise_stationary(psd_list, fs, segment_lengths):
    """
    Build a BlockDiagonalToeplitzSolver for piecewise-stationary noise.

    Parameters
    ----------
    psd_list : list of 1D arrays
        One-sided PSD for each segment (evaluated at rfft frequencies).
    fs : float
        Sampling frequency.
    segment_lengths : list of int
        Number of samples in each segment.

    Returns
    -------
    BlockDiagonalToeplitzSolver
    """
    solvers = []
    slices = []
    offset = 0
    for psd, N_k in zip(psd_list, segment_lengths):
        duration_k = N_k / fs
        solver_k = make_solver_from_psd(psd, fs, duration_k, backend='toeplitz')
        solvers.append(solver_k)
        slices.append(slice(offset, offset + N_k))
        offset += N_k
    return BlockDiagonalToeplitzSolver(solvers, slices)
