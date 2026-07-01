#!/usr/bin/env python
# coding: utf-8
"""
Unit tests for CovarianceSolver backends.

Run with:
    conda run -n <your_env> python -m pytest tests/test_covariance_solver.py -v
"""
import sys
import os
import numpy as np
import pytest
from scipy.linalg import toeplitz, solve_toeplitz

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ACF_noise_and_covariance_matrix_data'))
from covariance_solver import (
    ToeplitzSolver, WoodburySolver, BlockDiagonalToeplitzSolver,
    CholeskySolver, PCGSolver, make_solver,
)


RNG = np.random.default_rng(42)
N = 128


def _random_toeplitz_spd(n, decay=0.5):
    """Return a symmetric positive definite Toeplitz matrix."""
    acf = decay ** np.arange(n)
    return toeplitz(acf), acf


def _gs_vectors(acf):
    e1 = np.zeros(len(acf))
    e1[0] = 1.0
    x = solve_toeplitz((acf, acf), e1)
    y = x[::-1]
    return x, y


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reference_solve(C, v):
    return np.linalg.solve(C, v)


def assert_close(a, b, rtol=1e-6, atol=1e-8, label=""):
    err = np.max(np.abs(a - b))
    nrm = max(np.max(np.abs(b)), 1e-12)
    assert err / nrm < rtol or err < atol, \
        f"{label}: max error={err:.3e}, ref norm={nrm:.3e}"


# ---------------------------------------------------------------------------
# ToeplitzSolver
# ---------------------------------------------------------------------------

class TestToeplitzSolver:

    def test_1d_vector(self):
        C, acf = _random_toeplitz_spd(N)
        x, y = _gs_vectors(acf)
        solver = ToeplitzSolver(x, y)
        v = RNG.standard_normal(N)
        got = solver.apply_C_inv(v)
        ref = _reference_solve(C, v)
        assert_close(got, ref, label="ToeplitzSolver 1D")

    def test_2d_matrix(self):
        C, acf = _random_toeplitz_spd(N)
        x, y = _gs_vectors(acf)
        solver = ToeplitzSolver(x, y)
        V = RNG.standard_normal((N, 5))
        got = solver.apply_C_inv(V)
        ref = _reference_solve(C, V)
        assert_close(got, ref, label="ToeplitzSolver 2D")

    def test_make_solver_factory(self):
        _, acf = _random_toeplitz_spd(N)
        x, y = _gs_vectors(acf)
        solver = make_solver('toeplitz', x=x, y=y)
        assert isinstance(solver, ToeplitzSolver)


# ---------------------------------------------------------------------------
# WoodburySolver
# ---------------------------------------------------------------------------

class TestWoodburySolver:

    def _build(self, r=3):
        C_T, acf = _random_toeplitz_spd(N)
        x, y = _gs_vectors(acf)
        ts = ToeplitzSolver(x, y)
        U = RNG.standard_normal((N, r))
        D = np.abs(RNG.standard_normal(r)) + 0.5
        C_full = C_T + U @ np.diag(D) @ U.T
        solver = WoodburySolver(ts, U, D)
        return C_full, solver

    def test_1d_vector(self):
        C, solver = self._build()
        v = RNG.standard_normal(N)
        assert_close(solver.apply_C_inv(v), _reference_solve(C, v), label="Woodbury 1D")

    def test_2d_matrix(self):
        C, solver = self._build()
        V = RNG.standard_normal((N, 4))
        assert_close(solver.apply_C_inv(V), _reference_solve(C, V), label="Woodbury 2D")

    def test_rank1(self):
        C, solver = self._build(r=1)
        v = RNG.standard_normal(N)
        assert_close(solver.apply_C_inv(v), _reference_solve(C, v), label="Woodbury rank-1")

    def test_make_solver_factory(self):
        _, acf = _random_toeplitz_spd(N)
        x, y = _gs_vectors(acf)
        U = RNG.standard_normal((N, 2))
        D = np.array([1.0, 2.0])
        solver = make_solver('woodbury', x=x, y=y, U=U, D=D)
        assert isinstance(solver, WoodburySolver)


# ---------------------------------------------------------------------------
# BlockDiagonalToeplitzSolver
# ---------------------------------------------------------------------------

class TestBlockDiagonalToeplitzSolver:

    def test_two_blocks(self):
        N1, N2 = 60, 68
        C1, acf1 = _random_toeplitz_spd(N1)
        C2, acf2 = _random_toeplitz_spd(N2, decay=0.3)

        C_block = np.zeros((N1 + N2, N1 + N2))
        C_block[:N1, :N1] = C1
        C_block[N1:, N1:] = C2

        solver = make_solver(
            'block_diagonal',
            acfs=[acf1, acf2],
            segment_slices=[slice(0, N1), slice(N1, N1 + N2)],
        )
        v = RNG.standard_normal(N1 + N2)
        assert_close(solver.apply_C_inv(v), _reference_solve(C_block, v), label="BlockDiag 1D")

    def test_2d_matrix(self):
        N1, N2 = 60, 68
        C1, acf1 = _random_toeplitz_spd(N1)
        C2, acf2 = _random_toeplitz_spd(N2, decay=0.3)
        C_block = np.zeros((N1 + N2, N1 + N2))
        C_block[:N1, :N1] = C1
        C_block[N1:, N1:] = C2
        solver = make_solver(
            'block_diagonal',
            acfs=[acf1, acf2],
            segment_slices=[slice(0, N1), slice(N1, N1 + N2)],
        )
        V = RNG.standard_normal((N1 + N2, 3))
        assert_close(solver.apply_C_inv(V), _reference_solve(C_block, V), label="BlockDiag 2D")


# ---------------------------------------------------------------------------
# CholeskySolver
# ---------------------------------------------------------------------------

class TestCholeskySolver:

    def test_toeplitz_C(self):
        C, _ = _random_toeplitz_spd(N)
        solver = CholeskySolver(C)
        v = RNG.standard_normal(N)
        assert_close(solver.apply_C_inv(v), _reference_solve(C, v), label="Cholesky Toeplitz")

    def test_non_toeplitz_C(self):
        C, _ = _random_toeplitz_spd(N)
        U = RNG.standard_normal((N, 5))
        C_nt = C + U @ U.T
        solver = CholeskySolver(C_nt)
        v = RNG.standard_normal(N)
        assert_close(solver.apply_C_inv(v), _reference_solve(C_nt, v), label="Cholesky non-Toeplitz")

    def test_2d(self):
        C, _ = _random_toeplitz_spd(N)
        solver = CholeskySolver(C)
        V = RNG.standard_normal((N, 6))
        assert_close(solver.apply_C_inv(V), _reference_solve(C, V), label="Cholesky 2D")


# ---------------------------------------------------------------------------
# PCGSolver
# ---------------------------------------------------------------------------

class TestPCGSolver:

    def test_toeplitz_precond(self):
        C, acf = _random_toeplitz_spd(N)
        x, y = _gs_vectors(acf)
        precond = ToeplitzSolver(x, y)
        solver = PCGSolver(lambda v: C @ v, precond, N=N, tol=1e-10)
        v = RNG.standard_normal(N)
        assert_close(solver.apply_C_inv(v), _reference_solve(C, v), rtol=1e-6, label="PCG Toeplitz precond")

    def test_perturbed_C(self):
        C_T, acf = _random_toeplitz_spd(N)
        U = RNG.standard_normal((N, 3)) * 0.1
        C = C_T + U @ U.T
        x, y = _gs_vectors(acf)
        precond = ToeplitzSolver(x, y)
        solver = PCGSolver(lambda v: C @ v, precond, N=N, tol=1e-10)
        v = RNG.standard_normal(N)
        assert_close(solver.apply_C_inv(v), _reference_solve(C, v), rtol=1e-5, label="PCG perturbed C")


# ---------------------------------------------------------------------------
# Cross-backend consistency
# ---------------------------------------------------------------------------

class TestCrossBackend:

    def test_all_backends_agree_on_toeplitz(self):
        C, acf = _random_toeplitz_spd(N)
        x, y = _gs_vectors(acf)
        v = RNG.standard_normal(N)
        ref = _reference_solve(C, v)

        ts = ToeplitzSolver(x, y)
        cs = CholeskySolver(C)
        ps = PCGSolver(lambda w: C @ w, ts, N=N, tol=1e-10)

        assert_close(ts.apply_C_inv(v), ref, label="Toeplitz vs ref")
        assert_close(cs.apply_C_inv(v), ref, label="Cholesky vs ref")
        assert_close(ps.apply_C_inv(v), ref, rtol=1e-5, label="PCG vs ref")

    def test_woodbury_vs_cholesky_non_toeplitz(self):
        C_T, acf = _random_toeplitz_spd(N)
        x, y = _gs_vectors(acf)
        U = RNG.standard_normal((N, 4))
        D = np.abs(RNG.standard_normal(4)) + 0.5
        C = C_T + U @ np.diag(D) @ U.T
        v = RNG.standard_normal(N)

        w_sol = WoodburySolver(ToeplitzSolver(x, y), U, D)
        c_sol = CholeskySolver(C)

        assert_close(w_sol.apply_C_inv(v), c_sol.apply_C_inv(v), label="Woodbury vs Cholesky")
