#!/usr/bin/env python
# coding: utf-8
"""
Pure-math unit tests — no LAL, bilby, or pycbc required.

Tests the Gohberg-Semencul (GS) formula and binning geometry directly,
without importing from the codebase, so they run in a minimal uv environment.

Run with:
    uv run pytest tests/test_math.py -v
"""
import numpy as np
import pytest
from scipy.linalg import toeplitz, solve_toeplitz, matmul_toeplitz

RNG = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# GS formula (copy of TimeDomainLikelihoodBase.Inner_product_C_inv_vector)
# ---------------------------------------------------------------------------

def gs_apply_C_inv(x, y, v):
    """C^{-1} v via the Gohberg-Semencul representation."""
    N = len(x)
    xf = np.concatenate(([x[0]], np.zeros(N - 1)))
    ys = np.concatenate(([0.0], y[:-1]))
    zs = np.zeros(N)
    return (1.0 / x[0]) * (
        matmul_toeplitz((x, xf), matmul_toeplitz((xf, x), v))
        - matmul_toeplitz((ys, zs), matmul_toeplitz((zs, ys), v))
    )


def _make_toeplitz_spd(n, decay=0.99):
    acf = decay ** np.arange(n)
    C = toeplitz(acf)
    return C, acf


def _gs_vectors(acf):
    e1 = np.zeros(len(acf))
    e1[0] = 1.0
    x = solve_toeplitz((acf, acf), e1)
    return x, x[::-1]


# ---------------------------------------------------------------------------
# GS formula correctness
# ---------------------------------------------------------------------------

class TestGSFormula:

    @pytest.mark.parametrize("n", [64, 128, 512])
    def test_vector_matches_numpy(self, n):
        C, acf = _make_toeplitz_spd(n)
        x, y = _gs_vectors(acf)
        v = RNG.standard_normal(n)
        got = gs_apply_C_inv(x, y, v)
        ref = np.linalg.solve(C, v)
        np.testing.assert_allclose(got, ref, rtol=1e-5)

    def test_matrix_matches_numpy(self):
        n = 128
        C, acf = _make_toeplitz_spd(n)
        x, y = _gs_vectors(acf)
        V = RNG.standard_normal((n, 6))
        got = gs_apply_C_inv(x, y, V)
        ref = np.linalg.solve(C, V)
        np.testing.assert_allclose(got, ref, rtol=1e-5)

    def test_symmetry_of_weighted_inner_product(self):
        n = 128
        _, acf = _make_toeplitz_spd(n)
        x, y = _gs_vectors(acf)
        h1 = RNG.standard_normal(n)
        h2 = RNG.standard_normal(n)
        ip12 = np.dot(h1, gs_apply_C_inv(x, y, h2))
        ip21 = np.dot(h2, gs_apply_C_inv(x, y, h1))
        np.testing.assert_allclose(ip12, ip21, rtol=1e-10)

    def test_positive_definiteness(self):
        n = 128
        _, acf = _make_toeplitz_spd(n)
        x, y = _gs_vectors(acf)
        for _ in range(5):
            h = RNG.standard_normal(n)
            assert np.dot(h, gs_apply_C_inv(x, y, h)) > 0

    def test_identity_on_first_column(self):
        # C^{-1} e_1 = x by construction
        n = 64
        _, acf = _make_toeplitz_spd(n)
        x, y = _gs_vectors(acf)
        e1 = np.zeros(n); e1[0] = 1.0
        got = gs_apply_C_inv(x, y, e1)
        np.testing.assert_allclose(got, x, atol=1e-10)

    @pytest.mark.parametrize("decay", [0.5, 0.9, 0.999])
    def test_various_decay_rates(self, decay):
        n = 128
        C, acf = _make_toeplitz_spd(n, decay=decay)
        x, y = _gs_vectors(acf)
        v = RNG.standard_normal(n)
        got = gs_apply_C_inv(x, y, v)
        ref = np.linalg.solve(C, v)
        np.testing.assert_allclose(got, ref, rtol=1e-4)

    def test_round_trip(self):
        # C @ (C^{-1} v) should give back v
        n = 128
        C, acf = _make_toeplitz_spd(n)
        x, y = _gs_vectors(acf)
        v = RNG.standard_normal(n)
        C_inv_v = gs_apply_C_inv(x, y, v)
        recovered = C @ C_inv_v
        np.testing.assert_allclose(recovered, v, rtol=1e-5)


# ---------------------------------------------------------------------------
# Binning geometry (pure numpy, no waveform calls)
# ---------------------------------------------------------------------------

GAMMA = np.array([5 / 8, 3 / 8, 1 / 4, -3 / 8])
CHI = 1


def _setup_bins_inspiral(Time_array, time_split, epsilon_array, gamma=GAMMA, chi=CHI):
    """Standalone replica of RelativeBinningLikelihood22.setup_bins_inspiral."""
    d_alpha = []
    for g in gamma:
        if g < 0:
            d_alpha.append(2 * np.pi * chi / (min(abs(Time_array))) ** g)
        else:
            d_alpha.append(2 * np.pi * chi / (max(abs(Time_array))) ** g)
    d_alpha = np.array(d_alpha)

    d_phi_f = np.sum(
        np.sign(gamma) * d_alpha * (abs(Time_array[:, None]) ** gamma), axis=1
    )

    mask1 = Time_array <= time_split
    d_phi_f_regions = [d_phi_f[mask1], d_phi_f[~mask1]]
    idx_offsets = [0, np.sum(mask1)]

    bin_edge_indices = []
    for region_idx, (d_phi_f_reg, e) in enumerate(zip(d_phi_f_regions, epsilon_array)):
        n_bins = int(abs((d_phi_f_reg[-1] - d_phi_f_reg[0]) / e))
        region_bin_edges = []
        for i in range(n_bins + 1):
            bin_idx = np.where(
                d_phi_f_reg - d_phi_f_reg[0]
                >= (i / n_bins) * (d_phi_f_reg[-1] - d_phi_f_reg[0])
            )[0]
            region_bin_edges.append(bin_idx[-1] if len(bin_idx) > 0 else 0)
        bin_edge_indices.extend(np.array(region_bin_edges) + idx_offsets[region_idx])

    bin_edge_index = np.unique(np.array(bin_edge_indices))
    time_on_edge = Time_array[bin_edge_index]
    bin_width = time_on_edge[1:] - time_on_edge[:-1]
    bin_centre = (time_on_edge[1:] + time_on_edge[:-1]) / 2
    return len(bin_centre), bin_edge_index, bin_centre, bin_width, time_on_edge


class TestBinningGeometry:

    FS = 4096.0
    T = np.arange(-1.0 + 1.0 / 4096.0, 1.0 / 4096.0, 1.0 / 4096.0)

    def test_edges_monotone(self):
        T_insp = self.T[self.T < 0]
        _, edges, *_ = _setup_bins_inspiral(T_insp, -0.1, np.array([0.5, 0.1]))
        assert np.all(np.diff(edges) > 0)

    def test_widths_positive(self):
        T_insp = self.T[self.T < 0]
        _, _, _, widths, _ = _setup_bins_inspiral(T_insp, -0.1, np.array([0.5, 0.1]))
        assert np.all(widths > 0)

    def test_bin_count_consistent(self):
        T_insp = self.T[self.T < 0]
        n, edges, centres, widths, t_edges = _setup_bins_inspiral(
            T_insp, -0.1, np.array([0.5, 0.1])
        )
        assert n == len(centres)
        assert n == len(widths)
        assert len(edges) == n + 1

    def test_centres_within_time_range(self):
        T_insp = self.T[self.T < 0]
        _, _, centres, _, _ = _setup_bins_inspiral(T_insp, -0.1, np.array([0.5, 0.1]))
        assert np.all(centres >= T_insp[0])
        assert np.all(centres <= T_insp[-1])

    @pytest.mark.parametrize("epsilon", [0.5, 0.2, 0.1])
    def test_finer_epsilon_gives_more_bins(self, epsilon):
        T_insp = self.T[self.T < 0]
        eps_coarse = np.array([0.5, 0.5])
        eps_fine = np.array([epsilon, epsilon])
        n_coarse, *_ = _setup_bins_inspiral(T_insp, -0.1, eps_coarse)
        n_fine, *_ = _setup_bins_inspiral(T_insp, -0.1, eps_fine)
        assert n_fine >= n_coarse

    def test_time_split_partitions_correctly(self):
        T_insp = self.T[self.T < 0]
        _, edges, _, _, t_edges = _setup_bins_inspiral(T_insp, -0.1, np.array([0.5, 0.1]))
        # All edge times should lie within the input time range
        assert t_edges[0] >= T_insp[0]
        assert t_edges[-1] <= T_insp[-1]
