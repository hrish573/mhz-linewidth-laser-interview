"""Tests for model.py: parameter definitions and derived quantities."""

import numpy as np
import pytest
from src.model import LaserParams, SR87_PARAMS, HBAR, OMEGA_A_SR87


class TestSR87Params:
    """Check that the reference 87-Sr parameters match the paper."""

    def test_gamma(self):
        assert SR87_PARAMS.gamma == 0.01

    def test_gamma_T2(self):
        assert SR87_PARAMS.gamma_T2 == 1.0

    def test_Omega(self):
        assert SR87_PARAMS.Omega == 37.0

    def test_kappa(self):
        assert SR87_PARAMS.kappa == 9.4e5

    def test_cooperativity(self):
        """C = Omega^2 / (kappa * gamma) ≈ 0.146."""
        C = SR87_PARAMS.C
        assert abs(C - 37.0**2 / (9.4e5 * 0.01)) < 1e-10
        # Paper says C ≈ 0.15
        assert 0.14 < C < 0.16

    def test_T2(self):
        assert SR87_PARAMS.T2 == 1.0


class TestDerivedQuantities:
    """Test formulas for derived parameters."""

    p = SR87_PARAMS

    def test_d0_limits(self):
        """d0 = (w-gamma)/(w+gamma): goes from -1 (w=0) toward +1 (w>>gamma)."""
        # w = 0 gives d0 = -1
        assert abs(self.p.d0(0.0) - (-1.0)) < 1e-15
        # w = gamma gives d0 = 0
        assert abs(self.p.d0(self.p.gamma)) < 1e-15
        # w >> gamma gives d0 -> 1
        assert abs(self.p.d0(1e6) - 1.0) < 1e-4

    def test_Gamma(self):
        """Gamma = w + gamma + 2/T2."""
        w = 10.0
        expected = w + self.p.gamma + 2.0 * self.p.gamma_T2
        assert abs(self.p.Gamma(w) - expected) < 1e-15

    def test_Gamma_half(self):
        """Gamma_half = (w+gamma)/2 + 1/T2."""
        w = 10.0
        expected = (w + self.p.gamma) / 2.0 + self.p.gamma_T2
        assert abs(self.p.Gamma_half(w) - expected) < 1e-15

    def test_w_max(self):
        """w_max = N*C*gamma, Eq.(7)."""
        N = 1_000_000
        w_max = self.p.w_max(N)
        expected = N * self.p.C * self.p.gamma
        assert abs(w_max - expected) < 1e-10
        # For N=1e6, w_max ≈ 1460 s^-1
        assert 1000 < w_max < 2000

    def test_N_crit(self):
        """N_crit = 2/(C*gamma*T2) — minimum atoms for collective emission."""
        N_crit = self.p.N_crit()
        expected = 2.0 / (self.p.C * self.p.gamma * self.p.T2)
        assert abs(N_crit - expected) < 1e-10
        # Should be order 10^3
        assert 1000 < N_crit < 2000

    def test_P_max(self):
        """P_max = hbar*omega_a*N^2*C*gamma/8, Eq.(8)."""
        N = 1_000_000
        P_max = self.p.P_max(N)
        expected = HBAR * OMEGA_A_SR87 * N**2 * self.p.C * self.p.gamma / 8.0
        assert abs(P_max - expected) / expected < 1e-10

    def test_P_max_order_of_magnitude(self):
        """For N=1e6, P_max should be of order 1e-11 to 1e-10 W."""
        P = self.p.P_max(1_000_000)
        assert 1e-12 < P < 1e-9

    def test_timescale_hierarchy(self):
        """kappa >> 1/T2 >> gamma — the bad-cavity regime."""
        assert self.p.kappa > self.p.gamma_T2 > self.p.gamma
        assert self.p.kappa / self.p.gamma_T2 > 1e4
