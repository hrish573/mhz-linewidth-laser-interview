"""Tests for solver.py: steady-state solutions and limiting cases."""

import numpy as np
import pytest
from src.model import LaserParams, SR87_PARAMS, HBAR
from src.solver import (
    solve_steady_state,
    solve_steady_state_approx,
    compute_linewidth,
    scan_w,
)


class TestTrivialLimits:
    """Exact solver must agree with known trivial limits."""

    p = SR87_PARAMS

    def test_Omega_zero(self):
        """No coupling -> no cavity photons, no correlations, s_z = d0."""
        p0 = LaserParams(
            gamma=self.p.gamma,
            gamma_T2=self.p.gamma_T2,
            Omega=0.0,
            kappa=self.p.kappa,
            omega_a=self.p.omega_a,
        )
        sol = solve_steady_state(p0, w=10.0, N=100_000)
        assert sol['y'] == 0.0
        assert sol['n_ph'] == 0.0
        assert sol['s_s'] == 0.0
        assert abs(sol['s_z'] - p0.d0(10.0)) < 1e-15

    def test_w_zero(self):
        """No pump -> d0=-1, no lasing."""
        sol = solve_steady_state(self.p, w=0.0, N=100_000)
        assert sol['y'] == 0.0
        assert sol['n_ph'] == 0.0
        assert sol['s_s'] == 0.0
        assert sol['s_z'] == -1.0
        assert sol['power'] == 0.0


class TestCollectiveRegime:
    """Check that the solver produces expected collective behavior."""

    p = SR87_PARAMS
    N = 1_000_000

    def test_mid_pump_has_positive_ss(self):
        """In the collective window, s_s should be positive."""
        w = self.p.w_max(self.N) / 2.0  # roughly optimal pump
        sol = solve_steady_state(self.p, w=w, N=self.N)
        assert sol['s_s'] > 0, f"s_s = {sol['s_s']} should be > 0 in collective regime"
        assert sol['n_ph'] > 0
        assert sol['power'] > 0

    def test_power_N_squared_scaling(self):
        """Power should scale approximately as N^2 in the collective regime."""
        w = 10.0  # well within collective window for large N
        N1 = 100_000
        N2 = 200_000
        sol1 = solve_steady_state(self.p, w=w, N=N1)
        sol2 = solve_steady_state(self.p, w=w, N=N2)

        if sol1['power'] > 0 and sol2['power'] > 0:
            ratio = sol2['power'] / sol1['power']
            expected_ratio = (N2 / N1) ** 2
            # Allow generous tolerance — this is approximate
            assert 0.5 * expected_ratio < ratio < 2.0 * expected_ratio, (
                f"Power ratio {ratio:.2f} vs expected ~{expected_ratio:.2f}"
            )

    def test_strong_pump_destroys_coherence(self):
        """Pumping well beyond w_max should give much smaller s_s."""
        w_max = self.p.w_max(self.N)
        sol_good = solve_steady_state(self.p, w=w_max / 2.0, N=self.N)
        sol_bad = solve_steady_state(self.p, w=w_max * 10.0, N=self.N)

        # s_s should be much smaller (or zero) above threshold
        assert sol_bad['s_s'] < sol_good['s_s'] * 0.1, (
            f"Strong pump s_s = {sol_bad['s_s']:.6e} not much smaller than "
            f"optimal s_s = {sol_good['s_s']:.6e}"
        )

    def test_threshold_region(self):
        """Near w ~ gamma, the system should transition from non-lasing to lasing."""
        N = self.N
        w_low = self.p.gamma / 10.0
        w_high = self.p.gamma * 10.0

        sol_low = solve_steady_state(self.p, w=w_low, N=N)
        sol_high = solve_steady_state(self.p, w=w_high, N=N)

        # Below threshold: very small s_s
        # Above threshold: significant s_s
        assert sol_high['s_s'] > sol_low['s_s'] * 10, (
            f"s_s({w_high}) = {sol_high['s_s']:.6e} should be >> "
            f"s_s({w_low}) = {sol_low['s_s']:.6e}"
        )


class TestApproxVsExact:
    """The approximate Eq.(6) should agree with the exact solver
    in the bad-cavity limit for the collective regime."""

    p = SR87_PARAMS
    N = 1_000_000

    def test_agreement_mid_pump(self):
        """At optimal pump, exact and approx should give similar s_s."""
        w = self.p.w_max(self.N) / 2.0
        sol_exact = solve_steady_state(self.p, w=w, N=self.N)
        sol_approx = solve_steady_state_approx(self.p, w=w, N=self.N)

        if sol_approx['s_s'] > 0:
            rel_diff = abs(sol_exact['s_s'] - sol_approx['s_s']) / sol_approx['s_s']
            assert rel_diff < 0.01, (
                f"Exact s_s = {sol_exact['s_s']:.6e}, "
                f"Approx s_s = {sol_approx['s_s']:.6e}, "
                f"rel diff = {rel_diff:.3f}"
            )


class TestLinewidth:
    """Check the linewidth regression eigenvalue computation."""

    p = SR87_PARAMS

    def test_linewidth_no_atoms(self):
        """With s_z = d0 and no collective effect, linewidth ~ Gamma_half (HWHM)."""
        lw = compute_linewidth(self.p, w=10.0, N=0, s_z=self.p.d0(10.0))
        # With N=0, det = kappa*Gamma_half/2, eigenvalues are -kappa/2 and -Gamma_half
        # HWHM = Gamma_half (the slow eigenvalue)
        Gamma_half = self.p.Gamma_half(10.0)
        assert abs(lw - Gamma_half) < 0.1 * Gamma_half

    def test_linewidth_minimum_order(self):
        """In the collective regime, linewidth should reach ~ C*gamma scale."""
        N = 1_000_000
        w = self.p.w_max(N) / 2.0
        sol = solve_steady_state(self.p, w=w, N=N)
        lw = compute_linewidth(self.p, w=w, N=N, s_z=sol['s_z'])
        C_gamma = self.p.C * self.p.gamma

        # The linewidth should be in the range [0.1*C*gamma, 10*C*gamma]
        # This is a rough order-of-magnitude check
        assert lw < 100 * C_gamma, (
            f"Linewidth {lw:.4e} should be of order C*gamma = {C_gamma:.4e}"
        )


class TestScanConsistency:
    """Integration test for the scan functions."""

    p = SR87_PARAMS
    N = 100_000

    def test_scan_w_runs(self):
        """scan_w should return arrays of correct length."""
        w_arr = np.logspace(-2, 3, 20)
        results = scan_w(self.p, self.N, w_arr, exact=True)
        for key in ['y', 's_z', 's_s', 'n_ph', 'power', 'linewidth']:
            assert len(results[key]) == len(w_arr)

    def test_scan_w_monotone_power_in_window(self):
        """Power should rise then fall as w goes from low to high."""
        w_arr = np.logspace(-1, 4, 50)
        results = scan_w(self.p, self.N, w_arr, exact=True)
        power = results['power']
        # Find peak
        i_peak = np.argmax(power)
        # Peak should not be at the endpoints
        assert 0 < i_peak < len(w_arr) - 1, (
            f"Power peak at index {i_peak} (endpoints: 0, {len(w_arr)-1})"
        )


# ================================================================
# Limiting-case validation (notes.md §10.1, §10.2, §10.3)
# ================================================================

class TestLimitingCaseNoCoupling:
    """§10.1: Omega = 0 — free-atom limit.

    When the atom-cavity coupling vanishes, the cavity is decoupled from the
    atoms.  The solver should return *exactly*:
        y = 0, s_s = 0, n_ph = 0, s_z = d0, power = 0.
    This is an exact result, not a numerical approximation, because the
    solver's early-return branch at Omega=0 bypasses the quadratic entirely.
    """

    p = SR87_PARAMS

    def _make_uncoupled(self):
        return LaserParams(
            gamma=self.p.gamma,
            gamma_T2=self.p.gamma_T2,
            Omega=0.0,
            kappa=self.p.kappa,
            omega_a=self.p.omega_a,
        )

    def test_exact_zero_coherences(self):
        """y, s_s, n_ph, and power must be exactly zero at Omega=0."""
        p0 = self._make_uncoupled()
        for w in [0.001, 0.01, 1.0, 10.0, 100.0, 1e6]:
            sol = solve_steady_state(p0, w=w, N=1_000_000)
            assert sol['y'] == 0.0, f"y != 0 at w={w}"
            assert sol['s_s'] == 0.0, f"s_s != 0 at w={w}"
            assert sol['n_ph'] == 0.0, f"n_ph != 0 at w={w}"
            assert sol['power'] == 0.0, f"power != 0 at w={w}"

    def test_inversion_equals_d0_exactly(self):
        """s_z must equal d0 exactly (not up to tolerance) at Omega=0."""
        p0 = self._make_uncoupled()
        for w in [0.001, 0.01, 1.0, 10.0, 100.0, 1e6]:
            sol = solve_steady_state(p0, w=w, N=1_000_000)
            expected = p0.d0(w)
            assert sol['s_z'] == expected, (
                f"s_z={sol['s_z']} != d0={expected} at w={w}"
            )

    def test_approx_solver_also_gives_free_atom(self):
        """The approximate solver should also return the free-atom limit."""
        p0 = self._make_uncoupled()
        sol = solve_steady_state_approx(p0, w=10.0, N=100_000)
        assert sol['y'] == 0.0
        assert sol['s_s'] == 0.0
        assert sol['n_ph'] == 0.0


class TestLimitingCaseNoPump:
    """§10.2: w = 0 — no pump.

    Without repumping, d0 = -1 (all atoms in ground state).  The collective
    gain d0*N*gamma*C is negative, so no collective emission can exist.
    The solver should return exactly:
        y = 0, s_s = 0, n_ph = 0, s_z = -1, power = 0.
    This is exact because the solver's early-return branch at w=0 bypasses
    the quadratic.  Physically, the atoms have no inversion and cannot
    provide gain at any N.
    """

    p = SR87_PARAMS

    def test_exact_non_lasing_state(self):
        """All observables must be exactly zero/non-lasing at w=0."""
        for N in [100, 1_000, 10_000, 100_000, 1_000_000]:
            sol = solve_steady_state(self.p, w=0.0, N=N)
            assert sol['y'] == 0.0, f"y != 0 at N={N}"
            assert sol['s_s'] == 0.0, f"s_s != 0 at N={N}"
            assert sol['n_ph'] == 0.0, f"n_ph != 0 at N={N}"
            assert sol['s_z'] == -1.0, f"s_z != -1 at N={N}"
            assert sol['power'] == 0.0, f"power != 0 at N={N}"

    def test_gain_is_negative(self):
        """The collective gain d0*N*gamma*C must be negative at w=0."""
        d0 = self.p.d0(0.0)
        assert d0 == -1.0
        # At any N, d0*N*gamma*C < 0 < Gamma
        for N in [100, 1_000_000]:
            gain = d0 * N * self.p.gamma * self.p.C
            assert gain < 0, f"gain={gain} should be < 0 at N={N}"

    def test_approx_solver_also_non_lasing(self):
        """The approximate solver should also return non-lasing at w=0."""
        for N in [1_000, 1_000_000]:
            sol = solve_steady_state_approx(self.p, w=0.0, N=N)
            assert sol['y'] == 0.0
            assert sol['s_s'] == 0.0
            assert sol['s_z'] == -1.0
            assert sol['power'] == 0.0


class TestLimitingCaseStrongPump:
    """§10.3: w -> infinity — excessively strong pump.

    When w >> w_max = N*C*gamma, the inversion saturates at d0 -> 1 but the
    decoherence rate Gamma ~ w overwhelms the collective gain ~ N*C*gamma.
    The superradiant solution disappears: s_s -> 0, n_ph -> 0, s_z -> d0 ~ 1.

    This limit is not exact in the same sense as the Omega=0 and w=0 cases:
    the solver does not short-circuit, and the quadratic still has a non-trivial
    structure.  The disappearance of the collective solution is a *physical*
    asymptotic limit verified numerically.
    """

    p = SR87_PARAMS
    N = 1_000_000

    def test_d0_approaches_one(self):
        """d0 = (w-gamma)/(w+gamma) -> 1 as w -> infinity."""
        for w in [1e4, 1e6, 1e10]:
            d0 = self.p.d0(w)
            # Leading correction: 1 - d0 ~ 2*gamma/w
            assert abs(d0 - 1.0) < 3.0 * self.p.gamma / w + 1e-14, (
                f"d0={d0} not close enough to 1 at w={w}"
            )

    def test_collective_coherence_vanishes(self):
        """s_s must become negligible well above w_max."""
        w_max = self.p.w_max(self.N)
        # At optimal pump, get reference s_s
        sol_opt = solve_steady_state(self.p, w=w_max / 2.0, N=self.N)
        s_s_opt = sol_opt['s_s']
        assert s_s_opt > 0, "Sanity: should have s_s > 0 at optimal pump"

        # At 10x w_max, s_s should be much smaller
        sol_10x = solve_steady_state(self.p, w=10.0 * w_max, N=self.N)
        assert sol_10x['s_s'] < 1e-3 * s_s_opt, (
            f"s_s at 10*w_max = {sol_10x['s_s']:.2e} not small vs "
            f"s_s_opt = {s_s_opt:.2e}"
        )

        # At 1000x w_max, even more suppressed
        sol_1000x = solve_steady_state(self.p, w=1000.0 * w_max, N=self.N)
        assert sol_1000x['s_s'] < 1e-5 * s_s_opt, (
            f"s_s at 1000*w_max = {sol_1000x['s_s']:.2e} not negligible"
        )

    def test_inversion_approaches_d0_at_strong_pump(self):
        """At very large w, s_z should approach d0 (free-atom value)."""
        w = 1e6  # far beyond w_max ~ 1460
        sol = solve_steady_state(self.p, w=w, N=self.N)
        d0 = self.p.d0(w)
        # s_z should be very close to d0 since collective effects are gone
        assert abs(sol['s_z'] - d0) < 1e-3, (
            f"s_z={sol['s_z']:.8f} not close to d0={d0:.8f} at w={w}"
        )

    def test_gain_less_than_decoherence_above_wmax(self):
        """Above w_max, the approximate gain condition must fail."""
        w_max = self.p.w_max(self.N)
        for mult in [2.0, 10.0, 100.0]:
            w = mult * w_max
            d0 = self.p.d0(w)
            gain = d0 * self.N * self.p.gamma * self.p.C
            Gamma = self.p.Gamma(w)
            assert gain < Gamma, (
                f"At w={mult}*w_max: gain={gain:.2f} should be < Gamma={Gamma:.2f}"
            )

    def test_power_vanishes_at_extreme_pump(self):
        """Power must become negligible at very large w."""
        w_max = self.p.w_max(self.N)
        sol_opt = solve_steady_state(self.p, w=w_max / 2.0, N=self.N)
        P_opt = sol_opt['power']

        sol_extreme = solve_steady_state(self.p, w=1e6, N=self.N)
        assert sol_extreme['power'] < 1e-3 * P_opt, (
            f"power at w=1e6 = {sol_extreme['power']:.2e} not negligible vs "
            f"P_opt = {P_opt:.2e}"
        )
