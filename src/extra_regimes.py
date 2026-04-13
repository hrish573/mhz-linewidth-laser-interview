"""
Extra-regime studies beyond the main paper figures.

These studies reuse the core model + solver infrastructure to explore
parameter regimes not shown in the paper's Figs. 2–3, but accessible
from the same cumulant equations.

Usage:
    python -m src.extra_regimes
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from .model import LaserParams, SR87_PARAMS, HBAR
from .solver import solve_steady_state, solve_steady_state_approx, scan_w, compute_linewidth


OUTDIR = "figures/extra_regimes"


# ── 1. Exact vs approximate solver comparison ─────────────────────────

def plot_exact_vs_approx(
    p: LaserParams = SR87_PARAMS,
    N: int = 1_000_000,
    w_range=(1e-3, 1e5),
    nw=600,
    savepath=None,
):
    """
    1D power vs pump rate: exact quadratic solver overlaid with
    the approximate Eq.(6) solver, at fixed N.

    Shows where the bad-cavity approximation breaks down.
    """
    w_arr = np.logspace(np.log10(w_range[0]), np.log10(w_range[1]), nw)

    exact = scan_w(p, N, w_arr, exact=True)
    approx = scan_w(p, N, w_arr, exact=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(w_arr, exact['power'], 'b-', linewidth=1.5, label='Exact quadratic')
    ax.loglog(w_arr, approx['power'], 'r--', linewidth=1.5, label='Approx (Eq. 6)')

    # Reference lines
    ax.axvline(p.gamma, color='gray', linestyle=':', linewidth=1, label=r'$w=\gamma$')
    ax.axvline(p.w_max(N), color='gray', linestyle='--', linewidth=1,
               label=rf'$w_{{\max}}={p.w_max(N):.0f}$ s$^{{-1}}$')
    ax.axhline(p.P_max(N), color='green', linestyle='-.', linewidth=1, alpha=0.6,
               label=rf'$P_{{\max}}={p.P_max(N):.2e}$ W')

    ax.set_xlabel(r'Pump rate $w$ [s$^{-1}$]')
    ax.set_ylabel(r'Output power $P$ [W]')
    ax.set_title(rf'Exact vs approximate solver ($N = {N}$)')
    ax.legend(fontsize=8)
    plt.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=150)
    return fig, ax


# ── 2. N-scaling of power at optimal pump ─────────────────────────────

def plot_power_N_scaling(
    p: LaserParams = SR87_PARAMS,
    N_range=(1e3, 2e6),
    nN=80,
    savepath=None,
):
    """
    Power at optimal pump (w = w_max/2) as a function of N.

    Tests the predicted N^2 superradiant scaling from Eq.(8).
    """
    N_arr = np.logspace(np.log10(N_range[0]), np.log10(N_range[1]), nN).astype(int)

    powers = []
    for N in N_arr:
        w_opt = p.w_max(N) / 2.0
        if w_opt < p.gamma:
            w_opt = p.gamma * 2.0  # ensure we are above lower threshold
        sol = solve_steady_state(p, w=w_opt, N=int(N))
        powers.append(sol['power'])
    powers = np.array(powers)

    # Analytic prediction: P_max = hbar * omega_a * N^2 * C * gamma / 8
    P_analytic = p.P_max(N_arr)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(N_arr, powers, 'b-o', markersize=3, linewidth=1.5, label=r'Numerical $P(w_{\mathrm{opt}}, N)$')
    ax.loglog(N_arr, P_analytic, 'r--', linewidth=1.5, label=r'$P_{\max} = \hbar\omega_a N^2 C\gamma/8$')

    # Show N^2 slope reference
    N_ref = np.array([N_arr[0], N_arr[-1]])
    P_ref = powers[0] * (N_ref / N_arr[0])**2
    ax.loglog(N_ref, P_ref, 'k:', linewidth=1, alpha=0.4, label=r'$\propto N^2$ reference')

    ax.set_xlabel(r'Atom number $N$')
    ax.set_ylabel(r'Output power $P$ [W]')
    ax.set_title(r'Power scaling with atom number at optimal pump')
    ax.legend(fontsize=9)
    plt.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=150)
    return fig, ax


# ── 3. Cooperativity scan ─────────────────────────────────────────────

def plot_cooperativity_scan(
    N: int = 1_000_000,
    C_range=(0.01, 10.0),
    nC=80,
    savepath=None,
):
    """
    Minimum linewidth and peak power as a function of single-atom cooperativity C.

    Varies C by scaling Omega while keeping kappa fixed.
    Shows that the linewidth floor scales as C*gamma.
    """
    p0 = SR87_PARAMS
    C_arr = np.logspace(np.log10(C_range[0]), np.log10(C_range[1]), nC)

    min_lw = []
    peak_power = []

    for C_val in C_arr:
        # Scale Omega to achieve target C: Omega = sqrt(C * kappa * gamma)
        Omega_new = np.sqrt(C_val * p0.kappa * p0.gamma)
        p = LaserParams(
            gamma=p0.gamma,
            gamma_T2=p0.gamma_T2,
            Omega=Omega_new,
            kappa=p0.kappa,
            omega_a=p0.omega_a,
        )

        w_max = p.w_max(N)
        if w_max < p.gamma * 2:
            min_lw.append(np.nan)
            peak_power.append(0.0)
            continue

        # Scan over the collective window
        w_arr = np.logspace(np.log10(p.gamma), np.log10(w_max), 200)
        results = scan_w(p, N, w_arr, exact=True)

        min_lw.append(np.min(results['linewidth']))
        peak_power.append(np.max(results['power']))

    min_lw = np.array(min_lw)
    peak_power = np.array(peak_power)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: minimum linewidth vs C
    ax1.loglog(C_arr, min_lw, 'b-o', markersize=3, linewidth=1.5, label=r'Numerical $\Delta\nu_{\min}$')
    ax1.loglog(C_arr, C_arr * p0.gamma, 'r--', linewidth=1.5, label=r'$C\gamma$')
    ax1.set_xlabel(r'Cooperativity $C$')
    ax1.set_ylabel(r'Minimum linewidth $\Delta\nu_{\min}$ [s$^{-1}$]')
    ax1.set_title(r'Linewidth floor vs cooperativity')
    ax1.axvline(p0.C, color='gray', linestyle=':', linewidth=1, alpha=0.6,
                label=rf'$C_{{\mathrm{{Sr87}}}}={p0.C:.3f}$')
    ax1.legend(fontsize=8)

    # Right: peak power vs C
    ax2.loglog(C_arr, peak_power, 'b-o', markersize=3, linewidth=1.5, label=r'Numerical $P_{\max}$')
    P_analytic = HBAR * p0.omega_a * N**2 * C_arr * p0.gamma / 8.0
    ax2.loglog(C_arr, P_analytic, 'r--', linewidth=1.5, label=r'$\hbar\omega_a N^2 C\gamma/8$')
    ax2.set_xlabel(r'Cooperativity $C$')
    ax2.set_ylabel(r'Peak power $P_{\max}$ [W]')
    ax2.set_title(r'Peak power vs cooperativity')
    ax2.axvline(p0.C, color='gray', linestyle=':', linewidth=1, alpha=0.6,
                label=rf'$C_{{\mathrm{{Sr87}}}}={p0.C:.3f}$')
    ax2.legend(fontsize=8)

    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150)
    return fig, (ax1, ax2)


# ── 4. 1D power scan with thresholds ─────────────────────────────────

def plot_power_1d_scan(
    p: LaserParams = SR87_PARAMS,
    N: int = 1_000_000,
    w_range=(1e-3, 1e5),
    nw=600,
    savepath=None,
):
    """
    1D power vs pump rate at fixed N, with threshold markers.

    Shows the full pump-rate dependence including the sub-threshold
    spontaneous-emission tail visible only in the exact solver.
    """
    w_arr = np.logspace(np.log10(w_range[0]), np.log10(w_range[1]), nw)
    results = scan_w(p, N, w_arr, exact=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(w_arr, results['power'], 'b-', linewidth=1.5, label='Exact solver')

    ax.axvline(p.gamma, color='gray', linestyle=':', linewidth=1, label=r'$w=\gamma$')
    ax.axvline(p.gamma_T2, color='orange', linestyle='-.', linewidth=1, label=r'$w=1/T_2$')
    ax.axvline(p.w_max(N), color='gray', linestyle='--', linewidth=1,
               label=rf'$w_{{\max}}={p.w_max(N):.0f}$ s$^{{-1}}$')
    ax.axhline(p.P_max(N), color='green', linestyle='-.', linewidth=1, alpha=0.6,
               label=rf'$P_{{\max}}={p.P_max(N):.2e}$ W')

    ax.set_xlabel(r'Pump rate $w$ [s$^{-1}$]')
    ax.set_ylabel(r'Output power $P$ [W]')
    ax.set_title(rf'Output power vs pump rate ($N = {N}$)')
    ax.legend(fontsize=8)
    plt.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=150)
    return fig, ax


# ── 5. Dephasing sensitivity ──────────────────────────────────────────

def plot_dephasing_sensitivity(
    N: int = 1_000_000,
    T2_values=(0.1, 0.5, 1.0, 2.0, 10.0),
    w_range=(1e-3, 1e5),
    nw=400,
    savepath=None,
):
    """
    Linewidth vs pump rate for several values of 1/T2.

    Shows how the linewidth floor and collective window depend on
    inhomogeneous dephasing — a knob not varied in the paper's main figures.
    """
    p0 = SR87_PARAMS

    fig, ax = plt.subplots(figsize=(8, 5))
    w_arr = np.logspace(np.log10(w_range[0]), np.log10(w_range[1]), nw)

    for T2 in T2_values:
        gamma_T2 = 1.0 / T2
        p = LaserParams(
            gamma=p0.gamma,
            gamma_T2=gamma_T2,
            Omega=p0.Omega,
            kappa=p0.kappa,
            omega_a=p0.omega_a,
        )
        results = scan_w(p, N, w_arr, exact=True)
        ax.loglog(w_arr, results['linewidth'], linewidth=1.5,
                  label=rf'$T_2 = {T2}$ s ($1/T_2 = {gamma_T2}$ s$^{{-1}}$)')

    # C*gamma reference
    ax.axhline(p0.C * p0.gamma, color='red', linestyle='--', linewidth=1, alpha=0.5,
               label=rf'$C\gamma = {p0.C * p0.gamma:.1e}$ s$^{{-1}}$')

    ax.set_xlabel(r'Pump rate $w$ [s$^{-1}$]')
    ax.set_ylabel(r'Linewidth $\Delta\nu$ [s$^{-1}$]')
    ax.set_title(rf'Linewidth vs pump rate for different $T_2$ ($N = {N}$)')
    ax.set_ylim(1e-4, 1e6)
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=150)
    return fig, ax


# ── 6. Observables vs pump (multi-panel) ──────────────────────────────

def plot_observables_vs_w(
    p: LaserParams = SR87_PARAMS,
    N: int = 1_000_000,
    w_range=(1e-3, 1e5),
    nw=600,
    savepath=None,
):
    """
    All four cumulant observables (s_z, s_s, n_ph, y) vs pump rate
    at fixed N, showing the full internal structure of the steady state.
    """
    w_arr = np.logspace(np.log10(w_range[0]), np.log10(w_range[1]), nw)
    results = scan_w(p, N, w_arr, exact=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # s_z
    ax = axes[0, 0]
    ax.semilogx(w_arr, results['s_z'], 'b-', linewidth=1.5)
    d0_arr = np.array([p.d0(w) for w in w_arr])
    ax.semilogx(w_arr, d0_arr, 'r--', linewidth=1, alpha=0.6, label=r'$d_0(w)$ (free atom)')
    ax.axvline(p.gamma, color='gray', linestyle=':', linewidth=0.8)
    ax.axvline(p.w_max(N), color='gray', linestyle='--', linewidth=0.8)
    ax.set_xlabel(r'Pump rate $w$ [s$^{-1}$]')
    ax.set_ylabel(r'$s_z$')
    ax.set_title(r'Inversion per atom')
    ax.legend(fontsize=8)

    # s_s
    ax = axes[0, 1]
    ax.loglog(w_arr, np.maximum(results['s_s'], 1e-30), 'b-', linewidth=1.5)
    ax.axvline(p.gamma, color='gray', linestyle=':', linewidth=0.8)
    ax.axvline(p.w_max(N), color='gray', linestyle='--', linewidth=0.8)
    ax.axhline(1.0/8, color='red', linestyle='--', linewidth=1, alpha=0.5, label=r'$s_s = 1/8$')
    ax.set_xlabel(r'Pump rate $w$ [s$^{-1}$]')
    ax.set_ylabel(r'$s_s$')
    ax.set_title(r'Interatomic coherence')
    ax.legend(fontsize=8)

    # n_ph
    ax = axes[1, 0]
    ax.loglog(w_arr, np.maximum(results['n_ph'], 1e-30), 'b-', linewidth=1.5)
    ax.axvline(p.gamma, color='gray', linestyle=':', linewidth=0.8)
    ax.axvline(p.w_max(N), color='gray', linestyle='--', linewidth=0.8)
    ax.set_xlabel(r'Pump rate $w$ [s$^{-1}$]')
    ax.set_ylabel(r'$n_{\mathrm{ph}}$')
    ax.set_title(r'Cavity photon number')

    # y (atom-field coherence)
    ax = axes[1, 1]
    ax.loglog(w_arr, np.maximum(results['y'], 1e-30), 'b-', linewidth=1.5)
    ax.axvline(p.gamma, color='gray', linestyle=':', linewidth=0.8)
    ax.axvline(p.w_max(N), color='gray', linestyle='--', linewidth=0.8)
    ax.set_xlabel(r'Pump rate $w$ [s$^{-1}$]')
    ax.set_ylabel(r'$y$')
    ax.set_title(r'Atom-field coherence (imaginary part)')

    for ax in axes.flat:
        ax.grid(True, alpha=0.3)

    fig.suptitle(rf'All steady-state observables ($N = {N}$)', fontsize=13)
    plt.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=150)
    return fig, axes


# ── 7. Strong-pump asymptotic regime ──────────────────────────────────

def plot_strong_pump_asymptotic(
    p: LaserParams = SR87_PARAMS,
    N: int = 1_000_000,
    nw: int = 800,
    savepath=None,
):
    """
    Two-panel figure illustrating the over-pumped asymptotic regime.

    Top panel: s_s vs w  — collective coherence dies at large w.
    Bottom panel: s_z vs w — inversion saturates toward d0 -> 1.

    Physics: as w -> inf, d0 -> 1, Gamma > d0*N*C*gamma, so the
    superradiant collective branch shuts off: s_s -> 0, s_z -> d0.
    """
    w_max = p.w_max(N)
    # Range from well below threshold to 100 * w_max
    w_arr = np.logspace(np.log10(p.gamma * 0.1), np.log10(100 * w_max), nw)
    results = scan_w(p, N, w_arr, exact=True)

    d0_arr = np.array([p.d0(w) for w in w_arr])

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    # ── Top panel: s_s ──
    ax_top.semilogx(w_arr, results['s_s'], 'b-', linewidth=1.5,
                    label=r'$s_s$ (exact solver)')
    ax_top.axvline(w_max, color='gray', linestyle='--', linewidth=1,
                   label=rf'$w_{{\max}} = NCγ = {w_max:.0f}$ s$^{{-1}}$')
    ax_top.axvline(p.gamma, color='gray', linestyle=':', linewidth=0.8,
                   label=r'$w = \gamma$')
    ax_top.set_ylabel(r'$s_s$ (interatomic coherence)', fontsize=12)
    ax_top.set_ylim(-0.01, None)
    ax_top.legend(fontsize=9, loc='upper right')
    ax_top.grid(True, alpha=0.3)
    ax_top.set_title(
        rf'Over-pumped asymptotic regime ($N = {N:,}$, $C = {p.C:.4f}$)',
        fontsize=13,
    )

    # ── Bottom panel: s_z with d0 reference ──
    ax_bot.semilogx(w_arr, results['s_z'], 'b-', linewidth=1.5,
                    label=r'$s_z$ (exact solver)')
    ax_bot.semilogx(w_arr, d0_arr, 'r--', linewidth=1.2, alpha=0.8,
                    label=r'$d_0(w) = (w-\gamma)/(w+\gamma)$')
    ax_bot.axvline(w_max, color='gray', linestyle='--', linewidth=1)
    ax_bot.axvline(p.gamma, color='gray', linestyle=':', linewidth=0.8)
    ax_bot.axhline(1.0, color='black', linestyle=':', linewidth=0.5, alpha=0.4)
    ax_bot.set_xlabel(r'Pump rate $w$ [s$^{-1}$]', fontsize=12)
    ax_bot.set_ylabel(r'$s_z$ (inversion per atom)', fontsize=12)
    ax_bot.set_ylim(-1.1, 1.15)
    ax_bot.legend(fontsize=9, loc='lower right')
    ax_bot.grid(True, alpha=0.3)

    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150)
    return fig, (ax_top, ax_bot)


# ── 8. Presentation plot: over-pumped coherence collapse ────────────

def plot_overpumped_coherence_decay(
    p: LaserParams = SR87_PARAMS,
    N: int = 1_000_000,
    w_multiple_max: float = 20.0,
    nw: int = 800,
    savepath=None,
):
    """
    Presentation-friendly plot of the over-pumped regime.

    Shows only the interatomic coherence s_s so the upper-threshold physics
    is visually clear: coherence peaks inside the collective window and then
    collapses once the pump exceeds w_max = N C gamma.
    """
    w_max = p.w_max(N)
    w_arr = np.logspace(np.log10(p.gamma), np.log10(w_multiple_max * w_max), nw)
    results = scan_w(p, N, w_arr, exact=True)

    fig, ax = plt.subplots(figsize=(8, 4.8))

    s_s_arr = np.maximum(results['s_s'], 1e-12)

    ax.loglog(w_arr, s_s_arr, 'b-', linewidth=1.8, label=r'Exact $s_s$')
    ax.axvline(p.gamma, color='gray', linestyle=':', linewidth=0.9)
    ax.axvline(w_max, color='gray', linestyle='--', linewidth=1.1,
               label=rf'$w_{{\max}} = NC\gamma = {w_max:.0f}$ s$^{{-1}}$')
    ax.axhline(1.0 / 8.0, color='red', linestyle='--', linewidth=1.0, alpha=0.5)
    ax.axvspan(w_max, w_arr[-1], color='gray', alpha=0.08)
    ax.text(0.77, 0.89, 'Over-pumped', transform=ax.transAxes, fontsize=10,
            color='0.35', bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='0.8', alpha=0.9))

    ax.set_xlabel(r'Pump rate $w$ [s$^{-1}$]')
    ax.set_ylabel(r'$s_s$ (interatomic coherence)')
    ax.set_title(rf'Collective coherence collapses beyond $w_{{\max}}$ ($N = {N:,}$)')
    ax.set_ylim(1e-12, 3e-1)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=9, loc='lower left')

    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150)
    return fig, ax


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    plt.style.use('default')

    print("Generating extra-regime plots...")

    print("  1. Exact vs approximate solver...")
    plot_exact_vs_approx(savepath=f"{OUTDIR}/exact_vs_approx.png")

    print("  2. N-scaling of power...")
    plot_power_N_scaling(savepath=f"{OUTDIR}/power_N_scaling.png")

    print("  3. Cooperativity scan...")
    plot_cooperativity_scan(savepath=f"{OUTDIR}/cooperativity_scan.png")

    print("  4. 1D power scan...")
    plot_power_1d_scan(savepath=f"{OUTDIR}/power_1d_scan.png")

    print("  5. Dephasing sensitivity...")
    plot_dephasing_sensitivity(savepath=f"{OUTDIR}/dephasing_sensitivity.png")

    print("  6. All observables vs pump...")
    plot_observables_vs_w(savepath=f"{OUTDIR}/observables_vs_w.png")

    print("  7. Strong-pump asymptotic regime...")
    plot_strong_pump_asymptotic(savepath=f"{OUTDIR}/strong_pump_asymptotic.png")

    print("  8. Over-pumped coherence decay...")
    plot_overpumped_coherence_decay(savepath=f"{OUTDIR}/overpumped_coherence_decay.png")

    print(f"\nAll plots saved to {OUTDIR}/")


if __name__ == "__main__":
    main()
