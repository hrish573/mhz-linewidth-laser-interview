"""
Entry point for reproducing the main results of the paper.

Usage:
    python -m src.main
"""

import numpy as np
import os
from .model import SR87_PARAMS
from .solver import solve_steady_state, compute_linewidth_metrics, scan_w
from .plotting import plot_power_landscape, plot_linewidth_landscape, plot_linewidth_cut


def print_reference_values():
    """Print key reference values for checking."""
    p = SR87_PARAMS
    N = 1_000_000

    print("=" * 60)
    print("Reference parameters (87-Sr)")
    print("=" * 60)
    print(f"  gamma      = {p.gamma} s^-1")
    print(f"  1/T2       = {p.gamma_T2} s^-1")
    print(f"  Omega      = {p.Omega} s^-1")
    print(f"  kappa      = {p.kappa:.2e} s^-1")
    print(f"  C          = {p.C:.4f}")
    print(f"  N_crit     = {p.N_crit():.0f}")
    print(f"  w_max(N=1e6) = {p.w_max(N):.1f} s^-1")
    print(f"  P_max(N=1e6) = {p.P_max(N):.3e} W")
    print(f"  C*gamma    = {p.C * p.gamma:.4e} s^-1")
    print()

    # Solve at optimal pump
    w_opt = p.w_max(N) / 2.0
    sol = solve_steady_state(p, w=w_opt, N=N)
    print(f"Steady state at w = w_max/2 = {w_opt:.1f} s^-1, N = {N}:")
    print(f"  y     = {sol['y']:.6e}")
    print(f"  s_z   = {sol['s_z']:.6f}")
    print(f"  s_s   = {sol['s_s']:.6e}")
    print(f"  n_ph  = {sol['n_ph']:.6e}")
    print(f"  power = {sol['power']:.3e} W")
    print()

    # Linewidth at optimal pump
    lw_metrics = compute_linewidth_metrics(p, w=w_opt, N=N, s_z=sol['s_z'])
    lw_hwhm = lw_metrics['hwhm']
    lw_fwhm = lw_metrics['fwhm']
    Gamma_half = p.Gamma_half(w_opt)
    NCs_z = N * p.C * p.gamma * sol['s_z']
    print(f"Linewidth at w = w_max/2:")
    print(f"  Gamma_half           = {Gamma_half:.4f} s^-1")
    print(f"  N*C*gamma*s_z/2      = {NCs_z/2:.4f} s^-1")
    print(f"  Re(lambda_slow)      = -{lw_hwhm:.6e} s^-1")
    print(f"  HWHM (paper Δν)      = {lw_hwhm:.6e} s^-1  = {lw_hwhm*1e3:.4f} mHz")
    print(f"  FWHM (comparison)    = {lw_fwhm:.6e} s^-1  = {lw_fwhm*1e3:.4f} mHz")
    print(f"  Ratio to C*gamma     = {lw_hwhm / (p.C * p.gamma):.4f}")
    print()


def main():
    print_reference_values()

    os.makedirs("figures", exist_ok=True)

    print("Generating Fig. 2 (power landscape)...")
    plot_power_landscape(savepath="figures/fig2_power.png")
    print("  -> figures/fig2_power.png")

    print("Generating Fig. 3a (linewidth landscape)...")
    plot_linewidth_landscape(savepath="figures/fig3a_linewidth.png")
    print("  -> figures/fig3a_linewidth.png")

    print("Generating Fig. 3b (linewidth cut at N=1e6)...")
    plot_linewidth_cut(savepath="figures/fig3b_linewidth_cut.png")
    print("  -> figures/fig3b_linewidth_cut.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
