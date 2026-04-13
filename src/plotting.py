"""
Plotting routines for reproducing Figs. 2 and 3 from the paper.

Fig. 2: Output power P(w, N) — 2D color map.
Fig. 3a: Linewidth Delta_nu(w, N) — 2D color map.
Fig. 3b: Linewidth vs w at fixed N — 1D log-log cut.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from .model import LaserParams, SR87_PARAMS
from .solver import scan_wN, scan_w


def plot_power_landscape(
    p: LaserParams = SR87_PARAMS,
    w_range=(1e-3, 1e5),    N_range=(1e3, 1e6),
    nw=140,
    nN=140,
    exact=True,
    savepath=None,
):
    """
    Reproduce Fig. 2: log10(P / W) vs (log w, log N).
    """
    w_arr = np.logspace(np.log10(w_range[0]), np.log10(w_range[1]), nw)
    N_arr = np.logspace(np.log10(N_range[0]), np.log10(N_range[1]), nN).astype(int)

    data = scan_wN(p, w_arr, N_arr, exact=exact)
    power = data['power']

    # Mask zero power for log plot
    # LogNorm cannot display non-positive values; convert zero-power points to NaN
    # so they are skipped in both the heatmap and contour calculations.
    power_masked = np.where(power > 0, power, np.nan)

    fig, ax = plt.subplots(figsize=(8, 6))
    # pcolormesh draws a colored 2D field over (w, N); keep the mappable in `im`
    # so the colorbar uses the exact same normalization and colormap.
    im = ax.pcolormesh(
        w_arr, N_arr, power_masked,
        shading='auto', cmap='inferno',
        norm=LogNorm(vmin=np.nanmin(power_masked), vmax=np.nanmax(power_masked)),
    )
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Pump rate $w$ [s$^{-1}$]')
    ax.set_ylabel(r'Atom number $N$')
    ax.set_title(r'Output power $P$ [W]')
    plt.colorbar(im, ax=ax, label=r'$P$ [W]')

    # Iso-power contours are constant-power lines (same P at all points on a line).
    # Dense spacing makes slope changes and regime boundaries easier to see.
    log_power = np.log10(power_masked)
    contour_levels = np.arange(-21, -9.5, 0.5)  # log10(P/W) levels
    cs = ax.contour(
        w_arr, N_arr, log_power,
        levels=contour_levels, colors='k', linewidths=0.55,
    )
    label_levels = contour_levels[::2]
    label_map = {level: rf'$10^{{{int(level)}}}$' for level in label_levels}
    ax.clabel(cs, levels=label_levels, fmt=label_map, fontsize=7, inline=True)

    # Overlay upper threshold line: w_max = N*C*gamma
    # Build a smooth N grid only for drawing the analytic threshold guide.
    N_line = np.logspace(np.log10(N_range[0]), np.log10(N_range[1]), 200)
    w_max_line = N_line * p.C * p.gamma
    # Keep only the segment that lies inside the current plotting window.
    mask = (w_max_line >= w_range[0]) & (w_max_line <= w_range[1])
    ax.plot(w_max_line[mask], N_line[mask], 'w--', linewidth=1.5, label=r'$w_\mathrm{max}=NC\gamma$')

    # Overlay lower threshold line: w ~ gamma
    ax.axvline(p.gamma, color='white', linestyle=':', linewidth=1, label=r'$w=\gamma$')

    ax.legend(loc='upper left', fontsize=9)
    plt.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=150)
    return fig, ax


def plot_linewidth_landscape(
    p: LaserParams = SR87_PARAMS,
    w_range=(1e-3, 1e5),    N_range=(1e3, 1e6),
    nw=180,
    nN=180,
    exact=True,
    savepath=None,
):
    """
    Reproduce Fig. 3a: log10(Delta_nu / s^-1) vs (log w, log N).
    """
    w_arr = np.logspace(np.log10(w_range[0]), np.log10(w_range[1]), nw)
    N_arr = np.logspace(np.log10(N_range[0]), np.log10(N_range[1]), nN).astype(int)

    data = scan_wN(p, w_arr, N_arr, exact=exact)
    lw = data['linewidth']

    # Same masking logic as power: log scaling requires strictly positive data.
    lw_masked = np.where(lw > 0, lw, np.nan)

    fig, ax = plt.subplots(figsize=(8, 6))
    log_lw = np.log10(lw_masked)
    # 2D linewidth landscape; store as `im` so colorbar shares this exact mapping.
    im = ax.pcolormesh(
        w_arr, N_arr, lw_masked,
        shading='auto', cmap='inferno',
        norm=LogNorm(vmin=np.nanmin(lw_masked), vmax=np.nanmax(lw_masked)),
    )
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Pump rate $w$ [s$^{-1}$]')
    ax.set_ylabel(r'Atom number $N$')
    ax.set_title(r'Linewidth $\Delta\nu$ [s$^{-1}$]')
    plt.colorbar(im, ax=ax, label=r'$\Delta\nu$ [s$^{-1}$]')

    # Iso-linewidth contours: each contour is a constant Delta_nu line.
    # Using log10 values gives visually even spacing across decades.
    contour_levels = np.arange(-3, 5.25, 0.25)
    cs = ax.contour(
        w_arr, N_arr, log_lw,
        levels=contour_levels, colors='k', linewidths=0.55,
    )
    label_levels = contour_levels[::4]
    label_map = {level: rf'$10^{{{int(level)}}}$' for level in label_levels}
    ax.clabel(cs, levels=label_levels, fmt=label_map, fontsize=7, inline=True)

    # Reference lines
    # Separate helper grid for plotting the analytic upper-threshold guide.
    N_line = np.logspace(np.log10(N_range[0]), np.log10(N_range[1]), 200)
    w_max_line = N_line * p.C * p.gamma
    # Clip guide line to the visible x-range to avoid drawing off-panel segments.
    mask = (w_max_line >= w_range[0]) & (w_max_line <= w_range[1])
    ax.plot(w_max_line[mask], N_line[mask], 'w--', linewidth=1.5, label=r'$w_\mathrm{max}$')
    ax.axvline(p.gamma, color='white', linestyle=':', linewidth=1, label=r'$w=\gamma$')
    ax.axvline(p.gamma_T2, color='white', linestyle='-.', linewidth=1, label=r'$w=1/T_2$')

    ax.legend(loc='upper left', fontsize=9)
    plt.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=150)
    return fig, ax


def plot_linewidth_cut(
    p: LaserParams = SR87_PARAMS,
    N: int = 1_000_000,
    w_range=(1e-3, 1e5),    nw=500,
    exact=True,
    savepath=None,
):
    """
    Reproduce Fig. 3b: linewidth vs w at fixed N on log-log axes.
    """
    w_arr = np.logspace(np.log10(w_range[0]), np.log10(w_range[1]), nw)
    results = scan_w(p, N, w_arr, exact=exact)
    lw = results['linewidth']

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(w_arr, lw, 'b-', linewidth=1.5)

    # Reference lines
    ax.axhline(1e-3, color='r', linestyle='--', linewidth=1, alpha=0.5,
               label=r'$10^{-3}$ s$^{-1}$ (mHz scale)')
    ax.axvline(p.gamma, color='gray', linestyle=':', linewidth=1, label=r'$w=\gamma$')
    ax.axvline(p.gamma_T2, color='gray', linestyle='-.', linewidth=1, label=r'$w=1/T_2$')
    ax.axvline(p.w_max(N), color='gray', linestyle='--', linewidth=1, label=rf'$w_\mathrm{{max}}={p.w_max(N):.0f}$ s$^{{-1}}$')

    # Match paper's axis range: 10^-3 to 10^5 so the sub-mHz dip is visible
    ax.set_ylim(3e-4, 1e5)
    ax.set_xlabel(r'Pump rate $w$ [s$^{-1}$]')
    ax.set_ylabel(r'Linewidth $\Delta\nu$ [s$^{-1}$]')
    ax.set_title(rf'Linewidth vs pump rate ($N = {N}$)')
    ax.legend(fontsize=9)
    plt.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=150)
    return fig, ax
