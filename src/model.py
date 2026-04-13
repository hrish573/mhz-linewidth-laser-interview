"""
Parameters and derived quantities for the bad-cavity superradiant laser.

Paper: "Prospects for a mHz-linewidth laser" — Meiser, Ye, Carlson, Holland (2009)

Notation map (paper -> code):
    gamma       spontaneous decay rate                  s^{-1}
    gamma_T2    inhomogeneous dephasing rate 1/T2       s^{-1}
    Omega       single-atom vacuum Rabi frequency       s^{-1}
    kappa       cavity linewidth (FWHM photon loss)     s^{-1}
    C           single-atom cooperativity               dimensionless
    w           effective incoherent repump rate         s^{-1}
    N           number of atoms                         integer
    delta       atom-cavity detuning (omega_c - omega_a) s^{-1}
    d0          free-atom equilibrium inversion          dimensionless
    Gamma       total atomic dipole decay rate           s^{-1}
    omega_a     atomic transition frequency              rad/s

All quantities in SI (seconds, Watts, etc.) unless noted.
"""

import numpy as np
from dataclasses import dataclass


# 87-Sr clock transition frequency (1S0 -> 3P0)
OMEGA_A_SR87 = 2.0 * np.pi * 429.228e12  # rad/s
HBAR = 1.0545718e-34  # J·s


@dataclass
class LaserParams:
    """Physical parameters for the bad-cavity superradiant laser."""
    gamma: float      # spontaneous decay rate [s^-1]
    gamma_T2: float   # inhomogeneous dephasing rate 1/T2 [s^-1]
    Omega: float      # vacuum Rabi frequency [s^-1]
    kappa: float      # cavity decay rate [s^-1]
    omega_a: float    # atomic transition frequency [rad/s]

    @property
    def C(self) -> float:
        """Single-atom cooperativity: C = Omega^2 / (kappa * gamma)."""
        return self.Omega**2 / (self.kappa * self.gamma)

    @property
    def T2(self) -> float:
        """Inhomogeneous coherence time [s]."""
        return 1.0 / self.gamma_T2

    def d0(self, w: float) -> float:
        """Free-atom equilibrium inversion: d0 = (w - gamma) / (w + gamma)."""
        return (w - self.gamma) / (w + self.gamma)

    def Gamma(self, w: float) -> float:
        """Total atomic dipole relaxation rate: Gamma = w + gamma + 2/T2."""
        return w + self.gamma + 2.0 * self.gamma_T2

    def Gamma_half(self, w: float) -> float:
        """Half-rate for atom-field coherence decay: (w+gamma)/2 + 1/T2."""
        return (w + self.gamma) / 2.0 + self.gamma_T2

    def w_max(self, N: int) -> float:
        """Upper pump threshold: w_max = N * C * gamma."""
        return N * self.C * self.gamma

    def N_crit(self) -> float:
        """Critical atom number: N_crit = 2 / (C * gamma * T2)."""
        return 2.0 / (self.C * self.gamma * self.T2)

    def P_max(self, N: int) -> float:
        """Maximum outcoupled power: P_max = hbar * omega_a * N^2 * C * gamma / 8."""
        return HBAR * self.omega_a * N**2 * self.C * self.gamma / 8.0


# Reference parameters for 87-Sr from the paper
SR87_PARAMS = LaserParams(
    gamma=0.01,         # s^-1
    gamma_T2=1.0,       # s^-1  (1/T2)
    Omega=37.0,         # s^-1
    kappa=9.4e5,        # s^-1
    omega_a=OMEGA_A_SR87,
)
