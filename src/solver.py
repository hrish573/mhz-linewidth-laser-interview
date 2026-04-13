"""
Steady-state solver for the bad-cavity superradiant laser.

Solves the exact second-order cumulant equations (paper Eqs. 2-5) at delta=0.

Key physics:
    At resonance (delta=0), the atom-field coherence is purely imaginary:
        <a† sigma^-> = i*y    with y real.

    The 4 steady-state relations are:
        s_z  = d0 - 2*Omega*y / (w+gamma)                  from Eq.(2)
        y    = (Omega/2)*(n_ph*s_z + (s_z+1)/2 + (N-1)*s_s)/D  from Eq.(3)
        s_s  = Omega*s_z*y / Gamma                          from Eq.(4)
        n_ph = N*Omega*y / kappa                             from Eq.(5)

    where D = (w+gamma)/2 + 1/T2 + kappa/2.

    Substituting the other three relations into Eq.(3) gives a quadratic in y.

    The bad-cavity approximate version (Eq. 6) uses two steps.

    First, bad cavity means the atom-field coherence is a fast variable whose
    decay is dominated by kappa/2, so it can be adiabatically eliminated.

    Second, to isolate the collective superradiant branch, Eq.(3) drops the
    non-collective source terms: the (s_z+1)/2 spontaneous-emission seed and
    the n_ph*s_z cavity-stimulated term, keeping only the collective term
    (N-1)*s_s. These terms are not exactly zero in the bad-cavity limit; they
    are neglected because Eq.(6) is intended as a simplified collective-threshold
    approximation. That version has a trivial s_s=0 branch + a quadratic. The
    exact system generically has no y=0 root because spontaneous emission
    always seeds some coherence.
thi
Linewidth:
    From Eq.(9), the 2x2 regression matrix on resonance is:
        M = [[-kappa/2,         i*N*Omega/2    ],
            [-i*Omega*s_z/2,   -Gamma_half    ]]

    where Gamma_half = (w+gamma)/2 + 1/T2 (the atomic part of the
    atom-field coherence decay, without kappa/2).

    The linewidth is |Re(lambda_slow)|, the real part of the slower eigenvalue.
"""

import numpy as np
from .model import LaserParams, HBAR

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
def _build_quadratic_coeffs(p: LaserParams, w: float, N: int):
    """
    Build coefficients of a2*y^2 + a1*y + a0 = 0 from the exact
    steady-state system at delta=0.

    Derivation:
        Let alpha = 2*Omega/(w+gamma), so s_z = d0 - alpha*y.
        Let D = (w+gamma)/2 + 1/T2 + kappa/2.
        Let Gam = w + gamma + 2/T2.

        Eq.(3) at steady state, delta=0, with a_sigma = iy:
            y*D = (Omega/2) * [n_ph*s_z + (s_z+1)/2 + (N-1)*s_s]

        Substituting n_ph = N*Omega*y/kappa, s_s = Omega*s_z*y/Gam,
        s_z = d0 - alpha*y, and expanding:

        RHS = (Omega/2) * {
            (d0+1)/2                                               [y^0]
          + y * [N*Omega*d0/kappa - alpha/2 + (N-1)*Omega*d0/Gam]  [y^1]
          + y^2 * [-N*Omega*alpha/kappa - (N-1)*Omega*alpha/Gam]   [y^2]
        }

        Moving LHS = D*y to the right: 0 = a0 + a1*y + a2*y^2.

    Returns (a0, a1, a2).
    """
    d0 = p.d0(w)
    Gam = p.Gamma(w)       # w + gamma + 2/T2
    D = p.Gamma_half(w) + p.kappa / 2.0  # (w+gamma)/2 + 1/T2 + kappa/2
    alpha = 2.0 * p.Omega / (w + p.gamma)
    Om = p.Omega
    Om_half = Om / 2.0     # the outer factor in Eq.(3)

    # Constant term (y^0): (Omega/2) * (d0+1)/2
    a0 = Om_half * (d0 + 1.0) / 2.0 # spontaneous emission seed

    # Coefficient of y^1 (from RHS minus D*y from LHS)
    a1 = Om_half * (N * Om * d0 / p.kappa - alpha / 2.0 + (N - 1) * Om * d0 / Gam) - D

    # Coefficient of y^2
    a2 = -Om_half * (N * Om * alpha / p.kappa + (N - 1) * Om * alpha / Gam)

    return a0, a1, a2


def solve_steady_state(p: LaserParams, w: float, N: int):
    """
    Solve the exact steady-state cumulant equations at delta=0.

    Returns dict with keys: y, s_z, s_s, n_ph, power.

    The equation is a2*y^2 + a1*y + a0 = 0. We pick the physical root
    (real, and giving s_s >= 0 in the collective regime).

    When w=0, d0=-1 and a0=0, so y=0 is a root (non-lasing).
    When Omega=0, a0=a1=a2=0 trivially; return y=0.
    """
    if p.Omega == 0.0 or w == 0.0:
        d0 = p.d0(w) if w > 0 else -1.0
        return dict(y=0.0, s_z=d0, s_s=0.0, n_ph=0.0, power=0.0)

    a0, a1, a2 = _build_quadratic_coeffs(p, w, N)

    # Solve a2*y^2 + a1*y + a0 = 0
    if abs(a2) < 1e-30 * (abs(a1) + abs(a0)):
        # Degenerate: linear equation a1*y + a0 = 0
        if abs(a1) < 1e-30:
            y = 0.0
        else:
            y = -a0 / a1
    else:
        disc = a1**2 - 4.0 * a2 * a0
        if disc < 0:
            # No real roots — shouldn't happen for physical parameters
            # Fall back to y=0-like behavior
            y = 0.0
        else:
            sqrt_disc = np.sqrt(disc)
            y1 = (-a1 + sqrt_disc) / (2.0 * a2)
            y2 = (-a1 - sqrt_disc) / (2.0 * a2)

            # Physical constraints:
            #   y >= 0  (since n_ph = N*Omega*y/kappa >= 0)
            #   |s_z| <= 1
            # Above threshold: prefer root with largest s_s > 0 (collective emission)
            # Below threshold: prefer root with smallest y >= 0 (spontaneous seed only)
            candidates = []
            for yc in [y1, y2]:
                if np.isfinite(yc) and yc >= -1e-15:  # allow tiny negative from numerics
                    yc = max(yc, 0.0)
                    sz_c = p.d0(w) - 2.0 * p.Omega * yc / (w + p.gamma)
                    ss_c = p.Omega * sz_c * yc / p.Gamma(w)
                    if abs(sz_c) <= 1.0 + 1e-10:
                        candidates.append((yc, sz_c, ss_c))

            # Prefer the root with s_s > 0 (lasing); among those, pick largest s_s
            lasing = [(yc, sz_c, ss_c) for yc, sz_c, ss_c in candidates if ss_c >= 0]
            if lasing:
                best = max(lasing, key=lambda c: c[2])
                y = best[0]
            elif candidates:
                # Below threshold: pick smallest y (weak spontaneous emission)
                y = min(candidates, key=lambda c: c[0])[0]
            else:
                y = 0.0

    # Back-substitute to get all observables
    s_z = p.d0(w) - 2.0 * p.Omega * y / (w + p.gamma)
    Gam = p.Gamma(w)
    s_s = p.Omega * s_z * y / Gam
    n_ph = N * p.Omega * y / p.kappa
    power = HBAR * p.omega_a * p.kappa * n_ph

    return dict(y=y, s_z=s_z, s_s=s_s, n_ph=n_ph, power=power)


def solve_steady_state_approx(p: LaserParams, w: float, N: int):
    """
    Solve the approximate (Eq. 6) steady-state equations.

    This uses the bad-cavity approximation to adiabatically eliminate the
    fast atom-field coherence, and then makes an additional collective-only
    approximation in Eq.(3): it drops (s_z+1)/2 and n_ph*s_z, keeping only
    the collective term (N-1)*s_s ≈ N*s_s.

    The result is Eq.(6): s_s * (-Gamma + d0*N*gamma*C - 2*N^2*gamma^2*C^2*s_s/(w+gamma)) = 0

    Non-trivial solution: s_s = (w+gamma)*(d0*N*gamma*C - Gamma) / (2*N^2*gamma^2*C^2)

    Returns dict with keys: y, s_z, s_s, n_ph, power.
    """
    d0 = p.d0(w)
    Gam = p.Gamma(w)
    NgC = N * p.gamma * p.C  # = N * Omega^2 / kappa

    # Gain condition: d0 * NgC > Gamma
    gain = d0 * NgC - Gam

    if gain <= 0 or w <= 0:
        # Non-lasing branch
        return dict(y=0.0, s_z=d0 if w > 0 else -1.0, s_s=0.0, n_ph=0.0, power=0.0)

    # Non-trivial branch from Eq.(6)
    s_s = (w + p.gamma) * gain / (2.0 * NgC**2)

    # Back-substitute: s_z from Eq.(2)
    s_z = d0 - 2.0 * NgC * s_s / (w + p.gamma)

    # y from the adiabatic elimination: a_sigma = i*N*Omega*s_s/kappa, so y = N*Omega*s_s/kappa
    y = N * p.Omega * s_s / p.kappa

    # n_ph from Eq.(5)
    n_ph = N * p.Omega * y / p.kappa

    # Power
    power = HBAR * p.omega_a * p.kappa * n_ph

    return dict(y=y, s_z=s_z, s_s=s_s, n_ph=n_ph, power=power)


def compute_linewidth(p: LaserParams, w: float, N: int, s_z: float):
    """
    Compute the laser linewidth from the 2x2 regression matrix (Eq. 9).

    The matrix is:
        M = [[-kappa/2,            i*N*Omega/2    ],
             [-i*Omega*s_z/2,      -Gamma_half    ]]

    where Gamma_half = (w+gamma)/2 + 1/T2.

    The eigenvalue with the smallest |Re| gives the half-width at half-maximum
    (HWHM) of the Lorentzian power spectrum.  We return |Re(lambda_slow)|,
    consistent with the paper's plotted Δν (Fig. 3).

    On resonance, the eigenvalues are:
        lambda = (Tr/2) ± sqrt((Tr/2)^2 - det)

    with Tr = -(kappa/2 + Gamma_half),
         det = kappa*Gamma_half/2 - N*Omega^2*s_z/4.

    Note the MINUS sign in det: M12*M21 = (iNΩ/2)(-iΩs_z/2) = NΩ²s_z/4,
    so det = M11*M22 - M12*M21 = kappa*Gamma_half/2 - NΩ²s_z/4.

    Returns linewidth in s^-1 (HWHM, as plotted in the paper).
    """
    Gamma_half = p.Gamma_half(w)  # (w+gamma)/2 + 1/T2

    tr = -(p.kappa / 2.0 + Gamma_half)
    det = p.kappa * Gamma_half / 2.0 - N * p.Omega**2 * s_z / 4.0

    # Eigenvalues: lambda = tr/2 ± sqrt((tr/2)^2 - det)
    half_tr = tr / 2.0
    disc = half_tr**2 - det

    if disc >= 0:
        sqrt_disc = np.sqrt(disc)
        lam1 = half_tr + sqrt_disc
        lam2 = half_tr - sqrt_disc
    else:
        # Complex eigenvalues — both have same real part
        lam1 = half_tr
        lam2 = half_tr

    # The slow eigenvalue has smaller |Re|
    if abs(lam1.real if isinstance(lam1, complex) else lam1) < abs(lam2.real if isinstance(lam2, complex) else lam2):
        lam_slow = lam1
    else:
        lam_slow = lam2

    # HWHM = |Re(lambda_slow)| — matches paper's plotted Δν (Fig. 3)
    return abs(lam_slow.real if isinstance(lam_slow, complex) else lam_slow)


def compute_linewidth_metrics(p: LaserParams, w: float, N: int, s_z: float):
    """
    Return linewidth metrics in s^-1 for direct comparison.

    Returns dict with:
        linewidth: HWHM (paper convention)
        hwhm: HWHM
        fwhm: 2*HWHM
    """
    hwhm = compute_linewidth(p, w, N, s_z)
    return dict(linewidth=hwhm, hwhm=hwhm, fwhm=2.0 * hwhm)


def scan_w(p: LaserParams, N: int, w_array, exact=True):
    """
    Scan the pump rate w and return steady-state observables.

    Returns dict of arrays: y, s_z, s_s, n_ph, power, linewidth.
    """
    solver = solve_steady_state if exact else solve_steady_state_approx

    results = dict(y=[], s_z=[], s_s=[], n_ph=[], power=[], linewidth=[])

    for w in w_array:
        sol = solver(p, float(w), N)
        for key in ['y', 's_z', 's_s', 'n_ph', 'power']:
            results[key].append(sol[key])
        lw = compute_linewidth(p, float(w), N, sol['s_z'])
        results['linewidth'].append(lw)

    return {k: np.array(v) for k, v in results.items()}


def scan_wN(p: LaserParams, w_array, N_array, exact=True):
    """
    2D scan over pump rate w and atom number N.

    Returns dict of 2D arrays (shape [len(N_array), len(w_array)]):
        power, linewidth, s_s, s_z, n_ph.
    """
    nN = len(N_array)
    nw = len(w_array)
    solver = solve_steady_state if exact else solve_steady_state_approx

    power = np.zeros((nN, nw))
    linewidth = np.zeros((nN, nw))
    s_s = np.zeros((nN, nw))
    s_z = np.zeros((nN, nw))
    n_ph = np.zeros((nN, nw))

    for i, N in enumerate(N_array):
        for j, w in enumerate(w_array):
            sol = solver(p, float(w), int(N))
            power[i, j] = sol['power']
            s_s[i, j] = sol['s_s']
            s_z[i, j] = sol['s_z']
            n_ph[i, j] = sol['n_ph']
            linewidth[i, j] = compute_linewidth(p, float(w), int(N), sol['s_z'])

    return dict(power=power, linewidth=linewidth, s_s=s_s, s_z=s_z, n_ph=n_ph)
