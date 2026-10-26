#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["mpmath>=1.3"]
# ///
"""Certificate that the Gaussian copula has the largest copula entropy on two elliptical families.

Why this certificate exists
    The adequacy argument rests on one theorem: at fixed R the Gaussian copula has the largest copula entropy in the elliptical class, so kappa <= 0 along any elliptical path, with equality only at the Gaussian point.
    This script checks the theorem in arbitrary precision on two elliptical families, the multivariate Student-t and Pearson VII, independently of the MATLAB code.
    It shares no code with src/: the entropies are closed forms in digamma and log-gamma, and rows A-C verify those closed forms against direct numerical quadrature before any inequality is asserted on them.

The object
    At R = I the copula entropy of a spherical law is minus its total correlation, so it is non-positive and zero only for the Gaussian generator:

        S(g) = h_joint(d) - d * h_marginal = -TC(Z_0) <= 0

Rows
    A  h_t closed form vs polar quadrature of -int f log f (d=2, nu=5)
    B  h_pvii closed form vs polar quadrature (d=2, N=3)
    C  both densities integrate to 1 (the quadrature in A and B is only meaningful if its measure is right)
    D  Student-t: S < 0 on the grid, and S -> 0 as nu -> infinity
    E  Pearson VII with the true marginal parameter N-(d-1)/2: S < 0, and it agrees with the equivalent t copula at nu = 2N-d
    F  Power check: the naive same-N margin gives a positive S on at least one cell, violating the theorem.
       The 1-dim margin of a d-dim Pearson VII(N) has parameter N-(d-1)/2, not N, so the same-N margin is not a margin of the law.
       That this wrong margin is detectably wrong shows that rows D and E can see the error they exist to catch.

Run:  uv run tests/oracles/ellipticalEntropyMaxCheck.py
It exits nonzero and names every failed row; otherwise it prints a PASS summary.
"""

from __future__ import annotations

import sys

import mpmath as mp

mp.mp.dps = 40

psi = mp.digamma
lg = mp.loggamma

# Quadrature reproduces a dps=40 closed form to far better than this; the level is set well above the observed residual so a tolerance never has to move to keep a row green.
# Measured residuals are printed beside it on every run.
TOL_CLOSED_FORM = mp.mpf("1e-25")
TOL_MASS = mp.mpf("1e-25")
TOL_CROSS = mp.mpf("1e-25")
# S -> 0 only in the limit, so the approach is checked at a large but finite nu.
LIMIT_PARAM = mp.mpf("1e5")
TOL_LIMIT = mp.mpf("1e-3")

DIMS = (2, 3)
T_DOF = (3, 5, 10, 30, 100)
PVII_N = (2, 3, 5, 20)


def h_t(d, nu):
    """Differential entropy of the d-dim multivariate Student-t, scale = I."""
    d, nu = mp.mpf(d), mp.mpf(nu)
    return lg(nu / 2) - lg((nu + d) / 2) + (d / 2) * mp.log(nu * mp.pi) + ((nu + d) / 2) * (psi((nu + d) / 2) - psi(nu / 2))


def h_pvii(d, N):
    """Differential entropy of d-dim Pearson VII, density prop [1+x'x]^-N, N > d/2."""
    d, N = mp.mpf(d), mp.mpf(N)
    return N * (psi(N) - psi(N - d / 2)) - (lg(N) - (d / 2) * mp.log(mp.pi) - lg(N - d / 2))


def s_t(d, nu):
    """Copula entropy of the d-dim t, whose 1-dim margin is univariate t with the same nu."""
    return h_t(d, nu) - d * h_t(1, nu)


def s_pvii(d, N):
    """Copula entropy of the d-dim Pearson VII, using the true marginal parameter."""
    return h_pvii(d, N) - d * h_pvii(1, mp.mpf(N) - (mp.mpf(d) - 1) / 2)


def s_pvii_naive(d, N):
    """The same N on the margin, which is not a copula entropy; used by row F only."""
    return h_pvii(d, N) - d * h_pvii(1, N)


def _polar_entropy_2d(density, log_density):
    """h = -int_R2 f log f = -int_0^inf 2 pi r f(r) log f(r) dr for a radial f."""
    return mp.quad(lambda r: -2 * mp.pi * r * density(r) * log_density(r), [0, 1, 5, 20, 100, mp.inf])


def _polar_mass_2d(density):
    """Total mass int_0^inf 2 pi r f(r) dr of a radial density on R^2."""
    return mp.quad(lambda r: 2 * mp.pi * r * density(r), [0, 1, 5, 20, 100, mp.inf])


def main() -> int:
    """Run rows A-F and return 1 when any row fails, 0 otherwise."""
    failures: list[str] = []

    def check(row: str, ok: bool, message: str) -> None:
        status = "PASS" if ok else "FAIL"
        print(f"[ellipticalEntropyMax] {row}    {status}  {message}")
        if not ok:
            failures.append(row)

    # --- the two densities rows A-C integrate --------------------------------
    nu_a = mp.mpf(5)
    c_t = mp.gamma((nu_a + 2) / 2) / (mp.gamma(nu_a / 2) * nu_a * mp.pi)
    dens_t = lambda r: c_t * (1 + r**2 / nu_a) ** (-(nu_a + 2) / 2)
    logf_t = lambda r: mp.log(c_t) - ((nu_a + 2) / 2) * mp.log(1 + r**2 / nu_a)

    n_b = mp.mpf(3)
    c_p = mp.gamma(n_b) / (mp.pi * mp.gamma(n_b - 1))
    dens_p = lambda r: c_p * (1 + r**2) ** (-n_b)
    logf_p = lambda r: mp.log(c_p) - n_b * mp.log(1 + r**2)

    # A -----------------------------------------------------------------------
    closed, numeric = h_t(2, 5), _polar_entropy_2d(dens_t, logf_t)
    residual = abs(closed - numeric)
    check("A", residual <= TOL_CLOSED_FORM, f"h_t(2,5) closed {mp.nstr(closed, 12)} vs quadrature {mp.nstr(numeric, 12)} (|diff| = {mp.nstr(residual, 3)}, tol {mp.nstr(TOL_CLOSED_FORM, 3)})")

    # B -----------------------------------------------------------------------
    closed, numeric = h_pvii(2, 3), _polar_entropy_2d(dens_p, logf_p)
    residual = abs(closed - numeric)
    check("B", residual <= TOL_CLOSED_FORM, f"h_pvii(2,3) closed {mp.nstr(closed, 12)} vs quadrature {mp.nstr(numeric, 12)} (|diff| = {mp.nstr(residual, 3)}, tol {mp.nstr(TOL_CLOSED_FORM, 3)})")

    # C -----------------------------------------------------------------------
    mass_t, mass_p = _polar_mass_2d(dens_t), _polar_mass_2d(dens_p)
    worst_mass = max(abs(mass_t - 1), abs(mass_p - 1))
    check("C", worst_mass <= TOL_MASS, f"both densities integrate to 1 (worst |mass-1| = {mp.nstr(worst_mass, 3)}, tol {mp.nstr(TOL_MASS, 3)})")

    # D -----------------------------------------------------------------------
    worst_s = max(s_t(d, nu) for d in DIMS for nu in T_DOF)
    limits = [s_t(d, LIMIT_PARAM) for d in DIMS]
    ok = worst_s < 0 and all(value < 0 and abs(value) <= TOL_LIMIT for value in limits)
    check("D", ok, f"Student-t: max S over d in {list(DIMS)}, nu in {list(T_DOF)} is {mp.nstr(worst_s, 8)} < 0; at nu = 1e5 S = {', '.join(mp.nstr(v, 4) for v in limits)} (-> 0 from below)")

    # E -----------------------------------------------------------------------
    worst_s = max(s_pvii(d, N) for d in DIMS for N in PVII_N)
    worst_cross = max(abs(s_pvii(d, N) - s_t(d, 2 * mp.mpf(N) - d)) for d in DIMS for N in PVII_N)
    check("E", worst_s < 0 and worst_cross <= TOL_CROSS, f"Pearson VII with the true marginal N-(d-1)/2: max S = {mp.nstr(worst_s, 8)} < 0, and it matches the equivalent t at nu = 2N-d to {mp.nstr(worst_cross, 3)} (tol {mp.nstr(TOL_CROSS, 3)})")

    # F -- power check --------------------------------------------------------
    naive = [(d, N, s_pvii_naive(d, N)) for d in DIMS for N in PVII_N]
    positive = [(d, N, value) for d, N, value in naive if value > 0]
    check("F", bool(positive), f"POWER CHECK: the naive same-N margin violates the theorem on {len(positive)} of {len(naive)} cells, worst S = {mp.nstr(max((v for _, _, v in positive), default=mp.mpf(0)), 8)} > 0 (so rows D/E can see a wrong marginal)")

    if failures:
        print(f"\n[ellipticalEntropyMax] FAILED rows: {', '.join(failures)}", file=sys.stderr)
        return 1
    print("[ellipticalEntropyMax] PASS: the entropy-maximum theorem holds on both non-PE elliptical families, and their closed forms match quadrature.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
