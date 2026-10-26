#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["mpmath>=1.3"]
# ///
"""Arbitrary-precision reference for the t-copula shape term kappa_t(d, nu), written as a CSV that the MATLAB kernel tests assert against.

Why this reference exists
    `tCopulaEntropy` computes kappa_t(d, nu) in double precision from log-gamma and digamma terms, and the quantity is a difference of two O(d)-sized entropies whose limit is 0.
    As nu grows the two terms agree to more and more digits, and the difference is what survives: at nu = 1e6 about twelve significant digits cancel.
    A double-precision implementation cannot be its own witness there, and comparing it with a second double-precision implementation of the same formula would only show that the algebra was retyped correctly.

    This script computes the same closed form at mp.dps = 40, so the cancellation costs digits the reference has to spare and the surviving value is exact far beyond double precision.
    It reads no repository code except to cross-check itself (see below), and it is the only reference the kernel tests use for kappa.

The object
    The differential entropy of the standard d-variate Student-t with nu degrees of freedom and scale I:

        h_t(d, nu) = lgamma(nu/2) - lgamma((nu+d)/2) + (d/2) log(nu pi)
                     + ((nu+d)/2) (psi((nu+d)/2) - psi(nu/2))

    Every 1-dim margin of the d-variate t is univariate t with the same nu, which gives the copula shape term:

        kappa_t(d, nu) = h_t(d, nu) - d h_t(1, nu)

    kappa_t is the whole beyond-Gaussian part of the t-copula entropy, H_c(R; d, nu) = 1/2 log det R + kappa_t(d, nu).
    It is <= 0 with equality only in the Gaussian limit nu -> infinity (p = 1/nu -> 0), which is the entropy-maximum theorem `ellipticalEntropyMaxCheck.py` checks.

    The CSV also carries kappa_t(d-1, nu), the leave-one-out margin term the DTC decomposition needs.
    For the t family it is kappa_t evaluated at d-1 by the same code path, and at d = 2 it is identically 0 because kappa_t(1, nu) = h_t(1, nu) - h_t(1, nu).

Self-checks before anything is written
    Row A imports `s_t` from `ellipticalEntropyMaxCheck.py` beside this script, an independent transcription of the same closed form written for rows D and E of that certificate, and requires agreement below 1e-30 on eight cells spread over the grid.
    A typo in either transcription fails this before the CSV exists.
    That certificate sets mp.dps = 40 at import, the precision this script wants anyway; this script states the assignment itself rather than inheriting it, so reordering the imports can never silently lower it.
    Row B requires kappa_t < 0 and kappa_t(d-1, nu) <= 0 on every cell, and |kappa_t| < 1e-3 at nu = 1e6.

Run:  uv run tests/oracles/t_copula_kappa_reference.py
It writes t_copula_kappa_reference.csv beside this script, every value at 20 significant digits.
On any disagreement it exits nonzero, names the failed row and writes nothing; otherwise it prints a PASS summary.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import mpmath as mp

mp.mp.dps = 40

psi = mp.digamma
lg = mp.loggamma

# The grid: d samples the study's channel-set sizes from 2 to 20, and nu runs from the heavy-tailed end (nu = 1 is Cauchy, where the t has no mean) to nu = 1e6, a cell that exists only to exercise the cancellation.
DIMS = (2, 3, 4, 5, 6, 8, 10, 12, 16, 20)
DOF = ("1", "1.5", "2", "3", "4", "5", "7", "10", "15", "20", "30", "50", "100", "300", "1000", "1e4", "1e6")

# Two dps=40 transcriptions of one closed form differ by rounding only, so the level is set far below anything a retyping error could hide under and far above the observed residual.
# The worst residual is printed on every run.
TOL_CROSS = mp.mpf("1e-30")
CROSS_CELLS = ((2, "3"), (2, "100"), (3, "5"), (5, "10"), (5, "1e6"), (20, "1"), (20, "30"), (20, "1e6"))

SIGNIFICANT_DIGITS = 20
# The certificate this script cross-checks against sits beside it.
CERTIFICATE = Path(__file__).resolve().parent / "ellipticalEntropyMaxCheck.py"


def h_t(d, nu):
    """Differential entropy of the d-variate standard Student-t, scale = I."""
    d, nu = mp.mpf(d), mp.mpf(nu)
    return lg(nu / 2) - lg((nu + d) / 2) + (d / 2) * mp.log(nu * mp.pi) + ((nu + d) / 2) * (psi((nu + d) / 2) - psi(nu / 2))


def kappa_t(d, nu):
    """Copula shape term h_t(d, nu) - d h_t(1, nu); 0 at d = 1 by construction."""
    return h_t(d, nu) - mp.mpf(d) * h_t(1, nu)


def load_certificate_s_t():
    """Import s_t from the certificate ellipticalEntropyMaxCheck.py by file path, not by package."""
    if not CERTIFICATE.is_file():
        raise FileNotFoundError(f"expected the existing certificate at {CERTIFICATE}; the cross-check has no second transcription to compare against")
    spec = importlib.util.spec_from_file_location("ellipticalEntropyMaxCheck", CERTIFICATE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # The certificate sets mp.dps = 40 at import.
    # Restate it so this file's precision is a property of this file.
    mp.mp.dps = 40
    return module.s_t


def main() -> int:
    """Run rows A and B, write the CSV only when both pass, and return the exit code."""
    failures: list[str] = []

    def check(row: str, ok: bool, message: str) -> None:
        status = "PASS" if ok else "FAIL"
        print(f"[tCopulaKappaReference] {row}    {status}  {message}")
        if not ok:
            failures.append(row)

    # --- A: this transcription against the certificate's -----------------------
    s_t = load_certificate_s_t()
    residuals = [(d, nu, abs(kappa_t(d, mp.mpf(nu)) - s_t(d, mp.mpf(nu)))) for d, nu in CROSS_CELLS]
    worst_d, worst_nu, worst = max(residuals, key=lambda item: item[2])
    check("A", worst <= TOL_CROSS, f"kappa_t agrees with ellipticalEntropyMaxCheck.s_t on {len(residuals)} cells; worst |diff| = {mp.nstr(worst, 3)} at (d={worst_d}, nu={worst_nu}) (tol {mp.nstr(TOL_CROSS, 3)})")

    # --- B: the sign and limit structure the CSV's consumers rely on ----------
    table = []
    for d in DIMS:
        for nu_text in DOF:
            nu = mp.mpf(nu_text)
            kappa = kappa_t(d, nu)
            margin = mp.mpf(0) if d == 2 else kappa_t(d - 1, nu)
            table.append((d, nu_text, nu, kappa, margin))

    worst_kappa = max(row[3] for row in table)
    limit_cells = [row for row in table if row[1] == "1e6"]
    worst_limit = max(abs(row[3]) for row in limit_cells)
    ok = worst_kappa < 0 and all(row[3] < 0 for row in table) and all(row[4] <= 0 for row in table) and worst_limit < mp.mpf("1e-3")
    check("B", ok, f"kappa_t < 0 on all {len(table)} cells (max = {mp.nstr(worst_kappa, 8)}), kappa_t(d-1, nu) <= 0 on all of them, and |kappa_t| <= {mp.nstr(worst_limit, 3)} at nu = 1e6")

    # --- C: write the CSV -----------------------------------------------------
    if failures:
        print(f"\n[tCopulaKappaReference] FAILED rows: {', '.join(failures)}; no CSV written", file=sys.stderr)
        return 1

    destination = Path(__file__).resolve().with_suffix(".csv")
    lines = ["d,nu,p,kappa_t,kappa_t_dm1"]
    for d, nu_text, nu, kappa, margin in table:
        lines.append(",".join([str(d), mp.nstr(nu, SIGNIFICANT_DIGITS), mp.nstr(1 / nu, SIGNIFICANT_DIGITS), mp.nstr(kappa, SIGNIFICANT_DIGITS), mp.nstr(margin, SIGNIFICANT_DIGITS)]))
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[tCopulaKappaReference] C       PASS  wrote {len(table)} rows to {destination.name} at {SIGNIFICANT_DIGITS} significant digits")

    print("[tCopulaKappaReference] PASS: kappa_t(d, nu) computed at dps = 40, cross-checked against the existing certificate, CSV written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
