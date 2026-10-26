# t-copula kappa reference

**What this is.**
`t_copula_kappa_reference.csv` is an arbitrary-precision table of the t-copula shape term kappa_t(d, nu), the whole beyond-Gaussian part of the t-copula entropy H_c(R; d, nu) = 1/2 log det R + kappa_t(d, nu).
It has 170 rows: d in {2, 3, 4, 5, 6, 8, 10, 12, 16, 20} crossed with nu in {1, 1.5, 2, 3, 4, 5, 7, 10, 15, 20, 30, 50, 100, 300, 1000, 1e4, 1e6}.
The columns are `d, nu, p, kappa_t, kappa_t_dm1`, with p = 1/nu and `kappa_t_dm1` = kappa_t(d-1, nu), the leave-one-out margin term (identically 0 at d = 2, because kappa_t(1, nu) = h_1 - h_1).
Every value is printed at 20 significant digits, well past what any double-precision consumer can use, because the file is a reference, not a set of expected outputs.

**Why it exists.**
kappa_t is a small difference of terms of order d log nu whose limit is 0, so the cancellation grows with nu.
At nu = 1e6 the table's values run from -5.0e-13 (d = 2) to -9.5e-11 (d = 20), so about twelve significant digits cancel from inputs of order 150.
A double-precision implementation cannot be its own witness there, and checking it against a second double-precision transcription of the same formula would only show that the algebra was retyped correctly.

**Method.**
`t_copula_kappa_reference.py` (PEP 723, `mpmath>=1.3`, `mp.dps = 40`) evaluates the closed form:

    h_t(d, nu) = lgamma(nu/2) - lgamma((nu+d)/2) + (d/2) log(nu pi)
                 + ((nu+d)/2) (psi((nu+d)/2) - psi(nu/2))
    kappa_t(d, nu) = h_t(d, nu) - d h_t(1, nu)

The closed form is exact for the t family because every 1-dim margin of the d-variate t is univariate t with the same nu.
That closure is also what makes the margin column a re-evaluation at d-1 rather than a separate integral.
The generator reads no repository code except for its own cross-check.

**Self cross-check, and the two properties asserted before anything is written.**
Row A imports `s_t` from `tests/oracles/ellipticalEntropyMaxCheck.py` by file path (`importlib.util`) and requires agreement below 1e-30 on eight spread cells; the observed residual is exactly 0.0, both transcriptions being identical at dps = 40.
Row B requires kappa_t < 0 on all 170 cells (worst -4.9999967e-13, at d = 2, nu = 1e6), kappa_t(d-1, nu) <= 0 on all of them, and |kappa_t| < 1e-3 at nu = 1e6: the sign and limit structure of the entropy-maximum theorem, which is the only thing a consumer may assume without reading the values.
The CSV is written only if both rows pass, so a typo in the closed form fails before the file exists rather than after it is trusted.

**Command.**

    uv run tests/oracles/t_copula_kappa_reference.py

The run is deterministic and overwrites `t_copula_kappa_reference.csv` beside the script; when a row fails it exits nonzero, names that row and writes nothing.
Run it through `uv` as shown, so that the script's PEP 723 header supplies mpmath.

**What consumes it.**
`tests/tCopulaKernelTest.m`, methods `copulaEntropyMatchesTheMpmathOracle` and `copulaEntropyShapeTermHasItsStructuralProperties`.
The first asserts `tCopulaEntropy(d, 1/nu).kappa` against `kappa_t` and `.kappa_margin` against `kappa_t_dm1` at 1e-10 absolute on every row, nu = 1e6 included.
The second uses the CSV only for its (d, nu) grid and asserts structure (sign, the exact zero at p = 0, the O(p^2) approach) against no external value at all.
Nothing else in the repository reads this file.

**Scope limit.**
This is a reference for the shape term only.
It says nothing about `tTailQuantile`, whose accuracy below nu = 1e5 is bounded by MATLAB's `betaincinv` and is documented at the assertions in `tCopulaKernelTest.m` rather than here.
