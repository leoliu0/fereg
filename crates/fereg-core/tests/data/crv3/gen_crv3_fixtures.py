"""pyfixest 0.60.0 CRV3 reference for `crates/fereg-core/src/crv3/tests.rs` (`matches_pyfixest`).

Run on uts: ~/fereg-dev/pyenv/bin/python gen_crv3_fixtures.py (writes data.csv, expected.txt here).

Cases: OLS without FE (pyfixest's closed-form fast path), OLS with one FE not nested in the
cluster (slow refit path), Poisson with that FE (slow path), each unweighted and with aweights.
One FE keeps pyfixest's own demeaning exact (alternating projections converge in one sweep), so
its refits, which run at pyfixest's default FE tolerance, are not limited by the FE solver.
pyfixest's IRLS default (iwls_tol 1e-8) leaves ~1e-9 relative error in its Poisson coefficients
and ~1e-8 in the jackknife meat; `fepois` is therefore wrapped with iwls_tol 1e-12 (the CRV3
code itself is pyfixest's: its slow path looks `fepois` up in `pyfixest.estimation` per refit,
so the main fit and every refit use the tight tolerance).
expected.txt lines: `<case> <key> <values...>`, keys coef, vcov (k x k row-major), df_k, N, G.
"""

import functools
import os

import numpy as np
import pandas as pd
import pyfixest as pf
import pyfixest.estimation as pfe

assert pf.__version__ == "0.60.0", pf.__version__
pfe.fepois = functools.partial(pfe.fepois, iwls_tol=1e-12, iwls_maxiter=200)

here = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(20261013)
n, n_cl, n_fe = 600, 20, 15
cl = rng.integers(0, n_cl, n)
fe1 = rng.integers(0, n_fe, n)
alpha = rng.normal(size=n_fe)
shock = rng.normal(size=n_cl)
x1 = rng.normal(size=n) + 0.3 * alpha[fe1]
x2 = rng.normal(size=n) + 0.5 * shock[cl]
w = rng.uniform(0.5, 2.0, n)
y = 1.0 + 0.5 * x1 - 0.3 * x2 + alpha[fe1] + rng.normal(size=n) + 0.6 * shock[cl]
yp = rng.poisson(np.exp(0.2 + 0.3 * x1 - 0.2 * x2 + 0.3 * alpha[fe1] + 0.2 * shock[cl])).astype(float)
df = pd.DataFrame({"y": y, "yp": yp, "x1": x1, "x2": x2, "fe1": fe1, "cl": cl, "w": w})
assert df.groupby("fe1").yp.sum().min() > 0 and df.fe1.value_counts().min() > 1
assert sorted(df.cl.unique()) == list(range(n_cl)) and sorted(df.fe1.unique()) == list(range(n_fe))
df.to_csv(os.path.join(here, "data.csv"), index=False, float_format="%.17g")

cases = [
    ("ols_nofe", pfe.feols, "y ~ x1 + x2", None),
    ("ols_nofe_w", pfe.feols, "y ~ x1 + x2", "w"),
    ("ols_fe", pfe.feols, "y ~ x1 + x2 | fe1", None),
    ("ols_fe_w", pfe.feols, "y ~ x1 + x2 | fe1", "w"),
    ("pois_fe", pfe.fepois, "yp ~ x1 + x2 | fe1", None),
    ("pois_fe_w", pfe.fepois, "yp ~ x1 + x2 | fe1", "w"),
]
fmt = lambda v: " ".join(repr(float(x)) for x in np.ravel(v))
lines = []
for name, est, fml, weights in cases:
    m = est(fml, data=df, vcov={"CRV3": "cl"}, weights=weights)
    assert m._N == n
    lines += [
        f"{name} coef {fmt(m.coef().to_numpy())}",
        f"{name} vcov {fmt(m._vcov)}",
        f"{name} df_k {fmt([m._df_k])}",
        f"{name} N {fmt([m._N])}",
        f"{name} G {fmt([m._G[0]])}",
    ]
with open(os.path.join(here, "expected.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))
