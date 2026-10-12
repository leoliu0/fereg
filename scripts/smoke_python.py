"""Release smoke test of an installed fereg wheel: version and a tiny feols.

Usage: python scripts/smoke_python.py [expected-version]
Run it outside the repository's python/ directory so the installed package is imported.
"""

import sys

import numpy as np
import pandas as pd

import fereg

expected = sys.argv[1] if len(sys.argv) > 1 else None
print(f"fereg {fereg.__version__} from {fereg.__file__} on Python {sys.version.split()[0]}")
if expected is not None and fereg.__version__ != expected:
    sys.exit(f"installed fereg {fereg.__version__} != expected {expected}")

# Deterministic data with two fixed effects; the reference is OLS on explicit dummies.
n = 2000
i = np.arange(n)
f1 = i % 37
f2 = (i * 7) % 11
x = np.sin(0.37 * i) + 0.1 * (i % 5)
y = 1.5 * x + 0.3 * f1 - 0.2 * f2 + 0.05 * np.cos(1.3 * i)
df = pd.DataFrame({"y": y, "x": x, "f1": f1, "f2": f2})

design = np.column_stack(
    [x, (f1[:, None] == np.arange(37)).astype(float), (f2[:, None] == np.arange(1, 11)).astype(float)]
)
ref = np.linalg.lstsq(design, y, rcond=None)[0][0]

fit = fereg.feols("y ~ x | f1 + f2", df)
got = float(fit.coef()["x"])
print(f"coef(x): fereg {got:.12f}, dummy OLS {ref:.12f}")
if not np.isclose(got, ref, rtol=1e-7, atol=0):
    sys.exit("fereg.feols coefficient does not match the dummy-variable OLS reference")
if fit._N != n:
    sys.exit(f"nobs {fit._N} != {n}")
print("python smoke test passed")
