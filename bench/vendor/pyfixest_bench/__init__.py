"""Verbatim copies of pyfixest v0.60.0 ``benchmarks/modular/{dgp_functions,akm_dgp}.py``.

Source: https://github.com/py-econometrics/pyfixest/tree/v0.60.0/benchmarks/modular
License: MIT, Copyright (c) 2022 pyfixest authors (see ``LICENSE`` in this directory).
``dgp_functions.py`` is itself adapted from kylebutts/fixest_benchmarks (MIT) and
``akm_dgp.py`` from tlamadon/bipartitepandas (MIT), as their headers state.

The two modules are unmodified so that ``verify_dgp.py`` can check them against
upstream by SHA-256 and check ``dgp.py`` output against them value-for-value.
"""

UPSTREAM_REF = "v0.60.0"
UPSTREAM_SHA256 = {
    "dgp_functions.py": "9ed8c5768a227c0d15b910635e146651ad88fdff2425c217fd2a804a708980e2",
    "akm_dgp.py": "963d516e11aa5f7676cfc0661613f63612f5daceac56431e46895907bd9891c9",
}
