"""pyfixest benchmark runner (plan Phase 8.3).

Usage: python run_pyfixest.py data=<parquet> fml=<formula> method=<feols|fepois|feglm_logit>
       vcov=<iid|hetero|cluster:<var>> demeaner=<map|lsmr> threads=<T> reps=<timed runs>
       [fixef_maxiter=<n>]

RAYON_NUM_THREADS (and the BLAS/OpenMP thread variables) are set to T before
pyfixest is imported. Prints one ``RESULT {json}`` line per run (burn-in first);
only the estimation call is timed (time.perf_counter) on an in-memory DataFrame.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time

args = dict(a.split("=", 1) for a in sys.argv[1:])
threads = args["threads"]
for var in ("RAYON_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[var] = threads

import pandas as pd  # noqa: E402
import pyfixest as pf  # noqa: E402


def main() -> None:
    reps = int(args["reps"])
    vcov: str | dict = args["vcov"]
    if vcov.startswith("cluster:"):
        vcov = {"CRV1": vcov.split(":", 1)[1]}
    kw: dict = {"vcov": vcov}
    dm = args["demeaner"]
    dkw = {"fixef_maxiter": int(args["fixef_maxiter"])} if "fixef_maxiter" in args else {}
    kw["demeaner"] = pf.MapDemeaner(**dkw) if dm == "map" else pf.LsmrDemeaner(**dkw)
    method, fml = args["method"], args["fml"]
    df = pd.read_parquet(args["data"])

    def estimate():
        if method == "feols":
            return pf.feols(fml, df, **kw)
        if method == "fepois":
            return pf.fepois(fml, df, **kw)
        if method == "feglm_logit":
            return pf.feglm(fml, df, family="logit", **kw)
        raise ValueError(method)

    nrun = 1 + reps
    i = 0
    while i < nrun:
        phase = "burnin" if i == 0 else "timed"
        t0 = time.perf_counter()
        try:
            fit = estimate()
            t = time.perf_counter() - t0
        except Exception as e:  # report and stop: later runs would fail the same way
            msg = f"{type(e).__name__}: {e}"[:500]
            print("RESULT " + json.dumps({"run": phase, "idx": i, "ok": False, "error": msg}), flush=True)
            break
        coef = fit.coef()
        coefs = {str(k): float(v) for k, v in coef.items() if math.isfinite(v)}
        rec = {"run": phase, "idx": i, "ok": True, "time": t, "n_obs": int(fit._N), "coefs": coefs}
        print("RESULT " + json.dumps(rec), flush=True)
        if i == 0 and t > 60:
            nrun = 2
        i += 1


if __name__ == "__main__":
    main()
