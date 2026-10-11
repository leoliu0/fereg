"""Check dgp.py against upstream pyfixest v0.60.0 (plan Phase 8.1).

1. The vendored modules are byte-identical to upstream (SHA-256 of the files
   fetched from GitHub at the pinned tag).
2. ``dgp.base_dgp_latent`` reproduces upstream ``base_dgp(n, type_, k=10, seed)``
   column-for-column, bit-for-bit (dtype and raw bytes), for small n.
3. ``dgp.akm_config`` equals the config upstream ``dgps.py`` builds for every
   scenario, and ``dgp.akm_dgp`` equals upstream ``simulate_akm_panel`` output.

Usage: uv run python verify_dgp.py
"""

from __future__ import annotations

import hashlib
import importlib
import sys
import tempfile
import urllib.request
from dataclasses import asdict
from pathlib import Path

import numpy as np

import dgp
from vendor import pyfixest_bench

BASE = "https://raw.githubusercontent.com/py-econometrics/pyfixest/{ref}/benchmarks/modular/{f}"
FILES = ["dgp_functions.py", "akm_dgp.py", "dgps.py", "interfaces.py"]


def fetch(tmp: Path) -> None:
    for f in FILES:
        url = BASE.format(ref=pyfixest_bench.UPSTREAM_REF, f=f)
        (tmp / f).write_bytes(urllib.request.urlopen(url, timeout=60).read())


def same_frame(a, b, label: str) -> None:
    assert list(a.columns) == list(b.columns), (label, list(a.columns), list(b.columns))
    for c in a.columns:
        x, y = a[c].to_numpy(), b[c].to_numpy()
        assert x.dtype == y.dtype, (label, c, x.dtype, y.dtype)
        assert x.tobytes() == y.tobytes(), (label, c)


def main() -> None:
    here = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        fetch(tmp)
        for f, sha in pyfixest_bench.UPSTREAM_SHA256.items():
            up = hashlib.sha256((tmp / f).read_bytes()).hexdigest()
            local = hashlib.sha256((here / "vendor/pyfixest_bench" / f).read_bytes()).hexdigest()
            assert up == sha == local, (f, up, sha, local)
        print("vendored files: sha256 identical to upstream")

        sys.path.insert(0, str(tmp))
        up_fun = importlib.import_module("dgp_functions")
        up_akm = importlib.import_module("akm_dgp")
        up_dgps = importlib.import_module("dgps")

        for n in (1_000, 10_000, 100_000):
            for t in ("simple", "difficult"):
                seed = dgp.fixest_seed(n, t)
                ref = up_fun.base_dgp(n=n, type_=t, k=dgp.MAX_K, seed=seed)
                port, _ = dgp.base_dgp_latent(n, t, seed)
                same_frame(port, ref, f"base_dgp {t} n={n}")
                full = dgp.fixest_dgp(n, t)
                same_frame(full[ref.columns], ref, f"fixest_dgp base cols {t} n={n}")
        print("fixest DGP: port identical to upstream base_dgp (n = 1e3, 1e4, 1e5)")

        sweep = {d.dgp_name: d for d in up_dgps.get_akm_sweep_scenarios(tmp)}
        occ = {d.dgp_name: d for d in up_dgps.get_akm_occupation_scenarios(tmp)}
        for name, _, _ in dgp.AKM_SCENARIOS:
            upd = occ[name.replace("akmocc_", "akm_")] if name.startswith("akmocc_") else sweep[name]
            for n in (10_000, 1_000_000):
                assert asdict(upd._build_config(n=n)) == asdict(dgp.akm_config(name, n)), (name, n)
        print(f"AKM: configs identical to upstream dgps.py for all {len(dgp.AKM_SCENARIOS)} scenarios")

        for name in ("akm_baseline", "akm_freeze_3", "akmocc_occsize_2"):
            n = 20_000
            ref = up_akm.simulate_akm_panel(dgp.akm_config(name, n), seed=dgp.akm_seed(name, n))
            same_frame(dgp.akm_dgp(name, n), ref, f"akm {name}")
        print("AKM: dgp.akm_dgp identical to upstream simulate_akm_panel (n = 2e4)")

    df = dgp.fixest_dgp(100_000, "simple")
    assert np.isfinite(df[["z1", "z2", "xe", "y_iv"]].to_numpy()).all()
    print("IV extension: finite; corr(xe, z1) = %.3f" % np.corrcoef(df.xe, df.z1)[0, 1])
    print("VERIFY OK")


if __name__ == "__main__":
    main()
