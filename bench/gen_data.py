"""Lazy, cached benchmark datasets (plan Phase 8.2).

A dataset name is either ``fx_<simple|difficult>_<n>`` (fixest DGP incl. the IV
columns) or ``<akm scenario>_<n>`` (e.g. ``akm_sorting_2_1000000``).
Per dataset, under ``bench/data/``:

* ``<name>.parquet``  canonical copy, read by pyfixest
* ``<name>.csv``      read by fixest via ``data.table::fread``
* ``<name>.dta``      Stata (pandas ``to_stata(version=118)``)
* ``<name>.meta.json`` generator parameters; a mismatch regenerates everything

Files are written to a temporary name and renamed, so an interrupted run never
leaves a truncated file behind.

Usage: uv run python gen_data.py NAME [NAME ...] [--formats parquet,csv,dta]
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

import dgp

DATA_DIR = Path(__file__).resolve().parent / "data"
FORMATS = ("parquet", "csv", "dta")
# Bump when the content of any generated column changes.
GENERATOR_VERSION = 1

_FX = re.compile(r"^fx_(simple|difficult)_(\d+)$")
_AKM = re.compile(r"^(akm(?:occ)?_[a-z]+(?:_\d+)?)_(\d+)$")


def dataset_name_fx(type_: str, n: int) -> str:
    return f"fx_{type_}_{n}"


def dataset_name_akm(scenario: str, n: int) -> str:
    return f"{scenario}_{n}"


def _params(name: str) -> dict:
    if m := _FX.match(name):
        t, n = m.group(1), int(m.group(2))
        return {"kind": "fixest", "type": t, "n": n, "seed": dgp.fixest_seed(n, t)}
    if (m := _AKM.match(name)) and m.group(1) in dgp.AKM_INDEX:
        s, n = m.group(1), int(m.group(2))
        cfg = dgp.akm_config(s, n)
        return {
            "kind": "akm",
            "scenario": s,
            "n": n,
            "seed": dgp.akm_seed(s, n),
            "config": {k: list(v) if isinstance(v, tuple) else v for k, v in vars(cfg).items()},
        }
    raise ValueError(f"unknown dataset name {name!r}")


def _generate(p: dict) -> pd.DataFrame:
    if p["kind"] == "fixest":
        return dgp.fixest_dgp(p["n"], p["type"])
    return dgp.akm_dgp(p["scenario"], p["n"])


def _path(name: str, fmt: str) -> Path:
    return DATA_DIR / f"{name}.{fmt}"


def _atomic(path: Path, write) -> None:
    tmp = path.with_name(path.name + ".tmp")
    write(tmp)
    os.replace(tmp, path)


def _for_stata(df: pd.DataFrame) -> pd.DataFrame:
    """int64 -> int32 (Stata long) or int8 (byte) where the values fit."""
    out = {}
    for c in df.columns:
        v = df[c].to_numpy()
        if v.dtype.kind in "iub":
            lo, hi = (int(v.min()), int(v.max())) if len(v) else (0, 0)
            if -127 <= lo and hi <= 100:
                v = v.astype(np.int8)
            elif -2_147_483_647 <= lo and hi <= 2_147_483_620:
                v = v.astype(np.int32)
            else:
                v = v.astype(np.float64)
        out[c] = v
    return pd.DataFrame(out)


def ensure(name: str, formats: tuple[str, ...] | list[str] = FORMATS) -> dict[str, Path]:
    """Generate (if missing or stale) and return the paths of the requested formats."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    p = {**_params(name), "generator_version": GENERATOR_VERSION}
    meta = _path(name, "meta.json")
    fresh = meta.exists() and json.loads(meta.read_text()) == p
    if not fresh:
        for f in FORMATS:
            _path(name, f).unlink(missing_ok=True)
        meta.unlink(missing_ok=True)

    df: pd.DataFrame | None = None
    pq_path = _path(name, "parquet")
    if not pq_path.exists():
        print(f"[data] generating {name}", flush=True)
        df = _generate(p)
        _atomic(pq_path, lambda t: df.to_parquet(t, index=False))
        meta.write_text(json.dumps(p, sort_keys=True))
    for f in formats:
        path = _path(name, f)
        if f == "parquet" or path.exists():
            continue
        if df is None:
            df = pd.read_parquet(pq_path)
        print(f"[data] writing {path.name}", flush=True)
        if f == "csv":
            opts = pacsv.WriteOptions(quoting_style="none")
            _atomic(path, lambda t: pacsv.write_csv(pa.Table.from_pandas(df, preserve_index=False), t, opts))
        elif f == "dta":
            sdf = _for_stata(df)
            _atomic(path, lambda t: sdf.to_stata(t, write_index=False, version=118))
        else:
            raise ValueError(f"unknown format {f!r}")
    return {f: _path(name, f) for f in formats}


def num_rows(name: str) -> int:
    return pq.read_metadata(ensure(name, ("parquet",))["parquet"]).num_rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="+")
    ap.add_argument("--formats", default=",".join(FORMATS))
    a = ap.parse_args()
    for name in a.names:
        for f, path in ensure(name, a.formats.split(",")).items():
            print(f"{name} {f} {path} {path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
