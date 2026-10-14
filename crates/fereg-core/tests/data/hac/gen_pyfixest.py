"""pyfixest 0.60.0 references for hac.rs (NW panel / time series, DK).

Run (uts): ~/fereg-dev/pyenv/bin/python gen_pyfixest.py   (after gen_data.py)
Writes pyfixest_ref.csv (case,name,i,j,value).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf

assert pf.__version__ == "0.60.0", pf.__version__
HERE = Path(__file__).resolve().parent
panel = pd.read_csv(HERE / "panel.csv")
ts = pd.read_csv(HERE / "ts.csv")

rows = []


def put(case, name, value, i=0, j=0):
    rows.append(f"{case},{name},{i},{j},{float(value)!r}")


def run(case, data, vcov, kwargs, weights=None):
    fit = pf.feols("y ~ x1 + x2", data=data, vcov=vcov, vcov_kwargs=kwargs, weights=weights)
    v = np.asarray(fit._vcov)
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            put(case, "V", v[i, j], i, j)
    for j, b in enumerate(np.asarray(fit.coef())):
        put(case, "b", b, 0, j)
    put(case, "N", fit._N)
    put(case, "df_k", fit._df_k)
    put(case, "lag", -1 if fit._lag is None else fit._lag)
    put(case, "T", data[kwargs["time_id"]].nunique())


run("nw_panel_L2", panel, "NW", {"time_id": "t", "panel_id": "id", "lag": 2})
run("nw_panel_def", panel, "NW", {"time_id": "t", "panel_id": "id"})
run("nw_panel_L20", panel, "NW", {"time_id": "t", "panel_id": "id", "lag": 20})
run("nw_panel_w_L3", panel, "NW", {"time_id": "t", "panel_id": "id", "lag": 3}, weights="w")
run("nw_ts_L4", ts, "NW", {"time_id": "t", "lag": 4})
run("nw_ts_w_L2", ts, "NW", {"time_id": "t", "lag": 2}, weights="w")
run("dk_L2", panel, "DK", {"time_id": "t", "panel_id": "id", "lag": 2})
run("dk_def", panel, "DK", {"time_id": "t", "panel_id": "id"})

(HERE / "pyfixest_ref.csv").write_text("case,name,i,j,value\n" + "\n".join(rows) + "\n")
print("pyfixest_ref.csv:", len(rows), "rows")
