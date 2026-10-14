"""Input data of the HAC/Conley fixtures (hac.rs, conley.rs unit tests).

Run: python gen_data.py   (numpy only; writes ts.csv, panel.csv, geo.csv next to this file)
The reference scripts (gen_fixest.R, gen_pyfixest.py, gen_stata.do) read these files.
"""

from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
rng = np.random.default_rng(20261013)


def ar1(n, rho, scale=1.0):
    e = rng.normal(scale=scale, size=n)
    out = np.empty(n)
    out[0] = e[0]
    for i in range(1, n):
        out[i] = rho * out[i - 1] + e[i]
    return out


def write(name, cols):
    keys = list(cols)
    n = len(cols[keys[0]])
    with open(HERE / name, "w") as f:
        f.write(",".join(keys) + "\n")
        for i in range(n):
            row = []
            for k in keys:
                v = cols[k][i]
                row.append(str(int(v)) if k in ("t", "t2", "t3", "id") else repr(float(v)))
            f.write(",".join(row) + "\n")


# Time series with calendar gaps: 120 of the periods 1..150.
t = np.sort(rng.choice(np.arange(1, 151), size=120, replace=False))
n = len(t)
x1 = ar1(n, 0.6)
x2 = ar1(n, -0.3) + 0.5 * x1
y = 1.0 + 0.5 * x1 - 0.3 * x2 + ar1(n, 0.5)
write("ts.csv", {"t": t, "t2": 1980 + 2 * t, "x1": x1, "x2": x2, "y": y, "w": rng.uniform(0.5, 2.0, n)})


# Longer, less persistent series for bw(auto), so the Newey-West (1994) choice is below its cap
# mstar: 400 of the periods 1..520; t3 has calendar step 3.
t = np.sort(rng.choice(np.arange(1, 521), size=400, replace=False))
n = len(t)
xa = ar1(n, 0.2)
xb = ar1(n, 0.4) + 0.3 * xa
write(
    "ts_auto.csv",
    {
        "t": t,
        "t3": 2000 + 3 * t,
        "xa": xa,
        "xb": xb,
        "ya": 1.0 + xa - xb + ar1(n, 0.05),
        "yb": 1.0 + xa - xb + ar1(n, 0.3),
        "yc": 1.0 + xa - xb + ar1(n, 0.6),
    },
)

# Unbalanced panel with gaps: 40 units, periods 1..12, each present with probability 0.8.
ids, ts = [], []
for u in range(1, 41):
    keep = rng.uniform(size=12) < 0.8
    keep[rng.integers(12)] = True
    for p in np.nonzero(keep)[0]:
        ids.append(u)
        ts.append(p + 1)
ids = np.array(ids)
ts = np.array(ts)
# Rows in a scrambled order (the kernels must not rely on the input order).
perm = rng.permutation(len(ids))
ids, ts = ids[perm], ts[perm]
n = len(ids)
unit_fe = rng.normal(size=41)[ids]
time_shock = rng.normal(size=13)[ts]
x1 = rng.normal(size=n) + 0.5 * time_shock
x2 = rng.normal(size=n) + 0.3 * unit_fe
y = 2.0 + 0.7 * x1 + 0.2 * x2 + unit_fe + time_shock + rng.normal(size=n)
write("panel.csv", {"id": ids, "t": ts, "x1": x1, "x2": x2, "y": y, "w": rng.uniform(0.5, 2.0, n)})

# Spatial data: 260 sites (some repeated), 3 clusters; lon also on [0, 360] and lat also > 90
# (fixest shifts both).
sites = 260
centers = np.array([[41.0, -3.0], [48.5, 9.0], [44.0, 24.0]])
c = rng.integers(3, size=sites)
slat = np.round(centers[c, 0] + rng.normal(scale=1.6, size=sites), 3)
slon = np.round(centers[c, 1] + rng.normal(scale=2.4, size=sites), 3)
site = rng.integers(sites, size=600)
lat, lon = slat[site], slon[site]
n = len(site)
geo_shock = np.sin(lat / 2.0) + np.cos(lon / 3.0)
x1 = rng.normal(size=n) + geo_shock
x2 = rng.normal(size=n)
y = 0.5 + x1 - 0.5 * x2 + 2.0 * geo_shock + rng.normal(size=n)
write("geo.csv", {"lat": lat, "lon": lon, "lon360": lon + 170.0, "lat90": lat + 60.0, "x1": x1, "x2": x2, "y": y})
