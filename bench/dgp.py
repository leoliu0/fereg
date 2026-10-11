"""Benchmark data-generating processes (plan Phase 8.1).

* ``fixest_dgp(n, type_)``: exact port of pyfixest v0.60.0
  ``benchmarks/modular/dgp_functions.py::base_dgp`` (itself adapted from
  kylebutts/fixest_benchmarks, MIT) with nb_year=10, nb_indiv_per_firm=23,
  max_k=10 and betas 1/j over all ten regressors (upstream ``BaseDGP`` calls it
  with ``k=max_k``). The port additionally returns the latent fixed effects so
  that the IV extension can reuse them; the base columns are identical to the
  vendored upstream function (checked by ``verify_dgp.py``).
* ``akm_dgp(scenario, n)``: the vendored, unmodified pyfixest
  ``akm_dgp.py::simulate_akm_panel`` with the benchmark defaults of upstream
  ``dgps.py`` (``_AKM_DEFAULTS`` / ``_AKM_OCCUPATION_DEFAULTS``) and scenarios.

Seeds are fixed (never ``hash()``):
* fixest DGP: ``n*100 + 42 + {simple: 0, difficult: 1}``, i.e. upstream
  ``_seed_for`` with iteration 0 (one dataset per (type, n) serves every run).
* AKM: ``n*100 + 42 + scenario_index``, where the index is the position in
  ``AKM_SCENARIOS`` (sweep scenarios 0..25, then occupation scenarios 26..34).
* IV extension: independent stream ``default_rng([seed, IV_STREAM])`` so the
  base columns stay byte-identical to upstream.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from vendor.pyfixest_bench.akm_dgp import AKMConfig, simulate_akm_panel

NB_YEAR = 10
NB_INDIV_PER_FIRM = 23
MAX_K = 10
THETA = 0.5
IV_STREAM = 0x1F
TYPE_OFFSET = {"simple": 0, "difficult": 1}


def fixest_seed(n: int, type_: str) -> int:
    return n * 100 + 42 + TYPE_OFFSET[type_]


def base_dgp_latent(
    n: int,
    type_: str,
    seed: int,
    nb_year: int = NB_YEAR,
    nb_indiv_per_firm: int = NB_INDIV_PER_FIRM,
    k: int = MAX_K,
    max_k: int = MAX_K,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Port of upstream ``base_dgp``; same RNG call sequence, plus latent FEs."""
    rng = np.random.default_rng(seed)

    nb_indiv = round(n / nb_year)
    nb_firm = round(nb_indiv / nb_indiv_per_firm)
    if nb_indiv < 1 or nb_firm < 1:
        raise ValueError(
            f"n={n} too small for nb_year={nb_year}, nb_indiv_per_firm={nb_indiv_per_firm}"
        )
    n_obs = nb_indiv * nb_year
    if k < 1 or k > max_k:
        raise ValueError(f"k={k} must satisfy 1 <= k <= max_k={max_k}")

    indiv_id = np.repeat(np.arange(1, nb_indiv + 1), nb_year)
    year = np.tile(np.arange(1, nb_year + 1), nb_indiv)

    if type_ == "simple":
        firm_id = rng.integers(1, nb_firm + 1, size=n_obs)
    elif type_ == "difficult":
        firm_id = np.tile(np.arange(1, nb_firm + 1), n_obs // nb_firm + 1)[:n_obs]
    else:
        raise ValueError(f"Unknown type of dgp: {type_!r}")

    x = rng.standard_normal((n_obs, max_k))
    betas = 1.0 / np.arange(1, k + 1, dtype=float)

    firm_fe = rng.standard_normal(nb_firm)[firm_id - 1]
    unit_fe = rng.standard_normal(nb_indiv)[indiv_id - 1]
    year_fe = rng.standard_normal(nb_year)[year - 1]
    mu = x[:, :k] @ betas + firm_fe + unit_fe + year_fe
    y = mu + rng.standard_normal(len(mu))

    exp_y = np.exp(y)
    nb_p = THETA / (THETA + exp_y)
    negbin_y = rng.negative_binomial(n=THETA, p=nb_p)

    data = {
        "indiv_id": indiv_id,
        "firm_id": firm_id,
        "year": year,
        "y": y,
        "exp_y": exp_y,
        "negbin_y": negbin_y,
        "binary_y": (y > 0).astype(int),
    }
    for j in range(max_k):
        data[f"x{j + 1}"] = x[:, j]
    latent = {"firm_fe": firm_fe, "unit_fe": unit_fe, "year_fe": year_fe}
    return pd.DataFrame(data), latent


def fixest_dgp(n: int, type_: str) -> pd.DataFrame:
    """fixest benchmark panel with the plan's IV extension columns appended."""
    seed = fixest_seed(n, type_)
    df, lat = base_dgp_latent(n, type_, seed)
    rng = np.random.default_rng([seed, IV_STREAM])
    m = len(df)
    z1 = rng.standard_normal(m)
    z2 = rng.standard_normal(m)
    u = rng.standard_normal(m)
    xe = 0.6 * z1 + 0.4 * z2 + 0.5 * u + lat["unit_fe"] + lat["firm_fe"] + rng.standard_normal(m)
    y_iv = (
        xe
        + 0.5 * df["x1"].to_numpy()
        + lat["unit_fe"]
        + lat["firm_fe"]
        + lat["year_fe"]
        + u
        + rng.standard_normal(m)
    )
    df["z1"] = z1
    df["z2"] = z2
    df["xe"] = xe
    df["y_iv"] = y_iv
    return df


# --- AKM (pyfixest benchmarks/modular/dgps.py defaults and scenarios) ---------

AKM_DEFAULTS: dict = {
    "n_time": 10,
    "n_firms": 50_000,
    "n_industries": 5,
    "var_alpha": 1.0,
    "var_psi": 0.5,
    "var_phi": 0.1,
    "var_epsilon": 1.0,
    "gamma": 1.0,
    "rho_size": 0.6,
    "rho": 1.0,
    "delta": 0.1,
    "lambda_": 0.8,
    "beta_x1": 0.5,
    "n_match_bins": 2048,
    "entry_exit_share": 0.0,
    "entry_exit_n_periods": 2,
}

AKM_OCCUPATION_DEFAULTS: dict = {
    **AKM_DEFAULTS,
    "n_occupations": 200,
    "var_occ": 0.3,
    "occ_menu_size": 5,
    "occ_lambda": 0.5,
    "occ_delta": 0.3,
}

# (name, defaults, overrides) in upstream order; the list position is the seed offset.
AKM_SCENARIOS: list[tuple[str, dict, dict]] = [
    ("akm_baseline", AKM_DEFAULTS, {}),
    ("akm_scale_1", AKM_DEFAULTS, {}),
    ("akm_scale_2", AKM_DEFAULTS, {}),
    ("akm_scale_3", AKM_DEFAULTS, {}),
    ("akm_scale_4", AKM_DEFAULTS, {}),
    ("akm_sorting_1", AKM_DEFAULTS, {"rho": 0.0}),
    ("akm_sorting_2", AKM_DEFAULTS, {"rho": 5.0}),
    ("akm_sorting_3", AKM_DEFAULTS, {"rho": 20.0}),
    ("akm_sorting_4", AKM_DEFAULTS, {"rho": 50.0}),
    ("akm_sorting_5", AKM_DEFAULTS, {"rho": 100.0}),
    ("akm_mobility_1", AKM_DEFAULTS, {"delta": 1.0}),
    ("akm_mobility_2", AKM_DEFAULTS, {"delta": 0.5}),
    ("akm_mobility_3", AKM_DEFAULTS, {"delta": 0.05}),
    ("akm_mobility_4", AKM_DEFAULTS, {"delta": 0.01}),
    ("akm_mobility_5", AKM_DEFAULTS, {"delta": 0.005}),
    ("akm_mobility_6", AKM_DEFAULTS, {"delta": 0.001}),
    ("akm_interaction_1", AKM_DEFAULTS, {"rho": 0.0, "delta": 0.5}),
    ("akm_interaction_2", AKM_DEFAULTS, {"rho": 20.0, "delta": 0.5}),
    ("akm_interaction_3", AKM_DEFAULTS, {"rho": 0.0, "delta": 0.02}),
    ("akm_interaction_4", AKM_DEFAULTS, {"rho": 20.0, "delta": 0.02}),
    ("akm_freeze_1", AKM_DEFAULTS, {"delta": (0.1,) * 5}),
    ("akm_freeze_2", AKM_DEFAULTS, {"delta": (0.1,) * 4 + (0.005,)}),
    ("akm_freeze_3", AKM_DEFAULTS, {"delta": (0.1,) * 3 + (0.005,) * 2}),
    ("akm_freeze_4", AKM_DEFAULTS, {"delta": (0.1,) * 2 + (0.005,) * 3}),
    ("akm_freeze_5", AKM_DEFAULTS, {"delta": (0.1,) + (0.005,) * 4}),
    ("akm_freeze_6", AKM_DEFAULTS, {"delta": (0.005,) * 5}),
    # occupation benchmark (upstream calls its baseline "akm_baseline" too)
    ("akmocc_baseline", AKM_OCCUPATION_DEFAULTS, {}),
    ("akmocc_occlambda_1", AKM_OCCUPATION_DEFAULTS, {"occ_lambda": 0.01}),
    ("akmocc_occlambda_2", AKM_OCCUPATION_DEFAULTS, {"occ_lambda": 0.2}),
    ("akmocc_occlambda_3", AKM_OCCUPATION_DEFAULTS, {"occ_lambda": 0.9}),
    ("akmocc_occlambda_4", AKM_OCCUPATION_DEFAULTS, {"occ_lambda": 1.0}),
    ("akmocc_occsize_1", AKM_OCCUPATION_DEFAULTS, {"n_occupations": 10}),
    ("akmocc_occsize_2", AKM_OCCUPATION_DEFAULTS, {"n_occupations": 50}),
    ("akmocc_occsize_3", AKM_OCCUPATION_DEFAULTS, {"n_occupations": 1_000}),
    ("akmocc_occsize_4", AKM_OCCUPATION_DEFAULTS, {"n_occupations": 5_000}),
]
AKM_INDEX = {name: i for i, (name, _, _) in enumerate(AKM_SCENARIOS)}


def akm_seed(scenario: str, n: int) -> int:
    return n * 100 + 42 + AKM_INDEX[scenario]


def akm_config(scenario: str, n: int) -> AKMConfig:
    _, defaults, overrides = AKM_SCENARIOS[AKM_INDEX[scenario]]
    params = {**defaults, **overrides}
    # Upstream _infer_worker_count gives exactly n/10 when entry_exit_share = 0.
    return AKMConfig(n_workers=n // params["n_time"], **params)


def akm_dgp(scenario: str, n: int) -> pd.DataFrame:
    return simulate_akm_panel(akm_config(scenario, n), seed=akm_seed(scenario, n))
