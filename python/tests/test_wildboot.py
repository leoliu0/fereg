"""Tests for wild bootstrap inference in fereg (Python)."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

import fereg


def _synth_data(n: int = 240, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    c1 = np.arange(n) % 12
    c2 = np.arange(n) % 8
    fe1 = np.arange(n) % 15
    fe2 = (np.arange(n) * 3) % 6
    z1 = rng.normal(size=n)
    z2 = rng.normal(size=n)
    x1 = rng.normal(size=n) + 0.3 * np.sin(c1)
    x2 = rng.normal(size=n) + 0.25 * np.cos(fe1)
    u = rng.normal(size=n)
    endog = 0.7 * z1 - 0.4 * z2 + 0.5 * x1 + 0.6 * u + rng.normal(scale=0.5, size=n)
    fe_eff = 0.4 * np.sin(fe1) + 0.3 * np.cos(fe2) + 0.35 * np.sin(c1)
    y = 0.8 + 1.2 * x1 - 0.7 * x2 + 0.9 * endog + fe_eff + u
    eta_p = -0.3 + 0.5 * x1 - 0.3 * x2 + 0.2 * np.sin(fe1)
    y_pois = rng.poisson(np.exp(np.clip(eta_p, -3, 3)))
    eta_l = -0.2 + 0.8 * x1 - 0.5 * x2 + 0.25 * np.sin(fe2)
    y_bin = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-eta_l))).astype(float)
    w = 0.5 + (np.arange(n) % 4) * 0.4
    fw = 1 + (np.arange(n) % 3)
    return pd.DataFrame(
        {
            "y": y,
            "x1": x1,
            "x2": x2,
            "endog": endog,
            "z1": z1,
            "z2": z2,
            "c1": c1,
            "c2": c2,
            "fe1": fe1,
            "fe2": fe2,
            "y_pois": y_pois,
            "y_bin": y_bin,
            "w": w,
            "fw": fw,
        }
    )


def _explicit_dummy_wls_oracle(
    df: pd.DataFrame,
    R: np.ndarray,
    r: np.ndarray,
    *,
    impose_null: bool,
    ssc_k_fixef: str = "nonnested",
) -> tuple[float, float, np.ndarray]:
    """Independent NumPy explicit-dummy WLS oracle for full-enumeration Rademacher WCR11 / WCU11."""
    n = len(df)
    fe1_d = pd.get_dummies(df["fe1"], drop_first=False, dtype=float).to_numpy()
    fe2_d = pd.get_dummies(df["fe2"], drop_first=True, dtype=float).to_numpy()
    D = np.column_stack([df["x1"].to_numpy(), df["x2"].to_numpy(), fe1_d, fe2_d])
    y = df["y"].to_numpy()
    w = df["w"].to_numpy()
    sw = np.sqrt(w)
    Dw = D * sw[:, None]
    yw = y * sw

    k = 2
    p_tot = D.shape[1]
    XtX = Dw.T @ Dw
    bread_full = np.linalg.pinv(XtX)
    beta_full = bread_full @ (Dw.T @ yw)
    u_unr = y - D @ beta_full

    R_full = np.zeros((R.shape[0], p_tot))
    R_full[:, :k] = R
    if impose_null:
        RBR = R_full @ bread_full @ R_full.T
        diff = R_full @ beta_full - r
        beta_dgp = beta_full - bread_full @ R_full.T @ np.linalg.solve(RBR, diff)
        u_dgp = y - D @ beta_dgp
    else:
        beta_dgp = beta_full
        u_dgp = u_unr

    clusters = df["cl"].to_numpy()
    uniq_cl = np.unique(clusters)
    G = len(uniq_cl)

    # Nonnested cluster SSC matching fereg / pyfixest
    df_k = k + (p_tot - k if ssc_k_fixef == "full" else (len(df["fe1"].unique()) + len(df["fe2"].unique()) - 2))
    if ssc_k_fixef == "nonnested":
        # Check which FEs are nested in cluster
        nested_fe1 = df.groupby("fe1")["cl"].nunique().max() == 1
        nested_fe2 = df.groupby("fe2")["cl"].nunique().max() == 1
        fe_k = (0 if nested_fe1 else len(df["fe1"].unique())) + (0 if nested_fe2 else len(df["fe2"].unique()))
        # Component adjustment if both or one kept: match connected component count when neither nested
        df_k = k + (fe_k - (0 if (nested_fe1 and nested_fe2) else 1))
    ssc_fac = ((n - 1) / (n - df_k)) * (G / (G - 1))

    # Observed stat
    scores_obs = np.zeros((G, p_tot))
    for gi, gval in enumerate(uniq_cl):
        mask = clusters == gval
        scores_obs[gi] = (Dw[mask].T @ (sw[mask] * u_unr[mask]))
    meat_obs = ssc_fac * (scores_obs.T @ scores_obs)
    V_obs_beta = bread_full[:k, :] @ meat_obs @ bread_full[:, :k]
    diff_obs = R @ beta_full[:k] - r
    RV = R @ V_obs_beta @ R.T
    if R.shape[0] == 1:
        se_obs = float(np.sqrt(RV[0, 0]))
        stat_obs = float(diff_obs[0] / se_obs)
    else:
        se_obs = np.nan
        stat_obs = float(diff_obs @ np.linalg.solve(RV, diff_obs) / R.shape[0])

    # Precompute cluster residual vectors in full space projected by bread_full * R'
    # Full enumeration over 2^G sign combinations
    M = Dw @ bread_full  # n x p_tot
    BR = bread_full[:, :k] @ R.T  # p_tot x q
    RVR = R @ V_obs_beta @ R.T
    t_boot = []
    # Cluster contributions in orthogonalized world: per draw s in {-1, +1}^G
    cl_masks = [clusters == gval for gval in uniq_cl]
    for signs in itertools.product((-1.0, 1.0), repeat=G):
        s_vec = np.empty(n)
        for msk, sg in zip(cl_masks, signs):
            s_vec[msk] = sg
        u_b = u_dgp * s_vec
        y_b = D @ beta_dgp + u_b
        yb_w = y_b * sw
        beta_b = bread_full @ (Dw.T @ yb_w)
        e_b = y_b - D @ beta_b
        sc = np.zeros((G, p_tot))
        for gi, msk in enumerate(cl_masks):
            sc[gi] = Dw[msk].T @ (sw[msk] * e_b[msk])
        V_b = (bread_full[:k, :] @ (ssc_fac * (sc.T @ sc)) @ bread_full[:, :k])
        diff_b = R @ beta_b[:k] - (R @ beta_dgp[:k])
        RV_b = R @ V_b @ R.T
        if R.shape[0] == 1:
            t_boot.append(float(diff_b[0] / np.sqrt(RV_b[0, 0])))
        else:
            t_boot.append(float(diff_b @ np.linalg.solve(RV_b, diff_b) / R.shape[0]))
    t_arr = np.asarray(t_boot, dtype=float)
    eps = 1e-12 * (1.0 + abs(stat_obs))
    if R.shape[0] == 1:
        pval = float(np.mean(np.abs(t_arr) > abs(stat_obs) + eps))
    else:
        pval = float(np.mean(t_arr > stat_obs + eps))
    return stat_obs, pval, np.sort(t_arr)


def test_explicit_dummy_oracle_parity():
    df = _synth_data(96, seed=7)
    # 4 clusters not nested with fe1 or fe2, 16 full-enumeration draws
    df["cl"] = np.arange(len(df)) % 4
    m = fereg.feols("y ~ x1 + x2 | fe1 + fe2", data=df, weights="w", vcov={"CRV1": "cl"})

    R1 = np.array([[1.0, 2.0]])
    r1 = np.array([0.2])
    stat_o, p_o, tb_o = _explicit_dummy_wls_oracle(df, R1, r1, impose_null=True)
    wb = m.wildboottest(R=R1, r=r1, reps=16, weights="rademacher", boot_type="11", impose_null=True, seed=1)
    assert wb.full_enumeration is True
    assert wb.reps == 16
    assert wb.statistic == pytest.approx(stat_o, rel=1e-10, abs=1e-10)
    assert wb.p_value == pytest.approx(p_o, abs=1e-12)
    np.testing.assert_allclose(np.sort(wb.t_boot), tb_o, rtol=1e-9, atol=1e-9)

    # Joint WCU11 (q=2)
    R2 = np.eye(2)
    r2 = np.array([0.1, -0.2])
    stat_j, p_j, tb_j = _explicit_dummy_wls_oracle(df, R2, r2, impose_null=False)
    wb_j = fereg.wildboottest(
        m, R=R2, r=r2, reps=16, weights="rademacher", boot_type="11", impose_null=False, seed=1
    )
    assert wb_j.q == 2
    assert wb_j.stat_type == "F"
    assert wb_j.statistic == pytest.approx(stat_j, rel=1e-10, abs=1e-10)
    assert wb_j.p_value == pytest.approx(p_j, abs=1e-12)
    np.testing.assert_allclose(np.sort(wb_j.t_boot), tb_j, rtol=1e-9, atol=1e-9)


def test_ols_mnw_weights_and_pval_types():
    df = _synth_data(240, seed=11)
    m = fereg.feols("y ~ x1 + x2 | fe1 + fe2", data=df, vcov={"CRV1": "c1"})

    for bt in ("11", "13", "31", "33"):
        res = m.wildboottest("x1", reps=99, boot_type=bt, seed=123, nthreads=1) if False else m.wildboottest(
            "x1", reps=99, boot_type=bt, seed=123
        )
        assert res.method == "wildcluster"
        assert res.boot_type == bt
        assert 0.0 <= res.p_value <= 1.0
        assert res.reps_feasible == 99

    # bootstrap_type alias
    r_alias = m.wildboottest("x1", reps=99, bootstrap_type="13", seed=123)
    r_main = m.wildboottest("x1", reps=99, boot_type="13", seed=123)
    assert r_alias.p_value == r_main.p_value
    assert r_alias.statistic == r_main.statistic
    np.testing.assert_array_equal(r_alias.t_boot, r_main.t_boot)

    # weights
    for wt in ("rademacher", "webb", "mammen", "normal"):
        r_wt = m.wildboottest("x1", reps=49, weights=wt, seed=5)
        assert r_wt.weights == wt
        assert len(r_wt.t_boot) == 49

    # pval_type + aliases
    p_two = m.wildboottest("x1", reps=99, pval_type="two-tailed", conf_int=False, seed=7).p_value
    p_eq = m.wildboottest("x1", reps=99, pval_type="equal-tailed", conf_int=False, seed=7).p_value
    p_gt = m.wildboottest("x1", reps=99, pval_type=">", conf_int=False, seed=7).p_value
    p_gt2 = m.wildboottest("x1", reps=99, pval_type="greater", conf_int=False, seed=7).p_value
    p_lt = m.wildboottest("x1", reps=99, pval_type="<", conf_int=False, seed=7).p_value
    p_lt2 = m.wildboottest("x1", reps=99, pval_type="less", conf_int=False, seed=7).p_value
    assert p_gt == p_gt2 and p_lt == p_lt2
    assert p_gt + p_lt == pytest.approx(1.0, abs=1e-12)
    assert p_eq == pytest.approx(2.0 * min(p_gt, p_lt), abs=1e-12)
    assert 0.0 <= p_two <= 1.0


def test_hetero_full_enum_and_fweights():
    df = _synth_data(120, seed=19)
    m = fereg.feols("y ~ x1 + x2 | fe1", data=df, vcov="HC1")
    for bt in ("11", "21", "31", "13", "33"):
        r_h = m.wildboottest("x1", reps=49, boot_type=bt, seed=9)
        assert r_h.method == "hetero"
        assert r_h.n_clusters is None
        assert r_h.boot_type == bt
        assert len(r_h.t_boot) == 49

    # Full enumeration under Webb with G = 3 -> 6^3 = 216
    df_w = df.copy()
    df_w["g3"] = np.arange(len(df_w)) % 3
    m3 = fereg.feols("y ~ x1 + x2 | fe1", data=df_w, vcov={"CRV1": "g3"})
    r_webb = m3.wildboottest("x1", reps=500, weights="webb", conf_int=False, seed=1)
    assert r_webb.full_enumeration is True
    assert r_webb.reps == 216

    # fweights vs expanded rows parity
    df_fw = _synth_data(60, seed=23)
    df_exp = df_fw.loc[df_fw.index.repeat(df_fw["fw"])].reset_index(drop=True)
    m_fw = fereg.feols("y ~ x1 + x2 | fe1", data=df_fw, weights="fw", weights_type="fweights", vcov={"CRV1": "c1"})
    m_ex = fereg.feols("y ~ x1 + x2 | fe1", data=df_exp, vcov={"CRV1": "c1"})
    r_fw = m_fw.wildboottest("x1", reps=49, seed=3, conf_int=False)
    r_ex = m_ex.wildboottest("x1", reps=49, seed=3, conf_int=False)
    assert r_fw.statistic == pytest.approx(r_ex.statistic, rel=1e-11, abs=1e-11)
    assert r_fw.p_value == pytest.approx(r_ex.p_value, abs=1e-12)


def test_multicluster_bootcluster_posthoc_and_polars():
    df = _synth_data(200, seed=31)
    m = fereg.feols("y ~ x1 + x2 | fe1", data=df, vcov={"CRV1": "c1 + c2"})
    r_2w = m.wildboottest("x1", reps=49, seed=17)
    assert r_2w.cluster == ["c1", "c2"]
    assert r_2w.n_clusters == 24

    r_bc = m.wildboottest("x1", reps=49, bootcluster="c1", seed=17)
    assert r_bc.bootcluster == ["c1"]
    assert r_bc.n_clusters == 12

    r_bc_list = m.wildboottest("x1", reps=49, bootcluster=["c1", "c2"], seed=17)
    assert r_bc_list.n_clusters == 24

    # Post-hoc vcov override should be respected
    m_iid = fereg.feols("y ~ x1 + x2 | fe1", data=df, vcov="iid")
    m_iid.vcov({"CRV1": "c1"})
    r_post = m_iid.wildboottest("x1", reps=49, seed=17)
    assert r_post.method == "wildcluster"
    assert r_post.cluster == ["c1"]
    assert r_post.n_clusters == 12

    # Cluster override from IID
    m_plain = fereg.feols("y ~ x1 + x2 | fe1", data=df, vcov="iid")
    r_ov = m_plain.wildboottest("x1", reps=49, cluster="c1", seed=17)
    assert r_ov.statistic == pytest.approx(r_post.statistic, rel=1e-12)
    assert r_ov.p_value == r_post.p_value

    # Polars input support
    pl = pytest.importorskip("polars")
    m_pl = fereg.feols("y ~ x1 + x2 | fe1", data=pl.from_pandas(df), vcov="iid")
    r_pl = m_pl.wildboottest("x1", reps=49, cluster="c1", seed=17)
    assert r_pl.statistic == pytest.approx(r_ov.statistic, rel=1e-12)
    assert r_pl.p_value == r_ov.p_value


def test_iv_glm_ci_inversion_threads_and_outputs(capsys):
    df = _synth_data(240, seed=41)

    # IV WRE & score
    iv = fereg.feols("y ~ x1 | fe1 | endog ~ z1 + z2", data=df, vcov={"CRV1": "c1"})
    r_wre = iv.wildboottest("endog", reps=99, method="wre", seed=5)
    assert r_wre.method == "wre"
    assert r_wre.stat_type == "t"
    assert r_wre.point_estimate == pytest.approx(float(iv.coef()["endog"]), rel=1e-12)

    r_score_iv = iv.wildboottest(["endog", "x1"], reps=99, method="score", seed=5)
    assert r_score_iv.method == "score"
    assert r_score_iv.q == 2
    assert r_score_iv.stat_type == "score F"

    # Poisson & Logit score
    p = fereg.fepois("y_pois ~ x1 + x2 | fe1", data=df, vcov={"CRV1": "c1"})
    rp = p.wildboottest("x1", reps=99, seed=8)
    assert rp.method == "score"
    assert rp.stat_type == "score t"
    assert rp.conf_int is not None

    lg = fereg.feglm("y_bin ~ x1 + x2 | fe2", data=df, family="logit", vcov={"CRV1": "c1"})
    rl = lg.wildboottest("x2", reps=99, seed=8)
    assert rl.method == "score"
    assert rl.stat_type == "score t"

    # CI inversion check: testing r = lo and r = hi gives p_val near alpha = 0.05
    m = fereg.feols("y ~ x1 + x2 | fe1", data=df, vcov={"CRV1": "c1"})
    r_ci = m.wildboottest("x1", reps=199, level=0.95, seed=11, tol=1e-6)
    lo, hi = r_ci.conf_int
    assert lo < r_ci.point_estimate < hi
    p_lo = m.wildboottest("x1", r=lo, reps=199, conf_int=False, seed=11).p_value
    p_hi = m.wildboottest("x1", r=hi, reps=199, conf_int=False, seed=11).p_value
    assert abs(p_lo - 0.05) <= 0.02
    assert abs(p_hi - 0.05) <= 0.02

    # WCU CI inversion consistency
    r_wcu = m.wildboottest("x1", reps=99, impose_null=False, level=0.90, seed=11)
    assert r_wcu.conf_int is not None and r_wcu.conf_int[0] < r_wcu.point_estimate < r_wcu.conf_int[1]

    # Thread bit-exactness
    r_t1 = m.wildboottest("x1", reps=99, boot_type="33", seed=99, threads=1)
    r_t4 = m.wildboottest("x1", reps=99, boot_type="33", seed=99, threads=4)
    assert r_t1.statistic == r_t4.statistic
    assert r_t1.p_value == r_t4.p_value
    assert r_t1.conf_int == r_t4.conf_int
    np.testing.assert_array_equal(r_t1.t_boot, r_t4.t_boot)

    # Display / summary / Series helpers
    s = r_ci.to_series()
    assert isinstance(s, pd.Series)
    assert s["p_value"] == r_ci.p_value
    assert "WildBootResult" in repr(r_ci)
    r_ci.summary()
    out_txt = capsys.readouterr().out
    assert "Wild bootstrap test:" in out_txt


def test_wildboot_errors():
    df = _synth_data(120, seed=53)
    df["x_collin"] = df["x1"] * 2.0
    m = fereg.feols("y ~ x1 + x_collin + x2 | fe1", data=df, vcov={"CRV1": "c1"})

    with pytest.raises(ValueError, match="collinearity"):
        m.wildboottest("x_collin", reps=19)
    with pytest.raises(ValueError, match="not found"):
        m.wildboottest("nonexistent", reps=19)
    with pytest.raises(ValueError, match="not both"):
        m.wildboottest("x1", R=np.eye(2), reps=19)
    with pytest.raises(ValueError, match="either param or R"):
        m.wildboottest(reps=19)
    with pytest.raises(ValueError, match="full row rank"):
        m.wildboottest(R=np.array([[1.0, 1.0], [2.0, 2.0]]), reps=19)
    with pytest.raises(ValueError, match="shape"):
        m.wildboottest(R=np.array([[1.0]]), reps=19)
    with pytest.raises(ValueError, match="two-tailed"):
        m.wildboottest(["x1", "x2"], pval_type=">", reps=19)
    with pytest.raises(ValueError, match="Conflicting"):
        m.wildboottest("x1", boot_type="13", bootstrap_type="31", reps=19)
    with pytest.raises(ValueError, match="level"):
        m.wildboottest("x1", level=1.2, reps=19)

    df_t = df.copy()
    df_t["t"] = np.arange(len(df_t))
    m_nw = fereg.feols("y ~ x1 + x2 | fe1", data=df_t, vcov="NW", vcov_kwargs={"time_id": "t", "lag": 2})
    with pytest.raises(ValueError, match="not defined for 'NW'"):
        m_nw.wildboottest("x1", reps=19)

    multi = fereg.feols("y + y_pois ~ x1 | fe1", data=df)
    with pytest.raises(TypeError, match="FixestMulti"):
        fereg.wildboottest(multi, "x1", reps=19)

    lg = fereg.feglm("y_bin ~ x1 + x2 | fe2", data=df, family="logit")
    bc = lg.bias_corr()
    with pytest.raises(ValueError, match="bias-corrected"):
        bc.wildboottest("x1", reps=19)
