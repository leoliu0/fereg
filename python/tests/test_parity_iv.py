"""IV/2SLS parity: pyfixest 0.60.0 (one endogenous regressor) and fixest 0.14.2 (two)."""

import json
import shutil
import subprocess

import numpy as np
import pytest
from _parity import TIGHT, assert_same_summary, compare, make_data, pf

import fereg

DATA = make_data()


def test_iv_summary_matches_pyfixest(capsys):
    fml = "y_iv ~ x1 | f1 + f2 | e1 ~ z1 + z2"
    assert_same_summary(fereg.feols(fml, DATA), pf.feols(fml, DATA, demeaner=TIGHT), capsys)


IV_CASES = [
    ("y_iv ~ x1 | f1 | e1 ~ z1", None),
    ("y_iv ~ x1 | f1 + f2 | e1 ~ z1 + z2", None),
    ("y_iv ~ x1 + x2 | e1 ~ z1 + z2", None),
    ("y_iv ~ x1 | f1 + f2 | e1 ~ z1 + z2", "hetero"),
    ("y_iv ~ x1 | f1 + f2 | e1 ~ z1 + z2", {"CRV1": "cl"}),
    ("y_iv ~ x1 | f1 + f2 | e1 ~ z1 + z2", {"CRV1": "f1"}),
    ("y_iv ~ x1 | f1 + f2 | e1 ~ z1 + z2", {"CRV1": "cl+cl2"}),
]


@pytest.mark.parametrize("fml,vcov", IV_CASES, ids=[f"{c[0]}|{c[1]}" for c in IV_CASES])
def test_iv_parity(fml, vcov):
    theirs = pf.feols(fml, DATA, vcov=vcov, demeaner=TIGHT)
    ours = fereg.feols(fml, DATA, vcov=vcov)
    assert isinstance(ours, fereg.Feiv)
    compare(ours, theirs)
    np.testing.assert_allclose(ours.resid(), theirs.resid(), rtol=1e-7, atol=1e-9)


def test_iv_weighted():
    fml = "y_iv ~ x1 | f1 + f2 | e1 ~ z1 + z2"
    theirs = pf.feols(fml, DATA, weights="w", vcov="hetero", demeaner=TIGHT)
    ours = fereg.feols(fml, DATA, weights="w", vcov="hetero")
    compare(ours, theirs)


def test_iv_first_stage():
    fml = "y_iv ~ x1 | f1 + f2 | e1 ~ z1 + z2"
    theirs = pf.feols(fml, DATA, vcov={"CRV1": "cl"}, demeaner=TIGHT)
    ours = fereg.feols(fml, DATA, vcov={"CRV1": "cl"})
    compare(ours._model_1st_stage, theirs._model_1st_stage)
    np.testing.assert_allclose(ours._f_stat_1st_stage, theirs._f_stat_1st_stage, rtol=1e-7)
    np.testing.assert_allclose(ours._p_value_1st_stage, theirs._p_value_1st_stage, rtol=1e-6, atol=1e-300)


_R_SCRIPT = r"""
suppressMessages(library(fixest))
d <- read.csv(commandArgs(TRUE)[1])
m <- feols(y_iv ~ x1 | f1 + f2 | e1 + e2 ~ z1 + z2 + x3, d, fixef.rm = "singleton",
           fixef.tol = 3e-12, vcov = "iid")
h <- summary(m, vcov = "hetero")
cat(paste0('{"names":["', paste(names(coef(m)), collapse = '","'), '"],',
    '"coef":[', paste(sprintf("%.17g", coef(m)), collapse = ","), '],',
    '"se":[', paste(sprintf("%.17g", se(m)), collapse = ","), '],',
    '"se_hetero":[', paste(sprintf("%.17g", se(h)), collapse = ","), '],',
    '"nobs":', nobs(m), '}'))
"""


@pytest.mark.skipif(shutil.which("Rscript") is None, reason="Rscript not available")
def test_two_endogenous_vs_fixest(tmp_path):
    csv = tmp_path / "d.csv"
    DATA.to_csv(csv, index=False, na_rep="NA")
    script = tmp_path / "iv2.R"
    script.write_text(_R_SCRIPT)
    run = subprocess.run(["Rscript", str(script), str(csv)], capture_output=True, text=True)
    if run.returncode != 0:
        pytest.skip(f"fixest unavailable: {run.stderr[-300:]}")
    ref = json.loads(run.stdout)
    ours = fereg.feols("y_iv ~ x1 | f1 + f2 | e1 + e2 ~ z1 + z2 + x3", DATA)
    names = [n.removeprefix("fit_") for n in ref["names"]]
    assert sorted(names) == sorted(ours._coefnames)
    assert ours._N == ref["nobs"]
    order = [ours._coefnames.index(n) for n in names]
    np.testing.assert_allclose(ours._beta_hat[order], ref["coef"], rtol=1e-8)
    np.testing.assert_allclose(ours._se[order], ref["se"], rtol=1e-8)
    ours.vcov("hetero")
    np.testing.assert_allclose(ours._se[order], ref["se_hetero"], rtol=1e-8)
