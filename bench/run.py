"""Benchmark driver and speed gate (plan Phase 8.3-8.5).

Builds the plan 8.4 grid (G1-G7), runs every (tool, config, T) as a subprocess
with a timeout (default 900 s); each subprocess loads the data once, does one
burn-in and 3 timed runs (1 when the burn-in exceeds 60 s) and prints one
``RESULT {json}`` line per run. Medians are merged into ``results/results.csv``
(keyed by config, T, tool; later runs replace earlier ones) and the whole CSV is
rendered into ``RESULTS.md``.

Gates (plan 8.5). Stata: PASS iff the fereg median is below the median of every
competitor that finished; a competitor timeout counts as slower when fereg
finished. Competitor errors are excluded from the comparison and shown as such.
The fereg Python (``fereg_py``) and R (``fereg_r``) packages have their own gates
with the same rule against fixest, pyfixest MAP and pyfixest LSMR (LANG_GATES);
they never enter the Stata gate.

Examples:
  uv run python run.py --grid smoke --tools reghdfe,fixest
  uv run python run.py --grid gate --groups G1,G4 --threads 16
  uv run python run.py --grid gate --tools fereg_py,fereg_r --missing
  uv run python run.py --report-only
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import signal
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import dgp
import gen_data

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
RAW_DIR = RESULTS_DIR / "raw"
RESULTS_CSV = RESULTS_DIR / "results.csv"
RESULTS_MD = HERE / "RESULTS.md"

ALL_TOOLS = ["fereg", "reghdfe", "ivreghdfe", "ppmlhdfe", "xhdfe", "fixest", "pyfixest_map", "pyfixest_lsmr",
             "fereg_py", "fereg_r"]
STATA_TOOLS = {"fereg", "reghdfe", "ivreghdfe", "ppmlhdfe", "xhdfe"}
R_TOOLS = {"fixest": "run_fixest.R", "fereg_r": "run_fereg.R"}  # tool -> Rscript runner (CSV input)
# Tools per group of the Stata gate (plan 8.4 table); fereg is the candidate in every group.
GROUP_TOOLS = {
    "G1": ["fereg", "reghdfe", "xhdfe", "fixest", "pyfixest_map", "pyfixest_lsmr"],
    "G2": ["fereg", "reghdfe", "xhdfe", "fixest", "pyfixest_map", "pyfixest_lsmr"],
    "G3": ["fereg", "reghdfe", "xhdfe", "fixest", "pyfixest_map", "pyfixest_lsmr"],
    "G4": ["fereg", "ivreghdfe", "xhdfe", "fixest", "pyfixest_map", "pyfixest_lsmr"],
    "G5": ["fereg", "ppmlhdfe", "fixest", "pyfixest_map", "pyfixest_lsmr"],
    "G6": ["fereg", "fixest", "pyfixest_map", "pyfixest_lsmr"],
    "G7": ["fereg", "reghdfe", "xhdfe", "fixest", "pyfixest_map", "pyfixest_lsmr"],
}
# Language-package gates: candidate -> (section title, competitors, coefficient reference).
# Both packages run in every group (fixest syntax covers IV, Poisson, logit and fe[x] slopes).
LANG_GATES = {
    "fereg_py": ("Python package `fereg`", ["pyfixest_map", "pyfixest_lsmr", "fixest"], "fixest"),
    "fereg_r": ("R package `fereg`", ["fixest", "pyfixest_map", "pyfixest_lsmr"], "fixest"),
}
GROUP_TITLES = {
    "G1": "G1 OLS (fixest DGP)",
    "G2": "G2 AKM",
    "G3": "G3 AKM with occupation FE",
    "G4": "G4 IV/2SLS",
    "G5": "G5 PPML",
    "G6": "G6 Logit",
    "G7": "G7 Heterogeneous slopes",
}
FE2 = ("indiv_id", "year")
FE3 = ("indiv_id", "year", "firm_id")
AKM_FE = ("indiv_id", "firm_id", "year")
CSV_FIELDS = [
    "group", "config", "dataset", "n", "threads", "tool", "status", "median_s", "n_timed",
    "times_s", "burnin_s", "n_obs", "coefs", "error", "timestamp",
]


@dataclass(frozen=True)
class Config:
    group: str
    dataset: str
    model: str  # ols | iv | ppml | logit
    depvar: str
    xvars: tuple[str, ...]
    fes: tuple[str, ...]
    vcov: str  # iid | hetero | cluster:<var>
    endog: tuple[str, ...] = ()
    inst: tuple[str, ...] = ()
    slope: tuple[str, str] | None = None  # (fe, var): fe##c.var / fe[var]
    fixef_maxiter: int | None = None

    @property
    def n(self) -> int:
        return int(self.dataset.rsplit("_", 1)[1])

    @property
    def key(self) -> str:
        """Stable row id, e.g. ``fx_simple_100000 y~x1+x2 fe=indiv_id+year vcov=iid`` (no '|': markdown)."""
        fe = "+".join(self.fes + ((f"{self.slope[0]}[{self.slope[1]}]",) if self.slope else ()))
        rhs = "+".join(self.xvars) + (f"+({'+'.join(self.endog)}={'+'.join(self.inst)})" if self.endog else "")
        return f"{self.dataset} {self.depvar}~{rhs} fe={fe} vcov={self.vcov}"


@dataclass
class Job:
    tool: str
    cfg: Config
    threads: int
    status: str = "pending"
    times: list[float] = field(default_factory=list)
    burnin: float | None = None
    n_obs: int | None = None
    coefs: dict[str, float] = field(default_factory=dict)
    error: str = ""


# --- grid ---------------------------------------------------------------------

def _xs(k: int) -> tuple[str, ...]:
    return tuple(f"x{j}" for j in range(1, k + 1))


def build_grid(grid: str) -> list[tuple[Config, list[int]]]:
    """Return (config, thread counts) pairs. T = 1 only for n <= 1e6 (plan 8.4)."""
    out: list[tuple[Config, list[int]]] = []

    def add(cfg: Config) -> None:
        out.append((cfg, [16, 1] if cfg.n <= 1_000_000 else [16]))

    fx = gen_data.dataset_name_fx
    if grid == "smoke":
        for t in ("simple", "difficult"):
            for n in (10_000, 100_000):
                for fes in (FE2, FE3):
                    for k in (1, 5):
                        add(Config("G1", fx(t, n), "ols", "y", _xs(k), fes, "iid"))
        add(Config("G2", gen_data.dataset_name_akm("akm_scale_1", 10_000), "ols", "y", ("x1",), AKM_FE, "iid",
                   fixef_maxiter=10_000))
        add(Config("G4", fx("simple", 100_000), "iv", "y_iv", ("x1",), FE2, "hetero", ("xe",), ("z1", "z2")))
        add(Config("G5", fx("simple", 100_000), "ppml", "negbin_y", _xs(1), FE2, "hetero"))
        add(Config("G6", fx("simple", 100_000), "logit", "binary_y", _xs(1), FE2, "iid"))
        add(Config("G7", fx("simple", 100_000), "ols", "y", ("x1",), FE2, "iid", slope=("firm_id", "x2")))
        return out
    if grid != "gate":
        raise ValueError(grid)

    # G1
    for t in ("simple", "difficult"):
        for n in (10_000, 100_000, 1_000_000, 10_000_000):
            for fes in (FE2, FE3):
                for k in (1, 5, 10):
                    add(Config("G1", fx(t, n), "ols", "y", _xs(k), fes, "iid"))
        for n in (1_000_000, 10_000_000):
            for fes in (FE2, FE3):
                add(Config("G1", fx(t, n), "ols", "y", _xs(5), fes, "cluster:indiv_id"))
    # G2: AKM sweep (upstream benchmark_akm_sweep.py: n = 1e6 except the scale scenarios)
    scale_n = {"akm_scale_1": 10_000, "akm_scale_2": 100_000, "akm_scale_3": 1_000_000, "akm_scale_4": 10_000_000}
    for name, _, _ in dgp.AKM_SCENARIOS:
        if name.startswith("akmocc_"):
            continue
        ds = gen_data.dataset_name_akm(name, scale_n.get(name, 1_000_000))
        add(Config("G2", ds, "ols", "y", ("x1",), AKM_FE, "iid", fixef_maxiter=10_000))
    # G3: AKM occupation (n = 1e6)
    for name, _, _ in dgp.AKM_SCENARIOS:
        if name.startswith("akmocc_"):
            ds = gen_data.dataset_name_akm(name, 1_000_000)
            add(Config("G3", ds, "ols", "y", ("x1",), AKM_FE + ("occ_id",), "iid", fixef_maxiter=10_000))
    # G4-G6
    for t in ("simple", "difficult"):
        for n in (100_000, 1_000_000, 10_000_000):
            for fes in (FE2, FE3):
                add(Config("G4", fx(t, n), "iv", "y_iv", ("x1",), fes, "hetero", ("xe",), ("z1", "z2")))
                for k in (1, 5):
                    add(Config("G5", fx(t, n), "ppml", "negbin_y", _xs(k), fes, "hetero"))
                    add(Config("G6", fx(t, n), "logit", "binary_y", _xs(k), fes, "iid"))
    # G7
    for t in ("simple", "difficult"):
        for n in (100_000, 1_000_000):
            add(Config("G7", fx(t, n), "ols", "y", ("x1",), FE2, "iid", slope=("firm_id", "x2")))
    return out


# --- command construction -------------------------------------------------------

def _stata_vce(tool: str, vcov: str) -> str:
    if vcov == "iid":
        return ""
    if vcov == "hetero":
        return "robust" if tool == "ivreghdfe" else "vce(robust)"
    var = vcov.split(":", 1)[1]
    return f"cluster({var})" if tool == "ivreghdfe" else f"vce(cluster {var})"


def stata_command(tool: str, c: Config, threads: int) -> str | None:
    absorb = " ".join(c.fes + ((f"{c.slope[0]}##c.{c.slope[1]}",) if c.slope else ()))
    x = " ".join(c.xvars)
    iv = f"({' '.join(c.endog)} = {' '.join(c.inst)})" if c.endog else ""
    vce = _stata_vce(tool, c.vcov)
    if tool == "fereg":
        cmd = {"ols": "fereg", "iv": "fereg", "ppml": "feppml", "logit": "felogit"}[c.model]
        return f"{cmd} {c.depvar} {x} {iv}, absorb({absorb}) {vce} threads({threads})"
    if tool == "reghdfe" and c.model == "ols":
        return f"reghdfe {c.depvar} {x}, absorb({absorb}) {vce}"
    if tool == "ivreghdfe" and c.model == "iv":
        return f"ivreghdfe {c.depvar} {x} {iv}, absorb({absorb}) {vce}"
    if tool == "ppmlhdfe" and c.model == "ppml":
        return f"ppmlhdfe {c.depvar} {x}, absorb({absorb}) {vce}"
    if tool == "xhdfe" and c.model in ("ols", "iv"):
        if c.model == "iv":
            return (f"xhdfe {c.depvar} {x} {' '.join(c.endog)}, absorb({absorb}) endogenous({' '.join(c.endog)}) "
                    f"instruments({' '.join(c.inst)}) {vce} numthreads({threads})")
        return f"xhdfe {c.depvar} {x}, absorb({absorb}) {vce} numthreads({threads})"
    return None


def r_formula(c: Config) -> str:
    fe = "+".join(c.fes + ((f"{c.slope[0]}[{c.slope[1]}]",) if c.slope else ()))
    rhs = "+".join(c.xvars) if c.xvars else "1"
    f = f"{c.depvar}~{rhs}|{fe}"
    if c.endog:
        f += f"|{'+'.join(c.endog)}~{'+'.join(c.inst)}"
    return f


def method(c: Config) -> str:
    return {"ols": "feols", "iv": "feols", "ppml": "fepois", "logit": "feglm_logit"}[c.model]


def supported(tool: str, c: Config) -> bool:
    if tool in STATA_TOOLS:
        return stata_command(tool, c, 1) is not None
    if tool.startswith("pyfixest"):
        return c.slope is None  # pyfixest 0.60.0 has no varying slopes (fe[x])
    return True


def stata_vars(c: Config) -> list[str]:
    vs = [c.depvar, *c.xvars, *c.endog, *c.inst, *c.fes]
    if c.slope:
        vs += list(c.slope)
    if c.vcov.startswith("cluster:"):
        vs.append(c.vcov.split(":", 1)[1])
    return list(dict.fromkeys(vs))


def thread_env(threads: int) -> dict[str, str]:
    env = dict(os.environ)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS",
                "NUMBA_NUM_THREADS"):
        env[var] = str(threads)
    return env


# --- execution -------------------------------------------------------------------

def _run_proc(argv: list[str], cwd: Path, env: dict, timeout: float, log: Path) -> tuple[str, bool]:
    with open(log, "w") as fh:
        p = subprocess.Popen(argv, cwd=cwd, env=env, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            p.wait(timeout=timeout)
            timed_out = False
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()
            timed_out = True
    return log.read_text(errors="replace"), timed_out


def _parse(text: str) -> list[dict]:
    recs = []
    for line in text.splitlines():
        if line.startswith("RESULT "):
            try:
                recs.append(json.loads(line[7:]))
            except json.JSONDecodeError as e:
                recs.append({"run": "parse", "ok": False, "error": f"bad RESULT line: {e}"})
    return recs


def run_job(job: Job, reps: int, timeout: float) -> None:
    c, T = job.cfg, job.threads
    slug = f"{c.group}_{c.dataset}_{hashlib.sha1(c.key.encode()).hexdigest()[:8]}_T{T}_{job.tool}"
    jd = RAW_DIR / slug
    if jd.exists():
        shutil.rmtree(jd)
    jd.mkdir(parents=True)
    env = thread_env(T)
    r_tool = job.tool in R_TOOLS
    paths = gen_data.ensure(c.dataset, ("dta",) if job.tool in STATA_TOOLS else
                            ("csv",) if r_tool else ("parquet",))
    if job.tool in STATA_TOOLS:
        out = jd / "results.txt"
        do = jd / "job.do"
        do.write_text(
            f'global RF_DATA "{paths["dta"]}"\n'
            f'global RF_VARS "{" ".join(stata_vars(c))}"\n'
            f'global RF_CMD `"{stata_command(job.tool, c, T)}"\'\n'
            f"global RF_T {T}\n"
            f"global RF_REPS {reps}\n"
            f'global RF_OUT "{out}"\n'
            f'do "{HERE / "run_stata.do"}"\n'
        )
        log_text, timed_out = _run_proc([os.environ.get("STATA", "stata-mp"), "-b", "do", "job.do"], jd, env,
                                        timeout, jd / "stdout.txt")
        text = out.read_text(errors="replace") if out.exists() else ""
        stata_log = jd / "job.log"
        tail = stata_log.read_text(errors="replace")[-2000:] if stata_log.exists() else log_text[-2000:]
    else:
        if r_tool:
            argv = ["Rscript", str(HERE / R_TOOLS[job.tool]), f"data={paths['csv']}"]
        elif job.tool == "fereg_py":
            argv = [sys.executable, str(HERE / "run_fereg_py.py"), f"data={paths['parquet']}"]
        else:
            argv = [sys.executable, str(HERE / "run_pyfixest.py"), f"data={paths['parquet']}",
                    f"demeaner={job.tool.split('_', 1)[1]}"]
        argv += [f"fml={r_formula(c)}", f"method={method(c)}", f"vcov={c.vcov}", f"threads={T}", f"reps={reps}"]
        if c.fixef_maxiter:
            argv.append(f"{'fixef_iter' if r_tool else 'fixef_maxiter'}={c.fixef_maxiter}")
        text, timed_out = _run_proc(argv, jd, env, timeout, jd / "stdout.txt")
        tail = text[-2000:]

    recs = _parse(text)
    for r in recs:
        if not r.get("ok"):
            job.error = str(r.get("error", ""))[:300]
            continue
        if r["run"] == "burnin":
            job.burnin = r["time"]
        else:
            job.times.append(r["time"])
        job.n_obs = r.get("n_obs")
        job.coefs = r.get("coefs") or {}
    if job.times:
        job.status = "ok"
    elif timed_out:
        job.status = "timeout"
        job.error = f"timeout after {timeout:.0f} s"
    else:
        job.status = "error"
        if not job.error:
            job.error = "no RESULT lines; log tail: " + " ".join(tail.split())[-300:]


# --- results ---------------------------------------------------------------------

def _row_key(r: dict) -> tuple[str, str, str]:
    return (r["config"], str(r["threads"]), r["tool"])


def load_results() -> list[dict]:
    if not RESULTS_CSV.exists():
        return []
    with open(RESULTS_CSV, newline="") as fh:
        return list(csv.DictReader(fh))


def save_results(rows: list[dict]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = RESULTS_CSV.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, RESULTS_CSV)


def job_row(job: Job) -> dict:
    c = job.cfg
    return {
        "group": c.group,
        "config": c.key,
        "dataset": c.dataset,
        "n": c.n,
        "threads": job.threads,
        "tool": job.tool,
        "status": job.status,
        "median_s": f"{statistics.median(job.times):.6f}" if job.times else "",
        "n_timed": len(job.times),
        "times_s": ";".join(f"{t:.6f}" for t in job.times),
        "burnin_s": "" if job.burnin is None else f"{job.burnin:.6f}",
        "n_obs": "" if job.n_obs is None else job.n_obs,
        "coefs": json.dumps(job.coefs, sort_keys=True) if job.coefs else "",
        "error": job.error,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }


def _fmt_t(r: dict | None) -> str:
    if r is None:
        return ""
    if r["status"] == "ok":
        t = float(r["median_s"])
        return f"{t:.3f}" if t >= 0.1 else f"{t * 1000:.1f}ms"
    return r["status"]


def _gap(a: dict, ref: dict) -> float | None:
    common = [k for k in ref if k in a]
    if not common:
        return None
    return max(abs(a[k] - ref[k]) / max(abs(ref[k]), 1e-300) for k in common)


def gate(rows_by_tool: dict[str, dict], cand: str = "fereg", comps: list[str] | None = None) -> tuple[str, str]:
    """(ratio cand/best finished competitor, PASS/FAIL/n/a); comps defaults to every other tool of the row."""
    rf = rows_by_tool.get(cand)
    comps_r = {t: r for t, r in rows_by_tool.items() if t != cand and (comps is None or t in comps)}
    finished = [float(r["median_s"]) for r in comps_r.values() if r["status"] == "ok"]
    best = min(finished) if finished else None
    if rf is None or not comps_r:
        return "", "n/a"
    if rf["status"] != "ok":
        return "", "FAIL"
    t = float(rf["median_s"])
    ratio = f"{t / best:.2f}" if best else ""
    return ratio, ("PASS" if best is None or t < best else "FAIL")


def _gap_line(cfg: str, T: str, byt: dict[str, dict], tools: list[str], refs: list[str]) -> str | None:
    """Coefficient-gap row versus the first tool of ``refs`` that has coefficients in the row."""
    ref_tool = next((t for t in refs if byt.get(t, {}).get("coefs")), None)
    if ref_tool is None:
        return None
    ref_c = json.loads(byt[ref_tool]["coefs"])
    gaps = []
    for t in tools:
        r = byt.get(t)
        gp = _gap(json.loads(r["coefs"]), ref_c) if r and r["coefs"] and t != ref_tool else None
        gaps.append("" if gp is None else f"{gp:.1e}")
    return f"| `{cfg}` | {T} | {ref_tool} | " + " | ".join(gaps) + " |"


def _gate_section(lines: list[str], by_group: dict, cand: str, comps: dict[str, list[str]] | None,
                  refs: list[str], head: str) -> tuple[int, int, list[tuple[str, str, str, str]]]:
    """Append one table per group for ``cand``; returns (n_pass, n_fail, [(group, config, T, ratio)] of FAIL rows).

    ``comps`` maps group -> competitor list (None: the Stata GROUP_TOOLS, every other tool of the row).
    """
    n_pass = n_fail = 0
    fails = []
    for g in sorted(by_group):
        cols = [cand] + (comps[g] if comps is not None else [t for t in GROUP_TOOLS[g] if t != cand])
        cells = {k: {t: r for t, r in v.items() if t in cols} for k, v in by_group[g].items()}
        cells = {k: v for k, v in cells.items() if v}
        if not cells:
            continue
        tools = [t for t in cols if any(t in v for v in cells.values())]
        lines += [f"{head} {GROUP_TITLES[g]}", ""]
        lines.append("| config | T | " + " | ".join(tools) + " | ratio | gate |")
        lines.append("|---|---:|" + "---:|" * len(tools) + "---:|---|")
        gap_lines = []
        for (cfg, T), byt in sorted(cells.items(), key=lambda kv: (kv[0][0], -int(kv[0][1]))):
            ratio, verdict = gate(byt, cand, None if comps is None else comps[g])
            n_pass += verdict == "PASS"
            n_fail += verdict == "FAIL"
            if verdict == "FAIL":
                fails.append((g, cfg, T, ratio))
            lines.append(f"| `{cfg}` | {T} | " + " | ".join(_fmt_t(byt.get(t)) for t in tools)
                         + f" | {ratio} | {verdict} |")
            gl = _gap_line(cfg, T, byt, tools, refs)
            if gl:
                gap_lines.append(gl)
        lines.append("")
        if gap_lines:
            lines += ["Coefficient gap:", "", "| config | T | vs | " + " | ".join(tools) + " |",
                      "|---|---:|---|" + "---:|" * len(tools)] + gap_lines + [""]
    return n_pass, n_fail, fails


def _thin_keys(rows: list[dict], margin: float) -> set[tuple[str, str, str]]:
    """Row keys of gate candidates whose row FAILs or has ratio > 1/margin (see ``--thin``)."""
    by_cell: dict[tuple[str, str], dict[str, dict]] = {}
    groups: dict[tuple[str, str], str] = {}
    for r in rows:
        by_cell.setdefault((r["config"], str(r["threads"])), {})[r["tool"]] = r
        groups[(r["config"], str(r["threads"]))] = r["group"]
    gates = {"fereg": None} | {c: comps for c, (_, comps, _) in LANG_GATES.items()}
    out = set()
    for (cfg, T), byt in by_cell.items():
        for cand, comps in gates.items():
            if comps is None:  # Stata gate: the group's Stata-gate tools
                comps = [t for t in GROUP_TOOLS[groups[(cfg, T)]] if t != cand]
            ratio, verdict = gate(byt, cand, comps)
            if verdict == "FAIL" or (ratio and float(ratio) > 1 / margin):
                out.add((cfg, T, cand))
    return out


def write_report(rows: list[dict]) -> None:
    lines = [
        "# fereg benchmark results",
        "",
        f"Generated {datetime.now().isoformat(timespec='seconds')} by `run.py` from `results/results.csv`.",
        "Times are medians of the timed runs in seconds (`ms` = milliseconds); estimation call only, data in memory.",
        "`ratio` = fereg / fastest finished competitor. Gate: PASS iff fereg is faster than every competitor that",
        "finished (a competitor timeout counts as slower); `n/a` = fereg not run. `error`/`timeout`/`unsupported`",
        "cells are explained in results.csv. Coefficient gap: max relative difference over shared non-constant",
        "coefficients versus fereg (versus fixest when fereg has no result for the row).",
        "",
    ]
    by_group: dict[str, dict[tuple[str, str], dict[str, dict]]] = {}
    for r in rows:
        by_group.setdefault(r["group"], {}).setdefault((r["config"], str(r["threads"])), {})[r["tool"]] = r
    n_pass, n_fail, _ = _gate_section(lines, by_group, "fereg", None, ["fereg", "fixest"], "##")
    summary = [f"**Gate summary:** {n_pass} PASS, {n_fail} FAIL.", ""]

    for cand, (title, comps, ref) in LANG_GATES.items():
        if not any(r["tool"] == cand for r in rows):
            continue
        sec = [
            f"# {title} gate",
            "",
            f"`ratio` = {cand} / fastest finished competitor among {', '.join(comps)}. Gate: PASS iff {cand} is",
            "faster than every one of them that finished (a timeout counts as slower; errors and `unsupported`",
            f"are excluded). Coefficient gap: max relative difference versus {ref} (versus the Stata fereg",
            f"when {ref} has no coefficients for the row).",
            "",
        ]
        body: list[str] = []
        p, f, fails = _gate_section(body, by_group, cand, {g: comps for g in GROUP_TOOLS}, [ref, "fereg"], "##")
        sec += [f"**{cand} gate summary:** {p} PASS, {f} FAIL.", ""]
        if fails:
            sec += ["FAIL rows:", "", "| group | config | T | ratio |", "|---|---|---:|---:|"]
            sec += [f"| {g} | `{cfg}` | {T} | {ratio or 'n/a'} |" for g, cfg, T, ratio in fails] + [""]
        lines += sec + body
        summary += [f"**{cand} gate summary:** {p} PASS, {f} FAIL (section below).", ""]
    lines[9:9] = summary
    RESULTS_MD.write_text("\n".join(lines))


# --- main ----------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", choices=["smoke", "gate"], default="smoke")
    ap.add_argument("--tools", default=",".join(ALL_TOOLS))
    ap.add_argument("--groups", default="G1,G2,G3,G4,G5,G6,G7")
    ap.add_argument("--threads", default="16,1")
    ap.add_argument("--max-n", type=float, default=math.inf, help="skip configs with n above this")
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true", help="list the jobs without running them")
    ap.add_argument("--missing", action="store_true", help="skip jobs that already have a row in results.csv (resume)")
    ap.add_argument("--thin", type=float, default=None, metavar="MARGIN",
                    help="re-run only gate candidates (fereg, fereg_py, fereg_r) whose row FAILs or wins by less "
                         "than MARGIN (e.g. 1.2: ratio above 1/1.2); competitors are never selected")
    ap.add_argument("--report-only", action="store_true", help="re-render RESULTS.md from results.csv")
    a = ap.parse_args()

    if a.report_only:
        write_report(load_results())
        return
    tools = [t.strip() for t in a.tools.split(",") if t.strip()]
    unknown = set(tools) - set(ALL_TOOLS)
    if unknown:
        sys.exit(f"unknown tools: {sorted(unknown)}; choose from {ALL_TOOLS}")
    groups = set(a.groups.split(","))
    threads = {int(t) for t in a.threads.split(",")}

    jobs: list[Job] = []
    for cfg, ts in build_grid(a.grid):
        if cfg.group not in groups or cfg.n > a.max_n:
            continue
        for T in ts:
            if T not in threads:
                continue
            for tool in GROUP_TOOLS[cfg.group] + list(LANG_GATES):
                if tool in tools:
                    jobs.append(Job(tool, cfg, T))
    if a.missing:
        done = {_row_key(r) for r in load_results()}
        jobs = [j for j in jobs if _row_key(job_row(j)) not in done]
    if a.thin is not None:
        thin = _thin_keys(load_results(), a.thin)
        jobs = [j for j in jobs if _row_key(job_row(j)) in thin]
    if a.dry_run:
        for j in jobs:
            print(f"{j.cfg.group} T={j.threads:<2} {j.tool:<14} {j.cfg.key}")
        print(f"{len(jobs)} jobs")
        return

    rows = {_row_key(r): r for r in load_results()}
    t_start = time.perf_counter()
    for i, job in enumerate(jobs, 1):
        if not supported(job.tool, job.cfg):
            job.status = "unsupported"
        else:
            run_job(job, a.reps, a.timeout)
        row = job_row(job)
        rows[_row_key(row)] = row
        save_results(list(rows.values()))
        med = row["median_s"] or "-"
        print(f"[{i}/{len(jobs)} {time.perf_counter() - t_start:7.0f}s] {job.cfg.group} T={job.threads:<2} "
              f"{job.tool:<14} {job.status:<11} {med:>10}  {job.cfg.key}"
              + (f"  ({job.error[:120]})" if job.status != "ok" and job.error else ""), flush=True)
    write_report(list(rows.values()))
    print(f"wrote {RESULTS_CSV} and {RESULTS_MD}")


if __name__ == "__main__":
    main()
