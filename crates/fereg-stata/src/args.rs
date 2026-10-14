//! `plugin call` argument parsing (protocol: `argv[0]` = command, then `key=value` tokens).

use std::collections::HashMap;
use std::str::FromStr;

use fereg_core::hac::Kernel;
use fereg_core::{AbsorbTerm, AbsorberKind, Bandwidth, ConleyVce, DofOptions, Family, KernelVce, Psd, RfError, TimeVars, Vce, WeightKind};

fn bad(msg: String) -> RfError {
    RfError::InvalidInput(format!("fereg plugin: {msg}"))
}

/// Parsed `key=value` arguments. Every key must be consumed by the command (unknown keys are
/// errors, which catches ado/plugin protocol drift).
pub struct Args {
    pub cmd: String,
    map: HashMap<String, String>,
    used: std::cell::RefCell<Vec<String>>,
}

impl Args {
    pub fn parse(argv: &[String]) -> Result<Args, RfError> {
        let Some((cmd, rest)) = argv.split_first() else {
            return Err(bad("no command given".into()));
        };
        let mut map = HashMap::new();
        for tok in rest {
            let Some((k, v)) = tok.split_once('=') else {
                return Err(bad(format!("argument `{tok}` is not key=value")));
            };
            if map.insert(k.to_string(), v.to_string()).is_some() {
                return Err(bad(format!("key `{k}` given twice")));
            }
        }
        Ok(Args { cmd: cmd.clone(), map, used: Default::default() })
    }

    fn get(&self, key: &str) -> Option<&str> {
        self.used.borrow_mut().push(key.to_string());
        self.map.get(key).map(String::as_str)
    }

    pub fn str_opt(&self, key: &str) -> Option<&str> {
        self.get(key)
    }

    pub fn str_or<'a>(&'a self, key: &str, default: &'a str) -> &'a str {
        self.get(key).unwrap_or(default)
    }

    pub fn str_req(&self, key: &str) -> Result<&str, RfError> {
        self.get(key).ok_or_else(|| bad(format!("missing key `{key}`")))
    }

    pub fn num_or<T: FromStr>(&self, key: &str, default: T) -> Result<T, RfError> {
        match self.get(key) {
            None => Ok(default),
            Some(v) => v.parse().map_err(|_| bad(format!("invalid value `{v}` for `{key}`"))),
        }
    }

    pub fn num_req<T: FromStr>(&self, key: &str) -> Result<T, RfError> {
        let v = self.str_req(key)?;
        v.parse().map_err(|_| bad(format!("invalid value `{v}` for `{key}`")))
    }

    pub fn flag(&self, key: &str, default: bool) -> Result<bool, RfError> {
        match self.get(key) {
            None => Ok(default),
            Some("0") => Ok(false),
            Some("1") => Ok(true),
            Some(v) => Err(bad(format!("invalid value `{v}` for `{key}` (0 or 1)"))),
        }
    }

    /// Errors on any key the command did not look at.
    pub fn finish(&self) -> Result<(), RfError> {
        let used = self.used.borrow();
        let mut unknown: Vec<&str> =
            self.map.keys().filter(|k| !used.iter().any(|u| u == *k)).map(String::as_str).collect();
        if unknown.is_empty() {
            return Ok(());
        }
        unknown.sort_unstable();
        Err(bad(format!("unknown key(s) for cmd {}: {}", self.cmd, unknown.join(" "))))
    }

    /// Common execution keys; `serialio` defaults to `serial_default`.
    pub fn common(&self, serial_default: bool) -> Result<Common, RfError> {
        let verbose: u8 = self.num_or("verbose", 0)?;
        if verbose > 2 {
            return Err(bad("verbose must be 0, 1 or 2".into()));
        }
        Ok(Common { threads: self.num_or("threads", 0)?, verbose, serialio: self.flag("serialio", serial_default)? })
    }

    pub fn absorber(&self) -> Result<AbsorberKind, RfError> {
        match self.str_or("absorber", "fast") {
            "fast" => Ok(AbsorberKind::Fast),
            "dense" => Ok(AbsorberKind::Dense),
            v => Err(bad(format!("invalid absorber `{v}` (fast|dense)"))),
        }
    }

    /// `rowread=auto|direct|kept` (debug/benchmark: force a numeric read path).
    pub fn row_read(&self) -> Result<RowRead, RfError> {
        match self.str_or("rowread", "auto") {
            "auto" => Ok(RowRead::Auto),
            "direct" => Ok(RowRead::Direct),
            "kept" => Ok(RowRead::Kept),
            v => Err(bad(format!("invalid rowread `{v}` (auto|direct|kept)"))),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Common {
    /// 0 = all logical CPUs.
    pub threads: usize,
    pub verbose: u8,
    /// Read Stata data on the calling thread only.
    pub serialio: bool,
}

/// How rows in solver order are read from Stata.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RowRead {
    /// Direct when the row order is local, else kept order + permutation.
    Auto,
    /// Read every row straight from its Stata observation.
    Direct,
    /// Read in kept (ascending observation) order, then permute rows in memory.
    Kept,
}

/// 1-based list `a*b*c` -> 0-based indices below `limit`.
fn index_list(s: &str, limit: usize, what: &str) -> Result<Vec<usize>, RfError> {
    s.split('*')
        .map(|t| match t.parse::<usize>() {
            Ok(i) if i >= 1 && i <= limit => Ok(i - 1),
            _ => Err(bad(format!("invalid {what} index `{t}` in `{s}`"))),
        })
        .collect()
}

fn bit(s: &str, what: &str) -> Result<bool, RfError> {
    match s {
        "0" => Ok(false),
        "1" => Ok(true),
        _ => Err(bad(format!("invalid {what} flag `{s}`"))),
    }
}

/// `absorb=` value: `none`, or `;`-separated `<factors>|<icpt>|<slopes>|<save>` terms.
pub fn parse_absorb(s: &str, nfac: usize, nslope: usize) -> Result<Vec<AbsorbTerm>, RfError> {
    if s == "none" {
        return Ok(Vec::new());
    }
    s.split(';')
        .map(|term| {
            let parts: Vec<&str> = term.split('|').collect();
            let [f, i, sl, sv] = parts[..] else {
                return Err(bad(format!("invalid absorb term `{term}`")));
            };
            let slopes = if sl == "-" { Vec::new() } else { index_list(sl, nslope, "slope")? };
            Ok(AbsorbTerm {
                factors: index_list(f, nfac, "factor")?,
                intercept: bit(i, "intercept")?,
                slopes,
                save: bit(sv, "save")?,
            })
        })
        .collect()
}

fn code(rc: i32, msg: &str) -> RfError {
    RfError::Code(rc, msg.to_string())
}

/// The variance keys of `cmd=prepare` (PROTOCOL.md, "Variance estimators"): `vce`, `cluster`,
/// `tvar`, `ivar`, `tdelta`, `dkbw`, `kernel`, `bw`, `kiefer`, `psd`, `cutoff`, `pixel`,
/// `distance`. Conley coordinates are the slope variables `nslope` and `nslope + 1`.
pub fn parse_vce(a: &Args, nfac: usize, nslope: usize, ncoord: usize) -> Result<Vce, RfError> {
    let vce = a.str_or("vce", "unadjusted");
    let dims = |s: &str| s.split(';').map(|d| index_list(d, nfac, "cluster factor")).collect::<Result<Vec<_>, _>>();
    let cluster = a.str_opt("cluster").map(dims).transpose()?;
    let factor_pos = |key: &str| -> Result<Option<usize>, RfError> { a.str_opt(key).map(|s| index_list(s, nfac, key).map(|v| v[0])).transpose() };
    let tvar = factor_pos("tvar")?;
    let ivar = factor_pos("ivar")?;
    let tdelta: f64 = a.num_or("tdelta", 1.0)?;
    let time = |what: &str| -> Result<TimeVars, RfError> {
        let time = tvar.ok_or_else(|| bad(format!("{what} needs tvar=")))?;
        Ok(TimeVars { time, panel: ivar, delta: tdelta })
    };
    let kernel = a.str_opt("kernel");
    let bw = a.str_opt("bw");
    let kiefer = a.flag("kiefer", false)?;
    let psd = match a.str_or("psd", "none") {
        "none" => Psd::None,
        "psd0" => Psd::Zero,
        "psda" => Psd::Abs,
        v => return Err(bad(format!("invalid psd `{v}` (none|psd0|psda)"))),
    };
    let ncoord_ok = |want: usize| if ncoord == want { Ok(()) } else { Err(bad(format!("vce={vce} needs ncoord={want}"))) };

    // ivreg2 kernel family: `kernel=`, `bw=` or `kiefer=1` with unadjusted (AC), robust (HAC) or
    // cluster (Driscoll-Kraay / Thompson).
    if kernel.is_some() || bw.is_some() || kiefer {
        let robust = match vce {
            "unadjusted" => false,
            "robust" => true,
            "cluster" => true,
            v => return Err(bad(format!("kernel= is not allowed with vce={v}"))),
        };
        if vce == "cluster" && cluster.is_none() {
            return Err(bad("vce=cluster needs cluster=".into()));
        }
        if vce != "cluster" && cluster.is_some() {
            return Err(bad("cluster= requires vce=cluster".into()));
        }
        ncoord_ok(0)?;
        let kernel = match kernel {
            Some(k) => Kernel::parse(k).ok_or_else(|| code(198, "invalid kernel"))?,
            None if kiefer => Kernel::Truncated,
            None => Kernel::Bartlett,
        };
        if kiefer && kernel != Kernel::Truncated {
            return Err(code(198, &format!("incompatible options: kiefer and kernel({})", kernel.name())));
        }
        let bw = match (bw, kiefer) {
            (Some(_), true) => return Err(code(198, "incompatible options: kiefer and bw")),
            (None, true) => Bandwidth::Fixed(f64::NAN),
            (None, false) => return Err(code(102, "bandwidth option bw() required for HAC-robust estimation")),
            (Some("auto"), false) => Bandwidth::Auto,
            (Some(s), false) => match s.parse::<f64>() {
                Ok(b) if b > 0.0 => Bandwidth::Fixed(b),
                Ok(_) => return Err(code(198, "invalid bandwidth in option bw() - must be real > 0")),
                Err(_) => return Err(code(102, "bandwidth option bw() required for HAC-robust estimation")),
            },
        };
        return Ok(Vce::Kernel(KernelVce { kernel, bw, robust, cluster: cluster.unwrap_or_default(), kiefer, psd, time: time("kernel=")? }));
    }
    if psd != Psd::None {
        return Err(bad("psd= requires a kernel".into()));
    }
    if vce != "conley" {
        ncoord_ok(0)?;
    }
    let needs_cluster = matches!(vce, "cluster" | "crv3");
    match (needs_cluster, &cluster) {
        (true, None) => return Err(bad(format!("vce={vce} needs cluster="))),
        (false, Some(_)) => return Err(bad("cluster= requires vce=cluster or vce=crv3".into())),
        _ => {}
    }
    Ok(match vce {
        "unadjusted" => Vce::Unadjusted,
        "robust" => Vce::Robust,
        "cluster" => Vce::Cluster(cluster.unwrap_or_default()),
        "crv3" => Vce::Crv3(cluster.unwrap_or_default()),
        "hc2" => Vce::Hc2,
        "hc3" => Vce::Hc3,
        "dkraay" => {
            let bw = match a.str_opt("dkbw") {
                None => None,
                Some(s) => match s.parse::<usize>() {
                    Ok(b) if b >= 1 => Some(b),
                    _ => return Err(code(198, "vce(dkraay #): # must be a positive integer (bandwidth = lags + 1)")),
                },
            };
            Vce::Dkraay { time: time("vce=dkraay")?, bw }
        }
        "conley" => {
            ncoord_ok(2)?;
            let cutoff = match a.str_opt("cutoff") {
                None => None,
                Some(s) => match s.parse::<f64>() {
                    Ok(c) if c > 0.0 => Some(c),
                    _ => return Err(bad(format!("invalid cutoff `{s}` (km > 0)"))),
                },
            };
            let pixel: f64 = a.num_or("pixel", 0.0)?;
            if !(pixel >= 0.0) {
                return Err(bad("pixel must be >= 0".into()));
            }
            let spherical = match a.str_or("distance", "triangular") {
                "triangular" => false,
                "spherical" => true,
                v => return Err(bad(format!("invalid distance `{v}` (triangular|spherical)"))),
            };
            Vce::Conley(ConleyVce { lat: nslope, lon: nslope + 1, cutoff, pixel, spherical })
        }
        v => return Err(bad(format!("invalid vce `{v}`"))),
    })
}

pub fn parse_wtype(s: &str) -> Result<WeightKind, RfError> {
    match s {
        "none" => Ok(WeightKind::None),
        "fw" => Ok(WeightKind::Fweight),
        "aw" => Ok(WeightKind::Aweight),
        "pw" => Ok(WeightKind::Pweight),
        _ => Err(bad(format!("invalid wtype `{s}`"))),
    }
}

pub fn parse_family(s: &str) -> Result<Family, RfError> {
    match s {
        "gaussian" => Ok(Family::Gaussian),
        "poisson" => Ok(Family::Poisson),
        "logit" => Ok(Family::Logit),
        _ => Err(bad(format!("invalid family `{s}`"))),
    }
}

/// `dof=` value: `none` or a comma list of pairwise,clusters,continuous,firstpair.
pub fn parse_dof(s: &str) -> Result<DofOptions, RfError> {
    if s == "none" {
        return Ok(DofOptions::none());
    }
    let mut d = DofOptions::none();
    for t in s.split(',') {
        match t {
            "pairwise" => d.pairwise = true,
            "clusters" => d.clusters = true,
            "continuous" => d.continuous = true,
            "firstpair" => d.firstpair = true,
            _ => return Err(bad(format!("invalid dof item `{t}`"))),
        }
    }
    Ok(d)
}

/// Output position list `k1,k2,...` or `0` (1-based among `nout` output variables; 0 = none).
pub fn parse_positions(s: &str, nout: usize, key: &str) -> Result<Vec<usize>, RfError> {
    if s == "0" {
        return Ok(Vec::new());
    }
    s.split(',')
        .map(|t| match t.parse::<usize>() {
            Ok(k) if k >= 1 && k <= nout => Ok(k),
            _ => Err(bad(format!("invalid output position `{t}` in {key}="))),
        })
        .collect()
}

/// Single output position (`0` = none).
pub fn parse_position(s: &str, nout: usize, key: &str) -> Result<Option<usize>, RfError> {
    match s.parse::<usize>() {
        Ok(0) => Ok(None),
        Ok(k) if k <= nout => Ok(Some(k)),
        _ => Err(bad(format!("invalid output position `{s}` for {key}="))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(v: &[&str]) -> Args {
        Args::parse(&v.iter().map(|s| s.to_string()).collect::<Vec<_>>()).unwrap()
    }

    #[test]
    fn absorb_grammar() {
        let t = parse_absorb("1|1|-|0;2*3|1|-|1;4|0|1*2|1", 4, 2).unwrap();
        assert_eq!(t.len(), 3);
        assert_eq!(t[1].factors, vec![1, 2]);
        assert!(t[1].save);
        assert_eq!(t[2], AbsorbTerm { factors: vec![3], intercept: false, slopes: vec![0, 1], save: true });
        assert!(parse_absorb("none", 0, 0).unwrap().is_empty());
        assert!(parse_absorb("5|1|-|0", 4, 0).is_err());
        assert!(parse_absorb("1|1|-", 4, 0).is_err());
        assert!(parse_absorb("1|2|-|0", 4, 0).is_err());
    }

    #[test]
    fn vce_and_dof() {
        let vce = |v: &[&str], ncoord: usize| {
            let mut t = vec!["prepare"];
            t.extend_from_slice(v);
            let a = args(&t);
            let r = parse_vce(&a, 3, 1, ncoord);
            if r.is_ok() {
                a.finish().unwrap();
            }
            r
        };
        assert_eq!(vce(&["vce=cluster", "cluster=1;2*3"], 0).unwrap(), Vce::Cluster(vec![vec![0], vec![1, 2]]));
        assert!(vce(&["vce=cluster"], 0).is_err());
        assert!(vce(&["vce=robust", "cluster=1"], 0).is_err());
        assert_eq!(vce(&["vce=crv3", "cluster=2"], 0).unwrap(), Vce::Crv3(vec![vec![1]]));
        assert_eq!(vce(&["vce=hc3"], 0).unwrap(), Vce::Hc3);
        let tv = TimeVars { time: 2, panel: Some(0), delta: 2.0 };
        assert_eq!(vce(&["vce=dkraay", "tvar=3", "ivar=1", "tdelta=2", "dkbw=4"], 0).unwrap(), Vce::Dkraay { time: tv.clone(), bw: Some(4) });
        assert_eq!(vce(&["vce=dkraay", "dkbw=0", "tvar=3"], 0).unwrap_err().rc(), 198);
        assert!(vce(&["vce=dkraay"], 0).is_err());
        let k = vce(&["vce=cluster", "cluster=3", "kernel=quadratic-spectral", "bw=2.5", "tvar=3", "ivar=1", "tdelta=2", "psd=psda"], 0).unwrap();
        assert_eq!(
            k,
            Vce::Kernel(KernelVce {
                kernel: Kernel::QuadraticSpectral,
                bw: Bandwidth::Fixed(2.5),
                robust: true,
                cluster: vec![vec![2]],
                kiefer: false,
                psd: Psd::Abs,
                time: tv
            })
        );
        let Vce::Kernel(k) = vce(&["vce=unadjusted", "kiefer=1", "tvar=3", "ivar=1"], 0).unwrap() else { panic!() };
        assert!(k.kiefer && !k.robust && k.kernel == Kernel::Truncated);
        let Vce::Kernel(k) = vce(&["vce=robust", "bw=auto", "tvar=3"], 0).unwrap() else { panic!() };
        assert!(k.robust && k.kernel == Kernel::Bartlett && k.bw == Bandwidth::Auto);
        assert_eq!(vce(&["vce=robust", "kernel=newey", "bw=2", "tvar=3"], 0).unwrap_err().rc(), 198);
        assert_eq!(vce(&["vce=robust", "kernel=parzen", "tvar=3"], 0).unwrap_err().rc(), 102);
        assert!(vce(&["vce=robust", "bw=2"], 0).is_err());
        assert_eq!(
            vce(&["vce=conley", "cutoff=100", "distance=spherical"], 2).unwrap(),
            Vce::Conley(ConleyVce { lat: 1, lon: 2, cutoff: Some(100.0), pixel: 0.0, spherical: true })
        );
        assert!(vce(&["vce=conley"], 0).is_err());
        assert!(vce(&["vce=robust"], 2).is_err());
        assert_eq!(parse_dof("pairwise,clusters,continuous").unwrap(), DofOptions::default());
        assert_eq!(parse_dof("none").unwrap(), DofOptions::none());
        assert!(parse_dof("foo").is_err());
    }

    #[test]
    fn unknown_keys_are_reported() {
        let a = args(&["ols", "nx=2", "bogus=1"]);
        assert_eq!(a.num_req::<usize>("nx").unwrap(), 2);
        assert!(a.finish().is_err());
        let a = args(&["ols", "nx=2"]);
        let _ = a.num_req::<usize>("nx");
        assert!(a.finish().is_ok());
        assert!(Args::parse(&["x".into(), "novalue".into()]).is_err());
    }

    #[test]
    fn positions() {
        assert_eq!(parse_positions("1,3", 3, "fe").unwrap(), vec![1, 3]);
        assert!(parse_positions("0", 3, "fe").unwrap().is_empty());
        assert!(parse_positions("4", 3, "fe").is_err());
        assert_eq!(parse_position("0", 2, "d").unwrap(), None);
        assert_eq!(parse_position("2", 2, "d").unwrap(), Some(2));
    }
}
