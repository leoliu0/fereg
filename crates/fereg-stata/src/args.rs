//! `plugin call` argument parsing (protocol: `argv[0]` = command, then `key=value` tokens).

use std::collections::HashMap;
use std::str::FromStr;

use fereg_core::{AbsorbTerm, AbsorberKind, DofOptions, Family, RfError, Vce, WeightKind};

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

/// 1-based list `a*b*c` → 0-based indices below `limit`.
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

/// `vce=` and `cluster=` values.
pub fn parse_vce(vce: &str, cluster: Option<&str>, nfac: usize) -> Result<Vce, RfError> {
    match (vce, cluster) {
        ("unadjusted", None) => Ok(Vce::Unadjusted),
        ("robust", None) => Ok(Vce::Robust),
        ("cluster", Some(c)) => {
            Ok(Vce::Cluster(c.split(';').map(|d| index_list(d, nfac, "cluster factor")).collect::<Result<_, _>>()?))
        }
        ("cluster", None) => Err(bad("vce=cluster needs cluster=".into())),
        (_, Some(_)) => Err(bad("cluster= requires vce=cluster".into())),
        (v, None) => Err(bad(format!("invalid vce `{v}`"))),
    }
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
        assert_eq!(parse_vce("cluster", Some("1;2*3"), 3).unwrap(), Vce::Cluster(vec![vec![0], vec![1, 2]]));
        assert!(parse_vce("cluster", None, 3).is_err());
        assert!(parse_vce("robust", Some("1"), 3).is_err());
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
