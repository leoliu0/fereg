# regfast

Fast high-dimensional fixed-effects estimation for Stata, powered by a Rust core.

| Command | Estimator | Mirrors |
|---|---|---|
| `regfast` | OLS and IV/2SLS with any number of absorbed fixed effects | `reghdfe` 6.13.1 / `ivreghdfe` 1.1.4 |
| `ppmlfast` | Poisson pseudo-maximum likelihood | `ppmlhdfe` 2.3.3 |
| `logitfast` | Logit maximum likelihood | `logit y x i.fe1 i.fe2 ...` |

All three support weights, robust and multi-way clustered standard errors, residuals and
`predict`, saved fixed effects, and heterogeneous slopes (`absorb(firm##c.x)`).

The solver eliminates the largest fixed effect exactly and solves the remaining system with a
preconditioned conjugate-gradient method on cell-compressed sufficient statistics.

## Install (Stata 16+)

```stata
net install regfast, from("https://raw.githubusercontent.com/leoliu0/regfast/gh-pages/stata") replace
```

Binaries are provided for Linux (x86-64), macOS (Apple Silicon and Intel) and Windows (x86-64).

## Build from source

Requires Rust (stable) and a C compiler.

```sh
scripts/build_plugin.sh      # builds target/release plugin and copies it to stata/
scripts/install_local.sh     # installs the ado files and plugin into your PERSONAL ado directory
```

The Rust library alone: `cargo test --workspace --release`.

## License

MIT
