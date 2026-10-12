# Publishing a release

One tag builds and publishes everything: the Stata `net install` tree (branch `gh-pages`), and a
GitHub Release with the Stata zip, 25 Python wheels (CPython 3.10–3.14 × Linux x86-64/aarch64,
macOS arm64/x86-64, Windows x64), the Python sdist, the R source tarball (Rust crates vendored)
and R binaries for macOS arm64/x86-64 and Windows. PyPI is published too once it is enabled
(below). The workflow is `.github/workflows/release.yml`.

## Every release

1. Replace the old version with the new one (e.g. 0.3.0 → 0.3.1) in every file that has it: the
   Cargo workspace version (Rust crates, Stata plugin, Python package), `r/fereg/DESCRIPTION`,
   `r/fereg/src/rust/Cargo.toml`, `local rf_version` and the `*! version` headers in `stata/`,
   the mock plugin and its test, `crates/fereg-stata/PROTOCOL.md`, and the release URLs in
   `README.md`. List them with `git grep -lF 0.3.0 -- ':!*.lock' ':!bench'` and check each hit.
   Then refresh both lock files and check that every version string agrees:

   ```sh
   cargo update -w
   sh scripts/sync-core.sh && (cd r/fereg/src/rust && cargo update -p fereg -p fereg-core)
   sh scripts/check-versions.sh v0.3.1
   ```

2. Commit, push `main`, and wait for CI (`.github/workflows/ci.yml`) to pass on all jobs.
   Optionally rehearse the release without publishing:
   `gh workflow run release.yml --ref main -f dry_run=true`.
3. Tag and push the tag:

   ```sh
   git tag -a v0.3.1 -m "fereg 0.3.1"
   git push origin v0.3.1
   ```

   The `version` job fails the run early if the tag and any version string disagree.

## Once: enable PyPI

1. At <https://pypi.org/manage/account/publishing/>, add a pending publisher (GitHub) with
   project name `fereg`, owner `leoliu0`, repository `fereg`, workflow `release.yml`,
   environment `pypi`. A pending publisher does not reserve the name, so do this shortly before
   the tag.
2. `gh variable set PYPI_PUBLISH --body true -R leoliu0/fereg`

Without the variable the `pypi` job is skipped and the rest of the release still runs. To publish
an existing tag later, set the variable and run the release workflow on the tag ref with
`dry_run=false` (this also repeats the gh-pages and GitHub Release steps). Then `pip install
fereg` works and the `--find-links` part of the README's install command can go.

## Once: enable r-universe

1. Create the public repository `leoliu0/leoliu0.r-universe.dev` containing `packages.json`:

   ```json
   [{"package": "fereg", "url": "https://github.com/leoliu0/fereg", "subdir": "r/fereg", "branch": "*release"}]
   ```

   (`"branch": "*release"` builds the latest GitHub release; without it r-universe builds `main`.)
2. Install the r-universe GitHub app for that account:
   <https://github.com/apps/r-universe/installations/new>.

Builds appear within about an hour at <https://leoliu0.r-universe.dev>, after which R users can
`install.packages("fereg", repos = c("https://leoliu0.r-universe.dev", "https://cloud.r-project.org"))`.
r-universe runs `r/fereg/bootstrap.R` (copies the Rust core into the package) before building.
