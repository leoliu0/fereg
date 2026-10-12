#!/bin/sh
# Checks that every version string of a release agrees:
#   Cargo workspace version (Rust crates, the Stata plugin and the Python package),
#   r/fereg/DESCRIPTION Version, r/fereg/src/rust/Cargo.toml version,
#   the workspace crates in Cargo.lock and r/fereg/src/rust/Cargo.lock (builds use --locked),
#   and `local rf_version` in the Stata ado files that compare it with the plugin.
# Usage: scripts/check-versions.sh [vX.Y.Z]   (with a tag argument it must equal v<version>)
# Prints the version on success; exits 1 on any mismatch.
# After a version bump: `cargo update -w` (root) and, in r/fereg/src/rust after
# scripts/sync-core.sh, `cargo update -p fereg -p fereg-core`.
set -eu
cd "$(dirname "$0")/.."

cargo_version=$(awk '
  /^\[workspace\.package\]/ { inpkg = 1; next }
  /^\[/ { inpkg = 0 }
  inpkg && /^version *=/ { gsub(/.*= *"|".*/, ""); print; exit }
' Cargo.toml)
r_version=$(sed -n 's/^Version: *//p' r/fereg/DESCRIPTION)
r_crate_version=$(awk '
  /^\[package\]/ { inpkg = 1; next }
  /^\[/ { inpkg = 0 }
  inpkg && /^version *=/ { gsub(/.*= *"|".*/, ""); print; exit }
' r/fereg/src/rust/Cargo.toml)

status=0
check() { # name value
  if [ "$2" != "$cargo_version" ]; then
    echo "version mismatch: $1 is '$2', Cargo workspace version is '$cargo_version'" >&2
    status=1
  fi
}

if [ -z "$cargo_version" ]; then
  echo "could not read [workspace.package] version from Cargo.toml" >&2
  exit 1
fi
check "r/fereg/DESCRIPTION Version" "$r_version"
check "r/fereg/src/rust/Cargo.toml version" "$r_crate_version"
for ado in stata/*.ado; do
  v=$(sed -n 's/^[[:space:]]*local rf_version[[:space:]]*\([^[:space:]]*\).*/\1/p' "$ado")
  [ -z "$v" ] || check "$ado local rf_version" "$v"
done
# Version of package $2 in lock file $1.
lock_version() {
  awk -v pkg="$2" '
    /^\[\[package\]\]/ { name = "" }
    /^name = / { name = $3; gsub(/"/, "", name) }
    /^version = / && name == pkg { v = $3; gsub(/"/, "", v); print v; exit }
  ' "$1"
}
for pkg in fereg-core fereg-stata fereg-py; do
  check "Cargo.lock $pkg" "$(lock_version Cargo.lock "$pkg")"
done
for pkg in fereg fereg-core; do
  check "r/fereg/src/rust/Cargo.lock $pkg" "$(lock_version r/fereg/src/rust/Cargo.lock "$pkg")"
done
if [ $# -ge 1 ]; then
  check "tag $1 (without the leading v)" "${1#v}"
  case "$1" in v*) ;; *) echo "tag '$1' does not start with v" >&2; status=1 ;; esac
fi

[ $status -eq 0 ] || exit 1
echo "$cargo_version"
