#!/bin/sh
# Copies the fereg-core crate into the R package (r/fereg/src/rust/fereg-core) as a self-contained
# crate: `cargo package` resolves the workspace inheritance (version, edition, dependencies), so
# the copy builds outside this repository (R CMD build tarballs, r-universe, CRAN).
# Tests, examples and benches are not needed by the R build and are removed with their manifest
# entries to keep the R source tarball small.
set -eu

root=$(cd "$(dirname "$0")/.." && pwd)
dest="$root/r/fereg/src/rust/fereg-core"

cd "$root"
version=$(cargo pkgid -p fereg-core | sed 's/.*[#@]//')
cargo package -p fereg-core --no-verify --allow-dirty --quiet

crate="$root/target/package/fereg-core-$version.crate"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
# -m: extracted files get the current time, so cargo rebuilds the copy after every sync.
tar -xzmf "$crate" -C "$tmp"
src="$tmp/fereg-core-$version"

rm -rf "$src/tests" "$src/examples" "$src/benches" "$src/Cargo.toml.orig" "$src/Cargo.lock" "$src/.cargo_vcs_info.json"
awk '
  /^\[\[(example|test|bench)\]\]/ { skip = 1; next }
  /^\[/ { skip = 0 }
  !skip
' "$src/Cargo.toml" > "$src/Cargo.toml.new"
mv "$src/Cargo.toml.new" "$src/Cargo.toml"

rm -rf "$dest"
mkdir -p "$(dirname "$dest")"
mv "$src" "$dest"
echo "fereg-core $version -> $dest"
