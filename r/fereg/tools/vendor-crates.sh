#!/bin/sh
# Release step (CRAN-style offline builds): vendors every crates.io dependency of src/rust into
# src/rust/vendor.tar.xz. Run after scripts/sync-core.sh and before `R CMD build`. The Makevars
# unpack the tarball and point cargo at it with src/rust/vendor-config.toml.
set -eu

pkg=$(cd "$(dirname "$0")/.." && pwd)
cd "$pkg/src/rust"

if [ ! -f fereg-core/Cargo.toml ]; then
  echo "src/rust/fereg-core is missing: run scripts/sync-core.sh from the repository root first" >&2
  exit 1
fi

rm -rf vendor vendor.tar.xz
cargo vendor --locked --versioned-dirs vendor > /dev/null
tar -cJf vendor.tar.xz vendor
rm -rf vendor
echo "wrote $(pwd)/vendor.tar.xz ($(du -h vendor.tar.xz | cut -f1))"
