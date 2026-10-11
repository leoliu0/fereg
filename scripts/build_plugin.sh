#!/usr/bin/env bash
# Builds the Stata plugin in release mode and copies it into stata/ as
#   stata/regfast.<linux64|macos-universal|win64>.plugin  (the name used by regfast.pkg)
#   stata/regfast.plugin                                   (the name `program ..., plugin` loads)
# Usage: scripts/build_plugin.sh [extra cargo args...]
# On macOS both architectures are built and merged with lipo (needs both rustc targets).
set -euo pipefail
cd "$(dirname "$0")/.."

os="$(uname -s)"
case "$os" in
  Linux)
    cargo build --release -p regfast-stata "$@"
    src="target/release/libregfast_stata.so"
    name="linux64"
    ;;
  Darwin)
    export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-11.0}"
    cargo build --release -p regfast-stata --target aarch64-apple-darwin "$@"
    MACOSX_DEPLOYMENT_TARGET=10.13 cargo build --release -p regfast-stata --target x86_64-apple-darwin "$@"
    mkdir -p target/universal
    lipo -create \
      target/aarch64-apple-darwin/release/libregfast_stata.dylib \
      target/x86_64-apple-darwin/release/libregfast_stata.dylib \
      -output target/universal/libregfast_stata.dylib
    codesign --force -s - target/universal/libregfast_stata.dylib
    src="target/universal/libregfast_stata.dylib"
    name="macos-universal"
    ;;
  MINGW*|MSYS*|CYGWIN*)
    cargo build --release -p regfast-stata --target x86_64-pc-windows-msvc "$@"
    src="target/x86_64-pc-windows-msvc/release/regfast_stata.dll"
    name="win64"
    ;;
  *)
    echo "build_plugin.sh: unsupported OS $os" >&2
    exit 1
    ;;
esac

mkdir -p stata
cp "$src" "stata/regfast.$name.plugin"
cp "$src" "stata/regfast.plugin"
echo "built stata/regfast.$name.plugin and stata/regfast.plugin"
