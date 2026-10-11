#!/usr/bin/env bash
# Build the fake regfast.plugin used by tests/stata/test_ado_mock.do (Linux/macOS).
# Needs the vendored stplugin.h (crates/regfast-stata/c) or STPLUGIN_DIR pointing to it.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../../.." && pwd)"
inc="${STPLUGIN_DIR:-$root/crates/regfast-stata/c}"
if [[ ! -f "$inc/stplugin.h" ]]; then
  echo "stplugin.h not found in $inc (set STPLUGIN_DIR)" >&2
  exit 1
fi
cc -shared -fPIC -O2 -Wall -DSYSTEM=OPUNIX -I"$inc" -o "$here/regfast.plugin" "$here/rfmock.c"
echo "built $here/regfast.plugin"
