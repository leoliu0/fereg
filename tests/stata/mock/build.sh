#!/usr/bin/env bash
# Build the fake fereg.plugin used by tests/stata/test_ado_mock.do (Linux/macOS).
# Needs the vendored stplugin.h (crates/fereg-stata/c) or STPLUGIN_DIR pointing to it.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../../.." && pwd)"
inc="${STPLUGIN_DIR:-$root/crates/fereg-stata/c}"
if [[ ! -f "$inc/stplugin.h" ]]; then
  echo "stplugin.h not found in $inc (set STPLUGIN_DIR)" >&2
  exit 1
fi
cc -shared -fPIC -O2 -Wall -DSYSTEM=OPUNIX -I"$inc" -o "$here/fereg.plugin" "$here/rfmock.c"
echo "built $here/fereg.plugin"
