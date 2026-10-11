#!/usr/bin/env bash
# Installs the ado/sthlp files and the plugin from stata/ into Stata's PERSONAL directory
# (c(sysdir_personal), queried from Stata). Run scripts/build_plugin.sh first.
# Usage: scripts/install_local.sh   (STATA=<stata binary> overrides the default stata-mp)
set -euo pipefail
cd "$(dirname "$0")/.."

stata="${STATA:-stata-mp}"
if [[ ! -f stata/regfast.plugin ]]; then
  echo "install_local.sh: stata/regfast.plugin missing; run scripts/build_plugin.sh" >&2
  exit 1
fi

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
printf 'display "RFDIR:" c(sysdir_personal)\nexit, clear\n' >"$tmp/q.do"
dir="$(cd "$tmp" && "$stata" -q do q.do | sed -n 's/^RFDIR://p' | tail -n 1)"
# Stata may report the directory as ~/...; expand it.
dir="${dir/#\~/$HOME}"
if [[ -z "$dir" ]]; then
  echo "install_local.sh: could not query c(sysdir_personal) from $stata" >&2
  exit 1
fi
mkdir -p "$dir"

shopt -s nullglob
files=(stata/*.ado stata/*.sthlp stata/regfast.plugin)
cp "${files[@]}" "$dir"
echo "installed ${#files[@]} files into $dir"
