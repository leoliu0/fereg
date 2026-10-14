#!/usr/bin/env python3
"""Release step for CRAN-style offline builds of the R package.

Run from anywhere after scripts/sync-core.sh and before `R CMD build`:

    python3 r/fereg/tools/vendor-crates.py

1. `cargo vendor --locked` copies every crates.io dependency of src/rust.
2. Crates that no supported platform compiles (e.g. `loom`, which only builds under
   `--cfg loom`, WASI and UEFI crates, i686 and MSVC import libraries) are replaced by stubs:
   their Cargo.toml is kept, so the lock file still resolves, and their sources are empty.
3. The other crates lose their tests, benches, examples and CI files, and the manifests lose
   the corresponding targets; windows-sys keeps only the API modules of the features some
   platform enables.
4. The `files` maps of every .cargo-checksum.json are emptied (cargo then skips the per-file
   checks; the package checksums in Cargo.lock still apply).
5. The result is packed into src/rust/vendor.tar.xz (sorted members, fixed mtime) and
   inst/AUTHORS is rewritten from `cargo metadata` (every vendored crate with version,
   authors and license).

The Makevars unpack the tarball and build with `cargo build --offline` (src/rust/vendor-config.toml).
Needs Python >= 3.8, cargo and rustc.
"""

import io
import json
import lzma
import os
import re
import shutil
import subprocess
import sys
import tarfile
import unicodedata
from pathlib import Path

PKG = Path(__file__).resolve().parent.parent
RUST = PKG / "src" / "rust"
VENDOR = RUST / "vendor"

# Platforms the package is built on (CRAN and r-universe Linux, macOS and Windows builders, plus
# other common Linux/BSD hosts). A crate needed by none of them is stubbed.
PLATFORMS = [
    "x86_64-unknown-linux-gnu",
    "aarch64-unknown-linux-gnu",
    "powerpc64le-unknown-linux-gnu",
    "s390x-unknown-linux-gnu",
    "riscv64gc-unknown-linux-gnu",
    "x86_64-unknown-linux-musl",
    "aarch64-unknown-linux-musl",
    "x86_64-unknown-freebsd",
    "x86_64-apple-darwin",
    "aarch64-apple-darwin",
    "x86_64-pc-windows-gnu",
    "x86_64-pc-windows-gnullvm",
    "aarch64-pc-windows-gnullvm",
]

PRUNE_DIRS = ["tests", "benches", "examples", ".github", "ci", "target"]
PRUNE_TARGETS = ("test", "bench", "example")


def cargo_metadata(*args):
    out = subprocess.run(
        ["cargo", "metadata", "--format-version", "1", "--locked", "--manifest-path", str(RUST / "Cargo.toml"), *args],
        check=True, capture_output=True, text=True,
    ).stdout
    return json.loads(out)


def needed_features():
    """Package id -> enabled features, over the packages some supported platform builds."""
    feats = {}
    for triple in PLATFORMS:
        meta = cargo_metadata("--filter-platform", triple)
        for node in meta["resolve"]["nodes"]:
            feats.setdefault(node["id"], set()).update(node["features"])
    return feats


def strip_targets(manifest: Path, kinds):
    """Drops the [[kind]] tables of `kinds` from a Cargo.toml (text-level, keeps everything else)."""
    out, skip = [], False
    for line in manifest.read_text(encoding="utf-8").splitlines(keepends=True):
        if line.lstrip().startswith("["):  # any table header ends the skipped table
            m = re.match(r"^\s*\[\[\s*([A-Za-z0-9_\-]+)\s*\]\]", line)
            skip = bool(m) and m.group(1) in kinds
        if not skip:
            out.append(line)
    text = "".join(out)
    # Keep cargo from auto-discovering targets in the removed directories.
    for key in ("autotests", "autobenches", "autoexamples"):
        text = re.sub(rf"^{key}\s*=.*\n", "", text, flags=re.M)
    text = re.sub(r"^\[package\]\n", "[package]\nautotests = false\nautobenches = false\nautoexamples = false\n", text,
                  count=1, flags=re.M)
    manifest.write_text(text, encoding="utf-8")


def clear_checksum(crate: Path):
    path = crate / ".cargo-checksum.json"
    data = json.loads(path.read_text())
    data["files"] = {}
    path.write_text(json.dumps(data, separators=(",", ":")))


def stub(crate: Path, pkg):
    """Empties a crate that no supported platform builds; keeps the manifest and checksum."""
    keep = {crate / "Cargo.toml", crate / ".cargo-checksum.json"}
    for p in sorted(crate.rglob("*"), reverse=True):
        if p in keep:
            continue
        if p.is_dir() and not p.is_symlink():
            p.rmdir()
        else:
            p.unlink()
    strip_targets(crate / "Cargo.toml", PRUNE_TARGETS + ("bin",))
    root = Path(pkg["manifest_path"]).parent  # the registry copy cargo metadata read
    for t in pkg["targets"]:
        if set(t["kind"]) & {"lib", "rlib", "proc-macro", "custom-build", "cdylib", "staticlib", "dylib"}:
            src = crate / Path(t["src_path"]).relative_to(root)
            src.parent.mkdir(parents=True, exist_ok=True)
            src.write_text("fn main() {}\n" if "custom-build" in t["kind"] else "")


def prune(crate: Path):
    for d in PRUNE_DIRS:
        shutil.rmtree(crate / d, ignore_errors=True)
    strip_targets(crate / "Cargo.toml", PRUNE_TARGETS)


def prune_windows_sys(crate: Path, features):
    """windows-sys compiles the module of namespace A::B::C (src/Windows/A/B/C) only with feature
    `A_B_C`: drops the modules of features no supported platform enables (most of the crate)."""
    root = crate / "src" / "Windows"
    for d in sorted(p for p in root.rglob("*") if p.is_dir()):
        if d.exists() and "_".join(d.relative_to(root).parts) not in features:
            shutil.rmtree(d)


def pack():
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for p in [VENDOR, *sorted(VENDOR.rglob("*"))]:
            info = tar.gettarinfo(str(p), arcname=str(p.relative_to(RUST)))
            info.mtime, info.uid, info.gid, info.uname, info.gname = 0, 0, 0, "", ""
            info.mode = 0o755 if p.is_dir() or os.access(p, os.X_OK) else 0o644
            if p.is_file():
                with open(p, "rb") as f:
                    tar.addfile(info, f)
            else:
                tar.addfile(info)
    out = RUST / "vendor.tar.xz"
    out.write_bytes(lzma.compress(buf.getvalue(), preset=9 | lzma.PRESET_EXTREME))
    return out


def write_authors(packages):
    lines = [
        "fereg bundles the Rust crate fereg-core (MIT license, Leo Liu) in src/rust/fereg-core and the",
        "crates.io crates below in src/rust/vendor.tar.xz (`cargo vendor`). Crates marked [stub] are",
        "needed only to resolve Cargo.lock: no supported platform compiles them, so the tarball keeps",
        "their Cargo.toml (with the authors and license) and none of their code.",
        "",
        "Generated by tools/vendor-crates.py from `cargo metadata`.",
        "",
    ]
    for p in packages:
        authors = ", ".join(re.sub(r"\s*<[^>]*>", "", a) for a in p["authors"]) or f"the {p['name']} authors"
        lines.append(f"{p['name']} {p['version']}{' [stub]' if p['stub'] else ''}")
        lines.append(f"  License: {p['license'] or 'see the crate license file'}")
        lines.append(f"  Authors: {authors}")
        if p.get("repository"):
            lines.append(f"  Source:  {p['repository']}")
        lines.append("")
    text = "\n".join(lines)
    # ASCII text (accents dropped: Schr\u00f6der -> Schroder).
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    (PKG / "inst" / "AUTHORS").write_text(text, encoding="ascii")


def main():
    if not (RUST / "fereg-core" / "Cargo.toml").exists():
        sys.exit("src/rust/fereg-core is missing: run scripts/sync-core.sh from the repository root first")
    shutil.rmtree(VENDOR, ignore_errors=True)
    (RUST / "vendor.tar.xz").unlink(missing_ok=True)
    subprocess.run(["cargo", "vendor", "--locked", "--versioned-dirs", str(VENDOR)], check=True,
                   cwd=RUST, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    meta = cargo_metadata()
    needed = needed_features()
    vendored = []
    for pkg in meta["packages"]:
        if pkg["source"] is None:  # fereg and fereg-core (path crates)
            continue
        crate = VENDOR / f"{pkg['name']}-{pkg['version']}"
        assert crate.is_dir(), crate
        is_stub = pkg["id"] not in needed
        if is_stub:
            stub(crate, pkg)
        else:
            prune(crate)
            if pkg["name"] == "windows-sys":
                prune_windows_sys(crate, needed[pkg["id"]])
        clear_checksum(crate)
        vendored.append({**pkg, "stub": is_stub})

    vendored.sort(key=lambda p: (p["name"], p["version"]))
    write_authors(vendored)
    out = pack()
    shutil.rmtree(VENDOR)
    nstub = sum(p["stub"] for p in vendored)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB): {len(vendored)} crates, {nstub} stubs")


if __name__ == "__main__":
    main()
