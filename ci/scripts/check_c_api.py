#!/usr/bin/env python3
"""Checks on the C API surface.

Two independent checks, either of which can be run alone:

``symbols``
    Every function declared ``CSP_C_API_EXPORT`` in ``cpp/csp/engine/c`` must actually be
    exported by the built extension module. Catches declarations that were never implemented,
    and implementations that fail to export on a given platform.

``headers``
    Every public header must compile standalone, as both C11 and C++20. Catches missing
    includes, C/C++ incompatibilities, and unbalanced ``extern "C"`` blocks.
"""

from __future__ import annotations

import argparse
import platform
import re
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
C_API_DIR = REPO_ROOT / "cpp" / "csp" / "engine" / "c"
PY_BRIDGE_DIR = REPO_ROOT / "cpp" / "csp" / "python" / "c"

# CSP_C_API_EXPORT <return type> ccsp_name(
DECL_RE = re.compile(r"CSP_C_API_EXPORT\s+[^;(]*?\b(ccsp_[A-Za-z0-9_]+)\s*\(", re.MULTILINE)


def declared_symbols() -> set[str]:
    symbols: set[str] = set()
    for header in sorted(C_API_DIR.glob("*.h")):
        symbols |= set(DECL_RE.findall(header.read_text()))
    return symbols


def exported_symbols(library: Path) -> set[str]:
    if platform.system() == "Windows":
        out = subprocess.run(
            ["dumpbin", "/EXPORTS", str(library)], capture_output=True, text=True, check=True
        ).stdout
        return set(re.findall(r"\b(ccsp_[A-Za-z0-9_]+)\b", out))

    out = subprocess.run(
        ["nm", "-gU" if platform.system() == "Darwin" else "-D", "--defined-only", str(library)]
        if platform.system() != "Darwin"
        else ["nm", "-gU", str(library)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    # macOS prefixes symbols with an underscore
    return {m.lstrip("_") for m in re.findall(r"\b_?(ccsp_[A-Za-z0-9_]+)\b", out)}


def find_library() -> Path | None:
    suffixes = ("*.so", "*.pyd", "*.dylib")
    for pattern in suffixes:
        for candidate in REPO_ROOT.rglob(f"_cspimpl{pattern[1:]}"):
            return candidate
    for pattern in suffixes:
        matches = sorted(REPO_ROOT.rglob(f"cspimpl{pattern[1:]}"))
        if matches:
            return matches[0]
    return None


def check_symbols(library: Path) -> int:
    declared = declared_symbols()
    if not declared:
        print("ERROR: no CSP_C_API_EXPORT declarations found; is the path correct?")
        return 1

    exported = exported_symbols(library)
    missing = sorted(declared - exported)

    print(f"declared: {len(declared)}  exported: {len(exported & declared)}  library: {library}")
    if missing:
        print(f"\nERROR: {len(missing)} declared C API symbols are not exported:")
        for name in missing:
            print(f"  {name}")
        return 1

    print("All declared C API symbols are exported.")
    return 0


def check_headers() -> int:
    headers = sorted(C_API_DIR.glob("*.h"))
    bridge_headers = sorted(PY_BRIDGE_DIR.glob("*.h"))
    if not headers:
        print("ERROR: no public headers found")
        return 1

    include_args = ["-I", str(REPO_ROOT / "cpp")]
    python_include = sysconfig.get_paths()["include"]

    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        for header in headers + bridge_headers:
            rel = header.relative_to(REPO_ROOT / "cpp").as_posix()
            # The Python bridge headers need Python.h; the pure C API headers must not.
            extra = ["-I", python_include] if header in bridge_headers else []

            for label, compiler, std in (
                ("C11", "cc", "-std=c11"),
                ("C++20", "c++", "-std=c++20"),
            ):
                src = tmpdir / f"probe.{'c' if label == 'C11' else 'cpp'}"
                src.write_text(f"#include <{rel}>\n")
                result = subprocess.run(
                    [compiler, std, "-fsyntax-only", "-Wall", *include_args, *extra, str(src)],
                    capture_output=True,
                    text=True,
                )
                status = "ok" if result.returncode == 0 else "FAIL"
                print(f"  [{status:4}] {label:5} {rel}")
                if result.returncode != 0:
                    failures.append((rel, label, result.stderr.strip()))

    if failures:
        print(f"\nERROR: {len(failures)} header compilations failed:")
        for rel, label, stderr in failures:
            print(f"\n--- {rel} ({label}) ---\n{stderr}")
        return 1

    print("\nAll public headers compile standalone as C11 and C++20.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("check", choices=("symbols", "headers", "all"))
    parser.add_argument("--library", type=Path, help="path to the built extension module")
    args = parser.parse_args()

    status = 0

    if args.check in ("headers", "all"):
        print("== header compilation ==")
        status |= check_headers()

    if args.check in ("symbols", "all"):
        print("\n== exported symbols ==")
        library = args.library or find_library()
        if library is None or not library.exists():
            print("ERROR: could not locate the built extension module; pass --library")
            return 1
        status |= check_symbols(library)

    return status


if __name__ == "__main__":
    sys.exit(main())
