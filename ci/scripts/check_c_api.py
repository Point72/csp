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
import struct
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


def pe_exported_symbols(library: Path) -> set[str]:
    """Read the export name table out of a PE image.

    dumpbin is the obvious tool but only exists inside a Visual Studio developer environment,
    which the CI runner does not enter, so parse the image directly instead.
    """
    data = library.read_bytes()
    pe = struct.unpack_from("<I", data, 0x3C)[0]
    if data[pe : pe + 4] != b"PE\0\0":
        raise ValueError(f"{library} is not a PE image")

    num_sections = struct.unpack_from("<H", data, pe + 6)[0]
    opt_size = struct.unpack_from("<H", data, pe + 20)[0]
    opt = pe + 24
    magic = struct.unpack_from("<H", data, opt)[0]
    # The export directory is the first data directory entry; PE32+ places it 16 bytes later.
    export_rva = struct.unpack_from("<I", data, opt + (112 if magic == 0x20B else 96))[0]
    if not export_rva:
        return set()

    sections = []
    sec = opt + opt_size
    for i in range(num_sections):
        off = sec + i * 40
        virt_addr, raw_size, raw_ptr = struct.unpack_from("<III", data, off + 12)
        sections.append((virt_addr, raw_size, raw_ptr))

    def to_offset(rva: int) -> int:
        for virt_addr, raw_size, raw_ptr in sections:
            if virt_addr <= rva < virt_addr + raw_size:
                return raw_ptr + (rva - virt_addr)
        raise ValueError(f"RVA {rva:#x} outside every section of {library}")

    export = to_offset(export_rva)
    num_names = struct.unpack_from("<I", data, export + 24)[0]
    names_rva = struct.unpack_from("<I", data, export + 32)[0]
    names = to_offset(names_rva)

    symbols = set()
    for i in range(num_names):
        name_rva = struct.unpack_from("<I", data, names + i * 4)[0]
        start = to_offset(name_rva)
        symbols.add(data[start : data.index(b"\0", start)].decode("ascii"))
    return symbols


def exported_symbols(library: Path) -> set[str]:
    if platform.system() == "Windows":
        return {s for s in pe_exported_symbols(library) if s.startswith("ccsp_")}

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
    # The repo's own build wins, so a developer checks what they just built rather than an
    # unrelated csp that happens to be installed. CI builds the examples against an installed
    # wheel with nothing built in-tree, so fall back to the package's lib directory.
    roots = [REPO_ROOT]
    try:
        import csp

        roots.append(Path(csp.get_lib_path()))
    except (ImportError, AttributeError):
        # csp not installed, or too old to expose get_lib_path; the repo scan is all we have.
        pass

    suffixes = (".so", ".pyd", ".dylib")
    for stem in ("_cspimpl", "cspimpl"):
        for root in roots:
            for suffix in suffixes:
                matches = sorted(root.rglob(f"{stem}{suffix}"))
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
