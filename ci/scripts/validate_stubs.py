#!/usr/bin/env python3
"""Validate that all .pyi stub files are correct."""

import ast
import subprocess
import sys
from pathlib import Path

# Modules to validate with stubtest (must not have source code type errors)
STUBTEST_MODULES = [
    "csp.adapters.status",
    "csp.adapters.utils",
    "csp.curve",
    "csp.impl.constants",
    "csp.impl.enum",
    "csp.impl.error_handling",
    "csp.impl.outputadapter",
    "csp.impl.struct",
    "csp.impl.types.numpy_type_resolver",
    "csp.impl.wiring.adapter",
    "csp.impl.wiring.adapter_wrapper",
    "csp.impl.wiring.base_wiring_node_parser",
    "csp.impl.wiring.cache_support",
    "csp.impl.wiring.caching",
    "csp.impl.wiring.config",
    "csp.impl.wiring.context",
    "csp.impl.wiring.feedback",
    "csp.impl.wiring.graph",
    "csp.impl.wiring.graph_wrapper",
    "csp.impl.wiring.input",
    "csp.impl.wiring.input_adapter",
    "csp.impl.wiring.node",
    "csp.impl.wiring.output",
    "csp.impl.wiring.output_adapter",
    "csp.impl.wiring.outputs",
    "csp.impl.wiring.pull_adapter",
    "csp.impl.wiring.push_adapter",
    "csp.impl.wiring.runtime",
    "csp.impl.wiring.scheduler",
    "csp.impl.wiring.special_output_names",
    "csp.impl.wiring.threaded_runtime",
    "csp.impl.wiring.utils",
    "csp.showgraph",
    "csp.typing",
]


def check_syntax(stubs: list[Path]) -> list[str]:
    """Check all stub files for syntax errors."""
    errors = []
    for stub_path in stubs:
        try:
            ast.parse(stub_path.read_text())
        except SyntaxError as e:
            errors.append(f"{stub_path}: line {e.lineno}: {e.msg}")
    return errors


def check_types(stubs: list[Path]) -> tuple[int, str]:
    """Run mypy on stub files to check for type errors."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy",
            "--ignore-missing-imports",
            "--no-error-summary",
            "--no-warn-unused-ignores",
            *[str(s) for s in stubs],
        ],
        capture_output=True,
        text=True,
    )
    return result.returncode, result.stdout + result.stderr


def check_stubtest(modules: list[str], allowlist: Path) -> tuple[int, str]:
    """Run stubtest to verify stubs match runtime."""
    cmd = [
        sys.executable,
        "-m",
        "mypy.stubtest",
        *modules,
        "--ignore-missing-stub",
    ]
    if allowlist.exists():
        cmd.extend(["--allowlist", str(allowlist)])
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode, result.stdout + result.stderr


def main():
    """Check all stub files for syntax, type, and runtime match errors."""
    stubs = list(Path("csp").rglob("*.pyi"))
    print(f"Checking {len(stubs)} stub files...\n")

    # Step 1: Check syntax
    syntax_errors = check_syntax(stubs)
    if syntax_errors:
        print("❌ Syntax errors found:")
        for error in syntax_errors:
            print(f"  {error}")
        sys.exit(1)
    print("✓ All stub files are syntactically valid")

    # Step 2: Check types with mypy
    returncode, output = check_types(stubs)
    if returncode != 0:
        print("\n❌ Type errors in stubs:")
        print(output)
        sys.exit(1)
    print("✓ All stub files pass mypy type checking")

    # Step 3: Run stubtest on select modules
    allowlist = Path(".stubtest-allowlist")
    print(f"\nRunning stubtest on {len(STUBTEST_MODULES)} modules...")
    returncode, output = check_stubtest(STUBTEST_MODULES, allowlist)
    if returncode != 0:
        print("\n❌ Stubtest errors (stubs don't match runtime):")
        # Filter out the deprecation warning
        lines = [l for l in output.split("\n") if "DeprecationWarning" not in l and "import mypy.build" not in l]
        print("\n".join(lines))
        sys.exit(1)
    print("✓ Stubs match runtime for tested modules")

    print("\n✅ All stub validations passed!")
    sys.exit(0)


if __name__ == "__main__":
    main()
