import runpy
import subprocess
import sys
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch


class TestBuild(unittest.TestCase):
    def test_arm64_linux_uses_arm64_vcpkg_triplet(self):
        skbuild = ModuleType("skbuild")
        skbuild.setup = Mock()
        setup_path = Path(__file__).parents[2] / "setup.py"

        with (
            patch.dict(sys.modules, {"skbuild": skbuild}),
            patch.object(sys, "platform", "linux"),
            patch("platform.machine", return_value="aarch64"),
            patch("platform.system", return_value="Linux"),
            patch.object(
                subprocess,
                "check_output",
                return_value=b"9c5c2a0ab75aff5bcd08142525f6ff7f6f7ddeee\n",
            ),
        ):
            runpy.run_path(setup_path, run_name="csp_setup_test")

        cmake_args = skbuild.setup.call_args.kwargs["cmake_args"]
        self.assertIn("-DVCPKG_TARGET_TRIPLET=arm64-linux", cmake_args)

    def test_init_helpers(self):
        """was a build issue where InitHelpers ran on a single instance across all libraries linking _cspimpl"""
        # Take a sample of libraries
        from csp.lib import _cspbaselibimpl, _cspimpl, _parquetadapterimpl

        self.assertTrue(hasattr(_cspimpl, "PyNode"))
        self.assertFalse(hasattr(_cspbaselibimpl, "PyNode"))
        self.assertFalse(hasattr(_parquetadapterimpl, "PyNode"))

        self.assertFalse(hasattr(_cspimpl, "merge"))
        self.assertTrue(hasattr(_cspbaselibimpl, "merge"))
        self.assertFalse(hasattr(_parquetadapterimpl, "merge"))

        self.assertFalse(hasattr(_cspimpl, "_parquet_input_adapter"))
        self.assertFalse(hasattr(_cspbaselibimpl, "_parquet_input_adapter"))
        self.assertTrue(hasattr(_parquetadapterimpl, "_parquet_input_adapter"))


if __name__ == "__main__":
    unittest.main()
