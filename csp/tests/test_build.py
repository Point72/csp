import unittest


class TestBuild(unittest.TestCase):
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
