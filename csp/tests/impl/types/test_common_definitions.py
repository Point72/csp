import warnings
from unittest import TestCase

from csp import ts
from csp.impl.types.common_definitions import Outputs


class TestOutputs(TestCase):
    def test_unnamed_output_does_not_warn(self):
        # A single unnamed output is keyed by None; it belongs in __annotations__
        # but not in the class namespace, where a non-string key raises a
        # RuntimeWarning on Python 3.13+.
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            out = Outputs(ts[int])
        self.assertEqual(list(out.__annotations__), [None])
        self.assertNotIn(None, vars(out))

    def test_named_outputs_do_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            out = Outputs(a=ts[int], b=ts[str])
        self.assertEqual(set(out.__annotations__), {"a", "b"})
