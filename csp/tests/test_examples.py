import glob
import inspect
import os
import subprocess
import sys

import pytest

from csp import examples


EXAMPLE_DIR = os.path.dirname(inspect.getfile(examples))
EXAMPLE_LIST = sorted(glob.glob(os.sep.join([EXAMPLE_DIR, 'e_*'])))
PYTHON_EXAMPLES = [os.path.basename(e) for e in EXAMPLE_LIST
                   if e.endswith(".py")]
EXTENSION_EXAMPLES = [os.path.basename(e) for e in EXAMPLE_LIST
                      if not e.endswith(".py")]

XFAILS = {
    'e_03_numba_simple_example.py': pytest.mark.xfail(
        reason="numba nodes not yet supported", strict=True),
    'e_08_kafka.py': pytest.mark.xfail(
        reason="kafka adapter example is buggy, see gh-23", strict=True),
    'e_11_websocket_output.py': pytest.mark.xfail(
        reason="websocket adapter example triggers a TypeError, see gh-24",
        strict=True),
}

if sys.platform == 'darwin':
    XFAILS['e_12_caching_example.py'] = pytest.mark.xfail(
        reason="Loading parquet files on macOS is not yet supported",
        strict=True)

PYTHON_EXAMPLES = [p if p not in XFAILS else pytest.param(p, marks=XFAILS[p])
                   for p in PYTHON_EXAMPLES]


@pytest.mark.parametrize("example", PYTHON_EXAMPLES)
def test_python_examples(example):
    args = [sys.executable, os.sep.join([EXAMPLE_DIR, example])]
    completed_process = subprocess.run(args, capture_output=True)
    assert completed_process.returncode == 0
    # TODO capture stdout, compare with reference
    # need to make output of all examples reproducible
