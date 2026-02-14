from unittest import TestCase

import numpy as np
from pydantic import TypeAdapter

from csp.typing import Numpy1DArray, NumpyNDArray


class TestNNumpy1DArray(TestCase):
    def test_Numpy1DArray(self):
        ta = TypeAdapter(Numpy1DArray[float])
        ta.validate_python(np.array([1.0, 2.0]))
        ta.validate_python(np.array([1.0, 2.0], dtype=np.float64))
        self.assertRaises(Exception, ta.validate_python, np.array([[1.0]]))
        self.assertRaises(Exception, ta.validate_python, np.array(["foo"]))
        ta.validate_python(np.array([1, 2]))  # gets coerced to correct type
        ta.validate_python(np.array([1.0, 2.0], dtype=np.float32))  # gets coerced to correct type

    def test_NumpyNDArray(self):
        ta = TypeAdapter(NumpyNDArray[float])
        ta.validate_python(np.array([1.0, 2.0]))
        ta.validate_python(np.array([1.0, 2.0], dtype=np.float64))
        ta.validate_python(np.array([[1.0, 2.0]]))
        ta.validate_python(np.array([[1.0, 2.0]], dtype=np.float64))
        self.assertRaises(Exception, ta.validate_python, np.array(["foo"]))
        ta.validate_python(np.array([1, 2]))  # gets coerced to correct type
        ta.validate_python(np.array([1.0, 2.0], dtype=np.float32))  # gets coerced to correct type
