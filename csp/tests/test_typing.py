import numpy as np
from pydantic import TypeAdapter
from unittest import TestCase

from csp.typing import Numpy1DArray, NumpyNDArray


class TestNNumpy1DArray(TestCase):
    def test_Numpy1DArray(self):
        ta = TypeAdapter(Numpy1DArray[float])
        ta.validate_python(np.array([1.0, 2.0]))
        ta.validate_python(np.array([1.0, 2.0], dtype=np.float64))
        self.assertRaises(Exception, ta.validate_python, np.array([[1.0]]))
        self.assertRaises(Exception, ta.validate_python, np.array(["foo"]))
        self.assertRaises(Exception, ta.validate_python, np.array([1, 2]))
        self.assertRaises(Exception, ta.validate_python, np.array([1.0, 2.0], dtype=np.float32))

    def test_NumpyNDArray(self):
        ta = TypeAdapter(NumpyNDArray[float])
        ta.validate_python(np.array([1.0, 2.0]))
        ta.validate_python(np.array([1.0, 2.0], dtype=np.float64))
        ta.validate_python(np.array([[1.0, 2.0]]))
        ta.validate_python(np.array([[1.0, 2.0]], dtype=np.float64))
        self.assertRaises(Exception, ta.validate_python, np.array(["foo"]))
        self.assertRaises(Exception, ta.validate_python, np.array([1, 2]))
        self.assertRaises(Exception, ta.validate_python, np.array([1.0, 2.0], dtype=np.float32))
