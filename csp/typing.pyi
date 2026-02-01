"""Type stubs for csp typing utilities."""

from typing import Generic, TypeVar

import numpy as np

T = TypeVar("T", bound=np.generic)

class NumpyNDArray(Generic[T], np.ndarray):
    """
    A typed numpy ndarray for use in csp Structs.

    Example:
        class MyStruct(csp.Struct):
            values: csp.typing.NumpyNDArray[np.float64]
    """

    ...

class Numpy1DArray(NumpyNDArray[T]):
    """
    A 1-dimensional typed numpy array for use in csp Structs.

    Example:
        class MyStruct(csp.Struct):
            prices: csp.typing.Numpy1DArray[np.float64]
    """

    ...
