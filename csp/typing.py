import numpy
import sys
from typing import TypeVar

T = TypeVar("T")

if sys.version_info.major > 3 or sys.version_info.minor >= 7:
    import typing

    class Numpy1DArray(typing.Generic[T], numpy.ndarray):
        pass

    class NumpyNDArray(typing.Generic[T], numpy.ndarray):
        pass
else:
    from typing import MutableSequence, TypeVar, _generic_new

    class Numpy1DArray(numpy.ndarray, MutableSequence[T], extra=numpy.ndarray):
        __slots__ = ()

        def __new__(cls, *args, **kwds):
            if cls._gorg is Numpy1DArray:
                raise TypeError("Type NumpyArray cannot be instantiated; " "use ndarray() instead")
            return _generic_new(list, cls, *args, **kwds)

    class NumpyNDArray(numpy.ndarray, MutableSequence[T], extra=numpy.ndarray):
        __slots__ = ()

        def __new__(cls, *args, **kwds):
            if cls._gorg is NumpyNDArray:
                raise TypeError("Type NumpyMultidmensionalArray cannot be instantiated; " "use ndarray() instead")
            return _generic_new(list, cls, *args, **kwds)
