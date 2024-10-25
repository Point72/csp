import numpy
from typing import Generic, TypeVar, get_args

T = TypeVar("T")


class NumpyNDArray(Generic[T], numpy.ndarray):
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        """Validation of NumpyNDArray for pydantic v2"""
        from pydantic_core import core_schema

        source_args = get_args(source_type)
        if not source_args:
            raise TypeError(f"Must provide a single generic argument to {cls}")

        def _validate(v):
            if not isinstance(v, numpy.ndarray):
                raise ValueError("value must be an instance of numpy.ndarray")
            if not numpy.issubdtype(v.dtype, source_args[0]):
                raise ValueError(f"dtype of array must be a subdtype of {source_args[0]}")
            return v

        return core_schema.no_info_plain_validator_function(_validate)


class Numpy1DArray(NumpyNDArray[T]):
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        """Validation of Numpy1DArray for pydantic v2"""
        from pydantic_core import core_schema

        source_args = get_args(source_type)
        if not source_args:
            raise TypeError(f"Must provide a single generic argument to {cls}")

        def _validate(v):
            if not isinstance(v, numpy.ndarray):
                raise ValueError("value must be an instance of numpy.ndarray")
            if not numpy.issubdtype(v.dtype, source_args[0]):
                raise ValueError(f"dtype of array must be a subdtype of {source_args[0]}")
            if len(v.shape) != 1:
                raise ValueError("array must be one dimensional")
            return v

        return core_schema.no_info_plain_validator_function(_validate)
