from typing import Any, Generic, TypeVar, get_args

import numpy

T = TypeVar("T")


def _get_validator_np(source_type):
    # Given a source type, gets the numpy array validator
    def _validate(v):
        subtypes = get_args(source_type)
        dtype = subtypes[0] if subtypes and subtypes[0] != Any else None
        try:
            if dtype:
                return numpy.asarray(v, dtype=dtype)
            return numpy.asarray(v)

        except TypeError:
            raise ValueError(f"Unable to convert {v} to an array.")

    return _validate


class NumpyNDArray(Generic[T], numpy.ndarray):
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        """Validation of NumpyNDArray for pydantic v2"""
        from pydantic_core import core_schema

        source_args = get_args(source_type)
        if not source_args:
            raise TypeError(f"Must provide a single generic argument to {cls}")

        validate_func = _get_validator_np(source_type=source_type)

        def _validate(v):
            v = validate_func(v)
            if not isinstance(v, numpy.ndarray):
                raise ValueError("value must be an instance of numpy.ndarray")
            if not numpy.issubdtype(v.dtype, source_args[0]):
                raise ValueError(f"dtype of array must be a subdtype of {source_args[0]}")
            return v

        return core_schema.no_info_before_validator_function(
            _validate,
            core_schema.any_schema(),
            serialization=core_schema.wrap_serializer_function_ser_schema(
                lambda val, handler: handler(val if val is None else val.tolist()),
                info_arg=False,
                return_schema=core_schema.list_schema(),
            ),
        )


class Numpy1DArray(NumpyNDArray[T]):
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        """Validation of Numpy1DArray for pydantic v2"""
        from pydantic_core import core_schema

        source_args = get_args(source_type)
        if not source_args:
            raise TypeError(f"Must provide a single generic argument to {cls}")
        validate_func = _get_validator_np(source_type=source_type)

        def _validate(v):
            v = validate_func(v)
            if not isinstance(v, numpy.ndarray):
                raise ValueError("value must be an instance of numpy.ndarray")
            if not numpy.issubdtype(v.dtype, source_args[0]):
                raise ValueError(f"dtype of array must be a subdtype of {source_args[0]}")
            if len(v.shape) != 1:
                raise ValueError("array must be one dimensional")
            return v

        return core_schema.no_info_before_validator_function(
            _validate,
            core_schema.any_schema(),
            serialization=core_schema.wrap_serializer_function_ser_schema(
                lambda val, handler: handler(val if val is None else val.tolist()),
                info_arg=False,
                return_schema=core_schema.list_schema(),
            ),
        )
