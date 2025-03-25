import inspect
from typing import Any, Generic, List, Optional, TypeVar, Union, get_args

import numpy
from typing_extensions import TypeAliasType

T = TypeVar("T")

# Since numpy arrays can be n-dimensional,
# we can have n nested lists (for arbitrary n).
# We use this recursive generic list type to represent
# the nested lists. However, we set our base-case to be the
# inner function. This is so we can apply it on scalars
# (which to numpy are just 0-dimensional arrays)
RecursiveGenericList = TypeAliasType(
    "RecursiveGenericList", Union[T, List["RecursiveGenericList[T]"]], type_params=(T,)
)


def _convert_to_numpy_array(v, true_type, numpy_dtype, handler=None):
    """
    Convert a value to a numpy array with appropriate type handling.
    """
    # Case 1: Target dtype is object
    if numpy_dtype == numpy.dtype(object):
        if true_type is object:
            # No conversion needed for Any/object types
            return numpy.asarray(v, dtype=object)
        else:
            if isinstance(v, numpy.ndarray):
                v = v.tolist()
            pydantic_res = handler(v)
            return numpy.array(pydantic_res, dtype=numpy_dtype)

    # Case 2: Target dtype is a specific type
    else:
        if isinstance(v, numpy.ndarray) and numpy.issubdtype(v.dtype, numpy_dtype):
            # Already correct type, no conversion needed
            return v
        else:
            # Convert to the target dtype
            return numpy.asarray(v, dtype=numpy_dtype)


def _get_numpy_array_schema(source_type, handler, additional_checks=None):
    """
    Common schema generator for numpy array types.

    Parameters:
    - source_type: The annotated type
    - additional_checks: Optional function for additional validation
    """
    from pydantic import PydanticSchemaGenerationError
    from pydantic_core import core_schema

    source_args = get_args(source_type)
    # swap out Any for object
    if source_args and source_args[0] == Any:
        source_args[0] = object
    if not source_args or not inspect.isclass(source_args[0]):
        raise TypeError(f"Must provide a single argument to {source_type} that is not generic")

    true_type = source_args[0]
    try:
        numpy_dtype = numpy.dtype(true_type)
    except Exception:  # give up easily
        numpy_dtype = object

    def _validate(v, handler):
        res = _convert_to_numpy_array(v, true_type, numpy_dtype, handler)
        return additional_checks(res) if additional_checks else res

    try:
        schema = handler.generate_schema(Optional[RecursiveGenericList[true_type]])
    except PydanticSchemaGenerationError:
        schema = core_schema.any_schema()  # can be anything

    return core_schema.no_info_wrap_validator_function(
        _validate,
        schema,
        serialization=core_schema.plain_serializer_function_ser_schema(
            lambda val: val if val is None else val.tolist(),
            info_arg=False,
            return_schema=schema,
        ),
    )


class NumpyNDArray(Generic[T], numpy.ndarray):
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        """Validation of NumpyNDArray for pydantic v2.
        This class relies on numpy conversions for basic types that numpy recognizes (including numpy dtypes).
        For other types (that numpy would classify as object), we convert rely on pydantic for the type validation, and then. This is not supported for use with generic types.
        """
        # No additional checks needed for n-dimensional arrays
        return _get_numpy_array_schema(source_type, handler)


class Numpy1DArray(NumpyNDArray[T]):
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        """Validation of Numpy1DArray for pydantic v2. Same as NumpyNDArray, but we check the shape has only 1 dimension."""

        # Add 1D dimension check
        def check_1d(v):
            if len(v.shape) != 1:
                raise ValueError("array must be one dimensional")
            return v

        return _get_numpy_array_schema(source_type, handler, check_1d)
