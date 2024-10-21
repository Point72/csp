# utils for dealing with typing types
import numpy
import sys
import types
import typing

import csp.typing

T = typing.TypeVar("T")


class FastList(typing.Generic[T]):
    def __init__(self):
        raise NotImplementedError("Can not init FastList class")


class CspTypingUtils37:
    _ORIGIN_COMPAT_MAP = {list: typing.List, set: typing.Set, dict: typing.Dict, tuple: typing.Tuple}
    _ARRAY_ORIGINS = (csp.typing.Numpy1DArray, csp.typing.NumpyNDArray)
    _GENERIC_ALIASES = (typing._GenericAlias,)

    @classmethod
    def is_type_spec(cls, val):
        return isinstance(val, type) or cls.is_generic_container(val)

    # to replace calls to .__origin__, assumes typ is something you can call __origin__ on
    @classmethod
    def get_origin(cls, typ):
        raw_origin = typ.__origin__
        return cls._ORIGIN_COMPAT_MAP.get(raw_origin, raw_origin)

    @classmethod
    def is_numpy_array_type(cls, typ):
        return CspTypingUtils.is_generic_container(typ) and CspTypingUtils.get_orig_base(typ) is numpy.ndarray

    @classmethod
    def is_numpy_nd_array_type(cls, typ):
        return cls.is_numpy_array_type(typ) and cls.get_origin(typ) is csp.typing.NumpyNDArray

    # is typ a standard generic container
    @classmethod
    def is_generic_container(cls, typ):
        return isinstance(typ, cls._GENERIC_ALIASES) and typ.__origin__ is not typing.Union

    @classmethod
    def is_union_type(cls, typ):
        return isinstance(typ, typing._GenericAlias) and typ.__origin__ is typing.Union

    @classmethod
    def is_forward_ref(cls, typ):
        return isinstance(typ, typing.ForwardRef)

    @classmethod
    def get_orig_base(cls, typ):
        res = typ.__origin__
        if res in cls._ARRAY_ORIGINS:
            return numpy.ndarray
        return res

    @classmethod
    def pretty_typename(cls, typ):
        if cls.is_generic_container(typ):
            return str(typ)
        elif cls.is_forward_ref(typ):
            return cls.pretty_typename(typ.__forward_arg__)
        elif isinstance(typ, type):
            return typ.__name__
        else:
            return str(typ)


CspTypingUtils = CspTypingUtils37

if sys.version_info >= (3, 9):

    class CspTypingUtils39(CspTypingUtils37):
        # To support PEP 585
        _GENERIC_ALIASES = (typing._GenericAlias, typing.GenericAlias)

    CspTypingUtils = CspTypingUtils39

if sys.version_info >= (3, 10):

    class CspTypingUtils310(CspTypingUtils39):
        # To support PEP 604
        @classmethod
        def is_union_type(cls, typ):
            return (isinstance(typ, typing._GenericAlias) and typ.__origin__ is typing.Union) or isinstance(
                typ, types.UnionType
            )

    CspTypingUtils = CspTypingUtils310


class TsTypeValidator:
    """Class to help validate the arg of TsType.
    For example, this is to make sure that:
        ts[List] can validate as ts[List[float]]
        ts[Dict[str, List[str]] won't validate as ts[Dict[str, List[float]]
        ts["T"], ts[TypeVar("T")], ts[List["T"]], etc are allowed
        ts[Optional[float]], ts[Union[float, int]], ts[Annotated[float, None]], etc are not allowed
        etc
    For validation of csp baskets, this piece becomes the bottleneck
    """

    _cache: typing.Dict[type, "TsTypeValidator"] = {}

    @classmethod
    def make_cached(cls, source_type: type):
        """Make and cache the instance by source_type"""
        if source_type not in cls._cache:
            cls._cache[source_type] = cls(source_type)
        return cls._cache[source_type]

    def __init__(self, source_type: type):
        from pydantic import TypeAdapter

        from csp.impl.types.pydantic_types import CspTypeVarType
        from csp.impl.types.tstype import TsType

        self._source_type = source_type
        # Use CspTypingUtils for 3.8 compatibility, to map list -> typing.List, so one can call List[float]
        self._source_origin = typing.get_origin(source_type)
        self._source_is_union = CspTypingUtils.is_union_type(source_type)
        self._source_args = typing.get_args(source_type)
        self._source_adapter = None
        if type(source_type) in (typing.ForwardRef, typing.TypeVar):
            pass  # Will handle these separately as part of type checking
        elif self._source_origin is None and isinstance(self._source_type, type):
            # self._source_adapter = TypeAdapter(typing.Type[source_type])
            pass
        elif self._source_origin is CspTypeVarType:  # Handles TVar resolution
            self._source_adapter = TypeAdapter(
                self._source_type, config={"arbitrary_types_allowed": True, "strict": True}
            )
        elif type(self._source_origin) is type:  # Catch other types like list, dict, set, etc
            self._source_args_validators = [TsTypeValidator.make_cached(arg) for arg in self._source_args]
        elif self._source_is_union:
            self._source_args_validators = [TsTypeValidator.make_cached(arg) for arg in self._source_args]
        elif self._source_origin is TsType:
            # Common mistake, so have good error message
            raise TypeError(f"Found nested ts type - this is not allowed (inner type: {source_type})")
        else:
            raise TypeError(
                f"Argument to ts must either be: a type, ForwardRef or TypeVar. Got {source_type} which is an instance of {type(source_type)}."
            )
        self._last_value_type = None
        self._last_context = None

    def validate(self, value_type, info=None):
        """Run the validation against a proposed input type"""

        # Note: while tempting to cache this function, functools.cache/lru_cache actually slows things down.
        # To improve performance, we implement some quick and rudimentary last value caching logic
        # In baskets, the same type is likely to be validated over and over again, so we check whether value_type
        # is equal to the last value_type, and if so, skip validation (as any errors would already have been thrown)
        # We also don't test equality on info, assuming that the same validation info object is used
        # for a given validation run.
        if value_type == self._last_value_type and info is not None and self._last_context is info.context:
            return value_type
        self._last_value_type = value_type
        self._last_context = info.context if info is not None else None

        # Fast path because while we could use the source adapter in the next block to validate,
        # it's about 10x faster to do a simple validation with issubclass, and this adds up on baskets
        if self._source_origin is None:
            # Want to allow int to be passed for float (i.e. in resolution of TVars)
            if self._source_type is float and value_type is int:
                return self._source_type
            try:
                if issubclass(value_type, self._source_type):
                    return value_type
            except TypeError:
                # So that List[float] validates as list
                value_origin = typing.get_origin(value_type)
                if issubclass(value_origin, self._source_type):
                    return value_type

            raise ValueError(
                f"{self._error_message(value_type)}: {value_type} is not a subclass of {self._source_type}."
            )
        elif self._source_adapter is not None:
            # Slower path, which would work for None origin, but is necessary to validate CspTypeVarType and
            # track the TVars
            return self._source_adapter.validate_python(value_type, context=info.context if info else None)
        elif self._source_origin is typing.Union:
            if CspTypingUtils.is_union_type(value_type):
                value_args = typing.get_args(value_type)
                if set(value_args) <= set(self._source_args):
                    return value_type
            else:  # Check whether the argument validates as one of the elements of the union
                for source_validator in self._source_args_validators:
                    try:
                        return source_validator.validate(value_type, info)
                    except Exception:
                        pass
        else:
            value_origin = typing.get_origin(value_type) or value_type
            if not issubclass(value_origin, self._source_origin):
                raise ValueError(
                    f"{self._error_message(value_type)}: {value_origin} is not a subclass of {self._source_origin}."
                )

            value_args = typing.get_args(value_type)
            if self._source_args and len(value_args) != len(self._source_args):
                raise ValueError(f"{self._error_message(value_type)}: inconsistent number of generic args.")

            new_args = tuple(
                source_validator.validate(value_arg, info)
                for value_arg, source_validator in zip(value_args, self._source_args_validators)
            )
            if sys.version_info >= (3, 9):
                return self._source_origin[new_args]
            else:
                # Because python 3.8 will return "list" for get_origin(List[float]), but you can't call list[(float,)]
                return CspTypingUtils._ORIGIN_COMPAT_MAP.get(self._source_origin, self._source_origin)[new_args]

        raise ValueError(f"{self._error_message(value_type)}.")

    def _error_message(self, value_type):
        return f"cannot validate ts[{CspTypingUtils.pretty_typename(value_type)}] as ts[{CspTypingUtils.pretty_typename(self._source_type)}]"
