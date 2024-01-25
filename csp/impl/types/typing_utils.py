# utils for dealing with typing types
import numpy
import typing

import csp.typing


class CspTypingUtils37:
    _ORIGIN_COMPAT_MAP = {list: typing.List, set: typing.Set, dict: typing.Dict, tuple: typing.Tuple}
    _ARRAY_ORIGINS = (csp.typing.Numpy1DArray, csp.typing.NumpyNDArray)

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
        return isinstance(typ, typing._GenericAlias) and typ.__origin__ is not typing.Union

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


# Current typing utilities were
# stabilized as of python 3.7
CspTypingUtils = CspTypingUtils37
