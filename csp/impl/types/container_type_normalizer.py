import typing

import numpy
import typing_extensions

import csp.typing
from csp.impl.types.typing_utils import CspTypingUtils, FastList


class ContainerTypeNormalizer:
    """A utility class that helps switcing between generic container type specifications and actual types
    Example use cases:
        - convert [int] to typing.List[int],
        - convert typing.List[int] to list
    """

    _NORMALIZED_TYPE_MAPPING = {
        typing.Dict: dict,
        typing.Set: set,
        typing.List: list,
        typing.Tuple: tuple,
        csp.typing.Numpy1DArray: numpy.ndarray,
        csp.typing.NumpyNDArray: numpy.ndarray,
    }

    @classmethod
    def _convert_containers_to_typing_generic_meta(cls, typ, is_within_container):
        if CspTypingUtils.is_generic_container(typ):
            return typ
            # cls._deep_convert_generic_meta_to_typing_generic_meta(typ, is_within_container)
        elif isinstance(typ, dict):
            # warn(
            #     "Using {K: V} syntax for type declaration is deprecated. Use Dict[K, V] instead.",
            #     DeprecationWarning,
            #     stacklevel=4,
            # )
            if type(typ) is not dict or len(typ) != 1:  # noqa: E721
                raise TypeError(f"Invalid type decorator: '{typ}'")
            t1, t2 = typ.items().__iter__().__next__()
            return typing.Dict[
                cls._convert_containers_to_typing_generic_meta(t1, True),
                cls._convert_containers_to_typing_generic_meta(t2, True),
            ]
        elif isinstance(typ, set):
            # warn(
            #     "Using {T} syntax for type declaration is deprecated. Use Set[T] instead.",
            #     DeprecationWarning,
            #     stacklevel=4,
            # )
            if type(typ) is not set or len(typ) != 1:  # noqa: E721
                raise TypeError(f"Invalid type decorator: '{typ}'")
            t = typ.__iter__().__next__()
            return typing.Set[cls._convert_containers_to_typing_generic_meta(t, True)]
        elif isinstance(typ, list):
            # warn(
            #     "Using [T] syntax for type declaration is deprecated. Use List[T] instead.",
            #     DeprecationWarning,
            #     stacklevel=4,
            # )
            if type(typ) is not list or len(typ) != 1:  # noqa: E721
                raise TypeError(f"Invalid type decorator: '{typ}'")
            t = typ.__iter__().__next__()
            return typing.List[cls._convert_containers_to_typing_generic_meta(t, True)]
        elif isinstance(typ, str) and is_within_container:
            return typing.TypeVar(typ)
        elif typ is numpy.ndarray:
            return csp.typing.NumpyNDArray[float]
        else:
            # Note we don't handle any other container here, i.e for example deque or numpy arrays will be handled as regular non
            # container objects
            return typ

    @classmethod
    def normalized_type_to_actual_python_type(cls, typ, level=0):
        if isinstance(typ, typing_extensions._AnnotatedAlias):
            typ = CspTypingUtils.get_origin(typ)

        if CspTypingUtils.is_generic_container(typ):
            origin = CspTypingUtils.get_origin(typ)
            if origin is FastList and level == 0:
                return [cls.normalized_type_to_actual_python_type(typ.__args__[0], level + 1), True]
            if origin is typing.List and level == 0:
                return [cls.normalized_type_to_actual_python_type(typ.__args__[0], level + 1)]
            return cls._NORMALIZED_TYPE_MAPPING.get(CspTypingUtils.get_origin(typ), typ)
        elif CspTypingUtils.is_union_type(typ):
            return object
        elif CspTypingUtils.is_literal_type(typ):
            # Import here to prevent circular import
            from csp.impl.types.instantiation_type_resolver import UpcastRegistry

            args = typing.get_args(typ)
            typ = type(args[0])
            for arg in args[1:]:
                typ = UpcastRegistry.instance().resolve_type(typ, type(arg), raise_on_error=False)
            if typ:
                return typ
            else:
                return object
        else:
            return typ

    @classmethod
    def normalize_type(cls, typ):
        return cls._convert_containers_to_typing_generic_meta(typ, False)
