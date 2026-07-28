import collections.abc
import types
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
    def canonicalize_builtin_generics(cls, typ):
        """Recursively canonicalize PEP 585 builtin generics to their ``typing`` equivalents so that,
        e.g., ``list[int]`` and ``typing.List[int]`` (and every nesting combination) normalize to a single
        representation that compares and hashes equal.

        Only the builtin container origins in ``_ORIGIN_COMPAT_MAP`` (list, set, dict, tuple) are remapped
        to their ``typing`` form. Every other generic wrapper (FastList, csp.typing numpy arrays,
        typing.Callable, typing.Mapping, custom generics, ...) keeps its own origin/flavor, but we still
        recurse into its arguments so nested builtin containers get canonicalized. Unions (both
        ``typing.Union``/``Optional`` and PEP 604 ``X | Y``) are traversed as well. When nothing actually
        changes, the original object is returned unchanged, both to avoid needless allocations and to
        preserve object identity for callers that rely on it.
        """
        if CspTypingUtils.is_union_type(typ):
            args = typing.get_args(typ)
            converted_args = tuple(cls._canonicalize_arg(arg) for arg in args)
            if converted_args == args:
                return typ
            return typing.Union[converted_args]

        if CspTypingUtils.is_generic_container(typ):
            # __args__ (rather than get_args) keeps Callable's flattened ([params], ret) shape, which is
            # what copy_with expects for faithful reconstruction.
            args = typ.__args__
            converted_args = tuple(cls._canonicalize_arg(arg) for arg in args)
            canonical_origin = CspTypingUtils._ORIGIN_COMPAT_MAP.get(typ.__origin__)
            if canonical_origin is not None:
                # list/set/dict/tuple: rewrite to the typing form. A builtin (types.GenericAlias) alias
                # must always be rebuilt (that is the whole point); an already-typing alias whose args did
                # not change is returned as-is to preserve identity.
                if converted_args == args and not isinstance(typ, types.GenericAlias):
                    return typ
                return canonical_origin[converted_args if len(converted_args) != 1 else converted_args[0]]
            # Preserved wrapper (FastList, numpy arrays, Callable, Mapping, custom generic, ...): keep the
            # outer origin/flavor but rebuild with canonicalized args when a child actually changed.
            if converted_args == args:
                return typ
            if hasattr(typ, "copy_with"):
                # typing._GenericAlias (typing.Callable, typing.Mapping, csp numpy arrays, ...): copy_with
                # faithfully preserves the alias flavor, including Callable's flattened arg shape.
                return typ.copy_with(converted_args)
            origin = typ.__origin__
            if origin is collections.abc.Callable:
                # PEP 585 collections.abc.Callable[[p1, ...], ret]: __args__ is flattened, so restore the
                # ([params], ret) subscription shape (an Ellipsis param list stays as Callable[..., ret]).
                *params, ret = converted_args
                if params == [Ellipsis]:
                    return origin[..., ret]
                return origin[list(params), ret]
            try:
                return origin[converted_args if len(converted_args) != 1 else converted_args[0]]
            except TypeError:
                # Exotic / non-subscriptable origin: leave it unchanged rather than fail normalization.
                return typ

        return typ

    @classmethod
    def _canonicalize_arg(cls, arg):
        # A bare string argument inside a generic (e.g. ``list["T"]``) is stored raw by PEP 585 builtins but
        # as a ForwardRef by typing generics; canonicalize to ForwardRef so both spellings match (and so we
        # do not mint a fresh TypeVar on every call, which would defeat equality/caching).
        if isinstance(arg, str):
            return typing.ForwardRef(arg)
        return cls.canonicalize_builtin_generics(arg)

    @classmethod
    def _convert_containers_to_typing_generic_meta(cls, typ, is_within_container):
        typ = cls.canonicalize_builtin_generics(typ)
        if CspTypingUtils.is_generic_container(typ):
            return typ
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
