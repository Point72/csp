import types
import typing
from inspect import isclass
from typing import Any, ForwardRef, Generic, Literal, Optional, Type, TypeVar, Union, get_args, get_origin

from pydantic import GetCoreSchemaHandler, ValidationInfo, ValidatorFunctionWrapHandler
from pydantic_core import CoreSchema, core_schema

from csp.impl.types.common_definitions import OutputBasket, OutputBasketContainer
from csp.impl.types.tstype import SnapKeyType, SnapType, isTsDynamicBasket
from csp.impl.types.typing_utils import TsTypeValidator

_K = TypeVar("T", covariant=True)
_T = TypeVar("T", covariant=True)


def _check_source_type(cls, source_type):
    """Helper function for CspTypeVarType and CspTypeVar"""
    args = get_args(source_type)
    if len(args) != 1:
        raise ValueError(f"Must pass a single generic argument to {cls.__name__}. Got {args}.")
    v = args[0]
    if type(v) is TypeVar:
        return v.__name__
    elif type(v) is ForwardRef:  # In case someone writes, i.e. CspTypeVar["T"]
        return v.__forward_arg__
    else:
        raise ValueError(f"Must pass either a TypeVar or a ForwardRef (string) to {cls.__name__}. Got {type(v)}.")


class CspTypeVarType(Generic[_T]):
    """A special type representing a template variable for csp.
    It behaves similarly to a ForwardRef, but where the type of the forward arg is *implied* by a passed type, i.e. "float".
    """

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: GetCoreSchemaHandler) -> CoreSchema:
        typ = _check_source_type(cls, source_type)
        if not typ or not typ[0].isalpha():
            if typ and typ[0] == "~":
                raise SyntaxError(
                    f"Invalid generic type: {typ}. The generic type annotation (i.e. `~T`) is only allowed at the top level (i.e. `value: '~T'`)."
                )
            raise SyntaxError(
                f"Invalid generic type: {typ}. Generic types in csp must start with an alphabetic character."
            )

        def _validator(v: Any, info: ValidationInfo) -> Any:
            # info.context should be an instance of TVarValidationContext, but we don't check for performance
            if info.context is None:
                raise TypeError("Must pass an instance of TVarValidationContext to validate CspTypeVarType")
            info.context.add_tvar_type_ref(typ, v)
            return v

        return core_schema.with_info_plain_validator_function(_validator)


class CspTypeVar(Generic[_T]):
    """A special type representing a template variable for csp.
    It behaves similarly to a ForwardRef, but where the type of the forward arg is *implied* by the type of the input,
    i.e. passing "1.0" implies a type of "float".
    """

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: GetCoreSchemaHandler) -> CoreSchema:
        tvar = _check_source_type(cls, source_type)
        if not tvar or not tvar[0].isalpha():
            if tvar and tvar[0] == "~":
                raise SyntaxError(
                    f"Invalid generic type: {tvar}. The generic type annotation (i.e. `~T`) is only allowed at the top level (i.e. `value: '~T'`)."
                )
            raise SyntaxError(
                f"Invalid generic type: {tvar}. Generic types in csp must start with an alphabetic character. "
            )

        def _validator(v: Any, info: ValidationInfo) -> Any:
            # info.context should be an instance of TVarValidationContext, but we don't check for performance
            if info.context is None:
                raise TypeError("Must pass an instance of TVarValidationContext to validate CspTypeVar")
            info.context.add_tvar_ref(tvar, v)
            return v

        return core_schema.with_info_plain_validator_function(_validator)


class DynamicBasketPydantic(Generic[_K, _T]):
    # TODO: This can go away once DynamicBasket is it's own class and not just an alias for Dict[ts[_K], ts[_T]].
    # We can then just add the validator on DynamicBasket directly.

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: GetCoreSchemaHandler) -> CoreSchema:
        from csp.impl.wiring.edge import Edge

        args = get_args(source_type)
        ts_validator_key = TsTypeValidator.make_cached(args[0])
        ts_validator_value = TsTypeValidator.make_cached(args[1])

        def _validator(v: Any, info: ValidationInfo):
            """Functional validator for dynamic baskets"""
            if not isinstance(v, Edge) or not isTsDynamicBasket(v.tstype):
                raise ValueError("value must be a DynamicBasket")
            ts_validator_key.validate(v.tstype.__args__[0].typ, info)
            ts_validator_value.validate(v.tstype.__args__[1].typ, info)
            return v

        return core_schema.with_info_plain_validator_function(_validator)


def make_snap_validator(inp_def_type):
    """Create a validator function to handle SnapType."""

    def snap_validator(v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        if isinstance(v, SnapType):
            if v.ts_type.typ is inp_def_type:
                return v
            raise ValueError(f"Expecting {inp_def_type} for csp.snap value, but getting {v.ts_type.typ}")
        if isinstance(v, SnapKeyType):
            if v.key_tstype.typ is inp_def_type:
                return v
            raise ValueError(f"Expecting {inp_def_type} for csp.snap_key value, but getting {v.key_tstype.typ}")
        return handler(v)

    return snap_validator


def adjust_annotations(
    annotation, top_level: bool = True, in_ts: bool = False, make_optional: bool = False, forced_tvars=None
):
    """This function adjusts type annotations to replace TVars (ForwardRef, TypeVar and str)
    with CspTypeVar and CspTypeVarType as appropriate so that the custom csp templating logic can be carried out by
    pydantic validation.
    Because csp input type validation allows for None to be passed to any static arguments, we also adjust annotations
    to make the type Optional if the flag is set.
    """
    # TODO: Long term we should disable the make_optional flag and force people to use Optional as python intended
    from csp.impl.types.tstype import TsType  # Avoid circular import

    forced_tvars = forced_tvars or {}
    origin = get_origin(annotation)
    args = get_args(annotation)
    if isinstance(annotation, str):
        annotation = TypeVar(annotation)
    elif isinstance(annotation, OutputBasketContainer) or (
        isclass(annotation) and issubclass(annotation, OutputBasket)
    ):
        return OutputBasket(
            typ=adjust_annotations(
                annotation.typ, top_level=False, in_ts=False, make_optional=False, forced_tvars=forced_tvars
            )
        )
    elif isinstance(annotation, list):  # Handle list, i.e. in Callable[[Annotation], Any]
        return [
            adjust_annotations(a, top_level=False, in_ts=in_ts, make_optional=False, forced_tvars=forced_tvars)
            for a in annotation
        ]

    if type(annotation) is ForwardRef:
        if in_ts:
            return CspTypeVarType[TypeVar(annotation.__forward_arg__)]
        else:
            return CspTypeVar[TypeVar(annotation.__forward_arg__)]
    elif isinstance(annotation, TypeVar):
        if top_level:
            if annotation.__name__[0] == "~":
                return CspTypeVar[TypeVar(annotation.__name__[1:])]
            else:
                return CspTypeVarType[annotation]
        else:
            if in_ts:
                return CspTypeVarType[annotation]
            else:
                return CspTypeVar[annotation]
    elif isTsDynamicBasket(annotation):
        # Validation of dynamic baskets does not follow the pattern of validating Dict[ts[K], ts[V]]
        annotation_key = adjust_annotations(
            args[0], top_level=False, in_ts=True, make_optional=False, forced_tvars=forced_tvars
        ).typ
        annotation_value = adjust_annotations(
            args[1], top_level=False, in_ts=True, make_optional=False, forced_tvars=forced_tvars
        ).typ
        return DynamicBasketPydantic[annotation_key, annotation_value]
    elif origin and args:
        if origin is types.UnionType:  # For PEP604, i.e. x|y
            origin = typing.Union
        if origin is TsType:
            return TsType[
                adjust_annotations(args[0], top_level=False, in_ts=True, make_optional=False, forced_tvars=forced_tvars)
            ]
        if origin is Literal:  # for literals, we stop converting
            return Optional[annotation] if make_optional else annotation
        else:
            try:
                if origin is CspTypeVar or origin is CspTypeVarType:
                    new_args = args
                else:
                    new_args = tuple(
                        adjust_annotations(
                            arg, top_level=False, in_ts=in_ts, make_optional=False, forced_tvars=forced_tvars
                        )
                        for arg in args
                    )
                new_annotation = origin[new_args]
                # Handle force_tvars.
                if forced_tvars and (origin is CspTypeVar or origin is CspTypeVarType):
                    if new_args[0].__name__ in forced_tvars:
                        new_annotation = forced_tvars[new_args[0].__name__]
                        if origin is CspTypeVarType and not in_ts:
                            if new_annotation is float:
                                new_annotation = Union[Type[float], Type[int]]
                            else:
                                new_annotation = Type[new_annotation]
                if make_optional:
                    new_annotation = Optional[new_annotation]
                return new_annotation
            except TypeError:
                raise TypeError(f"Could not adjust annotations for {origin}")
    else:
        if make_optional:
            return Optional[annotation]
        else:
            return annotation
