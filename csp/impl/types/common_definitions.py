import ast
from collections import namedtuple
from enum import Enum, IntEnum, auto
from typing import Dict, List, Optional, Union

from csp.impl.types.container_type_normalizer import ContainerTypeNormalizer
from csp.impl.types.tstype import isTsBasket
from csp.impl.types.typing_utils import CspTypingUtils


class OutputTypeError(TypeError):
    _message = "Output malformed"

    def __init__(self, msg="", *args, **kwargs):
        super().__init__(msg or self._message)


class OutputMixedNamedAndUnnamedError(OutputTypeError):
    _message = "Outputs must all be named or be a single unnamed output, cant be both"


class OutputBasketNotABasket(OutputTypeError):
    _message = "OutputBasket input must be a basket, got {}"

    def __init__(self, typ, *args, **kwargs):
        super().__init__(self._message.format(typ))


class OutputBasketMixedShapeAndShapeOf(OutputTypeError):
    _message = "OutputBasket must specify scalar `shape` or input name `shape_of`, not both"


class OutputBasketWrongShapeType(OutputTypeError):
    _message = "OutputBasket had wrong shape/shape_of type: expected {}, got {}"

    def __init__(self, expected, got, *args, **kwargs):
        super().__init__(self._message.format(expected, got))


class Outputs:
    def __new__(cls, *args, **kwargs):
        """we are abusing class construction here because we can't use classgetitem.
        the return value of the constructor should be a `type`"""
        if (len(args) and len(kwargs)) or len(args) > 1:
            raise OutputMixedNamedAndUnnamedError()

        if args:
            # TODO easier to support multiple unnamed args than before,
            # but we'll leave this for later
            kwargs[None] = args[0] if not isTsBasket(args[0]) else OutputBasket(args[0])

        if kwargs:
            kwargs = {k: v if not isTsBasket(v) else OutputBasket(v) for k, v in kwargs.items()}

        # stash for convenience later
        kwargs["__annotations__"] = kwargs.copy()
        try:
            _make_pydantic_outputs(kwargs)
        except ImportError:
            pass
        return type("Outputs", (Outputs,), kwargs)

    def __init__(self, *args, **kwargs):
        if args:
            raise Exception("Should not get here")
        ...


def _make_pydantic_outputs(kwargs):
    """Add pydantic functionality to Outputs, if necessary"""
    from pydantic import create_model
    from pydantic_core import core_schema

    from csp.impl.wiring.outputs import OutputsContainer

    if None in kwargs:
        typ = ContainerTypeNormalizer.normalize_type(kwargs[None])
        model_fields = {"out": (typ, ...)}
    else:
        model_fields = {
            name: (ContainerTypeNormalizer.normalize_type(annotation), ...)
            for name, annotation in kwargs["__annotations__"].items()
        }
    config = {"arbitrary_types_allowed": True, "extra": "forbid", "strict": True}
    kwargs["__pydantic_model__"] = create_model("OutputsModel", __config__=config, **model_fields)
    kwargs["__get_pydantic_core_schema__"] = classmethod(
        lambda cls, source_type, handler: core_schema.no_info_after_validator_function(
            lambda v: OutputsContainer(**v.model_dump()), handler(cls.__pydantic_model__)
        )
    )


class OutputBasket(object):
    def __new__(cls, typ, shape: Optional[Union[List, int, str]] = None, shape_of: Optional[str] = None):
        """we are abusing class construction here because we can't use classgetitem.
        the return value of the constructor should be a `type`"""
        # First, ensure typ is a list or dict basket
        typ = ContainerTypeNormalizer.normalize_type(typ)

        if not isTsBasket(typ):
            raise OutputBasketNotABasket(typ)

        kwargs = {"typ": typ}

        # Next, validate shape and shape_of
        if shape and shape_of:
            raise OutputBasketMixedShapeAndShapeOf()
        elif shape:
            if CspTypingUtils.get_origin(typ) is Dict and not isinstance(shape, (list, tuple, str)):
                raise OutputBasketWrongShapeType((list, tuple, str), shape)
            if CspTypingUtils.get_origin(typ) is List and not isinstance(shape, (int, str)):
                raise OutputBasketWrongShapeType((int, str), shape)
            kwargs["shape"] = shape
            kwargs["shape_func"] = "with_shape"
        elif shape_of:
            # validate shape_of
            # just ensure its a string, will figure out later
            if not isinstance(shape_of, str):
                raise OutputBasketWrongShapeType(str, shape_of)
            kwargs["shape"] = shape_of
            kwargs["shape_func"] = "with_shape_of"
        else:
            # pass trough the type
            # if shape is required, it will be enforced in the parser
            kwargs["shape"] = None
            kwargs["shape_func"] = None

        return type("OutputBasket", (OutputBasket,), kwargs)

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        from pydantic_core import core_schema

        def validate_shape(v, info):
            shape = cls.shape
            if isinstance(shape, int) and len(v) != shape:
                raise ValueError(f"Wrong shape: got {len(v)}, expecting {shape}")
            if isinstance(shape, (list, tuple)) and v.keys() != set(shape):
                raise ValueError(f"Wrong dict shape: got {v.keys()}, expecting {set(shape)}")
            return v

        return core_schema.with_info_after_validator_function(validate_shape, handler(cls.typ))


class OutputBasketContainer:
    SHAPE_FUNCS = None

    class EvalType(Enum):
        # __outputs__
        WITH_SHAPE = auto()
        WITH_SHAPE_OF = auto()

    def __init__(self, typ, shape, eval_type, shape_func=None, **ast_kwargs):
        self.typ = typ
        # For string shape we should be referring to an argument, transform to load of that var
        if isinstance(shape, str):
            shape = ast.Name(id=shape, ctx=ast.Load(), **ast_kwargs)
        self.shape = shape
        self._shape_func = shape_func
        self.eval_type = eval_type
        self.ast_kwargs = ast_kwargs

    @property
    def shape_func(self):
        return self._shape_func

    @shape_func.setter
    def shape_func(self, value):
        self._shape_func = value

    def _get_dict_type_and_shape(self, *args, **kwargs):
        shape = self.shape_func(*args, **kwargs)
        if self.eval_type == self.EvalType.WITH_SHAPE:
            if isinstance(shape, list):
                return self.typ.__args__[1].typ, shape
            elif isinstance(shape, tuple):
                return self.typ.__args__[1].typ, list(shape)
            else:
                raise TypeError("Type of the dictionary with_shape argument must be list or tuple")
        elif self.eval_type == self.EvalType.WITH_SHAPE_OF:
            if isinstance(shape, dict):
                return self.typ.__args__[1].typ, list(shape.keys())
            else:
                raise TypeError("Type of the  dictionary with_shape_of argument must be a dictionary basket")
        else:
            raise RuntimeError(f"Unexpected eval_type: {self.eval_type}")

    def _get_list_type_and_shape(self, *args, **kwargs):
        shape = self.shape_func(*args, **kwargs)
        if self.eval_type == self.EvalType.WITH_SHAPE:
            if isinstance(shape, int):
                return self.typ.__args__[0].typ, shape
            else:
                raise TypeError("Type of the list with_shape argument must int")
        elif self.eval_type == self.EvalType.WITH_SHAPE_OF:
            if isinstance(shape, list):
                return self.typ.__args__[0].typ, len(shape)
            else:
                raise TypeError("Type of the list with_shape_of argument must be a  basket")
        else:
            raise RuntimeError(f"Unexpected eval_type: {self.eval_type}")

    def get_type_and_shape(self, *args, **kwargs):
        if self.is_dict_basket():
            return self._get_dict_type_and_shape(*args, **kwargs)
        elif self.is_list_basket():
            return self._get_list_type_and_shape(*args, **kwargs)
        else:
            raise RuntimeError(f"Unexpected basket type {self.typ}")

    def is_dict_basket(self):
        return CspTypingUtils.get_origin(self.typ) is Dict

    def is_list_basket(self):
        return CspTypingUtils.get_origin(self.typ) is List

    def __str__(self):
        return f"OutputBasketContainer(typ={self.typ}, shape={self.shape}, eval_type={self.eval_type})"

    def __repr__(self):
        return str(self)

    @classmethod
    def create_wrapper(cls, eval_typ):
        return lambda typ, shape, **ast_kwargs: OutputBasketContainer(typ, shape, eval_typ, **ast_kwargs)


OutputBasketContainer.SHAPE_FUNCS = {
    "with_shape": OutputBasketContainer.create_wrapper(OutputBasketContainer.EvalType.WITH_SHAPE),
    "with_shape_of": OutputBasketContainer.create_wrapper(OutputBasketContainer.EvalType.WITH_SHAPE_OF),
}


InputDef = namedtuple("InputDef", ["name", "typ", "kind", "basket_kind", "ts_idx", "arg_idx"])
OutputDef = namedtuple("OutputDef", ["name", "typ", "kind", "ts_idx", "shape"])


class ArgKind(IntEnum):
    SCALAR = 0x1
    TS = 0x2
    BASKET_TS = TS | 0x4
    DYNAMIC_BASKET_TS = BASKET_TS | 0x8
    ALARM = TS | 0x10

    def is_any_ts(self):
        return self & ArgKind.TS

    def is_single_ts(self):
        return self == ArgKind.TS

    def is_scalar(self):
        return self == ArgKind.SCALAR

    def is_basket(self):
        """true for dynamic baskets as well"""
        return (self & ArgKind.BASKET_TS) == ArgKind.BASKET_TS

    def is_non_dynamic_basket(self):
        return self == ArgKind.BASKET_TS

    def is_dynamic_basket(self):
        return self == ArgKind.DYNAMIC_BASKET_TS

    def is_alarm(self):
        return self == ArgKind.ALARM


class BasketKind(Enum):
    LIST = auto()
    DICT = auto()
    DYNAMIC_DICT = auto()


class PushMode(IntEnum):
    """PushMode specifies how to process multiple ticks at the same time on input adapter
    for sim adapters:
        LAST_VALUE will collapse ticks with the same timestamp
        NON_COLLAPSING will tick all events with the same time in separate cycles
        BURST will provide all ticks wit hthe same timestamp as a list of values

    for realtime adapters:
        LAST_VALUE will tick the with latest value since last engine cycle ( conflating ticks if engine cant keep up )
        NON_COLLAPSING will provide all ticks queued up from the realtime input spread across subsequent engine cycles
        BURST will provide all ticks since last engine cycle as a single list of values
    """

    LAST_VALUE = 1
    NON_COLLAPSING = 2
    BURST = 3


class ReplayMode(IntEnum):
    """PushPull adapters can take a replay_mode option to specify how to replay data
    EARLIEST   will replay all available data (Note that data with timestamps before engine start will be forced to playback at starttime )
    LATEST     only run from latest data ( effectively, no replay )
    START_TIME playback all data from engine starttime
    """

    EARLIEST = 1
    LATEST = 2
    START_TIME = 3


class DuplicatePolicy(IntEnum):
    """An 'enum' that specifies the policy for handling the last value in functions like value_at."""

    # NOTE: it has a corresponding enum in c++ implementation and can't be changed independently
    LAST_VALUE = 1
    FIRST_VALUE = 2
