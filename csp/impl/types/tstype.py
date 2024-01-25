import typing
from typing import Protocol, TypeVar

from csp.impl.types.container_type_normalizer import ContainerTypeNormalizer
from csp.impl.types.typing_utils import CspTypingUtils

_TYPE_VAR = TypeVar("T", covariant=True)


class TsType(Protocol[_TYPE_VAR]):
    def __class_getitem__(cls, params):
        normalized_type = ContainerTypeNormalizer.normalize_type(params)
        res = super().__class_getitem__(normalized_type)
        object.__setattr__(res, "typ", res.__args__[0])
        return res

    # Note the original plan was to stub all the unary/binary ops here, but looks like simply adding a new method
    # to this class breaks PyCharm's type checking
    # def __bin__(self, rhs: Union[_TYPE_VAR, "TsType"]) -> Union[_TYPE_VAR, "TsType"]: ...
    # def __un__(self) -> Union[_TYPE_VAR, "TsType"]: ...
    # __add__ = __bin__
    # ...


ts = TsType


# This is just syntactic sugar, converts into typing.Dict[ ts[key_type], ts[value_type] ]
class DynamicBasketMeta(type):
    def __getitem__(self, args):
        if not isinstance(args, tuple) or len(args) != 2:
            raise ValueError("csp.DynamicBasket[] requires keys_type,value_type args")
        return typing.Dict[ts[args[0]], ts[args[1]]]


class DynamicBasket(metaclass=DynamicBasketMeta):
    pass


def isTsType(t):
    return CspTypingUtils.is_generic_container(t) and CspTypingUtils.get_origin(t) is TsType


def isTsBasket(t):
    return CspTypingUtils.is_generic_container(t) and (
        (CspTypingUtils.get_origin(t) is typing.Dict and isTsType(t.__args__[1]))
        or (CspTypingUtils.get_origin(t) is typing.List and isTsType(t.__args__[0]))
    )


def isTsDynamicBasket(t):
    return (
        CspTypingUtils.is_generic_container(t)
        and CspTypingUtils.get_origin(t) is typing.Dict
        and isTsType(t.__args__[1])
        and isTsType(t.__args__[0])
    )


def isTsStaticBasket(t):
    return isTsBasket(t) and not isTsDynamicBasket(t)


class _GenericTypesMeta(type):
    """Utility class that meta class for GenericTSTypes"""

    def __getitem__(cls, typ):
        if isinstance(typ, str):
            base = typing.Generic[typing.TypeVar(typ)]
        else:
            base = ContainerTypeNormalizer.normalize_type(typing.Generic[typ])

        t_var = base.__args__[0]

        ts_type = ts[t_var]
        list_basket_type = typing.List[ts_type]
        dict_basket_type = typing.Dict[str, ts_type]
        ts_or_basket_type = typing.Union[ts_type, list_basket_type, dict_basket_type]

        return type(
            f"GenericTSTypes[{typ}]",
            tuple(),
            {
                "T_VAR": t_var,
                "TS_TYPE": ts_type,
                "TS_LIST_BASKET_TYPE": list_basket_type,
                "TS_DICT_BASKET_TYPE": dict_basket_type,
                "TS_OR_BASKET_TYPE": ts_or_basket_type,
            },
        )


class GenericTSTypes(metaclass=_GenericTypesMeta):
    """Utility class that makes generic ts type decoration easier.
    The following types can be used as type decorators:
    GenericTSTypes['T'].T_VAR - a 'T' type var
    GenericTSTypes['T'].TS_TYPE - a time series with values of 'T' type
    GenericTSTypes['T'].TS_LIST_BASKET_TYPE - a list basket of time series with values of type 'T'
    GenericTSTypes['T'].TS_DICT_BASKET_TYPE - a dict basket of time series with values of type 'T'
    GenericTSTypes['T'].TS_OR_BASKET_TYPE - a value that can be either a ts o any ts basket with values of type 'T'
    """

    pass


# csp.snap
class SnapType:
    def __init__(self, edge):
        from csp.impl.wiring.edge import Edge

        if not isinstance(edge, Edge):
            raise TypeError("csp.snap() only supports snapping of single timeseries edge")

        self._edge = edge

    @property
    def edge(self):
        return self._edge

    @property
    def ts_type(self):
        return self._edge.tstype

    def __repr__(self):
        return f"csp.snap(ts[{self.ts_type.typ.__name__}])"


class SnapKeyType:
    def __init__(self):
        self._dyn_basket = None

    @property
    def dyn_basket(self):
        assert self._dyn_basket is not None
        return self._dyn_basket

    @property
    def key_tstype(self):
        return self.dyn_basket.tstype.__args__[0]

    def __repr__(self):
        return f"csp.snapkey(ts[{self.key_tstype.typ.__name__}])"


# csp.attach
class AttachType:
    def __init__(self):
        self._dyn_basket = None

    @property
    def dyn_basket(self):
        assert self._dyn_basket is not None
        return self._dyn_basket

    @property
    def key_tstype(self):
        return self.dyn_basket.tstype.__args__[0]

    @property
    def value_tstype(self):
        return self.dyn_basket.tstype.__args__[1]

    def __repr__(self):
        return f"csp.attach({{ ts[{self.key_tstype.typ.__name__}], ts[{self.value_tstype.typ.__name__}] }})"
