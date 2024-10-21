import sys
from typing import Dict, Generic, List, Optional, Type, TypeVar, Union, get_args, get_origin
from unittest import TestCase

import csp
from csp import ts
from csp.impl.types.common_definitions import OutputBasket, OutputBasketContainer
from csp.impl.types.pydantic_types import CspTypeVar, CspTypeVarType, DynamicBasketPydantic, adjust_annotations
from csp.impl.types.tstype import DynamicBasket

T = TypeVar("T")
K = TypeVar("K")


class MyGeneric(Generic[T]):
    pass


class TestAdjustAnnotations(TestCase):
    def assertAnnotationsEqual(self, annotation1, annotation2):
        origin1 = get_origin(annotation1)
        origin2 = get_origin(annotation2)
        self.assertEqual(origin1, origin2)
        if origin1 is None:
            if isinstance(annotation1, TypeVar) and isinstance(annotation2, TypeVar):
                self.assertEqual(annotation1.__name__, annotation2.__name__)
            elif issubclass(annotation1, OutputBasket) and issubclass(annotation2, OutputBasket):
                self.assertAnnotationsEqual(annotation1.typ, annotation2.typ)
            else:
                self.assertEqual(annotation1, annotation2)
            return
        args1 = get_args(annotation1)
        args2 = get_args(annotation2)
        if args1 is None and args2 is None:
            return
        self.assertEqual(len(args1), len(args2))
        for arg1, arg2 in zip(args1, args2):
            self.assertAnnotationsEqual(arg1, arg2)

    def test_tvar_top_level(self):
        self.assertAnnotationsEqual(adjust_annotations("T"), CspTypeVarType[T])
        self.assertAnnotationsEqual(adjust_annotations("~T"), CspTypeVar[T])
        self.assertAnnotationsEqual(adjust_annotations(T), CspTypeVarType[T])
        self.assertAnnotationsEqual(adjust_annotations(TypeVar("~T")), CspTypeVar[T])

    def test_tvar_container(self):
        self.assertAnnotationsEqual(adjust_annotations(List["T"]), List[CspTypeVar[T]])
        self.assertAnnotationsEqual(adjust_annotations(List[T]), List[CspTypeVar[T]])
        self.assertAnnotationsEqual(adjust_annotations(List[List["T"]]), List[List[CspTypeVar[T]]])
        if sys.version_info >= (3, 9):
            self.assertAnnotationsEqual(adjust_annotations(list["T"]), list[CspTypeVar[T]])
            self.assertAnnotationsEqual(adjust_annotations(list[T]), list[CspTypeVar[T]])

        self.assertAnnotationsEqual(adjust_annotations(Dict["K", "T"]), Dict[CspTypeVar[K], CspTypeVar[T]])
        self.assertAnnotationsEqual(adjust_annotations(Dict[K, T]), Dict[CspTypeVar[K], CspTypeVar[T]])

        self.assertAnnotationsEqual(adjust_annotations(MyGeneric["T"]), MyGeneric[CspTypeVar[T]])
        self.assertAnnotationsEqual(adjust_annotations(MyGeneric[T]), MyGeneric[CspTypeVar[T]])

    def test_tvar_ts_of_container(self):
        self.assertAnnotationsEqual(adjust_annotations(ts["T"]), ts[CspTypeVarType[T]])
        self.assertAnnotationsEqual(adjust_annotations(ts["~T"]), ts[CspTypeVarType[TypeVar("~T")]])
        self.assertAnnotationsEqual(adjust_annotations(ts[List["T"]]), ts[List[CspTypeVarType[T]]])
        self.assertAnnotationsEqual(adjust_annotations(ts[List[T]]), ts[List[CspTypeVarType[T]]])
        self.assertAnnotationsEqual(adjust_annotations(ts[List[List["T"]]]), ts[List[List[CspTypeVarType[T]]]])
        if sys.version_info >= (3, 9):
            self.assertAnnotationsEqual(adjust_annotations(ts[list["T"]]), ts[list[CspTypeVarType[T]]])
            self.assertAnnotationsEqual(adjust_annotations(ts[list[T]]), ts[list[CspTypeVarType[T]]])

        self.assertAnnotationsEqual(
            adjust_annotations(ts[Dict["K", "T"]]), ts[Dict[CspTypeVarType[K], CspTypeVarType[T]]]
        )
        self.assertAnnotationsEqual(adjust_annotations(ts[Dict[K, T]]), ts[Dict[CspTypeVarType[K], CspTypeVarType[T]]])

        self.assertAnnotationsEqual(
            adjust_annotations(ts[Union["K", "T"]]), ts[Union[CspTypeVarType[K], CspTypeVarType[T]]]
        )
        self.assertAnnotationsEqual(
            adjust_annotations(ts[Union[K, T]]), ts[Union[CspTypeVarType[K], CspTypeVarType[T]]]
        )

    def test_tvar_container_of_ts(self):
        self.assertAnnotationsEqual(adjust_annotations(List[ts["T"]]), List[ts[CspTypeVarType[T]]])
        self.assertAnnotationsEqual(adjust_annotations(List[ts[T]]), List[ts[CspTypeVarType[T]]])
        self.assertAnnotationsEqual(adjust_annotations(List[ts[List["T"]]]), List[ts[List[CspTypeVarType[T]]]])

        self.assertAnnotationsEqual(adjust_annotations(Dict["K", ts["T"]]), Dict[CspTypeVar[K], ts[CspTypeVarType[T]]])
        self.assertAnnotationsEqual(adjust_annotations(Dict["K", ts[T]]), Dict[CspTypeVar[K], ts[CspTypeVarType[T]]])
        self.assertAnnotationsEqual(
            adjust_annotations(Union[ts["K"], ts["T"]]), Union[ts[CspTypeVarType[K]], ts[CspTypeVarType[T]]]
        )
        self.assertAnnotationsEqual(
            adjust_annotations(Union[ts[K], ts[T]]), Union[ts[CspTypeVarType[K]], ts[CspTypeVarType[T]]]
        )

        self.assertAnnotationsEqual(
            adjust_annotations(MyGeneric[ts[MyGeneric[T]]]), MyGeneric[ts[MyGeneric[CspTypeVarType[T]]]]
        )

    def test_dynamic_basket(self):
        container = DynamicBasket[str, float]
        self.assertAnnotationsEqual(adjust_annotations(container), DynamicBasketPydantic[str, float])

        self.assertAnnotationsEqual(
            adjust_annotations(Dict[ts["K"], ts["T"]]), DynamicBasketPydantic[CspTypeVarType[K], CspTypeVarType[T]]
        )
        self.assertAnnotationsEqual(
            adjust_annotations(Dict[ts[K], ts[T]]), DynamicBasketPydantic[CspTypeVarType[K], CspTypeVarType[T]]
        )

        # TODO: Remove this part once support for declaring dynamic baskets as a dict type is removed
        container = Dict[ts[str], ts[float]]
        self.assertAnnotationsEqual(adjust_annotations(container), DynamicBasketPydantic[str, float])

    def test_output_basket(self):
        container = OutputBasketContainer(List[ts["T"]], shape=5, eval_type=OutputBasketContainer.EvalType.WITH_SHAPE)
        self.assertAnnotationsEqual(adjust_annotations(container), OutputBasket(typ=List[ts[CspTypeVarType[T]]]))

    def test_other(self):
        self.assertAnnotationsEqual(adjust_annotations(List[str]), List[str])
        self.assertAnnotationsEqual(adjust_annotations(Dict[str, float]), Dict[str, float])
        self.assertAnnotationsEqual(adjust_annotations(MyGeneric[str]), MyGeneric[str])
        self.assertAnnotationsEqual(adjust_annotations(MyGeneric[str]), MyGeneric[str])

    def test_union_pipe(self):
        if sys.version_info >= (3, 10):
            self.assertAnnotationsEqual(adjust_annotations(str | float), Union[str, float])

    def test_make_optional(self):
        self.assertAnnotationsEqual(adjust_annotations(float, make_optional=True), Optional[float])
        self.assertAnnotationsEqual(adjust_annotations(List[float], make_optional=True), Optional[List[float]])

    def test_force_tvars(self):
        self.assertAnnotationsEqual(adjust_annotations(CspTypeVar[T], forced_tvars={"T": str}), str)
        self.assertAnnotationsEqual(adjust_annotations(CspTypeVarType[T], forced_tvars={"T": str}), Type[str])
        # Float gets converted to Union of float and int due to the way TVar resolution works
        self.assertAnnotationsEqual(
            adjust_annotations(CspTypeVarType[T], forced_tvars={"T": float}), Union[Type[float], Type[int]]
        )
