import sys
from datetime import datetime, timedelta
from typing import (
    Any,
    Callable,
    Dict,
    ForwardRef,
    Generic,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
    get_args,
    get_origin,
)
from unittest import TestCase

import numpy as np
import pytest
from pydantic import TypeAdapter

import csp
from csp import dynamic_demultiplex, ts
from csp.impl.types.common_definitions import OutputBasket, Outputs
from csp.impl.types.container_type_normalizer import ContainerTypeNormalizer
from csp.impl.types.pydantic_type_resolver import TVarValidationContext
from csp.impl.types.pydantic_types import DynamicBasketPydantic
from csp.impl.types.tstype import TsType
from csp.impl.types.typing_utils import FastList
from csp.typing import Numpy1DArray, NumpyNDArray

T = TypeVar("T")
U = TypeVar("U")


class MyGeneric(Generic[T]):
    pass


class MyGeneric2(Generic[T, U]):
    pass


class TestTsTypeValidation(TestCase):
    def test_validation(self):
        ta = TypeAdapter(TsType[float])
        ta.validate_python(csp.null_ts(float))
        ta.validate_python(csp.null_ts(int))  # int-to-float works
        self.assertRaises(Exception, ta.validate_python, csp.null_ts(str))
        self.assertRaises(Exception, ta.validate_python, "foo")

    def test_not_edge(self):
        self.assertRaises(TypeError, TypeAdapter, TsType[0])

    def test_nested_ts_type(self):
        self.assertRaises(TypeError, TypeAdapter, TsType[TsType[float]])

    def test_list(self):
        ta = TypeAdapter(TsType[List[float]])
        ta.validate_python(csp.null_ts(List[float]))
        ta.validate_python(csp.null_ts(list[float]))
        ta.validate_python(csp.null_ts(list[np.float64]))
        ta.validate_python(csp.null_ts(list[int]))
        self.assertRaises(Exception, ta.validate_python, csp.null_ts(list[str]))

        ta = TypeAdapter(TsType[list])
        ta.validate_python(csp.null_ts(list))
        ta.validate_python(csp.null_ts(List[float]))
        ta.validate_python(csp.null_ts(List[str]))

    def test_nested(self):
        ta = TypeAdapter(TsType[Dict[str, List[float]]])
        ta.validate_python(csp.null_ts(Dict[str, List[float]]))
        ta.validate_python(csp.null_ts(dict[str, list[float]]))
        ta.validate_python(csp.null_ts(Dict[str, List[np.float64]]))
        ta.validate_python(csp.null_ts(Dict[str, List[int]]))
        self.assertRaises(Exception, ta.validate_python, csp.null_ts(Dict[int, List[float]]))

    def test_typevar(self):
        ta = TypeAdapter(TsType[T])
        self.assertRaises(Exception, ta.validate_python, csp.null_ts(float))

    def test_forward_ref(self):
        ta = TypeAdapter(TsType["T"])
        self.assertRaises(Exception, ta.validate_python, csp.null_ts(float))

    def test_custom_generic(self):
        ta = TypeAdapter(TsType[MyGeneric[float]])
        ta.validate_python(csp.null_ts(MyGeneric[float]))
        ta.validate_python(csp.null_ts(MyGeneric[np.float64]))
        self.assertRaises(Exception, ta.validate_python, csp.null_ts(MyGeneric[str]))

        ta = TypeAdapter(TsType[MyGeneric2[float, str]])
        ta.validate_python(csp.null_ts(MyGeneric2[float, str]))
        self.assertRaises(Exception, ta.validate_python, csp.null_ts(MyGeneric2[str, str]))

    def test_union_of_ts(self):
        ta = TypeAdapter(Union[TsType[float], TsType[str]])
        ta.validate_python(csp.null_ts(str))
        ta.validate_python(csp.null_ts(float))
        ta.validate_python(csp.null_ts(np.float64))
        self.assertRaises(Exception, ta.validate_python, csp.null_ts(List[str]))

    def test_test_of_union(self):
        ta = TypeAdapter(TsType[Union[float, int, str]])
        ta.validate_python(csp.null_ts(float))
        ta.validate_python(csp.null_ts(int))
        ta.validate_python(csp.null_ts(str))
        self.assertRaises(Exception, ta.validate_python, csp.null_ts(List[str]))

    def test_context(self):
        context = TVarValidationContext()
        ta = TypeAdapter(TsType[float])
        ta.validate_python(csp.null_ts(float), context=context)

    def test_allow_null(self):
        context = TVarValidationContext(allow_none_ts=True)
        ta = TypeAdapter(TsType[float])
        ta.validate_python(csp.null_ts(float), context=context)
        ta.validate_python(None, context=context)

    def test_any(self):
        ta = TypeAdapter(TsType[Any])
        ta.validate_python(csp.null_ts(float))
        ta.validate_python(csp.null_ts(object))
        ta.validate_python(csp.null_ts(List[str]))
        ta.validate_python(csp.null_ts(Dict[str, List[float]]))

        # https://docs.python.org/3/library/typing.html#the-any-type
        # "Notice that no type checking is performed when assigning a value of type Any to a more precise type."
        ta = TypeAdapter(TsType[float])
        ta.validate_python(csp.null_ts(Any))


class TestOutputValidation(TestCase):
    def test_validation(self):
        ta = TypeAdapter(Outputs(x=ts[float], y=ts[str]))
        ta.validate_python({"x": csp.null_ts(float), "y": csp.null_ts(str)})
        self.assertRaises(Exception, ta.validate_python, {"x": csp.null_ts(float)})
        self.assertRaises(Exception, ta.validate_python, {"x": csp.null_ts(float), "y": "foo"})
        self.assertRaises(
            Exception, ta.validate_python, {"x": csp.null_ts(float), "y": csp.null_ts(str), "z": csp.null_ts(float)}
        )


class TestOutputBasketValidation(TestCase):
    def test_validation(self):
        ta = TypeAdapter(OutputBasket(Dict[str, TsType[float]]))
        ta.validate_python({"x": csp.null_ts(float), "y": csp.null_ts(float)})

    def test_dict_shape_validation(self):
        self.assertRaises(Exception, OutputBasket, Dict[str, TsType[float]], shape=2)

        ta = TypeAdapter(OutputBasket(Dict[str, TsType[float]], shape=["x", "y"]))
        ta.validate_python({"x": csp.null_ts(float), "y": csp.null_ts(float)})
        self.assertRaises(Exception, ta.validate_python, {"x": csp.null_ts(float)})
        self.assertRaises(
            Exception, ta.validate_python, {"x": csp.null_ts(float), "y": csp.null_ts(float), "z": csp.null_ts(float)}
        )

        ta = TypeAdapter(OutputBasket(Dict[str, TsType[float]], shape=("x", "y")))
        ta.validate_python({"x": csp.null_ts(float), "y": csp.null_ts(float)})
        self.assertRaises(Exception, ta.validate_python, {"x": csp.null_ts(float)})
        self.assertRaises(
            Exception, ta.validate_python, {"x": csp.null_ts(float), "y": csp.null_ts(float), "z": csp.null_ts(float)}
        )

    def test_list_shape_validation(self):
        self.assertRaises(Exception, OutputBasket, List[TsType[float]], shape=["a", "b"])

        ta = TypeAdapter(OutputBasket(List[TsType[float]], shape=2))
        ta.validate_python([csp.null_ts(float)] * 2)
        self.assertRaises(Exception, ta.validate_python, [csp.null_ts(float)])
        self.assertRaises(Exception, ta.validate_python, [csp.null_ts(float)] * 3)
        self.assertRaises(Exception, ta.validate_python, {"x": csp.null_ts(float), "y": csp.null_ts(float)})


class TestDynamicBasketPydantic(TestCase):
    def test_validate(self):
        ta = TypeAdapter(DynamicBasketPydantic[str, float])
        dynamic_basket = dynamic_demultiplex(csp.const(1.0), csp.const("A"))
        ta.validate_python(dynamic_basket)
        self.assertRaises(Exception, ta.validate_python, {csp.const("A"): csp.const(1.0)})


class TestTsTypePep585Equivalence(TestCase):
    """PEP 585 builtin generics (list[int], dict[str, int], ...) must normalize to the same TsType as
    their typing equivalents (typing.List[int], typing.Dict[str, int], ...) so that tooling that rewrites
    typing.List -> list (e.g. ruff UP006) does not change TsType equality/hashing."""

    def test_equality_and_hash_parity(self):
        cases = [
            (list[int], List[int]),
            (dict[str, int], Dict[str, int]),
            (set[int], Set[int]),
            (tuple[int, str], Tuple[int, str]),
            (tuple[int, ...], Tuple[int, ...]),
        ]
        for builtin, typing_form in cases:
            with self.subTest(builtin=builtin):
                self.assertEqual(ts[builtin], ts[typing_form])
                self.assertEqual(hash(ts[builtin]), hash(ts[typing_form]))

    def test_nested_equality(self):
        self.assertEqual(ts[list[dict[str, int]]], ts[List[Dict[str, int]]])
        self.assertEqual(
            ts[dict[str, list[tuple[int, set[str]]]]],
            ts[Dict[str, List[Tuple[int, Set[str]]]]],
        )
        self.assertEqual(
            hash(ts[list[dict[str, int]]]),
            hash(ts[List[Dict[str, int]]]),
        )

    def test_canonicalizes_to_typing_form(self):
        # The canonical inner type is the typing form (the representation csp inference already emits).
        self.assertEqual(ts[list[int]].typ, List[int])
        self.assertEqual(ts[dict[str, int]].typ, Dict[str, int])
        self.assertEqual(ts[set[int]].typ, Set[int])
        self.assertEqual(ts[tuple[int, str]].typ, Tuple[int, str])

    def test_const_inference_matches_modernized_annotation(self):
        # csp.const infers typing.List[...]; a modernized (ruff UP006) annotation is ts[list[...]].
        # These previously compared unequal, which broke strict-equality checks (e.g. csp-gateway channels).
        self.assertEqual(csp.const([1, 2, 3]).tstype, ts[list[int]])
        self.assertEqual(csp.const({"a": 1}).tstype, ts[dict[str, int]])

    def test_preserves_non_builtin_generics(self):
        # FastList / csp numpy array types share machinery with typing generics but must NOT be collapsed.
        self.assertIs(ContainerTypeNormalizer.normalize_type(FastList[int]).__origin__, FastList)
        self.assertIs(ContainerTypeNormalizer.normalize_type(Numpy1DArray[float]).__origin__, Numpy1DArray)
        self.assertIs(ContainerTypeNormalizer.normalize_type(NumpyNDArray[float]).__origin__, NumpyNDArray)

    def test_node_binding_builtin_generic(self):
        # A builtin-generic-annotated node input accepts an edge whose inferred type uses the typing form.
        @csp.node
        def consume(x: ts[list[int]]) -> ts[int]:
            if csp.ticked(x):
                return len(x)

        @csp.graph
        def g():
            csp.add_graph_output("o", consume(csp.const([1, 2, 3])))

        results = csp.run(g, starttime=datetime(2020, 1, 1), endtime=timedelta(days=1))
        self.assertEqual(results["o"][0][1], 3)

    def test_builtin_generic_nested_in_union(self):
        # A builtin container nested inside a union (typing.Optional or PEP 604 X | None) is canonicalized.
        self.assertEqual(ts[list[int] | None], ts[Optional[List[int]]])
        self.assertEqual(ts[Optional[list[int]]], ts[Optional[List[int]]])
        self.assertEqual(ts[dict[str, list[int]] | None], ts[Optional[Dict[str, List[int]]]])
        self.assertEqual(hash(ts[list[int] | None]), hash(ts[Optional[List[int]]]))

    def test_builtin_generic_nested_in_preserved_wrapper(self):
        # The outer wrapper (Mapping / FastList / Callable) is preserved, but builtin containers nested
        # inside it are still canonicalized so equality holds after a typing.List -> list rewrite.
        self.assertEqual(ts[Mapping[str, list[int]]], ts[Mapping[str, List[int]]])
        self.assertEqual(ts[FastList[list[int]]], ts[FastList[List[int]]])
        self.assertEqual(ts[Callable[[list[int]], dict[str, int]]], ts[Callable[[List[int]], Dict[str, int]]])
        # ... and the wrapper origin is not collapsed into typing.List/dict/etc.
        self.assertIs(ContainerTypeNormalizer.normalize_type(FastList[list[int]]).__origin__, FastList)

    def test_string_arg_canonicalizes_to_forward_ref(self):
        # PEP 585 builtins store a bare string arg (list["T"]) while typing stores a ForwardRef; both must
        # normalize to the same stable ForwardRef (not a freshly-minted TypeVar on each call).
        self.assertEqual(ts[list["T"]], ts[list["T"]])
        self.assertEqual(ts[list["T"]], ts[List["T"]])
        self.assertEqual(ContainerTypeNormalizer.normalize_type(list["T"]).__args__[0], ForwardRef("T"))

    def test_identity_preserved_when_unchanged(self):
        # Nothing to canonicalize -> return the very same object, since some call sites (csp.snap
        # validation) compare normalized types with `is`.
        self.assertIs(ContainerTypeNormalizer.normalize_type(List[int]), List[int])
        self.assertIs(ContainerTypeNormalizer.normalize_type(List["Foo"]), List["Foo"])
        self.assertIs(ContainerTypeNormalizer.normalize_type(Optional[int]), Optional[int])
        self.assertIs(ContainerTypeNormalizer.normalize_type(Dict[str, List[int]]), Dict[str, List[int]])

    def test_empty_tuple(self):
        self.assertEqual(ts[tuple[()]], ts[Tuple[()]])

    def test_callable_inner_canonicalized(self):
        import collections.abc as abc

        normalize = ContainerTypeNormalizer.normalize_type
        # typing.Callable: builtin containers nested in params/return canonicalize identically for both
        # spellings, so the modernized and classic annotations stay equal.
        self.assertEqual(
            normalize(Callable[[list[int]], dict[str, int]]),
            normalize(Callable[[List[int]], Dict[str, int]]),
        )
        # collections.abc.Callable (the ruff UP006 rewrite of typing.Callable) must reconstruct without
        # error for finite, empty and ellipsis parameter lists, canonicalizing nested builtins while
        # keeping its own origin.
        self.assertEqual(normalize(abc.Callable[[list[int]], str]), abc.Callable[[List[int]], str])
        self.assertEqual(normalize(abc.Callable[[], list[int]]), abc.Callable[[], List[int]])
        self.assertEqual(normalize(abc.Callable[..., list[int]]), abc.Callable[..., List[int]])
