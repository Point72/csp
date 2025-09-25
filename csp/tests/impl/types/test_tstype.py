import sys
from typing import Any, Dict, ForwardRef, Generic, List, Mapping, TypeVar, Union, get_args, get_origin
from unittest import TestCase

import numpy as np
import pytest
from pydantic import TypeAdapter

import csp
from csp import dynamic_demultiplex, ts
from csp.impl.types.common_definitions import OutputBasket, Outputs
from csp.impl.types.pydantic_type_resolver import TVarValidationContext
from csp.impl.types.pydantic_types import DynamicBasketPydantic
from csp.impl.types.tstype import TsType

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
