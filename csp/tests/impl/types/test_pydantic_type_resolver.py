from typing import Dict, Generic, List, Set, TypeVar, get_args, get_origin
from unittest import TestCase

import numpy as np
from pydantic import BaseModel, TypeAdapter, ValidationInfo, field_validator, model_validator

import csp
import csp.typing
from csp import ts
from csp.impl.types.common_definitions import OutputBasket, OutputBasketContainer
from csp.impl.types.pydantic_type_resolver import TVarValidationContext
from csp.impl.types.pydantic_types import CspTypeVar, CspTypeVarType, adjust_annotations
from csp.impl.types.tstype import TsType

T = TypeVar("T")


class MyGeneric(Generic[T]):
    pass


class TestPydanticTypeResolver_CspTypeVar(TestCase):
    def test_one_value(self):
        context = TVarValidationContext()
        ta = TypeAdapter(CspTypeVar["T"])
        ta.validate_python(0.0, context=context)
        context.resolve_tvars()
        self.assertDictEqual(context.tvars, {"T": float})

    def test_nested_values(self):
        context = TVarValidationContext()
        ta = TypeAdapter(CspTypeVar["T"])
        ta.validate_python([0.0], context=context)
        context.resolve_tvars()
        self.assertDictEqual(context.tvars, {"T": List[float]})

        context = TVarValidationContext()
        ta.validate_python([[0.0]], context=context)
        context.resolve_tvars()
        self.assertDictEqual(context.tvars, {"T": List[List[float]]})

        context = TVarValidationContext()
        ta.validate_python(set([0.0]), context=context)
        context.resolve_tvars()
        self.assertDictEqual(context.tvars, {"T": Set[float]})

        context = TVarValidationContext()
        ta.validate_python({"a": 0.0}, context=context)
        context.resolve_tvars()
        self.assertDictEqual(context.tvars, {"T": Dict[str, float]})

        context = TVarValidationContext()
        ta.validate_python(np.array([1.0, 2.0]), context=context)
        context.resolve_tvars()
        self.assertDictEqual(context.tvars, {"T": csp.typing.Numpy1DArray[float]})

        context = TVarValidationContext()
        ta.validate_python(np.array([[1.0, 2.0]]), context=context)
        context.resolve_tvars()
        self.assertDictEqual(context.tvars, {"T": csp.typing.NumpyNDArray[float]})

        # TODO: Test exceptions, especially empty container!
        # TODO: What happens if all elements of the list don't match the first element! Should add validation

    def test_multiple_values(self):
        context = TVarValidationContext()
        ta = TypeAdapter(CspTypeVar["T"])
        ta.validate_python(0.0, context=context)
        ta.validate_python(1, context=context)
        context.resolve_tvars()
        self.assertDictEqual(context.tvars, {"T": float})

        ta.validate_python(2.0, context=context)
        context.resolve_tvars()  # Ok to add more and re-resolve

        ta.validate_python("foo", context=context)  # Will fail because of type
        self.assertRaises(Exception, context.resolve_tvars)

    def test_two_tvars(self):
        context = TVarValidationContext()
        ta = TypeAdapter(CspTypeVar["T"])
        ta.validate_python(5.0, context=context)
        ta = TypeAdapter(CspTypeVar["S"])
        ta.validate_python("foo", context=context)
        context.resolve_tvars()
        self.assertDictEqual(context.tvars, {"T": float, "S": str})

    def test_forced_tvar(self):
        context = TVarValidationContext(forced_tvars={"T": float})
        ta = TypeAdapter(CspTypeVar["T"])
        ta.validate_python(np.float64(0.0), context=context)
        context.resolve_tvars()
        self.assertDictEqual(context.tvars, {"T": float})

    def test_bad_variable_name(self):
        self.assertRaises(SyntaxError, lambda: CspTypeVar[""])
        self.assertRaises(SyntaxError, TypeAdapter, CspTypeVar["~T"])
        self.assertRaises(SyntaxError, TypeAdapter, CspTypeVar["1"])
        _ = TypeAdapter(CspTypeVar["T1"])


class TestPydanticTypeResolver_CspTypeVarType(TestCase):
    def test_one_value(self):
        context = TVarValidationContext()
        ta = TypeAdapter(CspTypeVarType["T"])
        ta.validate_python(float, context=context)
        context.resolve_tvars()
        self.assertDictEqual(context.tvars, {"T": float})

    def test_multiple_values(self):
        context = TVarValidationContext()
        ta = TypeAdapter(CspTypeVarType["T"])
        ta.validate_python(float, context=context)
        ta.validate_python(np.float64, context=context)
        context.resolve_tvars()
        self.assertDictEqual(context.tvars, {"T": float})

        ta.validate_python(float, context=context)
        context.resolve_tvars()  # Ok to add more and re-resolve

        ta.validate_python(str, context=context)  # Will fail because of type
        self.assertRaises(Exception, context.resolve_tvars)

    def test_two_tvars(self):
        context = TVarValidationContext()
        ta = TypeAdapter(CspTypeVarType["T"])
        ta.validate_python(float, context=context)
        ta = TypeAdapter(CspTypeVarType["S"])
        ta.validate_python(str, context=context)
        context.resolve_tvars()
        self.assertDictEqual(context.tvars, {"T": float, "S": str})

    def test_forced_tvar(self):
        context = TVarValidationContext(forced_tvars={"T": float})
        ta = TypeAdapter(CspTypeVarType["T"])
        ta.validate_python(np.float64, context=context)
        context.resolve_tvars()
        self.assertDictEqual(context.tvars, {"T": float})

    def test_CspTypeVarType(self):
        context = TVarValidationContext()
        ta = TypeAdapter(CspTypeVar["T"])
        ta.validate_python(5.0, context=context)
        ta = TypeAdapter(CspTypeVarType["T"])
        ta.validate_python(np.float64, context=context)
        context.resolve_tvars()
        self.assertDictEqual(context.tvars, {"T": float})

    def test_Generic(self):
        context = TVarValidationContext()
        ta = TypeAdapter(CspTypeVarType["T"])
        ta.validate_python(MyGeneric[float], context=context)
        context.resolve_tvars()
        self.assertDictEqual(context.tvars, {"T": MyGeneric[float]})

        ta.validate_python(MyGeneric, context=context)
        context.resolve_tvars()
        self.assertDictEqual(context.tvars, {"T": MyGeneric})

    def test_Generic_subclass(self):
        context = TVarValidationContext()
        ta = TypeAdapter(CspTypeVarType["T"])
        ta.validate_python(MyGeneric[float], context=context)
        context.resolve_tvars()
        self.assertDictEqual(context.tvars, {"T": MyGeneric[float]})

        ta.validate_python(MyGeneric[np.float64], context=context)
        # Doesn't currently resolve, though in theory it could
        self.assertRaises(Exception, context.resolve_tvars)

    def test_TsType(self):
        context = TVarValidationContext()
        ta = TypeAdapter(TsType[CspTypeVarType["T"]])
        ta.validate_python(csp.null_ts(float), context=context)
        context.resolve_tvars()
        self.assertDictEqual(context.tvars, {"T": float})

    def test_TsType_list(self):
        context = TVarValidationContext()
        ta = TypeAdapter(TsType[CspTypeVarType["T"]])
        ta.validate_python(csp.null_ts(List[float]), context=context)
        context.resolve_tvars()
        self.assertDictEqual(context.tvars, {"T": List[float]})

    def test_TsType_nested(self):
        context = TVarValidationContext()
        ta = TypeAdapter(TsType[List[CspTypeVarType["T"]]])
        ta.validate_python(csp.null_ts(List[float]), context=context)
        context.resolve_tvars()
        self.assertDictEqual(context.tvars, {"T": float})

    def test_bad_variable_name(self):
        self.assertRaises(SyntaxError, lambda: CspTypeVarType[""])
        self.assertRaises(SyntaxError, TypeAdapter, CspTypeVarType["~T"])
        self.assertRaises(SyntaxError, TypeAdapter, CspTypeVarType["1"])
        _ = TypeAdapter(CspTypeVarType["T1"])


T = TypeVar("T")


class MyModel(BaseModel):
    static_1: CspTypeVar[T]
    static_2: CspTypeVar[T]
    typ_1: CspTypeVarType[T]
    typ_2: CspTypeVarType[T]
    ts_1: TsType[CspTypeVarType[T]]
    ts_2: TsType[CspTypeVarType[T]]

    @model_validator(mode="after")
    def validate_tvars(self, info: ValidationInfo):
        info.context.resolve_tvars()
        return info.context.revalidate(self)

    @field_validator("*", mode="before")
    @classmethod
    def my_validator(cls, v, info):
        info.context.field_name = info.field_name
        return v


class TestValidation(TestCase):
    def test_revalidation(self):
        values = dict(
            static_1=float(1), static_2=int(2), typ_1=int, typ_2=float, ts_1=csp.const(float(1)), ts_2=csp.const(int(2))
        )
        context = TVarValidationContext()
        model = MyModel.model_validate(values, context=context)
        self.assertDictEqual(context.tvars, {"T": float})
        self.assertEqual(model.static_1, float(1))
        self.assertEqual(model.static_2, float(2))
        self.assertIsInstance(model.static_2, float)
        self.assertEqual(model.typ_1, int)
        self.assertEqual(model.typ_2, float)
        self.assertEqual(model.ts_1.tstype.typ, float)
        self.assertEqual(model.ts_2.tstype.typ, float)
