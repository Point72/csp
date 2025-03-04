import _pickle
import json
import unittest
from datetime import datetime, timedelta
from typing import Dict, List

import pytest
from pydantic import BaseModel, ConfigDict, RootModel

import csp
from csp import ts


class MyEnum(csp.Enum):
    A = 1
    B = 20
    C = 3


class MyEnum2(csp.Enum):
    C = 10
    D = 20

    # Methods and properties allowed
    def f1(self):
        return self.value * 2

    @property
    def p1(self):
        return self.f1()

    @classmethod
    def c1(cls):
        return cls.C.value * 3

    @staticmethod
    def s1():
        return 123


MyDEnum = csp.DynamicEnum("MyDEnum", ["A", "B", "C"])


class MyEnum3(csp.Enum):
    FIELD1 = csp.Enum.auto()
    FIELD2 = csp.Enum.auto()


class MyModel(BaseModel):
    enum: MyEnum3
    enum_default: MyEnum3 = MyEnum3.FIELD1


class MyDictModel(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    enum_dict: Dict[MyEnum3, int] = None


class TestCspEnum(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(MyEnum("A"), MyEnum.A)
        self.assertEqual(MyEnum(1), MyEnum.A)
        self.assertEqual(MyEnum("B"), MyEnum.B)
        self.assertEqual(MyEnum(20), MyEnum.B)
        self.assertNotEqual(MyEnum.A, MyEnum.B)
        self.assertNotEqual(MyEnum.A, 1)
        self.assertNotEqual(MyEnum.B, MyEnum2.D)

        self.assertEqual(MyEnum2.C.value, 10)
        self.assertEqual(MyEnum2.C.name, "C")

        self.assertEqual(list(MyEnum), [MyEnum.A, MyEnum.B, MyEnum.C])

        self.assertEqual(MyEnum2.C.f1(), 20)
        self.assertEqual(MyEnum2.D.p1, 40)
        self.assertEqual(MyEnum2.c1(), 30)
        self.assertEqual(MyEnum2.s1(), 123)

        with self.assertRaisesRegex(ValueError, "123 is not a valid value on csp.enum type MyEnum"):
            MyEnum(123)

        with self.assertRaisesRegex(ValueError, "ABC is not a valid value on csp.enum type MyEnum"):
            MyEnum("ABC")

        with self.assertRaisesRegex(TypeError, "csp.Enum expected int enum value, got str for field B"):

            class FOO(csp.Enum):
                A = 1
                B = "hey"

        # bracket access
        self.assertEqual(MyEnum["A"], MyEnum.A)

        # auto
        class MyEnum3(csp.Enum):
            A = csp.Enum.auto()
            B = 10
            C = csp.Enum.auto()

        self.assertEqual(list(MyEnum3), [MyEnum3.A, MyEnum3.B, MyEnum3.C])
        self.assertEqual(MyEnum3.A.value, 0)
        self.assertEqual(MyEnum3.B.value, 10)
        self.assertEqual(MyEnum3.C.value, 11)

    def test_python_enum_compatibility(self):
        self.assertEqual(dict(MyEnum.__members__), dict(MyEnum.__metadata__))

    def test_node(self):
        """test ability of node to convert to/from enum types properly"""

        @csp.node
        def enum_test_out(x: ts[int]) -> ts[MyEnum]:
            if csp.ticked(x):
                return MyEnum(x)

        @csp.node
        def enum_test_in(x: ts[MyEnum]) -> ts[int]:
            if csp.ticked(x):
                return x.value

        @csp.graph
        def graph():
            td = timedelta()
            x = csp.curve(int, [(td, 3), (td, 20), (td, 1)], push_mode=csp.PushMode.NON_COLLAPSING)

            eout = enum_test_out(x)
            ein = enum_test_in(eout)
            csp.add_graph_output("eout", eout)
            csp.add_graph_output("ein", ein)

        result = csp.run(graph, starttime=datetime.utcnow())
        self.assertEqual([v[1] for v in result["eout"]], [MyEnum(3), MyEnum(20), MyEnum(1)])
        self.assertEqual([v[1] for v in result["ein"]], [3, 20, 1])

    def test_enum_type_check(self):
        class S(csp.Struct):
            a: MyEnum

        with self.assertRaisesRegex(TypeError, "Invalid enum type, expected enum type MyEnum got MyEnum2"):
            S(a=MyEnum2.C)

    def test_dynamic_enum(self):
        DEnum = csp.DynamicEnum("DEnum", ["A", "B", "C"])
        DEnum2 = csp.DynamicEnum("DEnum", {"A": 5, "B": 6, "C": 20})

        self.assertEqual(list(DEnum), [DEnum.A, DEnum.B, DEnum.C])
        self.assertEqual(list(v.value for v in DEnum), [0, 1, 2])
        self.assertEqual(list(v.value for v in DEnum2), [5, 6, 20])

    def test_incref_crash(self):
        """original cut was missing incref"""
        for _ in range(5):
            _ = [MyEnum(1) for x in range(1000)]

    def test_pickle(self):
        x = _pickle.loads(_pickle.dumps(MyEnum.B))
        self.assertEqual(x, MyEnum.B)

        # Dynamic pickle
        x = _pickle.loads(_pickle.dumps(MyDEnum.C))
        self.assertEqual(x, MyDEnum.C)

    def test_subclassing(self):
        # subclassing an enum should raise an error similar to the standard Python enum.Enum
        class A(csp.Enum):
            RED = 1

        with self.assertRaises(TypeError) as cm:

            class B(A):
                GREEN = 2

        self.assertEqual("Cannot extend csp.Enum 'A': inheriting from an Enum is prohibited", str(cm.exception))

    def test_pydantic_validation(self):
        assert MyModel(enum="FIELD2").enum == MyEnum3.FIELD2
        assert MyModel(enum=0).enum == MyEnum3.FIELD1
        assert MyModel(enum=MyEnum3.FIELD1).enum == MyEnum3.FIELD1
        with pytest.raises(ValueError):
            MyModel(enum=3.14)

    def test_pydantic_dict(self):
        assert dict(MyModel(enum=MyEnum3.FIELD2)) == {"enum": MyEnum3.FIELD2, "enum_default": MyEnum3.FIELD1}
        assert MyModel(enum=MyEnum3.FIELD2).model_dump(mode="python") == {
            "enum": MyEnum3.FIELD2,
            "enum_default": MyEnum3.FIELD1,
        }
        assert MyModel(enum=MyEnum3.FIELD2).model_dump(mode="json") == {"enum": "FIELD2", "enum_default": "FIELD1"}

    def test_pydantic_serialization(self):
        assert "enum" in MyModel.model_fields
        assert "enum_default" in MyModel.model_fields
        tm = MyModel(enum=MyEnum3.FIELD2)
        assert json.loads(tm.model_dump_json()) == json.loads('{"enum": "FIELD2", "enum_default": "FIELD1"}')

    def test_enum_as_dict_key_json_serialization(self):
        class DictWrapper(RootModel[Dict[MyEnum3, int]]):
            model_config = ConfigDict(use_enum_values=True)

            def __getitem__(self, item):
                return self.root[item]

        class MyDictWrapperModel(BaseModel):
            model_config = ConfigDict(use_enum_values=True)

            enum_dict: DictWrapper

        dict_model = MyDictModel(enum_dict={MyEnum3.FIELD1: 8, MyEnum3.FIELD2: 19})
        assert dict_model.enum_dict[MyEnum3.FIELD1] == 8
        assert dict_model.enum_dict[MyEnum3.FIELD2] == 19

        assert json.loads(dict_model.model_dump_json()) == json.loads('{"enum_dict":{"FIELD1":8,"FIELD2":19}}')

        dict_wrapper_model = MyDictWrapperModel(enum_dict=DictWrapper({MyEnum3.FIELD1: 8, MyEnum3.FIELD2: 19}))

        assert dict_wrapper_model.enum_dict[MyEnum3.FIELD1] == 8
        assert dict_wrapper_model.enum_dict[MyEnum3.FIELD2] == 19
        assert json.loads(dict_wrapper_model.model_dump_json()) == json.loads('{"enum_dict":{"FIELD1":8,"FIELD2":19}}')

    def test_json_schema_csp(self):
        assert MyModel.model_json_schema() == {
            "properties": {
                "enum": {
                    "description": "An enumeration of MyEnum3",
                    "enum": ["FIELD1", "FIELD2"],
                    "title": "MyEnum3",
                    "type": "string",
                },
                "enum_default": {
                    "default": "FIELD1",
                    "description": "An enumeration of MyEnum3",
                    "enum": ["FIELD1", "FIELD2"],
                    "title": "MyEnum3",
                    "type": "string",
                },
            },
            "required": ["enum"],
            "title": "MyModel",
            "type": "object",
        }


if __name__ == "__main__":
    unittest.main()
