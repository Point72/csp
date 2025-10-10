import unittest
from datetime import datetime, timedelta
from typing import Optional

import csp
from csp import ts
from csp.impl.wiring.base_parser import CspParseError


class TestStrictStructs(unittest.TestCase):
    def test_backwards_compatibility(self):
        """quick test that existing struct behavior is unchanged"""

        class OldStruct(csp.Struct, allow_unset=True):
            a: int
            b: str

        s = OldStruct(a=5)
        self.assertFalse(hasattr(s, "b"))
        with self.assertRaisesRegex(AttributeError, "b"):
            _ = s.b
        del s.a
        self.assertFalse(hasattr(s, "a"))

    def test_strict_struct_initialization(self):
        """test initialization rules for strict structs.

        notably,
        * Setting fields works as expected
        * Initialize all non-default fields (including Optional)
        * Missing required fields fail
        """

        class MyStrictStruct(csp.Struct, allow_unset=False):
            req_int: int
            opt_str: str | None = None
            def_int: int = 123
            opt_str_2: Optional[str] = None

        # Valid initialization
        s1 = MyStrictStruct(req_int=10, opt_str="hello")
        self.assertEqual(s1.req_int, 10)
        self.assertEqual(s1.opt_str, "hello")
        self.assertEqual(s1.def_int, 123)
        self.assertIsNone(s1.opt_str_2)

        with self.assertRaisesRegex(
            ValueError, r"Struct MyStrictStruct is not valid; required fields \[req_int\] were not set on init"
        ):
            MyStrictStruct()

    def test_strict_struct_hasattr_delattr(self):
        """test hasattr and delattr behavior for strict structs"""

        class MyStrictStruct(csp.Struct, allow_unset=False):
            req_int: int
            opt_str: Optional[str] = None

        s = MyStrictStruct(req_int=10, opt_str="hello")
        r = MyStrictStruct(req_int=5)

        # hasattr will always be True for all defined fields
        self.assertTrue(hasattr(s, "req_int"))
        self.assertTrue(hasattr(s, "opt_str"))
        self.assertTrue(hasattr(r, "req_int"))
        self.assertTrue(hasattr(r, "opt_str"))

        # delattr is forbidden
        with self.assertRaisesRegex(
            AttributeError, "Strict struct MyStrictStruct does not allow the deletion of field req_int"
        ):
            del s.req_int

        with self.assertRaisesRegex(
            AttributeError, "Strict struct MyStrictStruct does not allow the deletion of field opt_str"
        ):
            del s.opt_str

    def test_strict_struct_serialization(self):
        """test to_dict and from_dict behavior"""

        class MyStrictStruct(csp.Struct, allow_unset=False):
            req_int: int
            opt_str: Optional[str] = None
            def_int: int = 100
            opt_str2: Optional[str] = None

        s = MyStrictStruct(req_int=50, opt_str2="NoneStr")
        expected_dict = {"req_int": 50, "opt_str": None, "def_int": 100, "opt_str2": "NoneStr"}
        self.assertEqual(s.to_dict(), expected_dict)

        with self.assertRaisesRegex(
            ValueError, r"Struct MyStrictStruct is not valid; required fields \[req_int\] were not set on init"
        ):
            MyStrictStruct.from_dict({"opt_str": "hello", "def_int": 13})

        MyStrictStruct.from_dict({"req_int": 60, "opt_str": None, "opt_str2": None})
        s2 = MyStrictStruct.from_dict({"def_int": 72, "req_int": 60, "opt_str2": None})
        self.assertEqual(s2.req_int, 60)
        self.assertIsNone(s2.opt_str)
        self.assertEqual(s2.def_int, 72)

    def test_strict_struct_wiring_access_1(self):
        """test accessing fields on a time series at graph wiring time"""

        class MyStrictStruct(csp.Struct, allow_unset=False):
            req_int: int
            opt_str: Optional[str] = None

        # check that at graph and wire time we are able to access required fields just fine:

        @csp.node
        def ok_node(x: csp.ts[MyStrictStruct]):
            int_val = x.req_int

        @csp.graph
        def g():
            s_ts = csp.const(MyStrictStruct(req_int=1))
            req_ts = s_ts.req_int
            csp.add_graph_output("req_ts", req_ts)

        res = csp.run(g, starttime=datetime(2023, 1, 1))
        self.assertEqual(res["req_ts"][0][1], 1)

        # check that at graph time we cannot access optional fields:

        @csp.graph
        def g_fail():
            s_ts = csp.const(MyStrictStruct(req_int=1))
            opt_ts = s_ts.opt_str

        with self.assertRaisesRegex(
            AttributeError,
            "Cannot access optional field 'opt_str' on strict struct object 'MyStrictStruct' at graph time",
        ):
            csp.run(g_fail, starttime=datetime(2023, 1, 1))

    def test_strict_struct_fromts(self):
        """fromts requires all non-defaulted fields to tick together"""

        class MyStrictStruct(csp.Struct, allow_unset=False):
            req_int1: int
            req_int2: int
            opt_str: Optional[str] = None
            req_default_str: str = "default"

        @csp.node
        def make_ts(x: csp.ts[int]) -> csp.ts[int]:
            if x % 2 == 0:
                return x

        @csp.graph
        def g():
            ts1 = make_ts(csp.const(2))
            ts2 = make_ts(csp.const(1))

            # ts1 and ts2 don't tick together
            s_ts = MyStrictStruct.fromts(req_int1=ts1, req_int2=ts2)
            csp.add_graph_output("output", s_ts)

        with self.assertRaisesRegex(
            ValueError, r"Struct MyStrictStruct is not valid; required fields \[req_int2\] did not tick"
        ):
            csp.run(g, starttime=datetime(2023, 1, 1))

        @csp.graph
        def g_ok():
            ts1 = csp.const(2)
            ts2 = csp.const(4)

            # ts1 and ts2 tick together
            s_ts = MyStrictStruct.fromts(req_int1=ts1, req_int2=ts2)
            csp.add_graph_output("output", s_ts)

        csp.run(g_ok, starttime=datetime(2023, 1, 1))

        @csp.graph
        def g_ok_with_optional():
            beat = csp.timer(timedelta(days=1))
            even = csp.eq(csp.mod(csp.count(beat), csp.const(2)), csp.const(0))

            int_ts1 = csp.sample(even, csp.const(1))
            int_ts2 = csp.sample(even, csp.const(2))
            str_ts = csp.sample(even, csp.const("Hello"))

            s_ts = MyStrictStruct.fromts(req_int1=int_ts1, req_int2=int_ts2, req_default_str=str_ts)
            csp.add_graph_output("output", s_ts)

        csp.run(g_ok_with_optional, starttime=datetime(2025, 1, 1), endtime=datetime(2025, 1, 5))

    def test_strict_struct_inheritance_and_nested(self):
        class BaseStrict(csp.Struct, allow_unset=False):
            base_req: int

        class DerivedStrict(BaseStrict, allow_unset=False):
            derived_req: int

        d_ok = DerivedStrict(base_req=1, derived_req=2)
        self.assertEqual(d_ok.base_req, 1)
        self.assertEqual(d_ok.derived_req, 2)

        with self.assertRaisesRegex(
            ValueError, r"Struct DerivedStrict is not valid; required fields \[derived_req\] were not set on init"
        ):
            DerivedStrict(base_req=10)
        with self.assertRaisesRegex(
            ValueError, r"Struct DerivedStrict is not valid; required fields \[base_req\] were not set on init"
        ):
            DerivedStrict(derived_req=20)

        # loose base & strict child:
        class LooseBase(csp.Struct, allow_unset=True):
            loose_req: int

        class StrictChild(LooseBase, allow_unset=False):
            child_req: int

        sc_ok = StrictChild(child_req=5, loose_req=10)
        self.assertEqual(sc_ok.child_req, 5)

        with self.assertRaisesRegex(
            ValueError, r"Struct StrictChild is not valid; required fields \[child_req\] were not set on init"
        ):
            StrictChild()
        with self.assertRaisesRegex(
            ValueError, r"Struct StrictChild is not valid; required fields \[child_req\] were not set on init"
        ):
            StrictChild(loose_req=10)

        StrictChild(child_req=5)

        # nested struct fields:
        class InnerStrict(csp.Struct, allow_unset=False):
            val: int
            val2: float

        class OuterStrict(csp.Struct, allow_unset=False):
            inner: InnerStrict

        os_ok = OuterStrict(inner=InnerStrict(val=42, val2=43))
        self.assertEqual(os_ok.inner.val, 42)
        self.assertEqual(os_ok.inner.val2, 43)

        with self.assertRaisesRegex(
            ValueError, r"Struct InnerStrict is not valid; required fields \[val, val2\] were not set on init"
        ):
            OuterStrict(inner=InnerStrict())
        with self.assertRaisesRegex(
            ValueError, r"Struct InnerStrict is not valid; required fields \[val2\] were not set on init"
        ):
            OuterStrict(inner=InnerStrict(val=42))

        # nested loose struct inside strict:
        class InnerLoose(csp.Struct, allow_unset=True):
            val: int

        class OuterStrict2(csp.Struct, allow_unset=False):
            inner: InnerLoose

        ol_ok = OuterStrict2(inner=InnerLoose())
        self.assertIsInstance(ol_ok.inner, InnerLoose)

        with self.assertRaisesRegex(
            ValueError, r"Struct OuterStrict2 is not valid; required fields \[inner\] were not set on init"
        ):
            OuterStrict2()

    def test_nonstrict_cannot_inherit_strict(self):
        """non-strict structs inheriting from strict bases should raise"""

        class StrictBase(csp.Struct, allow_unset=False):
            base_val: int

        with self.assertRaisesRegex(ValueError, "non-strict inheritance of strict base"):

            class NonStrictChild1(StrictBase, allow_unset=True):
                child_val1: Optional[int] = None

    def test_nonstrict_strict_nonstrict_inheritance_order(self):
        """inheritance order NonStrict -> Strict -> NonStrict raises an error"""

        class NonStrictBase(csp.Struct, allow_unset=True):
            base_val: int

        class StrictMiddle(NonStrictBase, allow_unset=False):
            middle_val: int

        with self.assertRaisesRegex(ValueError, "non-strict inheritance of strict base"):

            class NonStrictChild(StrictMiddle, allow_unset=True):
                child_val: Optional[int] = None

    def test_nested_struct_serialization(self):
        """to_dict / from_dict work with nested strict & non-strict structs"""

        class InnerStrict(csp.Struct, allow_unset=False):
            x: int

        class InnerLoose(csp.Struct, allow_unset=True):
            y: Optional[int] = None

        class OuterStruct(csp.Struct, allow_unset=False):
            strict_inner: InnerStrict
            loose_inner: InnerLoose

        o = OuterStruct(strict_inner=InnerStrict(x=5), loose_inner=InnerLoose())
        expected_dict = {"strict_inner": {"x": 5}, "loose_inner": {"y": None}}
        self.assertEqual(o.to_dict(), expected_dict)

        o2 = OuterStruct.from_dict({"strict_inner": {"x": 10}, "loose_inner": {"y": 20}})
        self.assertEqual(o2.strict_inner.x, 10)
        self.assertIsNotNone(o2.loose_inner)
        self.assertEqual(o2.loose_inner.y, 20)

        with self.assertRaisesRegex(
            ValueError, r"Struct OuterStruct is not valid; required fields \[strict_inner\] were not set on init"
        ):
            OuterStruct.from_dict({"loose_inner": {"y": 1}})

        with self.assertRaisesRegex(
            ValueError, r"Struct InnerStrict is not valid; required fields \[x\] were not set on init"
        ):
            OuterStruct.from_dict({"strict_inner": {}, "loose_inner": {"y": None}})

    def test_strict_struct_wiring_access_2(self):
        class Test(csp.Struct, allow_unset=False):
            name: str
            age: int
            is_active: bool | None = None

            def greet(self):
                return f"Hello, my name is {self.name} and I am {self.age} years old."

        @csp.node
        def test() -> csp.ts[Test]:
            return Test(name="John", age=30, is_active=True)

        @csp.graph
        def main_graph():
            res = test().is_active
            csp.print("", res)

        with self.assertRaisesRegex(
            AttributeError, "Cannot access optional field 'is_active' on strict struct object 'Test' at graph time"
        ):
            csp.build_graph(main_graph)

    def test_strict_struct_optional_field_validation_no_default(self):
        with self.assertRaisesRegex(TypeError, "Optional field bad_field must have a default value"):

            class InvalidStrictStruct(csp.Struct, allow_unset=False):
                req_field: int
                bad_field: Optional[str]


if __name__ == "__main__":
    unittest.main()
