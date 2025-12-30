import string
import unittest
from datetime import datetime, timedelta
from typing import Optional

import csp
from csp.impl.struct import define_struct


class TestStrictStructs(unittest.TestCase):
    def test_backwards_compatibility(self):
        """quick test that existing struct behavior is unchanged"""

        class OldStruct(csp.Struct, strict=False):
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

        class MyStrictStruct(csp.Struct, strict=True):
            req_int: int
            opt_str: str | None = None
            def_int: int = 123
            opt_bool: Optional[bool]  # no default

        # Valid initialization
        s1 = MyStrictStruct(req_int=10, opt_bool=True)
        self.assertEqual(s1.req_int, 10)
        self.assertEqual(s1.opt_bool, True)
        self.assertEqual(s1.def_int, 123)
        self.assertIsNone(s1.opt_str)

        meta = s1.metadata()
        self.assertEqual(meta["opt_str"], str)
        self.assertEqual(meta["opt_bool"], bool)

        with self.assertRaisesRegex(
            ValueError,
            r"Struct MyStrictStruct is not valid; required fields \[req_int, opt_bool\] were not set on init",
        ):
            MyStrictStruct()

        with self.assertRaisesRegex(
            ValueError, r"Struct MyStrictStruct is not valid; required fields \[opt_bool\] were not set on init"
        ):
            MyStrictStruct(req_int=10)

        with self.assertRaisesRegex(
            ValueError, r"Struct MyStrictStruct is not valid; required fields \[req_int\] were not set on init"
        ):
            MyStrictStruct(opt_bool=None)

    def test_strict_struct_hasattr_delattr(self):
        """test hasattr and delattr behavior for strict structs"""

        class MyStrictStruct(csp.Struct, strict=True):
            req_int: int
            opt_str: Optional[str] = None

        s = MyStrictStruct(req_int=10, opt_str="hello")
        r = MyStrictStruct(req_int=5)

        # hasattr will always be True for all defined fields
        self.assertTrue(hasattr(s, "req_int"))
        self.assertTrue(hasattr(s, "opt_str"))
        self.assertTrue(hasattr(r, "req_int"))
        self.assertTrue(hasattr(r, "opt_str"))
        self.assertIsNone(r.opt_str)

        # delattr is forbidden
        with self.assertRaisesRegex(
            AttributeError, "Strict struct MyStrictStruct does not allow the deletion of field req_int"
        ):
            del s.req_int

        with self.assertRaisesRegex(
            AttributeError, "Strict struct MyStrictStruct does not allow the deletion of field opt_str"
        ):
            del s.opt_str

    def test_strict_struct_set_none(self):
        class MyStrictStruct(csp.Struct):
            f1: int | None = None
            f2: bool

        mss = MyStrictStruct(f2=True)
        self.assertIsNone(mss.f1)
        self.assertTrue(mss.f2)

        mss.f1 = 42
        self.assertEqual(mss.f1, 42)

        mss.f2 = False
        mss.f1 = None
        self.assertIsNone(mss.f1)
        self.assertFalse(mss.f2)  # other field is not affected

    def test_strict_struct_serialization(self):
        """test to_dict, from_dict, to_json behavior"""

        # to_dict , to_json
        class MyStrictStruct(csp.Struct, strict=True):
            req_int: int
            opt_str: Optional[str] = None
            def_int: int = 100
            opt_str2: Optional[str]

        s = MyStrictStruct(req_int=50, opt_str2="AStr")
        expected_dict = {"req_int": 50, "opt_str": None, "def_int": 100, "opt_str2": "AStr"}
        self.assertEqual(s.to_dict(), expected_dict)
        expected_json = '{"opt_str":null,"opt_str2":"AStr","req_int":50,"def_int":100}'
        self.assertEqual(s.to_json(), expected_json)

        # from_dict
        with self.assertRaisesRegex(
            ValueError,
            r"Struct MyStrictStruct is not valid; required fields \[opt_str2, req_int\] were not set on init",
        ):
            MyStrictStruct.from_dict({"opt_str": "hello", "def_int": 13})

        MyStrictStruct.from_dict({"req_int": 60, "opt_str": None, "opt_str2": None})
        s2 = MyStrictStruct.from_dict({"def_int": 72, "req_int": 60, "opt_str2": None})

        meta = s2.metadata()
        self.assertEqual(meta["req_int"], int)
        self.assertEqual(meta["opt_str"], str)
        self.assertEqual(meta["def_int"], int)
        self.assertEqual(meta["opt_str2"], str)

        self.assertIsNone(s2.opt_str)
        self.assertIsNone(s2.opt_str2)
        self.assertEqual(s2.req_int, 60)
        self.assertEqual(s2.def_int, 72)

    def test_strict_struct_wiring_access_1(self):
        """test accessing fields on a time series at graph wiring time"""

        class MyStrictStruct(csp.Struct, strict=True):
            req_int: int
            opt_str: Optional[str] = None

        # check that at graph and wire time we are able to access required fields just fine
        @csp.node
        def ok_node(x: csp.ts[MyStrictStruct]) -> csp.ts[str]:
            int_val = x.req_int
            str_val = x.opt_str
            return str(int_val) + ";" + str(str_val)

        @csp.graph
        def g():
            s_ts = csp.const(MyStrictStruct(req_int=1))
            req_ts = s_ts.req_int
            csp.add_graph_output("req_ts", req_ts)
            y = ok_node(s_ts)
            csp.add_graph_output("y", y)

        res = csp.run(g, starttime=datetime(2023, 1, 1))
        self.assertEqual(res["req_ts"][0][1], 1)
        self.assertEqual(res["y"][0][1], "1;None")

        # check that at graph time we cannot access optional fields:
        @csp.graph
        def g_fail() -> csp.ts[str]:
            s_ts = csp.const(MyStrictStruct(req_int=1))
            opt_ts = s_ts.opt_str
            return opt_ts

        with self.assertRaisesRegex(
            AttributeError,
            "Cannot access optional field 'opt_str' on strict struct object 'MyStrictStruct' at graph time",
        ):
            csp.run(g_fail, starttime=datetime(2023, 1, 1))

        # ensure you can return a strict struct from a user-defined node and access required fields correctly
        class Test(csp.Struct, strict=True):
            name: str
            age: int
            is_active: Optional[bool] = None

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

    def test_strict_struct_fromts_collectts(self):
        """fromts requires all non-defaulted fields to be valid, collectts requires them to tick together"""

        st = datetime(2023, 1, 1)

        class MyStrictStruct(csp.Struct, strict=True):
            req_int1: int
            req_int2: int
            opt_str: Optional[str] = None
            req_default_str: str = "default"
            opt_array_type: list[int] | None

        def test_and_check_values(g):
            res = csp.run(g, starttime=st)
            print(res)
            mss = res[0][0][1]
            self.assertEqual(mss.req_int1, 1)
            self.assertEqual(mss.req_int2, 2)
            self.assertIsNone(mss.opt_str)
            self.assertEqual(mss.req_default_str, "default")
            self.assertIsNone(mss.opt_array_type)

        @csp.graph
        def fromts_invalid() -> csp.ts[MyStrictStruct]:
            ts1 = csp.const(1)
            ts2 = csp.const(2, delay=timedelta(seconds=1))

            # ts2 is not valid when ts1 ticks
            return MyStrictStruct.fromts(req_int1=ts1, req_int2=ts2)

        with self.assertRaisesRegex(
            ValueError, r"Struct MyStrictStruct is not valid; required fields \[opt_array_type, req_int2\] did not tick"
        ):
            csp.run(fromts_invalid, starttime=st)

        @csp.graph
        def fromts_valid() -> csp.ts[MyStrictStruct]:
            ts1 = csp.const(1)
            ts2 = csp.const(2, delay=timedelta(seconds=1))
            ts3 = csp.const(None, delay=timedelta(seconds=1))

            # all are now valid when trigger ticks
            trigger = csp.const(True, delay=timedelta(seconds=1))
            return MyStrictStruct.fromts(trigger, req_int1=ts1, req_int2=ts2, opt_array_type=ts3)

        test_and_check_values(fromts_valid)

        @csp.graph
        def collectts_invalid() -> csp.ts[MyStrictStruct]:
            ts1 = csp.const(1)
            ts2 = csp.const(2, delay=timedelta(seconds=1))
            ts3 = csp.const(None, delay=timedelta(seconds=1))

            # ts1 ticks out of seq with ts2, ts3
            return MyStrictStruct.collectts(req_int1=ts1, req_int2=ts2, opt_array_type=ts3)

        with self.assertRaisesRegex(
            ValueError, r"Struct MyStrictStruct is not valid; required fields \[opt_array_type, req_int2\] did not tick"
        ):
            csp.run(collectts_invalid, starttime=st)

        @csp.graph
        def collectts_valid() -> csp.ts[MyStrictStruct]:
            ts1 = csp.const(1, delay=timedelta(seconds=1))
            ts2 = csp.const(2, delay=timedelta(seconds=1))
            ts3 = csp.const(None, delay=timedelta(seconds=1))

            # all tick together, collectts is valid
            return MyStrictStruct.collectts(req_int1=ts1, req_int2=ts2, opt_array_type=ts3)

        test_and_check_values(collectts_valid)

    def test_strict_struct_inheritance_and_nested(self):
        class BaseStrict(csp.Struct, strict=True):
            base_req: int

        class DerivedStrict(BaseStrict, strict=True):
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

        # nested struct fields:
        class InnerStrict(csp.Struct, strict=True):
            val: int
            val2: float

        class OuterStrict(csp.Struct, strict=True):
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
        class InnerLoose(csp.Struct):
            val: int

        class OuterStrict2(csp.Struct, strict=True):
            inner: InnerLoose

        ol_ok = OuterStrict2(inner=InnerLoose())
        self.assertIsInstance(ol_ok.inner, InnerLoose)

        with self.assertRaisesRegex(
            ValueError, r"Struct OuterStrict2 is not valid; required fields \[inner\] were not set on init"
        ):
            OuterStrict2()

    def test_no_mixed_inheritance(self):
        """non-strict structs inheriting from strict bases should raise, and vice versa"""

        class StrictBase(csp.Struct, strict=True):
            base_val: int

        with self.assertRaisesRegex(
            ValueError, r"Struct NonStrictChild1 was declared non-strict but derives from StrictBase which is strict"
        ):

            class NonStrictChild1(StrictBase):
                child_val1: Optional[int] = None

            _ = NonStrictChild1

        class NonStrictBase(csp.Struct):
            base_val: int

        with self.assertRaisesRegex(
            ValueError, r"Struct StrictChild1 was declared strict but derives from NonStrictBase which is non-strict"
        ):

            class StrictChild1(NonStrictBase, strict=True):
                child_val1: Optional[int] = None

            _ = StrictChild1

    def test_nested_struct_serialization(self):
        """to_dict / from_dict work with nested strict & non-strict structs"""

        class InnerStrict(csp.Struct, strict=True):
            x: int

        class InnerLoose(csp.Struct):
            y: Optional[int] = None

        class OuterStruct(csp.Struct, strict=True):
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

    def test_str_repr(self):
        class StrictStruct(csp.Struct, strict=True):
            a: int | None = None
            b: bool
            c: str | None = None

        ss1 = StrictStruct(b=True, c="abc")
        ss2 = StrictStruct(a=42, b=False)
        self.assertEqual(repr(ss1), "StrictStruct( a=None, b=True, c=abc )")
        self.assertEqual(str(ss2), "StrictStruct( a=42, b=False, c=None )")

    def test_bitmask_edge_cases(self):
        # 4 optional fields mean we have a full byte of optional fields
        A = define_struct(
            "A",
            {c: Optional[int] for c in string.ascii_lowercase[:4]},
            {c: None for c in string.ascii_lowercase[:4]},
            strict=True,
        )

        def verify_first_byte(a):
            self.assertEqual(a.a, 1)
            self.assertIsNone(a.b)
            self.assertEqual(a.c, 3)
            self.assertIsNone(a.d)

        def verify_second_byte(a, is_e_none):
            if is_e_none:
                self.assertIsNone(a.e)
            else:
                self.assertEqual(a.e, 5)
            self.assertEqual(a.f, 6)
            self.assertEqual(a.g, 7)

        a = A(a=1, c=3)
        verify_first_byte(a)

        # 5 optional fields and 2 non-optional mean we have a full byte of optional fields, then a partial byte with 1 opt and 2 non-opt (4 bits)
        A = define_struct(
            "A",
            {c: Optional[int] for c in string.ascii_lowercase[:5]} | {c: int for c in string.ascii_lowercase[5:7]},
            {c: None for c in string.ascii_lowercase[:4]},  # do not default last field
            strict=True,
        )

        # test with last opt field set, none, or neither
        a = A(a=1, c=3, e=5, f=6, g=7)
        verify_first_byte(a)
        verify_second_byte(a, False)
        a = A(e=None, f=6, g=7)
        verify_second_byte(a, True)
        with self.assertRaisesRegex(ValueError, r"Struct A is not valid; required fields \[e\] were not set on init"):
            a = A(f=6, g=7)


if __name__ == "__main__":
    unittest.main()
