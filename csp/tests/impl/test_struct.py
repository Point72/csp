import enum
import json
import pickle
import sys
import unittest
from datetime import date, datetime, time, timedelta
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union

import numpy as np
import pytz
from pydantic import TypeAdapter, ValidationError
from pydantic_core import PydanticSerializationError
from typing_extensions import Annotated

import csp
from csp.impl.struct import define_nested_struct, define_struct, defineNestedStruct, defineStruct
from csp.impl.types.typing_utils import FastList
from csp.typing import Numpy1DArray


class MyEnum(csp.Enum):
    A = 1
    FOO = 5


class StructNoDefaults(csp.Struct):
    b: bool
    i: int
    f: float
    s: str
    bt: bytes
    o: object
    a1: FastList[int]
    a2: List[str]
    a3: FastList[object]
    a4: List[bytes]
    a5: Numpy1DArray[float]


class StructWithDefaults(csp.Struct):
    b: bool
    i: int = 123
    f: float
    s: str = "456"
    e: MyEnum = MyEnum.FOO
    o: object
    a1: FastList[int] = [1, 2, 3]
    a2: List[str] = ["1", "2", "3"]
    a3: List[object] = ["hey", 123, (1, 2, 3)]
    np_arr: np.ndarray = np.array([1, 9])


class DerivedStructWithDefaults(StructWithDefaults):
    x: int = 789


class StructWithMethods(csp.Struct):
    i: int
    i2: int

    def add(self):
        return self.i + self.i2


class BaseNative(csp.Struct):
    i: int
    b: bool
    f: float


class BaseNonNative(csp.Struct):
    s: str
    l: list
    o: object


class BaseMixed(csp.Struct):
    i: int
    b: bool
    f: float
    s: str
    l: list
    o: object
    a1: List[int]
    a2: FastList[str]
    a3: List[object]


class DerivedFullyNative(BaseNative):
    i2: int
    b2: bool
    f2: float


class DerivedDerived(DerivedFullyNative):
    i3: int


class DerivedPartialNative(BaseNonNative):
    i: int
    b: bool
    f: float


class DerivedMixed(BaseMixed):
    i2: int
    b2: bool
    f2: float
    s2: str
    l2: list
    o2: object


class DerivedMixedNoop(DerivedMixed):
    pass


class DerivedMixedAfterNoop(DerivedMixedNoop):
    i3: int
    l3: list


class StructWithStruct(csp.Struct):
    a: int
    s: BaseNative


class StructWithMutableStruct(csp.Struct):
    a: int
    s: BaseNonNative


class StructWithLists(csp.Struct):
    # native_list: List[int]
    struct_list: List[BaseNative]
    fast_list: FastList[BaseNative]
    dialect_generic_list: List[list]


class AllTypes(csp.Struct):
    b: bool = False
    i: int = 123
    d: float = 123.456
    dt: datetime = datetime(2022, 12, 6, 1, 2, 3)
    td: timedelta = timedelta(seconds=0.123)
    dte: date = date(2022, 12, 6)
    t: time = time(1, 2, 3)
    s: str = "hello hello"
    e: MyEnum = MyEnum.FOO
    struct: BaseNative = BaseNative(i=123, b=True, f=456.789)
    arr: List[int] = [1, 2, 3]
    fl: FastList[int] = [1, 2, 3, 4]
    o: object = {"k": "v"}


class SimpleStruct(csp.Struct):
    a: int


class AnotherSimpleStruct(csp.Struct):
    b: str


class SimpleEnum(csp.Enum):
    A = 1
    B = 2
    C = 3


class AnotherSimpleEnum(csp.Enum):
    D = 5


class SimpleClass:
    x: int

    def __init__(self, x: int):
        self.x = x


class SimpleStructForPickleList(csp.Struct):
    a: List[int]


class SimpleStructForPickleFastList(csp.Struct):
    a: FastList[int]


# Common set of values for Struct list field tests
# For each type:
# items[:-2] are normal values of the given type that should be handled,
# items[-2] is a normal value for non-generic and non-str types and None for generic and str types (the purpose is to test the raise of TypeError if a single object instead of a sequence is passed),
# items[-1] is a value of a different type that is not convertible to the give type for non-generic types and None for generic types (the purpose is to test the raise of TypeError if an object of the wrong type is passed).
struct_list_test_values = {
    int: [4, 2, 3, 5, 6, 7, 8, "s"],
    bool: [True, True, True, False, True, False, True, 2],
    float: [1.4, 3.2, 2.7, 1.0, -4.5, -6.0, -2.0, "s"],
    datetime: [
        datetime(2022, 12, 6, 1, 2, 3),
        datetime(2022, 12, 7, 2, 2, 3),
        datetime(2022, 12, 8, 3, 2, 3),
        datetime(2022, 12, 9, 4, 2, 3),
        datetime(2022, 12, 10, 5, 2, 3),
        datetime(2022, 12, 11, 6, 2, 3),
        datetime(2022, 12, 13, 7, 2, 3),
        timedelta(seconds=0.123),
    ],
    timedelta: [
        timedelta(seconds=0.123),
        timedelta(seconds=12),
        timedelta(seconds=1),
        timedelta(seconds=0.5),
        timedelta(seconds=123),
        timedelta(seconds=70),
        timedelta(seconds=700),
        datetime(2022, 12, 8, 3, 2, 3),
    ],
    date: [
        date(2022, 12, 6),
        date(2022, 12, 7),
        date(2022, 12, 8),
        date(2022, 12, 9),
        date(2022, 12, 10),
        date(2022, 12, 11),
        date(2022, 12, 13),
        timedelta(seconds=0.123),
    ],
    time: [
        time(1, 2, 3),
        time(2, 2, 3),
        time(3, 2, 3),
        time(4, 2, 3),
        time(5, 2, 3),
        time(6, 2, 3),
        time(7, 2, 3),
        timedelta(seconds=0.123),
    ],
    str: ["s", "pqr", "masd", "wes", "as", "m", None, 5],
    csp.Struct: [
        SimpleStruct(a=1),
        AnotherSimpleStruct(b="sd"),
        SimpleStruct(a=3),
        AnotherSimpleStruct(b="sdf"),
        SimpleStruct(a=-4),
        SimpleStruct(a=5),
        SimpleStruct(a=7),
        4,
    ],  # untyped struct list
    SimpleStruct: [
        SimpleStruct(a=1),
        SimpleStruct(a=3),
        SimpleStruct(a=-1),
        SimpleStruct(a=-4),
        SimpleStruct(a=5),
        SimpleStruct(a=100),
        SimpleStruct(a=1200),
        AnotherSimpleStruct(b="sd"),
    ],
    SimpleEnum: [
        SimpleEnum.A,
        SimpleEnum.C,
        SimpleEnum.B,
        SimpleEnum.B,
        SimpleEnum.B,
        SimpleEnum.C,
        SimpleEnum.C,
        AnotherSimpleEnum.D,
    ],
    list: [[1], [1, 2, 1], [6], [8, 3, 5], [3], [11, 8], None, None],  # generic type list
    SimpleClass: [
        SimpleClass(x=1),
        SimpleClass(x=5),
        SimpleClass(x=9),
        SimpleClass(x=-1),
        SimpleClass(x=2),
        SimpleClass(x=3),
        None,
        None,
    ],  # generic type user-defined
}
struct_list_annotation_types = (List, FastList)


class TestCspStruct(unittest.TestCase):
    def test_basic(self):
        s = StructNoDefaults(i=123, s="456", bt=b"ab\001\000c", a4=[b"ab\001\000c"])
        self.assertEqual(s.i, 123)
        self.assertEqual(s.s, "456")
        self.assertEqual(s.bt, b"ab\001\000c")
        self.assertEqual(s.a4, [b"ab\001\000c"])
        self.assertFalse(hasattr(s, "f"))

        s = StructWithDefaults(o=None)
        self.assertEqual(s.i, 123)
        self.assertEqual(s.s, "456")
        self.assertEqual(s.o, None)
        self.assertFalse(hasattr(s, "f"))

        s = StructWithDefaults(i=456)
        self.assertEqual(s.i, 456)
        self.assertEqual(s.s, "456")
        self.assertEqual(s.a1, [1, 2, 3])
        self.assertEqual(s.a2, ["1", "2", "3"])
        self.assertEqual(s.a3, ["hey", 123, (1, 2, 3)])

        s = StructWithMethods(i=5, i2=10)
        self.assertEqual(s.add(), 15)

    def test_all_types(self):
        s = AllTypes()
        for k, v in AllTypes.__defaults__.items():
            self.assertEqual(getattr(s, k), v, k)

        c = s.copy()
        self.assertEqual(s, c)

    def test_exceptions(self):
        with self.assertRaisesRegex(TypeError, "struct field 'a' expected field annotation as a type got 'str'"):

            class FOO(csp.Struct):
                a: "123"

        with self.assertRaisesRegex(TypeError, "expected long \\(int\\) got str"):

            class FOO(csp.Struct):
                a: int = "123"

        with self.assertRaisesRegex(TypeError, "expected long \\(int\\) got str"):

            class FOO(csp.Struct):
                a: int

            f = FOO()
            f.a = "123"

        with self.assertRaisesRegex(AttributeError, "object has no attribute 'x'"):

            class FOO(csp.Struct):
                a: int

            f = FOO()
            f.x = "123"

        with self.assertRaisesRegex(AttributeError, "a"):

            class FOO(csp.Struct):
                a: int

            FOO().a

        with self.assertRaisesRegex(TypeError, "Struct types must define at least 1 field"):

            class Foo(csp.Struct):
                pass

            _ = Foo()

    def test_raw_struct_type(self):
        class StructA(csp.Struct):
            b: int

        class StructB(csp.Struct):
            x: csp.Struct

        a = StructA(b=3)
        b = StructB()

        # bug would crash here when StructB wasn't well-defined
        b.x = a
        self.assertEqual(repr(b), "StructB( x=StructA( b=3 ) )")

    def test_basic_comparison(self):
        class FOO(csp.Struct):
            a: int
            b: str
            c: list

        a = FOO(a=1, b="2", c=[1, 2, 3])
        b = FOO(a=1, b="2", c=[1, 2, 3])

        self.assertEqual(a, b)
        self.assertFalse(a != b)

        # Unset fields
        a = FOO(c=[1])
        b = FOO(c=[1])

        self.assertEqual(a, b)
        b.a = 1
        self.assertNotEqual(a, b)

        # Was a bug with typed list of struct comparisons
        class BAR(csp.Struct):
            a: int
            b: List[FOO]

        a = BAR(a=123, b=[FOO(a=1, b="2", c=[1, 2, 3])])
        b = BAR(a=123, b=[FOO(a=1, b="2", c=[1, 2, 3])])

        self.assertEqual(a, b)
        self.assertFalse(a != b)

        b.b = [FOO(a=1, b="2", c=[1, 2, 3, 4])]
        self.assertNotEqual(a, b)
        self.assertFalse(a == b)

    def test_copy(self):
        values = {
            "i": 123,
            "f": 123.456,
            "b": True,
            "i2": 111,
            "f2": 111.222,
            "b2": False,
            "s": "str",
            "o": {},
            "l": [1, 2, 3],
            "s2": "str2",
            "o2": None,
            "l2": [4, 5, 6],
            "a1": list(range(20)),
            "a2": list(str(x) for x in range(30)),
            "a3": [[], "hey", {}, lambda x: 1],
            "i3": 333,
            "l3": list(str(x * x) for x in range(5)),
        }

        for typ in (
            BaseNative,
            BaseNonNative,
            BaseMixed,
            DerivedFullyNative,
            DerivedPartialNative,
            DerivedMixed,
            DerivedMixedNoop,
            DerivedMixedAfterNoop,
        ):
            self.assertEqual(typ(), typ().copy())
            o = typ()
            for key in typ.metadata().keys():
                setattr(o, key, values[key])
            self.assertEqual(o, o.copy())
            # Compare with unset field
            delattr(o, list(typ.metadata().keys())[0])
            self.assertEqual(o, o.copy())

        derived = DerivedDerived()
        derived.i = 1
        base = BaseNative()
        base.copy_from(derived)
        self.assertEqual(base.i, derived.i)

    def test_deepcopy(self):
        values = {
            "i": 123,
            "f": 123.456,
            "b": True,
            "i2": 111,
            "f2": 111.222,
            "b2": False,
            "s": "str",
            "o": {},
            "l": [1, 2, 3],
            "s2": "str2",
            "o2": None,
            "l2": [4, 5, 6],
            "a1": list(range(20)),
            "a2": list(str(x) for x in range(30)),
            "a3": [[], "hey", {}, lambda x: 1],
            "i3": 333,
            "l3": list(str(x * x) for x in range(5)),
            "native_list": [1, 2, 3],
            "struct_list": [BaseNative(i=1), BaseNative(i=2)],
            "dialect_generic_list": [[1], [2], (4, 5, 6)],
        }

        for typ in (
            BaseNative,
            BaseNonNative,
            BaseMixed,
            DerivedFullyNative,
            DerivedPartialNative,
            DerivedMixed,
            DerivedMixedNoop,
            DerivedMixedAfterNoop,
        ):
            self.assertEqual(typ(), typ().deepcopy())
            o = typ()
            for key in typ.metadata().keys():
                setattr(o, key, values[key])
            self.assertEqual(o, o.deepcopy())
            # Compare with unset field
            delattr(o, list(typ.metadata().keys())[0])
            self.assertEqual(o, o.deepcopy())

        o = StructWithMutableStruct()
        o.s = BaseNonNative()
        o.s.l = [1, 2, 3]

        o_deepcopy = o.deepcopy()
        self.assertEqual(o, o_deepcopy)

        o.s.l[0] = -1
        self.assertNotEqual(o, o_deepcopy)

        o = StructWithLists(
            struct_list=[BaseNative(i=123)], fast_list=[BaseNative(i=123)], dialect_generic_list=[{"a": 1}]
        )

        o_deepcopy = o.deepcopy()
        o.struct_list[0].i = -1
        o.fast_list[0].i = -2
        o.dialect_generic_list[0]["b"] = 2

        self.assertEqual(o.struct_list[0].i, -1)
        self.assertEqual(o.fast_list[0].i, -2)
        self.assertEqual(o.dialect_generic_list[0], {"a": 1, "b": 2})
        self.assertEqual(o_deepcopy.struct_list[0].i, 123)
        self.assertEqual(o_deepcopy.fast_list[0].i, 123)
        self.assertEqual(o_deepcopy.dialect_generic_list[0], {"a": 1})

        # TODO struct deepcopy doesnt actually account for this case right now, which relies on memo passing
        # deepcopy supports ensuring that object instances that appear multiple times in a container will remain
        # the same ( copied ) instance in the copy.  Uncomment final assert if/when this is fixed
        class Inner(csp.Struct):
            v: int

        class Outer(csp.Struct):
            a3: List[object]

        i = Inner(v=5)
        s = Outer(a3=[i, i, i])
        s.a3[0].v = 1
        self.assertTrue(all(x.v == 1 for x in s.a3))
        sd = s.deepcopy()
        sd.a3[0].v = 2
        # self.assertTrue(all(x == 1 for x in sd.a3))

    def test_derived_defaults(self):
        s1 = StructWithDefaults()
        s2 = DerivedStructWithDefaults()
        self.assertEqual(s1.i, s2.i)
        self.assertEqual(s1.s, s2.s)
        self.assertEqual(s2.x, 789)

    def test_comparison(self):
        values = {
            "i": 123,
            "f": 123.456,
            "b": True,
            "i2": 111,
            "f2": 111.222,
            "b2": False,
            "s": "str",
            "o": {},
            "l": [1, 2, 3],
            "s2": "str2",
            "o2": None,
            "l2": [4, 5, 6],
            "a1": list(range(20)),
            "a2": list(str(x) for x in range(30)),
            "a3": [[], "hey", {}, lambda x: 1],
            "i3": 333,
            "l3": list(str(x * x) for x in range(5)),
        }

        for typ in (
            BaseNative,
            BaseNonNative,
            BaseMixed,
            DerivedFullyNative,
            DerivedPartialNative,
            DerivedMixed,
            DerivedMixedNoop,
            DerivedMixedAfterNoop,
        ):
            self.assertEqual(typ(), typ())
            o = typ()
            o2 = typ()
            for key in typ.metadata().keys():
                setattr(o, key, values[key])
                self.assertNotEqual(o, o2)
                setattr(o2, key, values[key])
                self.assertEqual(o, o2)

    def test_hash(self):
        values = {
            "i": 123,
            "f": 123.456,
            "b": True,
            "i2": 111,
            "f2": 111.222,
            "b2": False,
            "s": "str",
            "o": (1, 2, 3),
            "s2": "str2",
            "o2": None,
            "a1": list(range(20)),
            "a2": list(str(x) for x in range(30)),
            "i3": 333,
        }

        for typ in (
            BaseNative,
            BaseNonNative,
            BaseMixed,
            DerivedFullyNative,
            DerivedPartialNative,
            DerivedMixed,
            DerivedMixedNoop,
            DerivedMixedAfterNoop,
        ):
            self.assertNotEqual(hash(typ()), 0, msg=str(typ))
            o = typ()
            prev = hash(o)
            for key in typ.metadata().keys():
                if key in values:
                    setattr(o, key, values[key])
                    self.assertNotEqual(hash(o), prev)
                    prev = hash(o)

    def test_hash_max_recursion(self):
        """ensure that if user tries to hash a cyclic data structure,
        we return an error rather than infinitely recurring and blowing through the stack"""

        class S(csp.Struct):
            a: tuple

        s = S()
        s.a = (s,)

        with self.assertRaisesRegex(RecursionError, "Exceeded max recursion depth.*"):
            hash(s)

    def test_struct_member(self):
        s = StructWithStruct(s=BaseNative(i=123))
        self.assertEqual(s.s.i, 123)
        self.assertNotEqual(hash(s), 0)
        self.assertEqual(s, s.copy())

    def test_clear(self):
        s = DerivedMixed(i=123, i2=456, s="hey", s2="hey2", l=[1, 2, 3], l2=[4, 5, 6])
        s2 = DerivedMixed(i2=456, s2="hey2", l2=[4, 5, 6])

        # Test that delattr leads to equality
        del s.i
        del s.s
        del s.l
        self.assertEqual(s, s2)

        ## Test that clear leads to equality as well
        s = DerivedMixed(i=123, i2=456, s="hey", s2="hey2", l=[1, 2, 3], l2=[4, 5, 6])
        s.clear()
        self.assertEqual(s, DerivedMixed())

    def test_copy_from(self):
        values = {
            "i": 123,
            "f": 123.456,
            "b": True,
            "i2": 111,
            "f2": 111.222,
            "b2": False,
            "s": "str",
            "o": {},
            "l": [1, 2, 3],
            "s2": "str2",
            "o2": None,
            "l2": [4, 5, 6],
            "a1": list(range(20)),
            "a2": list(str(x) for x in range(30)),
            "a3": [[], "hey", {}, lambda x: 1],
            "i3": 333,
            "l3": list(str(x * x) for x in range(5)),
        }

        for typ in (BaseNative, BaseNonNative, BaseMixed):
            for typ2 in (
                DerivedFullyNative,
                DerivedPartialNative,
                DerivedMixed,
                DerivedMixedNoop,
                DerivedMixedAfterNoop,
            ):
                if issubclass(typ2, typ):
                    blank = typ()
                    source = typ2()
                    for key in typ2.metadata().keys():
                        setattr(source, key, values[key])

                    blank.copy_from(source)
                    for key in blank.metadata().keys():
                        self.assertEqual(getattr(blank, key), getattr(source, key), (typ, typ2, key))

                    # Test other direction ( derived copy from base )
                    blank = typ2()
                    source = typ()
                    for key in typ.metadata().keys():
                        setattr(source, key, values[key])
                    blank.copy_from(source)
                    for key in source.metadata().keys():
                        self.assertEqual(getattr(blank, key), getattr(source, key), (typ, typ2, key))

    def test_copy_from_unsets(self):
        source = DerivedMixed(i=1, f=2.3, i2=4, s2="woodchuck")
        dest1 = DerivedMixed(i=5, b=True, l2=[1, 2, 3])
        dest2 = BaseMixed(i=6, s="banana", a1=[4, 5, 6])

        dest1.copy_from(source)
        self.assertTrue(dest1.i == 1)  # overrides already set value
        self.assertTrue(dest1.f == 2.3)  # adds
        self.assertTrue(dest1.i2 == 4)
        self.assertTrue(dest1.s2 == "woodchuck")
        self.assertFalse(hasattr(dest1, "b"))  # unsets
        self.assertFalse(hasattr(dest1, "l2"))
        self.assertEqual(dest1, source)  # this should actually cover the above

        dest2.copy_from(source)
        self.assertTrue(dest2.i == 1)  # overrides already set value
        self.assertTrue(dest2.f == 2.3)  # adds
        self.assertFalse(hasattr(dest2, "s"))  # unsets
        self.assertFalse(hasattr(dest2, "a1"))

        # from base class -> derived
        dest2 = BaseMixed(i=6, s="banana", a1=[4, 5, 6])
        dest3 = DerivedMixed(i=5, b=True)
        dest3.copy_from(dest2)
        self.assertTrue(dest3.i == 6)  # overrides already set value
        self.assertFalse(hasattr(dest3, "b"))  # unsets

    def test_deepcopy_from(self):
        source = StructWithLists(
            struct_list=[BaseNative(i=123)], fast_list=[BaseNative(i=123)], dialect_generic_list=[{"a": 1}]
        )

        blank = StructWithLists()
        blank.deepcopy_from(source)

        source.struct_list[0].i = -1
        source.fast_list[0].i = -2
        source.dialect_generic_list[0]["b"] = 2

        self.assertEqual(source.struct_list[0].i, -1)
        self.assertEqual(source.fast_list[0].i, -2)
        self.assertEqual(source.dialect_generic_list[0], {"a": 1, "b": 2})
        self.assertEqual(blank.struct_list[0].i, 123)
        self.assertEqual(blank.fast_list[0].i, 123)
        self.assertEqual(blank.dialect_generic_list[0], {"a": 1})

    def test_update_from(self):
        source = DerivedMixed(i=1, f=2.3, i2=4, s2="woodchuck")
        dest1 = DerivedMixed(i=5, b=True, l2=[1, 2, 3])
        dest2 = BaseMixed(i=6, s="banana", a1=[4, 5, 6])

        dest1.update_from(source)
        self.assertTrue(dest1.i == 1)  # overrides already set value
        self.assertTrue(dest1.f == 2.3)  # adds
        self.assertTrue(dest1.i2 == 4)
        self.assertTrue(dest1.s2 == "woodchuck")
        self.assertTrue(dest1.b)  # no unsets
        self.assertTrue(dest1.l2 == [1, 2, 3])
        self.assertNotEqual(dest1, source)

        dest2.update_from(source)
        self.assertTrue(dest2.i == 1)  # overrides already set value
        self.assertTrue(dest2.f == 2.3)  # adds
        self.assertTrue(dest2.s == "banana")  # no unsets
        self.assertTrue(dest2.a1 == [4, 5, 6])

        # update from base class
        dest3 = DerivedMixed()
        dest3.update_from(dest2)
        self.assertTrue(dest3.i == 1)  # overrides already set value
        self.assertTrue(dest3.f == 2.3)  # adds
        self.assertTrue(dest3.s == "banana")  # no unsets
        self.assertTrue(dest3.a1 == [4, 5, 6])

    def test_update(self):
        dest = DerivedMixed(i2=5, b=True, l2=[1, 2, 3], s="foo")
        dest.update(f2=5.5, s2="bar")
        self.assertEqual(dest, DerivedMixed(i2=5, b=True, l2=[1, 2, 3], s="foo", f2=5.5, s2="bar"))

        dest = BaseNative(b=True)
        dest.update(i=5)
        self.assertEqual(dest, BaseNative(b=True, i=5))

        dest = BaseNonNative(s="foo")
        dest.update(l=[3, 6, 7])
        self.assertEqual(dest, BaseNonNative(s="foo", l=[3, 6, 7]))

        dest = DerivedPartialNative(l=[2, 3, 4], f=3.14)
        dest.update(b=False, i=5, s="bar")
        self.assertEqual(dest, DerivedPartialNative(l=[2, 3, 4], f=3.14, b=False, i=5, s="bar"))

    def test_multibyte_mask(self):
        BigStruct = define_struct("BigStruct", {k: float for k in "abcdefghijklmnopqrdtuvwxyz"})

        s = BigStruct()
        for key in BigStruct.metadata().keys():
            self.assertFalse(hasattr(s, key))
            setattr(s, key, 1.0)
            self.assertTrue(hasattr(s, key))

            for other_key in BigStruct.metadata().keys():
                if other_key != key:
                    self.assertFalse(hasattr(s, other_key))

            delattr(s, key)

    def test_interned_from_dict(self):
        """was a bug where dict keys werent properly interned for struct lookup"""

        class MyStruct(csp.Struct):
            my_field: str

        k = "my"
        k += "_" + "field"
        s = MyStruct(**{k: "xxx"})
        self.assertEqual(s.my_field, "xxx")

    def test_from_dict_with_enum(self):
        struct = StructWithDefaults.from_dict({"e": MyEnum.A})
        self.assertEqual(MyEnum.A, getattr(struct, "e"))
        struct = StructWithDefaults.from_dict({"e": MyEnum.A}, use_pydantic=True)
        self.assertEqual(MyEnum.A, getattr(struct, "e"))

    def test_from_dict_with_list_derived_type(self):
        class ListDerivedType(list):
            def __init__(self, iterable=None):
                super().__init__(iterable)

        class StructWithListDerivedType(csp.Struct):
            ldt: ListDerivedType

        s1 = StructWithListDerivedType(ldt=ListDerivedType([1, 2]))
        self.assertTrue(isinstance(s1.to_dict()["ldt"], ListDerivedType))
        s2 = StructWithListDerivedType.from_dict(s1.to_dict())
        self.assertEqual(s1, s2)
        s3 = StructWithListDerivedType.from_dict(s1.to_dict(), use_pydantic=True)
        self.assertEqual(s1, s3)

    def test_from_dict_loop_no_defaults(self):
        looped = StructNoDefaults.from_dict(StructNoDefaults(a1=[9, 10]).to_dict())
        self.assertEqual(looped, StructNoDefaults(a1=[9, 10]))
        looped = StructNoDefaults.from_dict(StructNoDefaults(a1=[9, 10]).to_dict(), use_pydantic=True)
        self.assertEqual(looped, StructNoDefaults(a1=[9, 10]))

    def test_from_dict_loop_with_defaults(self):
        for use_pydantic in [True, False]:
            looped = StructWithDefaults.from_dict(StructWithDefaults().to_dict(), use_pydantic=use_pydantic)
            # Note that we cant compare numpy arrays, so we check them independently
            comp = StructWithDefaults()
            self.assertTrue(np.array_equal(looped.np_arr, comp.np_arr))

            del looped.np_arr
            del comp.np_arr
            self.assertEqual(looped, comp)

    def test_to_json_loop_with_no_defaults(self):
        for use_pydantic in [True, False]:
            base_struct = StructNoDefaults(
                a1=[9, 10],
                a5=np.array([1, 2, 3]),
                bt=b"ab\001\000c",
            )
            if not use_pydantic:
                # Need the callback to handle the numpy array type
                looped = StructNoDefaults.type_adapter().validate_json(base_struct.to_json(lambda x: x.tolist()))
            else:
                looped = StructNoDefaults.type_adapter().validate_json(
                    StructNoDefaults.type_adapter().dump_json(base_struct)
                )
            # Note that we cant compare numpy arrays, so we check them independently
            self.assertTrue(np.array_equal(looped.a5, base_struct.a5))

            del looped.a5
            del base_struct.a5
            self.assertEqual(looped, base_struct)
            self.assertFalse(isinstance(looped.a1, list))

    def test_from_dict_loop_with_generic_typing(self):
        class MyStruct(csp.Struct):
            foo: Set[int]
            bar: Tuple[str, ...]
            np_arr: csp.typing.NumpyNDArray[float]

        for use_pydantic in [True, False]:
            looped = MyStruct.from_dict(
                MyStruct(foo=set((9, 10)), bar=("a", "b"), np_arr=np.array([1, 3])).to_dict(), use_pydantic=use_pydantic
            )
            expected = MyStruct(foo=set((9, 10)), bar=("a", "b"), np_arr=np.array([1, 3]))
            self.assertEqual(looped.foo, expected.foo)
            self.assertEqual(looped.bar, expected.bar)
            self.assertTrue(np.all(looped.np_arr == expected.np_arr))

    def test_struct_yaml_serialization(self):
        class S1(csp.Struct):
            i: int
            f: float
            b: bool
            default_i: int = 42

        class S2(csp.Struct):
            value: Tuple[int]
            set_value: Set[str]

        class S(csp.Struct):
            d: Dict[str, S1]
            ls: List[int]
            lc: List[S2]

        input = """
        d:
            s1:
                i: 1
                f: 1.5
                b : true
                default_i: 0 # Comment
            s2:
                i: 2
                f: 2.5
                b: false
        ls:
            - 1
            - 2
            - 3
        lc:
            -
                value: [1,2,3]
                set_value: ["x","y","z"]
            -
                value:
                    - 4
        """

        loaded_struct = S.from_yaml(input)
        dumped_value = loaded_struct.to_yaml()
        loaded_struct2 = S.from_yaml(dumped_value)

        # Sanity check
        self.assertNotEqual(loaded_struct, S())
        self.assertEqual(loaded_struct, loaded_struct2)
        self.assertEqual(
            loaded_struct,
            S(
                d={"s1": S1(i=1, f=1.5, b=True, default_i=0), "s2": S1(i=2, f=2.5, b=False)},
                ls=[1, 2, 3],
                lc=[S2(value=(1, 2, 3), set_value={"x", "y", "z"}), S2(value=(4,))],
            ),
        )

        # test struct with Enum
        class E(csp.Enum):
            A = csp.Enum.auto()
            B = 2
            C = 30

        class S3(csp.Struct):
            i: int
            x: E

        for ev in [E.A, E.B, E.C]:
            s = S3(i=5, x=ev)
            s2 = S3.from_yaml(s.to_yaml())
            self.assertEqual(s2.x, ev)

        s = S3(i=5, x=E.A)
        exp_yaml = "i: 5\nx: A\n"

        self.assertEqual(s.to_yaml(), exp_yaml)

    def test_struct_type_check(self):
        class S1(csp.Struct):
            a: int
            b: dict

        class S2(csp.Struct):
            s: S1

        class FOO(csp.Struct):
            a: int

        with self.assertRaisesRegex(TypeError, "Invalid int type, expected long \\(int\\) got str"):
            S1(a="hey")

        with self.assertRaisesRegex(TypeError, "Invalid dict type, expected dict got list"):
            S1(b=[])

        # was a crash
        with self.assertRaisesRegex(TypeError, "Invalid struct type, expected struct S1 got int"):
            S2(s=1)

        with self.assertRaisesRegex(TypeError, "Invalid struct type, expected struct S1 got FOO"):
            S2(s=FOO())

    def test_list_field_set_iterator(self):
        class S(csp.Struct):
            l: List[object]

        s = S()
        expected = list(range(100))

        s.l = list(range(100))
        self.assertEqual(s.l, expected)

        s.l = range(100)
        self.assertEqual(s.l, expected)

        s.l = set(range(100))
        self.assertEqual(s.l, expected)

        def gen(c, should_raise):
            i = 0
            while i < c:
                yield i
                i += 1
                if should_raise:
                    raise RuntimeError("test")

        s.l = gen(100, False)
        self.assertEqual(s.l, expected)

        with self.assertRaisesRegex(RuntimeError, "test"):
            s.l = gen(100, True)

    def test_interned_fields(self):
        class Foo(csp.Struct):
            f1: int = 5

        f = Foo()
        self.assertEqual(getattr(f, "f1"), 5)
        self.assertEqual(getattr(f, f"f{1}"), 5)

    def test_struct_instance(self):
        """was a crash"""
        with self.assertRaisesRegex(TypeError, "csp.Struct cannot be instantiated"):
            _ = csp.Struct()

    def test_gc(self):
        # attempt to ensure that we are doing GC collection / ref counting correctly
        # had a refcount bug where pystruct would decref structs even if their struct ref
        # had other referrals

        # This test forces a struct through the engine where its stored as a straight c++ StructPtr
        # consume node then boxes it in a python PyStruct which then gets decrefed, which should not impact the struct

        import gc

        class FOO(csp.Struct):
            s: str
            l: list

        @csp.node
        def gen(x: csp.ts[bool]) -> csp.ts[FOO]:
            if csp.ticked(x):
                return FOO(s="test", l=[1, 2, 3, 4])

        @csp.node(memoize=False)
        def consume(x: csp.ts[FOO]):
            if csp.ticked(x):
                self.assertEqual(x.l[:4], [1, 2, 3, 4])
                x.l.append(x)
                del x
                gc.collect()

        def graph():
            x = csp.timer(timedelta(seconds=1), True)
            g = gen(x)
            for _ in range(20):
                consume(g)

        class AssertLeakFree:
            def __enter__(ctx):
                gc.collect()
                ctx._gc_start = len(gc.get_objects())

            def __exit__(ctx, exc_type, exc_val, exc_tb):
                gc.collect()
                gc_end = len(gc.get_objects())
                self.assertLessEqual(gc_end, ctx._gc_start + 1)

        csp.run(graph, starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=5))

        # try to check for mem leaks

        # straight struct with and without cycles
        with AssertLeakFree():
            for _ in range(10000):
                s = FOO(l=[np.ndarray(100000)])

            del s

        with AssertLeakFree():
            for _ in range(10000):
                s = FOO(l=[np.ndarray(100000)])
                s.l.append(s)

            del s

        @csp.node
        def gen2(x: csp.ts[bool]) -> csp.ts[FOO]:
            if csp.ticked(x):
                f = FOO(s="test", l=[np.ndarray(100000)])
                f.l.append(f)
                return f

        @csp.node(memoize=False)
        def consume2(x: csp.ts[FOO]):
            with csp.state():
                s_x = None

            if csp.ticked(x):
                # Keep reference around every3 ticks
                if csp.num_ticks(x) % 3 == 0:
                    s_x = x

        def graph2():
            x = csp.timer(timedelta(seconds=1), True)
            g = gen2(x)
            for _ in range(20):
                consume2(g)

        with AssertLeakFree():
            for _ in range(100):
                csp.run(graph2, starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=1000))

    def test_deprecated_defineStruct(self):
        metadata = {
            "a": float,
            "b": int,
        }
        defaults = {"a": 0.0, "b": 1}
        TestStruct = define_struct("TestStruct", metadata, defaults)
        TestStruct2 = defineStruct("TestStruct", metadata, defaults)
        self.assertEqual(TestStruct.metadata(), TestStruct2.metadata())
        self.assertEqual(TestStruct.__defaults__, TestStruct2.__defaults__)

    def test_define_nested_struct(self):
        metadata = {
            "a": float,
            "b": int,
            "c": {
                "x": MyEnum,
                "y": List[int],
            },
            "d": {"s": object, "t": FastList[object]},
        }

        normalized_metadata = {
            "a": float,
            "b": int,
            "c": {
                "x": MyEnum,
                "y": [int],
            },
            "d": {"s": object, "t": [object, True]},
        }
        TestStruct = define_nested_struct("TestStruct", metadata)
        self.assertEqual(TestStruct.__name__, "TestStruct")
        self.assertEqual(list(TestStruct.metadata().keys()), list(normalized_metadata.keys()))
        self.assertEqual(TestStruct.metadata()["a"], normalized_metadata["a"])
        self.assertEqual(TestStruct.metadata()["b"], normalized_metadata["b"])
        c = TestStruct.metadata()["c"]
        self.assertTrue(issubclass(c, csp.Struct))
        self.assertEqual(c.__name__, "TestStruct_c")
        self.assertEqual(c.metadata(), normalized_metadata["c"])
        d = TestStruct.metadata()["d"]
        self.assertTrue(issubclass(d, csp.Struct))
        self.assertEqual(d.__name__, "TestStruct_d")
        self.assertEqual(d.metadata(), normalized_metadata["d"])

        defaults = {"a": 0.0, "c": {"y": []}, "d": {}}
        TestStruct2 = define_nested_struct("TestStruct2", metadata, defaults)
        s = TestStruct2()
        self.assertEqual(s.a, 0.0)
        self.assertEqual(s.c, s.metadata()["c"]())
        self.assertEqual(s.c.y, [])
        self.assertEqual(s.d, s.metadata()["d"]())

        # Make sure deprecated function still works without raising
        TestStruct3 = defineNestedStruct("TestStruct3", metadata, defaults)
        s = TestStruct3()
        self.assertEqual(s.a, 0.0)
        self.assertEqual(s.c, s.metadata()["c"]())
        self.assertEqual(s.c.y, [])
        self.assertEqual(s.d, s.metadata()["d"]())

    def test_all_fields_set(self):
        types = [int, bool, list, str]
        for num_fields in range(1, 25):
            meta = {chr(ord("a") + x): types[x % len(types)] for x in range(num_fields)}
            stype = define_struct("foo", meta)
            s = stype()
            self.assertFalse(s.all_fields_set())
            keys = list(meta.keys())
            for k in keys[:-1]:
                setattr(s, k, meta[k]())
                self.assertFalse(s.all_fields_set())

            setattr(s, keys[-1], meta[keys[-1]]())
            self.assertTrue(s.all_fields_set())

            # Test derived structs
            meta2 = {k + "2": t for k, t in meta.items()}
            stype2 = define_struct("foo", meta2, base=stype)
            s2 = stype2()
            self.assertFalse(s2.all_fields_set())
            keys = list(stype2.metadata().keys())
            for k in keys[:-1]:
                setattr(s2, k, stype2.metadata()[k]())
                self.assertFalse(s2.all_fields_set())

            setattr(s2, keys[-1], stype2.metadata()[keys[-1]]())
            self.assertTrue(s2.all_fields_set())

    def test_struct_pickle(self):
        import pickle

        foo = StructNoDefaults(b=True, f=42.42, s="test", bt=b"ab\001\000c")
        foo_pickled = pickle.dumps(foo)
        foo_unpickled = pickle.loads(foo_pickled)
        self.assertEqual(foo, foo_unpickled)
        self.assertNotEqual(id(foo), id(foo_unpickled))

        # composable structs
        foo = StructWithStruct(a=123, s=BaseNative(i=456, b=True))
        foo_pickled = pickle.dumps(foo)
        foo_unpickled = pickle.loads(foo_pickled)
        self.assertEqual(foo, foo_unpickled)
        self.assertNotEqual(id(foo), id(foo_unpickled))

        # composable structs w derived type instance ( Was a bug )
        foo = StructWithStruct(a=123, s=DerivedFullyNative(i=456, b=True, f2=123.456))
        foo_pickled = pickle.dumps(foo)
        foo_unpickled = pickle.loads(foo_pickled)
        self.assertEqual(foo, foo_unpickled)
        self.assertNotEqual(id(foo), id(foo_unpickled))

    def test_struct_type_alloc(self):
        for i in range(1000):
            name = f"struct_{i}"
            fieldname = f"field{i}"
            S = define_struct(name, {fieldname: int})
            s = S()
            setattr(s, fieldname, i)
            ts = getattr(csp.const(s), fieldname)
            csp.run(ts, starttime=datetime.utcnow(), endtime=timedelta())

    def test_struct_printing(self):
        # simple test
        class StructA(csp.Struct):
            a: int
            b: str
            c: List[int]

        s1 = StructA(a=1, b="b", c=[1, 2])
        exp_repr_s1 = "StructA( a=1, b=b, c=[1, 2] )"
        self.assertEqual(repr(s1), exp_repr_s1)
        self.assertEqual(str(s1), exp_repr_s1)

        # cover all other types
        class ClassA:
            a = 1

            def __repr__(self):
                return f"ClassA(a={self.a})"

        class EnumA(csp.Enum):
            RED = 1
            BLUE = 2

        class StructB(csp.Struct):
            a: timedelta
            b: datetime
            c: bool
            d: float
            e: ClassA
            f: StructA
            g: EnumA
            h: float  # test a bunch of double cases
            i: float
            j: float
            k: float

        f1 = 1.0
        f2 = 1.23
        f3 = (4 / 3) * 10**9
        f4 = 123.456
        f5 = (4 / 3) * 10 ** (-9)
        s2 = StructB(
            a=timedelta(1), b=datetime(2020, 1, 1), c=False, d=f1, e=ClassA(), f=s1, g=EnumA.RED, h=f2, i=f3, j=f4, k=f5
        )
        exp_repr_s2 = f"StructB( a={repr(timedelta(1))}, b={repr(datetime(2020, 1, 1))}, c=False, d={repr(f1)}, e=ClassA(a=1), f=StructA( a=1, b=b, c=[1, 2] ), g=<EnumA.RED: 1>, h={repr(f2)}, i={repr(f3)}, j={repr(f4)}, k={repr(f5)} )"
        # repr and str are the same for floats
        self.assertEqual(repr(s2), exp_repr_s2)

        # test derived structs
        class StructC(StructA):
            d: int
            e: str

        class StructD(StructC):
            f: int

        s3 = StructC(a=1, b="b", c=[1, 2], d=1, e="e")
        exp_repr_s3 = "StructC( a=1, b=b, c=[1, 2], d=1, e=e )"
        self.assertEqual(repr(s3), exp_repr_s3)

        s4 = StructD(a=1, b="b", c=[1, 2], d=1, e="e", f=2)
        exp_repr_s4 = "StructD( a=1, b=b, c=[1, 2], d=1, e=e, f=2 )"
        self.assertEqual(repr(s4), exp_repr_s4)

        # test structs with struct, struct array fields
        class StructE(csp.Struct):
            a: StructA
            b: List[StructC]

        s5 = StructE(
            a=StructA(a=1, b="b", c=[1, 2]),
            b=[StructC(a=2, b="b", c=[3, 4], d=2, e="e"), StructC(a=3, b="b", c=[5, 6], d=3, e="e")],
        )
        exp_repr_s5 = "StructE( a=StructA( a=1, b=b, c=[1, 2] ), b=[StructC( a=2, b=b, c=[3, 4], d=2, e=e ), StructC( a=3, b=b, c=[5, 6], d=3, e=e )] )"
        self.assertEqual(repr(s5), exp_repr_s5)
        self.assertEqual(str(s5), exp_repr_s5)

        # test array fields
        class StructF(csp.Struct):
            a: List[int]
            b: List[bool]
            c: List[List[float]]
            d: List[ClassA]
            e: List[EnumA]
            f: List[StructC]  # leave unset for test

        # str (called by print) will show unset fields, repr (called in logging) will not
        s6 = StructF(a=[1], b=[True], c=[[1.0]], d=[ClassA()], e=[EnumA.RED, EnumA.BLUE])
        exp_str_s6 = (
            "StructF( a=[1], b=[True], c=[[1.0]], d=[ClassA(a=1)], e=[<EnumA.RED: 1>, <EnumA.BLUE: 2>], f=<unset> )"
        )
        exp_repr_s6 = "StructF( a=[1], b=[True], c=[[1.0]], d=[ClassA(a=1)], e=[<EnumA.RED: 1>, <EnumA.BLUE: 2>] )"
        self.assertEqual(str(s6), exp_str_s6)
        self.assertEqual(repr(s6), exp_repr_s6)

        # test unset in arrays/nested structs
        class StructG(csp.Struct):
            a: StructA
            b: List[StructA]
            c: ClassA

        s7 = StructG(a=StructA(), b=[StructA(), StructA()])
        unset_structA_str = "StructA( a=<unset>, b=<unset>, c=<unset> )"
        exp_str_s7 = f"StructG( a={unset_structA_str}, b=[{unset_structA_str}, {unset_structA_str}], c=<unset> )"
        exp_repr_s7 = "StructG( a=StructA(  ), b=[StructA(  ), StructA(  )] )"
        self.assertEqual(str(s7), exp_str_s7)
        self.assertEqual(repr(s7), exp_repr_s7)

        # test case where a struct field is defined as a base type but the actual instance is a derived type
        class BaseStruct(csp.Struct):
            a: int

        class DerivedStructA(BaseStruct):
            b: int

        class DerivedStructB(BaseStruct):
            c: int

        class TwiceDerivedStruct(DerivedStructA):
            d: int

        class StructH(csp.Struct):
            a: BaseStruct

        s8 = StructH(a=BaseStruct())
        s9 = StructH(a=DerivedStructA(a=1, b=2))
        s10 = StructH(a=DerivedStructB())
        s11 = StructH(a=TwiceDerivedStruct(a=1, b=2, d=3))

        s8_str = "StructH( a=BaseStruct( a=<unset> ) )"
        s9_str_repr = "StructH( a=DerivedStructA( a=1, b=2 ) )"
        s10_repr = "StructH( a=DerivedStructB(  ) )"
        s11_str_repr = "StructH( a=TwiceDerivedStruct( a=1, b=2, d=3 ) )"

        self.assertEqual(str(s8), s8_str)
        self.assertEqual(repr(s9), s9_str_repr)
        self.assertEqual(str(s9), s9_str_repr)
        self.assertEqual(repr(s10), s10_repr)
        self.assertEqual(str(s11), s11_str_repr)
        self.assertEqual(repr(s11), s11_str_repr)

        # test dictionary of structs as a struct field
        class StructI(csp.Struct):
            a: int
            b: str
            c: dict

        class StructK(csp.Struct):
            a: float
            b: float

        class StructJ(csp.Struct):
            a: int
            b: dict
            c: StructK

        s12 = StructI(
            a=1, b="b", c={1: StructJ(a=1, b={2: StructK(a=1.0)}, c=StructK()), 2: StructJ(a=2, b={3: StructK(a=2.0)})}
        )
        s12_str_repr = "StructI( a=1, b=b, c={1: StructJ( a=1, b={2: StructK( a=1.0 )}, c=StructK(  ) ), 2: StructJ( a=2, b={3: StructK( a=2.0 )} )} )"

        # note that unset fields are only shown in str calls for the struct and any nested structs...no good way to get it for structs within a dict field
        self.assertEqual(str(s12), s12_str_repr)
        self.assertEqual(repr(s12), s12_str_repr)

        class StructWBytes(csp.Struct):
            x: str
            y: bytes

        s_bytes = StructWBytes(x="test", y=b"\x9d_@2")
        s_bytes_repr = "StructWBytes( x=test, y=b'\\x9d_@2' )"
        self.assertEqual(str(s_bytes), s_bytes_repr)
        self.assertEqual(repr(s_bytes), s_bytes_repr)

    def test_recursive_repr(self):
        class StructB(csp.Struct):
            x: csp.Struct

        b = StructB()
        b.x = b
        self.assertEqual(repr(b), "StructB( x=( ... ) )")

    def test_disappearing_dynamic_types(self):
        """Was a BUG due to missing refcount: https://github.com/Point72/csp/issues/74"""

        class Outer(csp.Struct):
            s: csp.Struct

        all = []
        for i in range(10000):
            sType = define_struct("foo", {"a": dict})
            all.append(Outer(s=sType(a={"foo": "bar"})))
            repr(all)
            all = all[:100]

    def test_python_conversion_on_nested_base_struct(self):
        """Was a BUG due to the error message in fromPython trying to access the meta name of a base struct class"""

        class A(csp.Struct):
            a: csp.Struct

        # 1) in constructor
        with self.assertRaises(TypeError) as e:
            my_a = A(a=None)

        # 2) setting the member
        with self.assertRaises(TypeError) as e:
            my_a = A()
            my_a.a = None

    def test_bool_array(self):
        """Test [bool] specific functionality since its special cased as vector<uint8> in C++"""

        class A(csp.Struct):
            l: List[bool]

        raw = [True, False, True]
        a = A(l=raw)
        self.assertTrue(all(a.l[i] is raw[i] for i in range(3)))

        r = repr(a)
        self.assertTrue(repr(raw) in r)

    def test_to_dict_recursion(self):
        class MyStruct(csp.Struct):
            l1: list
            l2: list
            d1: dict
            d2: dict
            t1: tuple
            t2: tuple

        test_struct = MyStruct(l1=[1], l2=[2])
        result_dict = {"l1": [1], "l2": [2]}
        self.assertEqual(test_struct.to_dict(), result_dict)

        test_struct = MyStruct(l1=[1], l2=[2])
        test_struct.l1.append(test_struct.l2)
        test_struct.l2.append(test_struct.l1)
        with self.assertRaises(RecursionError):
            test_struct.to_dict()

        test_struct = MyStruct(l1=[1])
        test_struct.l1.append(test_struct.l1)
        with self.assertRaises(RecursionError):
            test_struct.to_dict()

        test_struct = MyStruct(l1=[1])
        test_struct.l1.append(test_struct)
        with self.assertRaises(RecursionError):
            test_struct.to_dict()

        test_struct = MyStruct(d1={1: 1}, d2={2: 2})
        result_dict = {"d1": {1: 1}, "d2": {2: 2}}
        self.assertEqual(test_struct.to_dict(), result_dict)

        test_struct = MyStruct(d1={1: 1}, d2={2: 2})
        test_struct.d1["d2"] = test_struct.d2
        test_struct.d2["d1"] = test_struct.d1
        with self.assertRaises(RecursionError):
            test_struct.to_dict()

        test_struct = MyStruct(d1={1: 1}, d2={2: 2})
        test_struct.d1["d1"] = test_struct.d1
        with self.assertRaises(RecursionError):
            test_struct.to_dict()

        test_struct = MyStruct(d1={1: 1}, d2={2: 2})
        test_struct.d1["d1"] = test_struct
        with self.assertRaises(RecursionError):
            test_struct.to_dict()

        test_struct = MyStruct(t1=(1, 1), t2=(2, 2))
        result_dict = {"t1": (1, 1), "t2": (2, 2)}
        self.assertEqual(test_struct.to_dict(), result_dict)

        test_struct = MyStruct(t1=(1, 1))
        test_struct.t1 = (1, 2, test_struct)
        with self.assertRaises(RecursionError):
            test_struct.to_dict()

    def test_to_dict_postprocess(self):
        class MySubStruct(csp.Struct):
            i: int = 0

            def postprocess_to_dict(self, obj):
                obj["postprocess_called"] = True
                obj["postprocess_val"] = self.i
                return obj

        class MyStruct(csp.Struct):
            i: int = 1
            mss: MySubStruct = MySubStruct()

            def postprocess_to_dict(self, obj):
                obj["postprocess_called"] = True
                obj["postprocess_val"] = self.i
                return obj

        test_struct = MyStruct(i=5)
        result_dict = {
            "i": 5,
            "postprocess_called": True,
            "postprocess_val": 5,
            "mss": {"i": 0, "postprocess_called": True, "postprocess_val": 0},
        }
        self.assertEqual(test_struct.to_dict(), result_dict)

    def test_to_dict_preserve_enums(self):
        class MyEnum(csp.Enum):
            A = 1
            B = 2
            C = 3

        class MySubEnum(csp.Enum):
            SUB_A = 1
            SUB_B = 2
            SUB_C = 3

        class MySubStruct(csp.Struct):
            a: List[MySubEnum]
            b: list[MySubEnum]
            c: list
            d: set
            e: tuple
            f: Dict[str, MySubEnum]
            g: dict
            h: MySubEnum
            i: MyEnum

        class MyStruct(csp.Struct):
            a: List[MyEnum]
            b: list[MyEnum]
            c: list
            d: set
            e: tuple
            f: Dict[str, MyEnum]
            g: dict
            h: MySubEnum
            i: MyEnum
            j: MySubStruct
            k: List[MySubStruct]

        test_sub_struct = MySubStruct(
            a=[MySubEnum.SUB_A, MySubEnum.SUB_B],
            b=[MySubEnum.SUB_B, MySubEnum.SUB_C],
            c=[MySubEnum.SUB_C, MySubEnum.SUB_A],
            d=set([MySubEnum.SUB_A, MySubEnum.SUB_B]),
            e=(MySubEnum.SUB_B, MySubEnum.SUB_C),
            f={"3": MySubEnum.SUB_C, "1": MySubEnum.SUB_A},
            g={"1": MySubEnum.SUB_A, "2": MySubEnum.SUB_B},
            h=MySubEnum.SUB_A,
            i=MyEnum.B,
        )
        result_sub_dict = {
            "a": ["SUB_A", "SUB_B"],
            "b": ["SUB_B", "SUB_C"],
            "c": ["SUB_C", "SUB_A"],
            "d": {"SUB_A", "SUB_B"},
            "e": ("SUB_B", "SUB_C"),
            "f": {"3": "SUB_C", "1": "SUB_A"},
            "g": {"1": "SUB_A", "2": "SUB_B"},
            "h": "SUB_A",
            "i": "B",
        }
        result_sub_dict_preserve_enums = {
            "a": [MySubEnum.SUB_A, MySubEnum.SUB_B],
            "b": [MySubEnum.SUB_B, MySubEnum.SUB_C],
            "c": [MySubEnum.SUB_C, MySubEnum.SUB_A],
            "d": {MySubEnum.SUB_A, MySubEnum.SUB_B},
            "e": (MySubEnum.SUB_B, MySubEnum.SUB_C),
            "f": {"3": MySubEnum.SUB_C, "1": MySubEnum.SUB_A},
            "g": {"1": MySubEnum.SUB_A, "2": MySubEnum.SUB_B},
            "h": MySubEnum.SUB_A,
            "i": MyEnum.B,
        }
        self.assertEqual(test_sub_struct.to_dict(), result_sub_dict)
        self.assertEqual(test_sub_struct.to_dict(preserve_enums=True), result_sub_dict_preserve_enums)

        test_struct = MyStruct(
            a=[MyEnum.A, MyEnum.B],
            b=[MyEnum.B, MyEnum.C],
            c=[MyEnum.C, MyEnum.A],
            d=set([MyEnum.A, MyEnum.B]),
            e=(MyEnum.B, MyEnum.C),
            f={"3": MyEnum.C, "1": MyEnum.A},
            g={"1": MyEnum.A, "2": MyEnum.B},
            h=MySubEnum.SUB_A,
            i=MyEnum.B,
            j=test_sub_struct,
            k=[test_sub_struct, test_sub_struct],
        )
        result_dict = {
            "a": ["A", "B"],
            "b": ["B", "C"],
            "c": ["C", "A"],
            "d": {"A", "B"},
            "e": ("B", "C"),
            "f": {"3": "C", "1": "A"},
            "g": {"1": "A", "2": "B"},
            "h": "SUB_A",
            "i": "B",
            "j": result_sub_dict,
            "k": [result_sub_dict, result_sub_dict],
        }
        result_dict_preserve_enums = {
            "a": [MyEnum.A, MyEnum.B],
            "b": [MyEnum.B, MyEnum.C],
            "c": [MyEnum.C, MyEnum.A],
            "d": {MyEnum.A, MyEnum.B},
            "e": (MyEnum.B, MyEnum.C),
            "f": {"3": MyEnum.C, "1": MyEnum.A},
            "g": {"1": MyEnum.A, "2": MyEnum.B},
            "h": MySubEnum.SUB_A,
            "i": MyEnum.B,
            "j": result_sub_dict_preserve_enums,
            "k": [result_sub_dict_preserve_enums, result_sub_dict_preserve_enums],
        }
        self.assertEqual(test_struct.to_dict(), result_dict)
        self.assertEqual(test_struct.to_dict(preserve_enums=True), result_dict_preserve_enums)

    def test_to_json_primitives(self):
        class MyStruct(csp.Struct):
            b: bool = True
            i: int = 123
            f: float = 3.14
            s: str = "456"

        test_struct = MyStruct()
        result_dict = {"b": True, "i": 123, "f": 3.14, "s": "456"}
        self.assertEqual(json.loads(test_struct.to_json()), result_dict)

        test_struct = MyStruct(b=False, i=456, f=1.73, s="789")
        result_dict = {"b": False, "i": 456, "f": 1.73, "s": "789"}
        self.assertEqual(json.loads(test_struct.to_json()), result_dict)

        test_struct = MyStruct(b=False, i=456, f=float("nan"), s="789")
        result_dict = {"b": False, "i": 456, "f": None, "s": "789"}
        self.assertEqual(json.loads(test_struct.to_json()), result_dict)

        test_struct = MyStruct(b=False, i=456, f=float("inf"), s="789")
        result_dict = {"b": False, "i": 456, "f": None, "s": "789"}
        self.assertEqual(json.loads(test_struct.to_json()), result_dict)

        test_struct = MyStruct(b=False, i=456, f=float("-inf"), s="789")
        result_dict = {"b": False, "i": 456, "f": None, "s": "789"}
        self.assertEqual(json.loads(test_struct.to_json()), result_dict)

    def test_to_json_enums(self):
        from enum import Enum as PyEnum

        class MyPyEnum(PyEnum):
            PyEnumA = 1
            PyEnumB = 2
            PyEnumC = 3
            NumPyEnum = 4

        def callback(obj):
            if isinstance(obj, PyEnum):
                return obj.name
            raise RuntimeError("Invalid type for callback")

        class MyCspEnum(csp.Enum):
            CspEnumA = 1
            CspEnumB = 2
            CspEnumC = 3
            NumCspEnum = 4

        class MyStruct(csp.Struct):
            i: int = 123
            csp_e: MyCspEnum
            py_e: MyPyEnum

        test_struct = MyStruct()
        result_dict = {"i": 123}
        self.assertEqual(json.loads(test_struct.to_json()), result_dict)

        test_struct = MyStruct(i=456, csp_e=MyCspEnum.CspEnumC)
        result_dict = {"i": 456, "csp_e": "CspEnumC"}
        self.assertEqual(json.loads(test_struct.to_json()), result_dict)
        self.assertEqual(json.loads(test_struct.to_json(callback)), result_dict)

        test_struct = MyStruct(i=456, py_e=MyPyEnum.PyEnumC)
        result_dict = {"i": 456, "py_e": "PyEnumC"}
        self.assertEqual(json.loads(test_struct.to_json(callback)), result_dict)

        test_struct = MyStruct(i=456, csp_e=MyCspEnum.CspEnumA, py_e=MyPyEnum.PyEnumB)
        result_dict = {"i": 456, "csp_e": "CspEnumA", "py_e": "PyEnumB"}
        self.assertEqual(json.loads(test_struct.to_json(callback)), result_dict)

    def test_to_json_datetime(self):
        class MyStruct(csp.Struct):
            i: int = 123
            dt: datetime
            d: date
            t: time
            td: timedelta

        test_struct = MyStruct()
        result_dict = {"i": 123}
        self.assertEqual(json.loads(test_struct.to_json()), result_dict)

        dt = None
        test_struct = MyStruct(i=456, dt=dt, d=None, t=None, td=None)
        result_dict = {"i": 456, "dt": None, "d": None, "t": None, "td": None}
        self.assertEqual(json.loads(test_struct.to_json()), result_dict)

        dt = datetime(2024, 3, 8)
        dt_utc = datetime(2024, 3, 8, tzinfo=pytz.utc)
        test_struct = MyStruct(i=456, dt=dt)
        result_dict = json.loads(test_struct.to_json())
        self.assertEqual(datetime.fromisoformat(result_dict["dt"]), dt_utc)

        dt = datetime.now(tz=pytz.utc)
        test_struct = MyStruct(i=456, dt=dt)
        result_dict = json.loads(test_struct.to_json())
        self.assertEqual(datetime.fromisoformat(result_dict["dt"]), dt)

        dt = datetime.now(tz=pytz.timezone("America/New_York"))
        dt_utc = dt.astimezone(pytz.utc)
        test_struct = MyStruct(i=456, dt=dt)
        result_dict = json.loads(test_struct.to_json())
        self.assertEqual(datetime.fromisoformat(result_dict["dt"]), dt_utc)
        self.assertEqual(datetime.fromisoformat(result_dict["dt"]), dt)

        d = date(2024, 3, 8)
        test_struct = MyStruct(i=456, d=d)
        result_dict = {"i": 456, "d": "2024-03-08"}
        self.assertEqual(json.loads(test_struct.to_json()), result_dict)

        t = time(5, 10, 12, 500)
        test_struct = MyStruct(i=456, t=t)
        result_dict = {"i": 456, "t": "05:10:12.000500"}
        self.assertEqual(json.loads(test_struct.to_json()), result_dict)

        td = timedelta(days=1, seconds=68400.05)
        seconds = td.total_seconds()
        microseconds = round((seconds - int(seconds)) * 1e6)
        test_struct = MyStruct(i=456, td=td)
        result_dict = {"i": 456, "td": f"{int(seconds):+}.{microseconds:06}"}
        self.assertEqual(json.loads(test_struct.to_json()), result_dict)

        td = timedelta(days=-1, seconds=68400.05)
        seconds = td.total_seconds()
        microseconds = abs(round((seconds - int(seconds)) * 1e6))
        test_struct = MyStruct(i=456, td=td)
        result_dict = {"i": 456, "td": f"{int(seconds):+}.{microseconds:06}"}
        self.assertEqual(json.loads(test_struct.to_json()), result_dict)

    def test_to_json_list(self):
        class MyStruct(csp.Struct):
            i: int = 123
            l_i: List[int]
            l_b: List[bool]
            l_dt: List[datetime]
            l_l_i: List[List[int]]
            l_tuple: Tuple[int, float, str]
            l_any: list

        test_struct = MyStruct()
        result_dict = {"i": 123}
        self.assertEqual(json.loads(test_struct.to_json()), result_dict)

        l_i = [1, 2, 3]
        test_struct = MyStruct(i=456, l_i=l_i)
        result_dict = {"i": 456, "l_i": l_i}
        self.assertEqual(json.loads(test_struct.to_json()), result_dict)

        l_b = [True, True, False]
        test_struct = MyStruct(i=456, l_b=l_b)
        result_dict = {"i": 456, "l_b": l_b}
        self.assertEqual(json.loads(test_struct.to_json()), result_dict)

        dt = datetime.now(tz=pytz.timezone("Europe/London"))
        l_dt = [dt, dt]
        test_struct = MyStruct(i=456, l_dt=l_dt)
        result_dict = json.loads(test_struct.to_json())
        self.assertEqual([datetime.fromisoformat(d_str) for d_str in result_dict["l_dt"]], l_dt)

        l_l_i = [[1, 2], [3, 4]]
        test_struct = MyStruct(i=456, l_l_i=l_l_i)
        result_dict = {"i": 456, "l_l_i": l_l_i}
        self.assertEqual(json.loads(test_struct.to_json()), result_dict)

        l_tuple = (1, 3.14, "hello world")
        test_struct = MyStruct(i=456, l_tuple=l_tuple)
        result_dict = {"i": 456, "l_tuple": list(l_tuple)}
        self.assertEqual(json.loads(test_struct.to_json()), result_dict)

        l_i = [1, 2, 3]
        test_struct = MyStruct(i=456, l_any=l_i)
        result_dict = {"i": 456, "l_any": l_i}
        self.assertEqual(json.loads(test_struct.to_json()), result_dict)

        dt = datetime.now(tz=pytz.timezone("Europe/London"))
        l_dt = [dt, dt]
        test_struct = MyStruct(i=456, l_any=l_dt)
        result_dict = json.loads(test_struct.to_json())
        self.assertEqual([datetime.fromisoformat(d_str) for d_str in result_dict["l_any"]], l_dt)

        l_l_i = [[1, 2], [3, 4]]
        test_struct = MyStruct(i=456, l_any=l_l_i)
        result_dict = {"i": 456, "l_any": l_l_i}
        self.assertEqual(json.loads(test_struct.to_json()), result_dict)

        l_any = [[1, float("nan")], [float("INFINITY"), float("-inf")]]
        test_struct = MyStruct(i=456, l_any=l_any)
        result_dict = {"i": 456, "l_any": [[1, None], [None, None]]}
        self.assertEqual(json.loads(test_struct.to_json()), result_dict)

        l_any = [[None], None, [1, 2, None]]
        test_struct = MyStruct(i=456, l_any=l_any)
        result_dict = {"i": 456, "l_any": l_any}
        self.assertEqual(json.loads(test_struct.to_json()), result_dict)

        l_any = [[1, 2], "hello", [4, 3.2, [6, [7], (None, True, 10.5, (11, [float("nan"), None, False]))]]]
        l_any_result = [[1, 2], "hello", [4, 3.2, [6, [7], [None, True, 10.5, [11, [None, None, False]]]]]]
        test_struct = MyStruct(i=456, l_any=l_any)
        result_dict = {"i": 456, "l_any": l_any_result}
        self.assertEqual(json.loads(test_struct.to_json()), result_dict)

    def test_to_json_dict(self):
        class MyStruct(csp.Struct):
            i: int = 123
            d_i: Dict[int, int]
            d_f: Dict[float, int]
            d_dt: Dict[str, datetime]
            d_d_s: Dict[str, Dict[str, str]]
            d_any: dict

        test_struct = MyStruct()
        result_dict = {"i": 123}
        self.assertEqual(json.loads(test_struct.to_json()), result_dict)

        d_i = {1: 2, 3: 4, 5: 6}
        d_i_res = {str(k): v for k, v in d_i.items()}
        test_struct = MyStruct(i=456, d_i=d_i)
        result_dict = {"i": 456, "d_i": d_i_res}
        self.assertEqual(json.loads(test_struct.to_json()), result_dict)

        d_f = {1.2: 2, 2.3: 4, 3.4: 6, 4.5: 7}
        d_f_res = {str(k): v for k, v in d_f.items()}
        test_struct = MyStruct(i=456, d_f=d_f)
        result_dict = {"i": 456, "d_f": d_f_res}
        self.assertEqual(json.loads(test_struct.to_json()), result_dict)

        dt = datetime.now(tz=pytz.utc)
        d_dt = {"d1": dt, "d2": dt}
        test_struct = MyStruct(i=456, d_dt=d_dt)
        result_dict = json.loads(test_struct.to_json())
        self.assertEqual({k: datetime.fromisoformat(d) for k, d in result_dict["d_dt"].items()}, d_dt)

        d_d_s = {"b1": {"d1": "k1", "d2": "k2"}, "b2": {"d3": "k3", "d4": "k4"}}
        test_struct = MyStruct(i=456, d_d_s=d_d_s)
        result_dict = {"i": 456, "d_d_s": d_d_s}
        self.assertEqual(json.loads(test_struct.to_json()), result_dict)

        d_i = {1: 2, 3: 4, 5: 6}
        d_i_res = {str(k): v for k, v in d_i.items()}
        test_struct = MyStruct(i=456, d_any=d_i)
        result_dict = {"i": 456, "d_any": d_i_res}
        self.assertEqual(json.loads(test_struct.to_json()), result_dict)

        d_f = {1.2: 2, 2.3: 4, 3.4: 6, 4.5: 7}
        d_f_res = {str(k): v for k, v in d_f.items()}
        test_struct = MyStruct(i=456, d_any=d_f)
        result_dict = {"i": 456, "d_any": d_f_res}
        self.assertEqual(json.loads(test_struct.to_json()), result_dict)

        dt = datetime.now(tz=pytz.utc)
        d_dt = {"d1": dt, "d2": dt}
        test_struct = MyStruct(i=456, d_any=d_dt)
        result_dict = json.loads(test_struct.to_json())
        self.assertEqual({k: datetime.fromisoformat(d) for k, d in result_dict["d_any"].items()}, d_dt)

        d_any = {
            "b1": {1: "k1", "d2": {4: 5.5}},
            "b2": {"d3": {}, "d4": {"d5": {"d6": {"d7": {}}}}, "d8": None},
            "b3": None,
        }
        d_any_res = {
            "b1": {"1": "k1", "d2": {"4": 5.5}},
            "b2": {"d3": {}, "d4": {"d5": {"d6": {"d7": {}}}}, "d8": None},
            "b3": None,
        }
        test_struct = MyStruct(i=456, d_any=d_any)
        result_dict = {"i": 456, "d_any": d_any_res}
        self.assertEqual(json.loads(test_struct.to_json()), result_dict)

        # Special floats not supported as keys
        d_f = {float("nan"): 2, 2.3: 4, 3.4: 6, 4.5: 7}
        d_f_res = {str(k): v for k, v in d_f.items()}
        test_struct = MyStruct(i=456, d_any=d_f)
        with self.assertRaises(ValueError):
            test_struct.to_json()

        d_f = {float("inf"): 2, 2.3: 4, 3.4: 6, 4.5: 7}
        d_f_res = {str(k): v for k, v in d_f.items()}
        test_struct = MyStruct(i=456, d_any=d_f)
        with self.assertRaises(ValueError):
            test_struct.to_json()

        d_f = {float("-inf"): 2, 2.3: 4, 3.4: 6, 4.5: 7}
        d_f_res = {str(k): v for k, v in d_f.items()}
        test_struct = MyStruct(i=456, d_any=d_f)
        with self.assertRaises(ValueError):
            test_struct.to_json()

        # None as key
        d_none = {
            None: 2,
        }
        d_none_res = {"null": 2}
        test_struct = MyStruct(i=456, d_any=d_none)
        result_dict = {"i": 456, "d_any": d_none_res}
        self.assertEqual(json.loads(test_struct.to_json()), result_dict)

        # Bool as key
        d_bool = {True: 2, False: "abc"}
        d_bool_res = {str(k): v for k, v in d_bool.items()}
        test_struct = MyStruct(i=456, d_any=d_bool)
        result_dict = {"i": 456, "d_any": d_bool_res}
        self.assertEqual(json.loads(test_struct.to_json()), result_dict)

        # Datetime as key
        dt = datetime.now(tz=pytz.utc)
        d_datetime = {dt: "datetime"}
        test_struct = MyStruct(i=456, d_any=d_datetime)
        result_dict = json.loads(test_struct.to_json())
        self.assertEqual({datetime.fromisoformat(k): v for k, v in result_dict["d_any"].items()}, d_datetime)

        dt = datetime.now(tz=pytz.utc)
        d_datetime = {dt.date(): "date"}
        test_struct = MyStruct(i=456, d_any=d_datetime)
        result_dict = json.loads(test_struct.to_json())
        self.assertEqual({date.fromisoformat(k): v for k, v in result_dict["d_any"].items()}, d_datetime)

        dt = datetime.now(tz=pytz.utc)
        d_datetime = {dt.time(): "time"}
        test_struct = MyStruct(i=456, d_any=d_datetime)
        result_dict = json.loads(test_struct.to_json())
        self.assertEqual({time.fromisoformat(k): v for k, v in result_dict["d_any"].items()}, d_datetime)

        # csp.Enum as key
        class MyCspEnum(csp.Enum):
            KEY1 = csp.Enum.auto()
            KEY2 = csp.Enum.auto()
            KEY3 = csp.Enum.auto()

        d_csp_enum = {MyCspEnum.KEY1: "key1", MyCspEnum.KEY2: "key2", MyCspEnum.KEY3: "key3"}
        d_csp_enum_res = {k.name: v for k, v in d_csp_enum.items()}
        test_struct = MyStruct(i=456, d_any=d_csp_enum)
        result_dict = {"i": 456, "d_any": d_csp_enum_res}
        self.assertEqual(json.loads(test_struct.to_json()), result_dict)

        # enum as key
        class MyPyEnum(enum.Enum):
            KEY1 = enum.auto()
            KEY2 = enum.auto()
            KEY3 = enum.auto()

        d_py_enum = {MyPyEnum.KEY1: "key1", MyPyEnum.KEY2: "key2", MyPyEnum.KEY3: "key3"}
        d_py_enum_res = {k.name: v for k, v in d_csp_enum.items()}
        test_struct = MyStruct(i=456, d_any=d_py_enum)
        result_dict = {"i": 456, "d_any": d_py_enum_res}
        self.assertEqual(json.loads(test_struct.to_json()), result_dict)

    def test_to_json_struct(self):
        class MySubSubStruct(csp.Struct):
            b: bool = True
            i: int = 123
            f: float = 3.14
            s: str = "MySubSubStruct"

        class MySubStruct(csp.Struct):
            b: bool = True
            i: int = 456
            f: float = 2.71
            s: str = "MySubStruct"
            msss: MySubSubStruct = MySubSubStruct()

        class MyStruct(csp.Struct):
            i: int = 789
            s: str = "MyStruct"
            mss: MySubStruct = MySubStruct()
            msss: MySubSubStruct

        test_struct = MySubSubStruct()
        result_dict = {"b": True, "i": 123, "f": 3.14, "s": "MySubSubStruct"}
        self.assertEqual(json.loads(test_struct.to_json()), result_dict)

        test_struct = MySubStruct()
        result_dict = {
            "b": True,
            "i": 456,
            "f": 2.71,
            "s": "MySubStruct",
            "msss": {"b": True, "i": 123, "f": 3.14, "s": "MySubSubStruct"},
        }
        self.assertEqual(json.loads(test_struct.to_json()), result_dict)

        test_struct = MyStruct()
        result_dict = {
            "i": 789,
            "s": "MyStruct",
            "mss": {
                "b": True,
                "i": 456,
                "f": 2.71,
                "s": "MySubStruct",
                "msss": {"b": True, "i": 123, "f": 3.14, "s": "MySubSubStruct"},
            },
        }
        self.assertEqual(json.loads(test_struct.to_json()), result_dict)

        msss = MySubSubStruct(s="MySubSubStructNew")
        mss = MySubStruct(s="MySubStructNew", msss=msss)
        ms = MyStruct(s="MyStructNew", mss=mss, msss=msss)

        result_dict = {"b": True, "i": 123, "f": 3.14, "s": "MySubSubStructNew"}
        self.assertEqual(json.loads(msss.to_json()), result_dict)

        result_dict = {
            "b": True,
            "i": 456,
            "f": 2.71,
            "s": "MySubStructNew",
            "msss": {"b": True, "i": 123, "f": 3.14, "s": "MySubSubStructNew"},
        }
        self.assertEqual(json.loads(mss.to_json()), result_dict)

        result_dict = {
            "i": 789,
            "s": "MyStructNew",
            "mss": {
                "b": True,
                "i": 456,
                "f": 2.71,
                "s": "MySubStructNew",
                "msss": {"b": True, "i": 123, "f": 3.14, "s": "MySubSubStructNew"},
            },
            "msss": {"b": True, "i": 123, "f": 3.14, "s": "MySubSubStructNew"},
        }
        self.assertEqual(json.loads(ms.to_json()), result_dict)

    def test_to_json_all(self):
        class MyEnum(csp.Enum):
            A = 1
            B = 2
            C = 3
            NUM = 4

        class NonCspStruct:
            pass

        class MySubSubStruct(csp.Struct):
            ncsp: NonCspStruct
            nparray: np.ndarray
            myenum: MyEnum

        class MySubStruct(csp.Struct):
            d_s_msss: dict
            l_ncsp: List[NonCspStruct]
            py_l_ncsp: list

        class MyStruct(csp.Struct):
            i: int = 789
            s: str = "MyStruct"
            ts: datetime
            l_mss: List[MySubStruct]
            l_msss: list
            d_i_ncsp: dict

        def custom_jsonifier(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, NonCspStruct):
                return {"ncsp_key": "ncsp_value", "ncsp_arr": np.array(["ncsp_arr_val1", "ncsp_arr_val2"])}
            else:
                return obj

        test_struct = MyStruct()
        result_dict = {"i": 789, "s": "MyStruct"}
        self.assertEqual(json.loads(test_struct.to_json()), result_dict)

        enum1 = MyEnum.A
        enum2 = MyEnum.C
        enum3 = MyEnum.NUM

        ncsp1 = NonCspStruct()
        ncsp2 = NonCspStruct()
        ncsp3 = NonCspStruct()

        msss1 = MySubSubStruct(ncsp=ncsp1, myenum=enum1)
        msss2 = MySubSubStruct(myenum=enum2)
        msss3 = MySubSubStruct(nparray=np.array([1, 9]), myenum=enum3)
        msss4 = MySubSubStruct(ncsp=ncsp2, nparray=np.array([1, 9]))

        mss1 = MySubStruct(d_s_msss={"msss1": msss1}, l_ncsp=[ncsp1, ncsp2], py_l_ncsp=[ncsp2, ncsp3])
        mss2 = MySubStruct(d_s_msss={"msss2": msss2}, l_ncsp=[ncsp2], py_l_ncsp=[ncsp3])

        dt = datetime.now(tz=pytz.utc)
        ms = MyStruct(
            i=123456789,
            s="NewMyStruct",
            ts=dt,
            l_mss=[mss2, mss1],
            l_msss=[msss3, msss1, msss2, msss4],
            d_i_ncsp={1: ncsp1, 2: ncsp2, 3: ncsp3},
        )
        test_struct = ms
        result_dict_ncsp1 = {"ncsp_key": "ncsp_value", "ncsp_arr": ["ncsp_arr_val1", "ncsp_arr_val2"]}
        result_dict_ncsp2 = {"ncsp_key": "ncsp_value", "ncsp_arr": ["ncsp_arr_val1", "ncsp_arr_val2"]}
        result_dict_ncsp3 = {"ncsp_key": "ncsp_value", "ncsp_arr": ["ncsp_arr_val1", "ncsp_arr_val2"]}
        result_dict_msss1 = {"ncsp": result_dict_ncsp1, "myenum": "A"}
        result_dict_msss2 = {"myenum": "C"}
        result_dict_msss3 = {"nparray": [1, 9], "myenum": "NUM"}
        result_dict_msss4 = {"ncsp": result_dict_ncsp2, "nparray": [1, 9]}

        result_dict_mss1 = {
            "d_s_msss": {"msss1": result_dict_msss1},
            "l_ncsp": [result_dict_ncsp1, result_dict_ncsp2],
            "py_l_ncsp": [result_dict_ncsp2, result_dict_ncsp3],
        }
        result_dict_mss2 = {
            "d_s_msss": {"msss2": result_dict_msss2},
            "l_ncsp": [result_dict_ncsp2],
            "py_l_ncsp": [result_dict_ncsp3],
        }
        result_dict_ms = {
            "i": 123456789,
            "s": "NewMyStruct",
            "ts": dt,
            "l_mss": [result_dict_mss2, result_dict_mss1],
            "l_msss": [result_dict_msss3, result_dict_msss1, result_dict_msss2, result_dict_msss4],
            "d_i_ncsp": {"1": result_dict_ncsp1, "2": result_dict_ncsp2, "3": result_dict_ncsp3},
        }

        self.assertEqual(json.loads(msss1.to_json(custom_jsonifier)), result_dict_msss1)
        self.assertEqual(json.loads(msss2.to_json(custom_jsonifier)), result_dict_msss2)
        self.assertEqual(json.loads(msss3.to_json(custom_jsonifier)), result_dict_msss3)
        self.assertEqual(json.loads(msss4.to_json(custom_jsonifier)), result_dict_msss4)

        self.assertEqual(json.loads(mss1.to_json(custom_jsonifier)), result_dict_mss1)
        self.assertEqual(json.loads(mss2.to_json(custom_jsonifier)), result_dict_mss2)

        result = json.loads(test_struct.to_json(custom_jsonifier))
        result["ts"] = datetime.fromisoformat(result["ts"])
        self.assertEqual(result, result_dict_ms)

    def test_to_json_callback(self):
        class MyExceptionNonCspStruct:
            pass

        class MyExceptionStruct(csp.Struct):
            e: MyExceptionNonCspStruct

        class MyNonCspStruct:
            s: str = "NonCspStruct"

        class MyStruct(csp.Struct):
            i: int = 123
            narray: np.ndarray
            mncsps: MyNonCspStruct

        def custom_jsonifier(obj):
            if isinstance(obj, np.ndarray):
                a = obj.tolist()
                return a
            elif isinstance(obj, MyNonCspStruct):
                return {"s": obj.s}
            elif isinstance(obj, MyExceptionNonCspStruct):
                raise TypeError("Testing exceptions in callback")
            else:
                raise Exception("Obj type cannot be jsonified")

        test_struct = MyStruct()
        result_dict = {"i": 123}
        self.assertEqual(json.loads(test_struct.to_json()), result_dict)

        narray = np.array([1, 9])
        test_struct = MyStruct(narray=narray)
        result_dict = {"i": 123, "narray": narray}
        result_dict_custom = {"i": 123, "narray": narray.tolist()}
        self.assertEqual(json.loads(test_struct.to_json(custom_jsonifier)), result_dict_custom)
        with self.assertRaises(ValueError):
            test_struct.to_json()

        narray = np.eye(10)
        test_struct = MyStruct(narray=narray)
        result_dict = {"i": 123, "narray": narray}
        result_dict_custom = {"i": 123, "narray": narray.tolist()}
        with self.assertRaises(ValueError):
            self.assertEqual(json.loads(test_struct.to_json()), result_dict)
        self.assertEqual(json.loads(test_struct.to_json(custom_jsonifier)), result_dict_custom)

        mncsps = MyNonCspStruct()
        test_struct = MyStruct(mncsps=mncsps)
        result_dict = {"i": 123, "mncsps": mncsps}
        result_dict_custom = {"i": 123, "mncsps": {"s": mncsps.s}}
        with self.assertRaises(ValueError):
            self.assertEqual(json.loads(test_struct.to_json()), result_dict)
        self.assertEqual(json.loads(test_struct.to_json(custom_jsonifier)), result_dict_custom)

        mncsps = MyNonCspStruct()
        mncsps.s = "NewNonCspStruct"
        mncsps.s_dynamic = "DynamicMetadata"
        test_struct = MyStruct(mncsps=mncsps)
        result_dict = {"i": 123, "mncsps": mncsps}
        result_dict_custom = {"i": 123, "mncsps": {"s": mncsps.s}}
        with self.assertRaises(ValueError):
            self.assertEqual(json.loads(test_struct.to_json()), result_dict)
        self.assertEqual(json.loads(test_struct.to_json(custom_jsonifier)), result_dict_custom)

        test_struct = MyExceptionStruct(e=MyExceptionNonCspStruct())
        with self.assertRaises(TypeError):
            json.loads(test_struct.to_json(custom_jsonifier))

    def test_list_field_append(self):
        """Was a BUG when the struct with list field was not recognizing changes made to this field in python"""
        for ann_typ in struct_list_annotation_types:
            for typ, v in struct_list_test_values.items():

                class A(csp.Struct):
                    a: ann_typ[typ]

                s = A(a=[])
                s.a.append(v[0])

                self.assertEqual(s.a, [v[0]])

                s.a.append(v[1])
                s.a.append(v[2])

                self.assertEqual(s.a, [v[0], v[1], v[2]])

                # Check if not generic type
                if v[-1] is not None:
                    with self.assertRaises(TypeError) as e:
                        s.a.append(v[-1])

    def test_list_field_insert(self):
        """Was a BUG when the struct with list field was not recognizing changes made to this field in python"""
        for ann_typ in struct_list_annotation_types:
            for typ, v in struct_list_test_values.items():

                class A(csp.Struct):
                    a: ann_typ[typ]

                s = A(a=[])
                s.a.insert(0, v[0])

                self.assertEqual(s.a, [v[0]])

                s.a.insert(1, v[1])
                s.a.insert(1, v[2])

                self.assertEqual(s.a, [v[0], v[2], v[1]])

                s.a.insert(-1, v[3])

                self.assertEqual(s.a, [v[0], v[2], v[3], v[1]])

                s.a.insert(100, v[4])
                s.a.insert(-100, v[5])

                self.assertEqual(s.a, [v[5], v[0], v[2], v[3], v[1], v[4]])

                # Check if not generic type
                if v[-1] is not None:
                    with self.assertRaises(TypeError) as e:
                        s.a.insert(-1, v[-1])

    def test_list_field_pop(self):
        """Was a BUG when the struct with list field was not recognizing changes made to this field in python"""
        for ann_typ in struct_list_annotation_types:
            for typ, v in struct_list_test_values.items():

                class A(csp.Struct):
                    a: ann_typ[typ]

                s = A(a=[v[0], v[1], v[2], v[3], v[4]])
                b = s.a.pop()

                self.assertEqual(s.a, [v[0], v[1], v[2], v[3]])
                self.assertEqual(b, v[4])

                b = s.a.pop(-1)

                self.assertEqual(s.a, [v[0], v[1], v[2]])
                self.assertEqual(b, v[3])

                b = s.a.pop(1)

                self.assertEqual(s.a, [v[0], v[2]])
                self.assertEqual(b, v[1])

                with self.assertRaises(IndexError) as e:
                    s.a.pop()
                    s.a.pop()
                    s.a.pop()

                s = A(a=[v[0], v[1], v[2], v[3], v[4]])

                b = s.a.pop(-3)

                self.assertEqual(s.a, [v[0], v[1], v[3], v[4]])
                self.assertEqual(b, v[2])

                with self.assertRaises(IndexError) as e:
                    s.a.pop(-5)

                with self.assertRaises(IndexError) as e:
                    s.a.pop(4)

    def test_list_field_set_item(self):
        """Was a BUG when the struct with list field was not recognizing changes made to this field in python"""
        for ann_typ in struct_list_annotation_types:
            for typ, v in struct_list_test_values.items():

                class A(csp.Struct):
                    a: ann_typ[typ]

                s = A(a=[v[0], v[1], v[2]])
                s.a.__setitem__(0, v[3])

                self.assertEqual(s.a, [v[3], v[1], v[2]])

                s.a[1] = v[4]

                self.assertEqual(s.a, [v[3], v[4], v[2]])

                s.a[-1] = v[5]

                self.assertEqual(s.a, [v[3], v[4], v[5]])

                with self.assertRaises(IndexError) as e:
                    s.a[100] = v[0]

                with self.assertRaises(IndexError) as e:
                    s.a[-100] = v[0]

                s.a[5:6] = [v[0], v[0], v[0]]

                self.assertEqual(s.a, [v[3], v[4], v[5], v[0], v[0], v[0]])

                s.a[4:] = [v[1], v[2]]

                self.assertEqual(s.a, [v[3], v[4], v[5], v[0], v[1], v[2]])

                s.a[2:4] = [v[2]]

                self.assertEqual(s.a, [v[3], v[4], v[2], v[1], v[2]])

                s.a[3:5] = [v[0], v[5]]

                self.assertEqual(s.a, [v[3], v[4], v[2], v[0], v[5]])

                s.a[1:10:2] = [v[1], v[2]]

                self.assertEqual(s.a, [v[3], v[1], v[2], v[2], v[5]])

                # Check if not str or generic type (as str is a sequence of str)
                if v[-2] is not None:
                    with self.assertRaises(TypeError) as e:
                        s.a[1:4] = v[-2]

                self.assertEqual(s.a, [v[3], v[1], v[2], v[2], v[5]])

                # Check if not generic type
                if v[-1] is not None:
                    with self.assertRaises(TypeError) as e:
                        s.a[1:4] = [v[-1]]

                self.assertEqual(s.a, [v[3], v[1], v[2], v[2], v[5]])

                with self.assertRaises(ValueError) as e:
                    s.a[1:10:2] = [v[0]]

                self.assertEqual(s.a, [v[3], v[1], v[2], v[2], v[5]])

                s.a[:2:-1] = [v[3], v[4]]

                self.assertEqual(s.a, [v[3], v[1], v[2], v[4], v[3]])

                with self.assertRaises(ValueError) as e:
                    s.a[-1:1:-2] = [v[0]]

                s.a[-1:1:-2] = [v[0], v[5]]

                self.assertEqual(s.a, [v[3], v[1], v[5], v[4], v[0]])

    def test_list_field_reverse(self):
        """Was a BUG when the struct with list field was not recognizing changes made to this field in python"""
        for ann_typ in struct_list_annotation_types:
            for typ, v in struct_list_test_values.items():

                class A(csp.Struct):
                    a: ann_typ[typ]

                s = A(a=[v[0], v[1], v[2], v[3]])
                s.a.reverse()

                self.assertEqual(s.a, [v[3], v[2], v[1], v[0]])

    def test_list_field_sort(self):
        """Was a BUG when the struct with list field was not recognizing changes made to this field in python"""
        # Not using pystruct_list_test_values, as sort() tests are of different semantics (order and sorting key existance matters).
        values = {
            int: [1, 5, 2, 2, -1, -5, "s"],
            float: [1.4, 5.2, 2.7, 2.7, -1.4, -5.2, "s"],
            datetime: [
                datetime(2022, 12, 6, 1, 2, 3),
                datetime(2022, 12, 8, 3, 2, 3),
                datetime(2022, 12, 7, 2, 2, 3),
                datetime(2022, 12, 7, 2, 2, 3),
                datetime(2022, 12, 5, 2, 2, 3),
                datetime(2022, 12, 3, 2, 2, 3),
                None,
            ],
            timedelta: [
                timedelta(seconds=1),
                timedelta(seconds=123),
                timedelta(seconds=12),
                timedelta(seconds=12),
                timedelta(seconds=0.1),
                timedelta(seconds=0.01),
                None,
            ],
            date: [
                date(2022, 12, 6),
                date(2022, 12, 8),
                date(2022, 12, 7),
                date(2022, 12, 7),
                date(2022, 12, 5),
                date(2022, 12, 3),
                None,
            ],
            time: [time(5, 2, 3), time(7, 2, 3), time(6, 2, 3), time(6, 2, 3), time(4, 2, 3), time(3, 2, 3), None],
            str: ["s", "xyz", "w", "w", "bds", "a", None],
        }

        for ann_typ in struct_list_annotation_types:
            for typ, v in values.items():

                class A(csp.Struct):
                    a: ann_typ[typ]

                s = A(a=[v[0], v[1], v[2], v[3], v[4], v[5]])

                s.a.sort()

                self.assertEqual(s.a, [v[5], v[4], v[0], v[2], v[3], v[1]])

                s.a.sort(reverse=True)

                self.assertEqual(s.a, [v[1], v[2], v[3], v[0], v[4], v[5]])

                with self.assertRaises(TypeError) as e:
                    s.a.sort(1)

                with self.assertRaises(TypeError) as e:
                    s.a.sort(key=abs, a=3)

                # Check if sorting key abs() is defined
                if v[6] is not None:
                    s.a.sort(key=abs)

                    self.assertEqual(s.a, [v[0], v[4], v[2], v[3], v[1], v[5]])

            class B(csp.Struct):
                a: List[MyEnum]

            s = B(a=[MyEnum.A, MyEnum.FOO])

            with self.assertRaises(TypeError) as e:
                s.a.sort()
            with self.assertRaises(TypeError) as e:
                s.a.sort(reverse=True)
            s.a.sort(reverse=True, key=str)
            self.assertEqual(s.a, [MyEnum.FOO, MyEnum.A])

    def test_list_field_extend(self):
        """Was a BUG when the struct with list field was not recognizing changes made to this field in python"""
        for ann_typ in struct_list_annotation_types:
            for typ, v in struct_list_test_values.items():

                class A(csp.Struct):
                    a: ann_typ[typ]

                s = A(a=[v[0], v[1], v[2]])
                s.a.extend([v[3]])

                self.assertEqual(s.a, [v[0], v[1], v[2], v[3]])

                s.a.extend([])
                s.a.extend((v[4], v[5]))

                self.assertEqual(s.a, [v[0], v[1], v[2], v[3], v[4], v[5]])

                # Check if not str or generic type (as str is a sequence of str)
                if v[-2] is not None:
                    with self.assertRaises(TypeError) as e:
                        s.a.extend(v[-2])

                # Check if not generic type
                if v[-1] is not None:
                    with self.assertRaises(TypeError) as e:
                        s.a.extend([v[-1]])

    def test_list_field_remove(self):
        """Was a BUG when the struct with list field was not recognizing changes made to this field in python"""
        for ann_typ in struct_list_annotation_types:
            for typ, v in struct_list_test_values.items():

                class A(csp.Struct):
                    a: ann_typ[typ]

                s = A(a=[v[0], v[1], v[0], v[2]])
                s.a.remove(v[0])

                self.assertEqual(s.a, [v[1], v[0], v[2]])

                s.a.remove(v[2])

                self.assertEqual(s.a, [v[1], v[0]])

                with self.assertRaises(ValueError) as e:
                    s.a.remove(v[3])

    def test_list_field_clear(self):
        """Was a BUG when the struct with list field was not recognizing changes made to this field in python"""
        for ann_typ in struct_list_annotation_types:
            for typ, v in struct_list_test_values.items():

                class A(csp.Struct):
                    a: ann_typ[typ]

                s = A(a=[v[0], v[1], v[2], v[3]])
                s.a.clear()

                self.assertEqual(s.a, [])

    def test_list_field_del(self):
        """Was a BUG when the struct with list field was not recognizing changes made to this field in python"""
        for ann_typ in struct_list_annotation_types:
            for typ, v in struct_list_test_values.items():

                class A(csp.Struct):
                    a: ann_typ[typ]

                s = A(a=[v[0], v[1], v[2], v[3]])
                del s.a[0]

                self.assertEqual(s.a, [v[1], v[2], v[3]])

                del s.a[1]

                self.assertEqual(s.a, [v[1], v[3]])

                s = A(a=[v[0], v[1], v[2], v[3]])
                del s.a[1:3]

                self.assertEqual(s.a, [v[0], v[3]])

                del s.a[5:100]

                self.assertEqual(s.a, [v[0], v[3]])

                del s.a[1:]

                self.assertEqual(s.a, [v[0]])

                s = A(a=[v[0], v[1], v[2], v[3]])
                del s.a[:1:-1]

                self.assertEqual(s.a, [v[0], v[1]])

                with self.assertRaises(IndexError) as e:
                    del s.a[5]

    def test_list_field_inplace_concat(self):
        """Was a BUG when the struct with list field was not recognizing changes made to this field in python"""
        for ann_typ in struct_list_annotation_types:
            for typ, v in struct_list_test_values.items():

                class A(csp.Struct):
                    a: ann_typ[typ]

                s = A(a=[v[0], v[1]])
                s.a.__iadd__([v[2], v[3]])

                self.assertEqual(s.a, [v[0], v[1], v[2], v[3]])

                s.a += (v[4], v[5])

                self.assertEqual(s.a, [v[0], v[1], v[2], v[3], v[4], v[5]])

                s.a += []

                self.assertEqual(s.a, [v[0], v[1], v[2], v[3], v[4], v[5]])

                with self.assertRaises(TypeError) as e:
                    s.a += v[-1]

                # Check if not generic type
                if v[-1] is not None:
                    with self.assertRaises(TypeError) as e:
                        s.a += [v[-1]]

                self.assertEqual(s.a, [v[0], v[1], v[2], v[3], v[4], v[5]])

    def test_list_field_inplace_repeat(self):
        """Was a BUG when the struct with list field was not recognizing changes made to this field in python"""
        for ann_typ in struct_list_annotation_types:
            for typ, v in struct_list_test_values.items():

                class A(csp.Struct):
                    a: ann_typ[typ]

                s = A(a=[v[0], v[1]])
                s.a.__imul__(1)

                self.assertEqual(s.a, [v[0], v[1]])

                s.a *= 2

                self.assertEqual(s.a, [v[0], v[1], v[0], v[1]])

                with self.assertRaises(TypeError) as e:
                    s.a *= [3]

                with self.assertRaises(TypeError) as e:
                    s.a *= "s"

                s.a *= 0

                self.assertEqual(s.a, [])

                s.a += [v[2], v[3]]

                self.assertEqual(s.a, [v[2], v[3]])

                s.a *= -1

                self.assertEqual(s.a, [])

    def test_list_field_lifetime(self):
        """Ensure that the lifetime of struct list field exceeds the lifetime of struct holding it"""
        for ann_typ in struct_list_annotation_types:

            class A(csp.Struct):
                a: ann_typ[int]

            s = A(a=[1, 2, 3])
            l = s.a
            del s

            self.assertEqual(l, [1, 2, 3])

    def test_list_field_len(self):
        """Ensure that non-list-modifying operations on list fields work fine"""
        for ann_typ in struct_list_annotation_types:
            for typ, v in struct_list_test_values.items():

                class A(csp.Struct):
                    a: ann_typ[typ]

                s = A(a=[v[0], v[1]])

                self.assertEqual(len(s.a), 2)

                s.a *= 2

                self.assertEqual(len(s.a), 4)

                s.a.clear()

                self.assertEqual(len(s.a), 0)

    def test_list_field_repr(self):
        """Ensure that non-list-modifying operations on list fields work fine"""
        for ann_typ in struct_list_annotation_types:
            for typ, v in struct_list_test_values.items():
                # Excluding str due to own repr implementation
                if typ is not str:

                    class A(csp.Struct):
                        a: ann_typ[typ]

                    s = A(a=[v[0], v[1]])

                    self.assertEqual(repr(s.a), f"[{repr(v[0])}, {repr(v[1])}]")

                    s.a *= 2

                    self.assertEqual(repr(s.a), f"[{repr(v[0])}, {repr(v[1])}, {repr(v[0])}, {repr(v[1])}]")

                    s.a.clear()

                    self.assertEqual(repr(s.a), "[]")

    def test_list_field_str(self):
        """Ensure that non-list-modifying operations on list fields work fine"""
        for ann_typ in struct_list_annotation_types:
            for typ, v in struct_list_test_values.items():
                # Excluding str due to own repr implementation
                if typ is not str:

                    class A(csp.Struct):
                        a: ann_typ[typ]

                    s = A(a=[v[0], v[1]])

                    self.assertEqual(str(s.a), f"[{repr(v[0])}, {repr(v[1])}]")

                    s.a *= 2

                    self.assertEqual(str(s.a), f"[{repr(v[0])}, {repr(v[1])}, {repr(v[0])}, {repr(v[1])}]")

                    s.a.clear()

                    self.assertEqual(str(s.a), "[]")

    def test_list_field_get_item(self):
        """Ensure that non-list-modifying operations on list fields work fine"""
        for ann_typ in struct_list_annotation_types:
            for typ, v in struct_list_test_values.items():

                class A(csp.Struct):
                    a: ann_typ[typ]

                s = A(a=[v[0], v[1], v[2]])

                self.assertEqual(s.a.__getitem__(0), v[0])

                self.assertEqual(s.a[1], v[1])

                self.assertEqual(s.a[-1], v[2])

                with self.assertRaises(IndexError) as e:
                    b = s.a[100]

                with self.assertRaises(IndexError) as e:
                    b = s.a[-100]

                self.assertEqual(s.a[5:6], [])

                self.assertEqual(s.a[1:], [v[1], v[2]])

                self.assertEqual(s.a[1:2], [v[1]])

                self.assertEqual(s.a[:10:2], [v[0], v[2]])

                self.assertEqual(s.a[-1::-1], [v[2], v[1], v[0]])

                self.assertEqual(s.a[-1:-3:-1], [v[2], v[1]])

    def test_list_field_copy(self):
        """Ensure that non-list-modifying operations on list fields work fine"""
        for ann_typ in struct_list_annotation_types:
            for typ, v in struct_list_test_values.items():

                class A(csp.Struct):
                    a: ann_typ[typ]

                s = A(a=[v[0], v[1], v[2]])

                self.assertEqual(s.a.copy(), [v[0], v[1], v[2]])

                b = s.a.copy()
                b.append(v[4])
                s.a.append(v[3])

                self.assertEqual(s.a, [v[0], v[1], v[2], v[3]])
                self.assertEqual(b, [v[0], v[1], v[2], v[4]])

                with self.assertRaises(TypeError) as e:
                    s.a.copy(1)

                with self.assertRaises(TypeError) as e:
                    s.a.copy(s=2)

                with self.assertRaises(IndexError) as e:
                    b = s.a[-100]

                self.assertEqual(type(s.a.copy()), list)

    def test_list_field_index(self):
        """Ensure that non-list-modifying operations on list fields work fine"""
        for ann_typ in struct_list_annotation_types:
            for typ, v in struct_list_test_values.items():

                class A(csp.Struct):
                    a: ann_typ[typ]

                s = A(a=[v[0], v[3], v[0]])

                self.assertEqual(s.a.index(v[0]), 0)

                self.assertEqual(s.a.index(v[3]), 1)

                self.assertEqual(s.a.index(v[0], 1), 2)

                self.assertEqual(s.a.index(v[0], 1, 3), 2)

                self.assertEqual(s.a.index(v[0], -2), 2)

                self.assertEqual(s.a.index(v[0], -100, -1), 0)

                self.assertEqual(s.a.index(v[3], -100, 100), 1)

                with self.assertRaises(ValueError) as e:
                    s.a.index(v[3], 2)

                with self.assertRaises(ValueError) as e:
                    s.a.index(v[3], -1, 100)

                with self.assertRaises(ValueError) as e:
                    s.a.index(v[-1])

    def test_list_field_count(self):
        """Ensure that non-list-modifying operations on list fields work fine"""
        for ann_typ in struct_list_annotation_types:
            for typ, v in struct_list_test_values.items():

                class A(csp.Struct):
                    a: ann_typ[typ]

                s = A(a=[v[0], v[3], v[0]])

                self.assertEqual(s.a.count(v[0]), 2)

                self.assertEqual(s.a.count(v[3]), 1)

                self.assertEqual(s.a.count(v[-1]), 0)

    def test_list_field_compare(self):
        """Ensure that non-list-modifying operations on list fields work fine"""
        # Not using pystruct_list_test_values, as comaprison tests are of different semantics (order and comparison key existance matters).
        values = {
            int: [1, 5, 2, 2, -1, -5, "s"],
            float: [1.4, 5.2, 2.7, 2.7, -1.4, -5.2, "s"],
            datetime: [
                datetime(2022, 12, 6, 1, 2, 3),
                datetime(2022, 12, 8, 3, 2, 3),
                datetime(2022, 12, 7, 2, 2, 3),
                datetime(2022, 12, 7, 2, 2, 3),
                datetime(2022, 12, 5, 2, 2, 3),
                datetime(2022, 12, 3, 2, 2, 3),
                None,
            ],
            timedelta: [
                timedelta(seconds=1),
                timedelta(seconds=123),
                timedelta(seconds=12),
                timedelta(seconds=12),
                timedelta(seconds=0.1),
                timedelta(seconds=0.01),
                None,
            ],
            date: [
                date(2022, 12, 6),
                date(2022, 12, 8),
                date(2022, 12, 7),
                date(2022, 12, 7),
                date(2022, 12, 5),
                date(2022, 12, 3),
                None,
            ],
            time: [time(5, 2, 3), time(7, 2, 3), time(6, 2, 3), time(6, 2, 3), time(4, 2, 3), time(3, 2, 3), None],
            str: ["s", "xyz", "w", "w", "bds", "a", None],
        }

        for ann_typ in struct_list_annotation_types:
            for typ, v in values.items():

                class A(csp.Struct):
                    a: ann_typ[typ]

                s1 = A(a=[v[0], v[1], v[2], v[5]])
                s1a = [v[0], v[1], v[2], v[5]]
                s2 = A(a=[v[0], v[1], v[3], v[4]])
                s2a = [v[0], v[1], v[3], v[4]]

                self.assertEqual(s1.a < s2.a, True)
                self.assertEqual(s1.a <= s2.a, True)
                self.assertEqual(s1.a > s2.a, False)
                self.assertEqual(s1.a >= s2.a, False)
                self.assertEqual(s1.a != s2.a, True)
                self.assertEqual(s1.a == s2.a, False)

                self.assertEqual(s1.a == s1a, True)
                self.assertEqual(s1.a != s1a, False)
                self.assertEqual(s1.a == s2a, False)
                self.assertEqual(s1.a != s2a, True)

                self.assertEqual(s1.a < s2a, True)
                self.assertEqual(s1.a <= s2a, True)
                self.assertEqual(s1.a > s2a, False)
                self.assertEqual(s1.a >= s2a, False)

                s3 = A(a=[v[0], v[1], v[2]])

                self.assertEqual(s3.a < s1.a, True)

                s4 = A(a=[v[0], v[1], v[2]])

                self.assertEqual(s4.a == s3.a, True)

            class B(csp.Struct):
                a: List[MyEnum]

            s = B(a=[MyEnum.A, MyEnum.FOO])
            t = B(a=[MyEnum.FOO, MyEnum.FOO])

            with self.assertRaises(TypeError) as e:
                s.a < t.a
            with self.assertRaises(TypeError) as e:
                s.a <= t.a
            with self.assertRaises(TypeError) as e:
                s.a > t.a
            with self.assertRaises(TypeError) as e:
                s.a >= t.a
            self.assertEqual(s.a == t.a, False)
            self.assertEqual(s.a != t.a, True)

    def test_list_field_iter(self):
        """Ensure that non-list-modifying operations on list fields work fine"""
        for ann_typ in struct_list_annotation_types:
            for typ, v in struct_list_test_values.items():

                class A(csp.Struct):
                    a: ann_typ[typ]

                s = A(a=[v[0], v[1], v[2]])

                b = iter(s.a)
                c = s.a.__iter__()
                d = s.a.__reversed__()

                self.assertEqual(next(b), v[0])
                self.assertEqual(next(c), v[0])
                self.assertEqual(next(d), v[2])

                self.assertEqual(next(b), v[1])
                self.assertEqual(next(c), v[1])
                self.assertEqual(next(d), v[1])

                self.assertEqual(next(b), v[2])
                self.assertEqual(next(c), v[2])
                self.assertEqual(next(d), v[0])

                with self.assertRaises(StopIteration) as e:
                    next(b)
                with self.assertRaises(StopIteration) as e:
                    next(c)
                with self.assertRaises(StopIteration) as e:
                    next(d)

    def test_list_field_concat(self):
        """Ensure that non-list-modifying operations on list fields work fine"""
        for ann_typ in struct_list_annotation_types:
            for typ, v in struct_list_test_values.items():

                class A(csp.Struct):
                    a: ann_typ[typ]

                s = A(a=[v[0], v[1]])
                t = A(a=[v[2], v[3]])

                self.assertEqual(s.a.__add__([v[2], v[3]]), [v[0], v[1], v[2], v[3]])
                self.assertEqual(s.a.__add__(t.a), [v[0], v[1], v[2], v[3]])

                self.assertEqual(s.a + [v[2], v[3]], [v[0], v[1], v[2], v[3]])
                self.assertEqual(s.a + t.a, [v[0], v[1], v[2], v[3]])

                self.assertEqual(s.a + [], [v[0], v[1]])

                with self.assertRaises(TypeError) as e:
                    tmp = s.a = v[-1]

                self.assertEqual(s.a, [v[0], v[1]])

    def test_list_field_repeat(self):
        """Ensure that non-list-modifying operations on list fields work fine"""
        for ann_typ in struct_list_annotation_types:
            for typ, v in struct_list_test_values.items():

                class A(csp.Struct):
                    a: ann_typ[typ]

                s = A(a=[v[0], v[1]])

                self.assertEqual(s.a.__mul__(1), [v[0], v[1]])

                self.assertEqual(s.a * 2, [v[0], v[1], v[0], v[1]])

                with self.assertRaises(TypeError) as e:
                    tmp = s.a * [3]

                with self.assertRaises(TypeError) as e:
                    tmp = s.a * "s"

                self.assertEqual(s.a * 0, [])

                self.assertEqual(s.a * -1, [])

    def test_list_field_correct_type_used(self):
        """Check that FastList and PyStructList types are used correctly"""

        class A(csp.Struct):
            a: List[int]

        with self.assertRaises(TypeError):

            class B(csp.Struct):
                a: List[int, False]

        class C(csp.Struct):
            a: List[int]

        class D(csp.Struct):
            a: FastList[int]

        with self.assertRaises(TypeError):

            class E(csp.Struct):
                a: List[int, True]

        p = A(a=[1, 2])
        r = C(a=[1, 2])
        s = D(a=[1, 2])

        self.assertEqual(str(type(p.a)), "<class '_cspimpl.PyStructList'>")
        self.assertEqual(str(type(r.a)), "<class '_cspimpl.PyStructList'>")
        self.assertEqual(str(type(s.a)), "<class '_cspimpl.PyStructFastList'>")

    def test_list_field_correct_type_passed(self):
        """Check that FastList can be passed to where Python list is expected"""

        class A(csp.Struct):
            a: List[int]

        class B(csp.Struct):
            a: FastList[int]

        p = csp.unroll(csp.const(A(a=[1, 2, 3])).a)
        q = csp.unroll(csp.const(B(a=[1, 2, 3])).a)

    def test_list_field_pickle(self):
        """Was a BUG when the struct with list field was not recognizing changes made to this field in python"""
        # Not using pystruct_list_test_values, as pickling tests are of different semantics (picklability of struct fields matters).
        v = [1, 5, 2]

        s = SimpleStructForPickleList(a=[v[0], v[1], v[2]])

        t = pickle.loads(pickle.dumps(s))

        self.assertEqual(t.a, s.a)
        self.assertEqual(type(t.a), type(s.a))

        b = pickle.loads(pickle.dumps(s.a))

        self.assertEqual(b, s.a)
        self.assertEqual(type(b), list)

        s = SimpleStructForPickleFastList(a=[v[0], v[1], v[2]])

        t = pickle.loads(pickle.dumps(s))

        self.assertEqual(t.a, s.a)
        self.assertEqual(type(t.a), type(s.a))

        b = pickle.loads(pickle.dumps(s.a))

        self.assertEqual(b, s.a)
        self.assertEqual(type(b), list)

    def test_dir(self):
        s = SimpleStruct(a=0)
        dir_output = dir(s)
        self.assertIn("a", dir_output)
        self.assertIn("to_dict", dir_output)
        self.assertIn("update", dir_output)
        self.assertIn("__metadata__", dir_output)
        self.assertEqual(dir_output, sorted(dir_output))

    def test_annotations(self):
        class StructWithAnnotations(csp.Struct):
            b: Annotated[float, "test"]
            d: Annotated[Dict[str, Annotated[int, "test_int"]], "test_dict"]
            s: str

        self.assertEqual(
            StructWithAnnotations.metadata(typed=True),
            {
                "b": Annotated[float, "test"],
                "d": Annotated[Dict[str, Annotated[int, "test_int"]], "test_dict"],
                "s": str,
            },
        )
        self.assertEqual(StructWithAnnotations.metadata(typed=False), {"b": float, "d": dict, "s": str})

    def test_literal(self):
        simple_class0 = SimpleClass(0)
        simple_class1 = SimpleClass(1)

        class StructWithLiteral(csp.Struct):
            s: Literal["foo", "bar"]
            f: Literal[0, 1.0]
            o: Literal["foo", 0]
            c: Literal[simple_class0, simple_class1]

        self.assertEqual(
            StructWithLiteral.metadata(typed=True),
            {
                "s": Literal["foo", "bar"],
                "f": Literal[0, 1.0],
                "o": Literal["foo", 0],
                "c": Literal[simple_class0, simple_class1],
            },
        )
        self.assertEqual(StructWithLiteral.metadata(typed=False), {"s": str, "f": float, "o": object, "c": SimpleClass})

    def test_union(self):
        class StructWithUnion(csp.Struct):
            o1: Union[int, float]
            o2: Optional[str]

        self.assertEqual(
            StructWithUnion.metadata(typed=True),
            {
                "o1": Union[int, float],
                "o2": Optional[str],
            },
        )
        self.assertEqual(StructWithUnion.metadata(typed=False), {"o1": object, "o2": object})

    def test_pydantic_validation(self):
        """Test Pydantic validation integration with CSP Structs"""

        # 1. Test basic validation
        class SimpleStruct(csp.Struct):
            value: int
            name: str = "default"
            scores: List[float]

        # Valid data
        valid_data = {"value": 11, "name": "ya", "scores": [1.1, 2.2, 3.3]}
        result = SimpleStruct.from_dict(valid_data, use_pydantic=True)
        self.assertIsInstance(result, SimpleStruct)
        self.assertEqual(result.value, 11)
        self.assertEqual(result.name, "ya")
        self.assertEqual(result.scores, [1.1, 2.2, 3.3])

        invalid_data = valid_data.copy()
        invalid_data["missing"] = False
        with self.assertRaises(ValidationError):
            SimpleStruct.from_dict(invalid_data, use_pydantic=True)  # extra fields throw an error

        # Test that we can validate existing structs
        existing = SimpleStruct(value=1, scores=[1])
        new = SimpleStruct.from_dict(existing, use_pydantic=True)
        self.assertTrue(existing is new)  # we do not revalidate
        self.assertEqual(existing.value, 1)

        # Test type coercion
        coercion_data = {
            "value": "42",  # string should convert to int
            "scores": ["1.1", 2, "3.3"],  # mixed types should convert to float
        }
        result = SimpleStruct.from_dict(coercion_data, use_pydantic=True)
        self.assertEqual(result.value, 42)
        self.assertEqual(result.scores, [1.1, 2.0, 3.3])

        # 2. Test nested validation
        class NestedStruct(csp.Struct):
            simple: SimpleStruct
            tags: List[str]

        nested_data = {"simple": {"value": 11, "name": "ya", "scores": [1.1, 2.2, 3.3]}, "tags": ["test1", "test2"]}
        result = NestedStruct.from_dict(nested_data, use_pydantic=True)
        self.assertIsInstance(result, NestedStruct)
        self.assertIsInstance(result.simple, SimpleStruct)
        self.assertEqual(result.simple.value, 11)
        self.assertEqual(result.tags, ["test1", "test2"])

        # 3. Test validation errors
        with self.assertRaises(ValidationError) as exc_info:
            SimpleStruct.from_dict({"value": "not an integer", "scores": [1.1, 2.2, "invalid"]}, use_pydantic=True)
        self.assertIn("Input should be a valid integer", str(exc_info.exception))

        # 4. Test with complex types
        class ComplexStruct(csp.Struct):
            dates: List[datetime]
            nested: Optional[SimpleStruct] = None
            mapping: Dict[str, float]

        complex_data = {
            "dates": ["2023-01-01", "2023-01-02"],  # strings should convert to datetime
            "mapping": {"a": "1.1", "b": 2.2},  # mixed types should convert to float
        }
        result = ComplexStruct.from_dict(complex_data, use_pydantic=True)
        self.assertIsInstance(result.dates[0], datetime)
        self.assertEqual(result.mapping, {"a": 1.1, "b": 2.2})

        # 5. Test with enums
        class MyEnum(csp.Enum):
            A = 1
            B = 2

        class EnumStruct(csp.Struct):
            enum_field: MyEnum
            enum_list: List[MyEnum]

        enum_data = {"enum_field": "A", "enum_list": ["A", "B", "A"]}
        result = EnumStruct.from_dict(enum_data, use_pydantic=True)
        self.assertEqual(result.enum_field, MyEnum.A)
        self.assertEqual(result.enum_list, [MyEnum.A, MyEnum.B, MyEnum.A])

    def test_non_serializable_pydantic(self):
        class DummyBlankClass: ...

        class StructWithDummy(csp.Struct):
            x: int
            y: DummyBlankClass
            z: List[DummyBlankClass]
            z1: Dict[DummyBlankClass, DummyBlankClass]
            z2: Optional[Dict[DummyBlankClass, int]]
            z3: List[List[DummyBlankClass]]
            z4: Optional[List[List[DummyBlankClass]]]

        val = DummyBlankClass()
        struct_as_dict = dict(x=12, y=val, z=[val], z1={val: val}, z2=None)
        new_struct = StructWithDummy.from_dict(struct_as_dict, use_pydantic=True)
        self.assertTrue(new_struct.y is val)
        self.assertTrue(new_struct.z[0] is val)
        self.assertTrue(new_struct.z1[val] is val)
        self.assertTrue(new_struct.z2 is None)
        self.assertEqual(TypeAdapter(StructWithDummy).dump_python(new_struct), struct_as_dict)

        for original_z2 in [None, {val: 12}]:
            z3_val = set([tuple([val, val])])
            struct_as_dict = dict(
                x=12,
                y=val,
                z=set([val]),  # type is off
                z1={val: val},
                z2=original_z2,
                z3=z3_val,
                z4=z3_val,
            )
            new_struct = StructWithDummy.from_dict(struct_as_dict, use_pydantic=True)
            self.assertTrue(new_struct.y is val)
            self.assertTrue(new_struct.z[0] is val)
            self.assertTrue(new_struct.z1[val] is val)
            self.assertTrue(new_struct.z2 is original_z2)
            self.assertTrue(new_struct.z3[0][0] is val)
            self.assertTrue(new_struct.z3[0][1] is val)
            self.assertEqual(new_struct.z4, z3_val)  # we don't actually validate

            new_struct_as_dict = TypeAdapter(StructWithDummy).dump_python(new_struct)
            self.assertEqual(new_struct_as_dict.pop("z"), [val])  # turned into a list!
            self.assertEqual(struct_as_dict.pop("z"), set([val]))  # remains a set

            # turned into a list of tuples! Note that the inner type is wrong, we do not error since we
            # pass an 'any_schema'. We maintain csp's behavior by passing the raw result to csp.
            self.assertEqual(new_struct_as_dict.pop("z3"), [tuple([val, val])])
            self.assertEqual(new_struct_as_dict.pop("z4"), z3_val)
            csp_struct = StructWithDummy(**struct_as_dict)
            self.assertEqual(csp_struct.z3, [tuple([val, val])])
            self.assertEqual(csp_struct.z4, z3_val)

            self.assertEqual(struct_as_dict.pop("z3"), z3_val)  # remains a set
            self.assertEqual(struct_as_dict.pop("z4"), z3_val)  # remains a set
            self.assertEqual(new_struct_as_dict, struct_as_dict)

            with self.assertRaises(PydanticSerializationError):
                TypeAdapter(StructWithDummy).dump_json(new_struct)

    def test_pydantic_validation_complex(self):
        """Test Pydantic validation with complex nested types and serialization"""

        # Define our enum types
        class StatusEnum(csp.Enum):
            ACTIVE = 1
            PENDING = 2
            CLOSED = 3

        class PriorityEnum(csp.Enum):
            HIGH = 3
            MEDIUM = 2
            LOW = 1

        # Define our nested structs
        class MetadataStruct(csp.Struct):
            created_at: datetime
            tags: Set[str]
            priority: PriorityEnum

        class TaskStruct(csp.Struct):
            title: str
            metadata: MetadataStruct
            subtasks: Optional[List[Dict[str, Any]]] = None

        # Complex struct with nested generics and enum keys
        class ProjectStruct(csp.Struct):
            name: str
            task_statuses: Dict[StatusEnum, List[TaskStruct]]
            priority_mapping: Dict[PriorityEnum, MetadataStruct]
            optional_tasks: Optional[List[TaskStruct]] = None

        # Test data
        task_data = {
            "title": "Main Task",
            "metadata": {
                "created_at": "2023-01-01T12:00:00",
                "tags": ["important", "urgent"],
                "priority": 3,  # HIGH
            },
            "subtasks": [{"name": "subtask1", "done": True}, {"name": "subtask2", "done": False}],
        }

        project_data = {
            "name": "Test Project",
            "task_statuses": {
                1: [task_data, task_data],  # ACTIVE
                2: [],  # PENDING
                3: [task_data],  # CLOSED
            },
            "priority_mapping": {
                3: {  # HIGH
                    "created_at": "2023-01-01T12:00:00",
                    "tags": ["critical"],
                    "priority": 3,
                },
                2: {  # MEDIUM
                    "created_at": "2023-01-02T12:00:00",
                    "tags": ["normal"],
                    "priority": 2,
                },
            },
        }

        # 1. Test validation
        result = ProjectStruct.from_dict(project_data, use_pydantic=True)

        # Verify the structure
        self.assertIsInstance(result, ProjectStruct)
        self.assertEqual(result.name, "Test Project")

        # Verify enum keys were properly converted
        self.assertIn(StatusEnum.ACTIVE, result.task_statuses)
        self.assertIn(PriorityEnum.HIGH, result.priority_mapping)

        # Verify nested structures
        active_tasks = result.task_statuses[StatusEnum.ACTIVE]
        self.assertEqual(len(active_tasks), 2)
        self.assertIsInstance(active_tasks[0], TaskStruct)
        self.assertIsInstance(active_tasks[0].metadata, MetadataStruct)

        # Verify nested enum values
        self.assertEqual(active_tasks[0].metadata.priority, PriorityEnum.HIGH)

        # 2. Test serialization and deserialization
        # First, serialize to JSON
        json_data = result.to_json(lambda x: list(x))  # need this for Set, csp complains

        # Then deserialize back
        restored = TypeAdapter(ProjectStruct).validate_json(json_data)

        # Verify the restored object matches the original
        self.assertEqual(restored.name, result.name)
        self.assertEqual(len(restored.task_statuses[StatusEnum.ACTIVE]), len(result.task_statuses[StatusEnum.ACTIVE]))

        # Verify nested structures were properly restored
        restored_task = restored.task_statuses[StatusEnum.ACTIVE][0]
        original_task = result.task_statuses[StatusEnum.ACTIVE][0]

        self.assertEqual(restored_task.title, original_task.title)
        self.assertEqual(restored_task.metadata.tags, original_task.metadata.tags)
        self.assertEqual(restored_task.metadata.priority, original_task.metadata.priority)

        # 3. Test validation errors with invalid enum values
        invalid_data = project_data.copy()
        invalid_data["task_statuses"][99] = []  # Invalid enum value

        with self.assertRaises(ValidationError) as exc_info:
            ProjectStruct.from_dict(invalid_data, use_pydantic=True)

        # 4. Test validation errors with invalid nested types
        invalid_task_data = project_data.copy()
        invalid_task_data["task_statuses"][1][0]["metadata"]["priority"] = 99  # Invalid priority

        with self.assertRaises(ValidationError) as exc_info:
            ProjectStruct.from_dict(invalid_task_data, use_pydantic=True)

    def test_pydantic_models_with_csp_structs(self):
        """Test Pydantic BaseModels containing CSP Structs as attributes"""
        from pydantic import BaseModel

        # Define some CSP Structs
        class LocationStruct(csp.Struct):
            latitude: float
            longitude: float
            name: str = "unknown"

        class MetricsStruct(csp.Struct):
            value: int
            timestamp: datetime
            tags: List[str]

        # Define Pydantic models that use these structs
        class SensorReading(BaseModel):
            id: str
            location: LocationStruct
            metrics: List[MetricsStruct]
            backup_locations: Dict[str, LocationStruct] = {}

        # Test data
        sensor_data = {
            "id": "sensor-123",
            "location": {"latitude": 40.7128, "longitude": -74.0060, "name": "New York"},
            "metrics": [
                {"value": 42, "timestamp": "2023-01-01T12:00:00", "tags": ["temperature", "indoor"]},
                {"value": 25, "timestamp": "2023-01-01T12:01:00", "tags": ["humidity", "indoor"]},
            ],
            "backup_locations": {"backup1": {"latitude": 34.0522, "longitude": -118.2437, "name": "Los Angeles"}},
        }

        # 1. Test validation
        result = SensorReading.model_validate(sensor_data)

        # Verify the structure
        self.assertIsInstance(result, SensorReading)
        self.assertIsInstance(result.location, LocationStruct)
        self.assertEqual(len(result.metrics), 2)
        self.assertIsInstance(result.metrics[0], MetricsStruct)

        # Verify values
        self.assertEqual(result.id, "sensor-123")
        self.assertEqual(result.location.name, "New York")
        self.assertEqual(result.metrics[0].value, 42)
        self.assertEqual(result.backup_locations["backup1"].name, "Los Angeles")

        # 2. Test serialization and deserialization
        # To JSON
        json_str = result.model_dump_json()

        # Back from JSON
        restored = SensorReading.model_validate_json(json_str)

        # Verify everything was preserved
        self.assertEqual(restored.id, result.id)
        self.assertEqual(restored.location.latitude, result.location.latitude)
        self.assertEqual(restored.metrics[0].value, result.metrics[0].value)
        self.assertEqual(restored.backup_locations["backup1"].name, result.backup_locations["backup1"].name)

        # 3. Test validation errors
        # Invalid location data
        invalid_data = sensor_data.copy()
        invalid_data["location"]["latitude"] = "not a float"

        with self.assertRaises(ValidationError) as exc_info:
            SensorReading.model_validate(invalid_data)
        self.assertIn("Input should be a valid number", str(exc_info.exception))

        # Invalid metrics data
        invalid_data = sensor_data.copy()
        invalid_data["metrics"][0]["value"] = "not an int"

        with self.assertRaises(ValidationError) as exc_info:
            SensorReading.model_validate(invalid_data)
        self.assertIn("Input should be a valid integer", str(exc_info.exception))

        # 4. Test nested model with multiple struct types
        class ComplexModel(BaseModel):
            name: str
            primary_location: LocationStruct
            secondary_locations: List[LocationStruct]
            metrics_by_location: Dict[str, List[MetricsStruct]]

        complex_data = {
            "name": "Complex Sensor",
            "primary_location": {"latitude": 40.7128, "longitude": -74.0060, "name": "New York"},
            "secondary_locations": [
                {"latitude": 34.0522, "longitude": -118.2437, "name": "Los Angeles"},
                {"latitude": 41.8781, "longitude": -87.6298, "name": "Chicago"},
            ],
            "metrics_by_location": {
                "New York": [{"value": 42, "timestamp": "2023-01-01T12:00:00", "tags": ["temperature"]}],
                "Los Angeles": [{"value": 75, "timestamp": "2023-01-01T12:00:00", "tags": ["temperature"]}],
            },
        }

        # Test validation of complex model
        result = ComplexModel.model_validate(complex_data)

        # Verify structure and values
        self.assertIsInstance(result.primary_location, LocationStruct)
        self.assertEqual(len(result.secondary_locations), 2)
        self.assertEqual(len(result.metrics_by_location["New York"]), 1)
        self.assertEqual(result.metrics_by_location["New York"][0].value, 42)

        # Test serialization of complex model
        json_str = result.model_dump_json()
        restored = ComplexModel.model_validate_json(json_str)

        self.assertEqual(restored.primary_location.name, result.primary_location.name)
        self.assertEqual(
            len(restored.metrics_by_location["Los Angeles"]), len(result.metrics_by_location["Los Angeles"])
        )

    def test_struct_with_annotated_validation(self):
        """Test CSP Struct with Annotated fields and validators"""
        from pydantic import BeforeValidator, WrapValidator

        # Simple validator that modifies the value and enforces value > 0
        def value_validator(v: Any) -> int:
            if isinstance(v, str):
                v = int(v)
            v = int(v)
            if v <= 0:
                raise ValueError("value must be positive")
            return v * 2

        # Wrap validator that can modify the whole struct
        def struct_validator(val, handler) -> Any:
            if isinstance(val, dict) and "description" not in val:
                # Expand simple dict to full structure
                val["description"] = "auto_generated"
            return handler(val)

        # Define our nested structs
        class InnerStruct(csp.Struct):
            value: Annotated[int, BeforeValidator(value_validator)]
            description: str = "default"
            z: int

        class OuterStruct(csp.Struct):
            name: str
            inner: Annotated[InnerStruct, WrapValidator(struct_validator)]

        # Test simple value validation
        inner = InnerStruct.from_dict({"value": "21"}, use_pydantic=True)
        self.assertEqual(inner.value, 42)  # "21" -> 21 -> 42
        self.assertEqual(inner.description, "default")
        self.assertFalse(hasattr(inner, "z"))

        # test existing instance
        inner_new = InnerStruct.from_dict(inner, use_pydantic=True)
        self.assertTrue(inner is inner_new)
        # No revalidation
        self.assertEqual(inner_new.value, 42)

        # Test validation with invalid value in existing instance
        inner.value = -5  # Set invalid value
        # No revalidation, no error
        self.assertTrue(inner is InnerStruct.from_dict(inner, use_pydantic=True))
        with self.assertRaises(ValidationError) as cm:
            InnerStruct.from_dict(inner.to_dict(), use_pydantic=True)
        self.assertIn("value must be positive", str(cm.exception))

        # Test simple value validation
        inner = InnerStruct.from_dict({"value": "21", "z": 17}, use_pydantic=True)
        self.assertEqual(inner.value, 42)  # "21" -> 21 -> 42
        self.assertEqual(inner.description, "default")
        self.assertEqual(inner.z, 17)

        # Test struct validation with expansion
        outer = OuterStruct.from_dict({"name": "test", "inner": {"value": 10, "z": 12}}, use_pydantic=True)
        self.assertEqual(outer.inner.value, 20)  # 10 -> 20 (doubled)
        self.assertEqual(outer.inner.description, "auto_generated")
        self.assertEqual(outer.inner.z, 12)

        # Test normal full structure still works
        outer = OuterStruct.from_dict(
            {"name": "test", "inner": {"value": "5", "description": "custom"}}, use_pydantic=True
        )
        self.assertEqual(outer.inner.value, 10)  # "5" -> 5 -> 10 (doubled)
        self.assertEqual(outer.inner.description, "custom")
        self.assertFalse(hasattr(outer.inner, "z"))  # make sure z is not set

    def test_struct_with_union(self):
        """Test CSP Struct with Union fields"""
        from typing import List, Union

        class MetricStruct(csp.Struct):
            # Union of different numeric types
            value: Union[int, float]
            # Union with None (equivalent to Optional)
            name: Optional[str] = None
            # More complex union with list
            tags: Union[str, List[str]] = "default"

        # Test with different value types
        metric1 = MetricStruct.from_dict(
            {
                "value": 42,  # int
            },
            use_pydantic=True,
        )
        self.assertEqual(metric1.value, 42)
        self.assertIsNone(metric1.name)
        self.assertEqual(metric1.tags, "default")

        metric2 = MetricStruct.from_dict(
            {
                "value": 42.5,  # float
                "name": "test",
                "tags": ["tag1", "tag2"],
            },
            use_pydantic=True,
        )
        self.assertEqual(metric2.value, 42.5)
        self.assertEqual(metric2.name, "test")
        self.assertEqual(metric2.tags, ["tag1", "tag2"])

        # Test with string that should convert to float
        metric3 = MetricStruct.from_dict(
            {
                "value": "42.5",  # should convert to float
                "tags": "single_tag",  # single string tag
            },
            use_pydantic=True,
        )
        self.assertEqual(metric3.value, 42.5)
        self.assertEqual(metric3.tags, "single_tag")

        # Test validation error with invalid type
        with self.assertRaises(ValidationError) as exc_info:
            MetricStruct.from_dict(
                {
                    "value": "not a number",
                },
                use_pydantic=True,
            )
        self.assertIn("Input should be a valid number", str(exc_info.exception))

        # Test with string that should convert to float
        metric3 = MetricStruct.from_dict(
            {
                "tags": "single_tag"  # single string tag
            },
            use_pydantic=True,
        )
        self.assertFalse(hasattr(metric3, "value"))
        self.assertEqual(metric3.tags, "single_tag")

    def test_struct_with_nested_union(self):
        """Test CSP Struct with Union of different Struct types"""

        class MetricStruct(csp.Struct):
            value: float
            unit: str

        class EventStruct(csp.Struct):
            name: str
            timestamp: datetime

        class DataPoint(csp.Struct):
            id: str
            # Union of different Struct types
            data: Union[MetricStruct, EventStruct]
            # Optional nested struct
            metadata: Union[MetricStruct, None] = None
            # List of union of structs
            history: List[Union[MetricStruct, EventStruct]] = []

        # Test with MetricStruct
        metric_data = {"id": "metric-1", "data": {"value": 42.5, "unit": "celsius"}}
        result = DataPoint.from_dict(metric_data, use_pydantic=True)
        self.assertIsInstance(result.data, MetricStruct)
        self.assertEqual(result.data.value, 42.5)
        self.assertEqual(result.data.unit, "celsius")

        # Test with EventStruct
        event_data = {
            "id": "event-1",
            "data": {"name": "system_start", "timestamp": "2023-01-01T12:00:00"},
            "history": [
                {"value": 10.0, "unit": "meters"},
                {"name": "previous_event", "timestamp": "2023-01-01T11:00:00"},
            ],
        }
        result = DataPoint.from_dict(event_data, use_pydantic=True)
        self.assertIsInstance(result.data, EventStruct)
        self.assertEqual(result.data.name, "system_start")
        self.assertIsInstance(result.history[0], MetricStruct)
        self.assertIsInstance(result.history[1], EventStruct)

        # Test serialization and deserialization
        result = DataPoint.from_dict(event_data, use_pydantic=True)
        json_data = result.to_json()
        restored = TypeAdapter(DataPoint).validate_json(json_data)

        self.assertIsInstance(restored.data, EventStruct)
        self.assertEqual(restored.data.name, result.data.name)
        self.assertEqual(len(restored.history), len(result.history))
        self.assertIsInstance(restored.history[0], MetricStruct)
        self.assertIsInstance(restored.history[1], EventStruct)

    def test_struct_union_with_inheritance(self):
        """Test Union with inherited Struct types to ensure proper type resolution"""

        class BaseMetric(csp.Struct):
            name: str
            value: float

        class TemperatureMetric(BaseMetric):
            unit: str = "celsius"
            precision: int = 2

        class PressureMetric(BaseMetric):
            unit: str = "pascal"
            altitude: float

        class DataPoint(csp.Struct):
            id: str
            # Union should resolve to most specific type
            metric: Union[BaseMetric, TemperatureMetric, PressureMetric]
            # List of metrics to test multiple instances
            history: List[Union[BaseMetric, TemperatureMetric, PressureMetric]] = []

        # Test with TemperatureMetric data
        temp_data = {
            "id": "temp-1",
            "metric": {
                "name": "temperature",
                "value": 25.5,
                "precision": 1,  # specific to TemperatureMetric
            },
        }
        result = DataPoint.from_dict(temp_data, use_pydantic=True)
        self.assertIsInstance(result.metric, TemperatureMetric)  # Should be TemperatureMetric, not BaseMetric
        self.assertEqual(result.metric.unit, "celsius")
        self.assertEqual(result.metric.precision, 1)

        # Test with PressureMetric data
        pressure_data = {"id": "pressure-1", "metric": {"name": "pressure", "value": 101.325, "altitude": 0.0}}
        result = DataPoint.from_dict(pressure_data, use_pydantic=True)
        self.assertIsInstance(result.metric, PressureMetric)  # Should be PressureMetric, not BaseMetric
        self.assertEqual(result.metric.unit, "pascal")
        self.assertEqual(result.metric.altitude, 0.0)

        # Test with mixed list of metrics
        mixed_data = {
            "id": "mixed-1",
            "metric": {
                "name": "current",
                "value": 100.0,  # just base metric
            },
            "history": [
                {"name": "temp", "value": 25.5, "precision": 1},
                {"name": "pressure", "value": 101.325, "altitude": 0.0},
                {
                    "name": "simple",
                    "value": 42.0,  # base metric
                },
            ],
        }
        result = DataPoint.from_dict(mixed_data, use_pydantic=True)
        self.assertIsInstance(result.metric, BaseMetric)  # Should be base metric
        self.assertIsInstance(result.history[0], TemperatureMetric)  # Should be temperature
        self.assertIsInstance(result.history[1], PressureMetric)  # Should be pressure
        self.assertIsInstance(result.history[2], BaseMetric)  # Should be base

        # Test serialization and deserialization preserves specific types
        json_data_csp = result.to_json()
        json_data_pydantic = TypeAdapter(DataPoint).dump_json(result).decode()
        self.assertEqual(json.loads(json_data_csp), json.loads(json_data_pydantic))
        restored = TypeAdapter(DataPoint).validate_json(json_data_csp)
        self.assertEqual(restored, result)

    def test_pydantic_custom_serialization(self):
        """Test that CustomStruct correctly serializes integers with comma formatting"""
        from pydantic.functional_serializers import PlainSerializer

        # Define the custom integer type with fancy formatting
        FancyInt = Annotated[int, PlainSerializer(lambda x: f"{x:,}", return_type=str, when_used="always")]

        # Simple struct with just the FancyInt
        class CustomStruct(csp.Struct):
            value: FancyInt

        # Test different integer values
        test_cases = [
            (1234, "1,234"),
            (1000000, "1,000,000"),
            (42, "42"),
        ]

        for input_value, expected_output in test_cases:
            # Create and serialize the struct
            s = CustomStruct(value=input_value)
            serialized = json.loads(TypeAdapter(CustomStruct).dump_json(s))

            # Verify the serialization
            self.assertEqual(
                serialized["value"],
                expected_output,
            )

    def test_pydantic_serialization_with_enums(self):
        """Test serialization behavior with enums using both native and Pydantic approaches"""

        class Color(csp.Enum):
            RED = 1
            GREEN = 2
            BLUE = 3

        class Shape(csp.Enum):
            CIRCLE = 1
            SQUARE = 2
            TRIANGLE = 3

        class DrawingStruct(csp.Struct):
            color: Color
            shape: Shape
            colors: List[Color]
            shapes: Dict[str, Shape]

        drawing = DrawingStruct(
            color=Color.RED,
            shape=Shape.CIRCLE,
            colors=[Color.RED, Color.GREEN, Color.BLUE],
            shapes={"a": Shape.SQUARE, "b": Shape.TRIANGLE},
        )

        # Test native serialization
        native_json = json.loads(drawing.to_json())
        self.assertEqual(native_json["color"], "RED")
        self.assertEqual(native_json["shape"], "CIRCLE")
        self.assertEqual(native_json["colors"], ["RED", "GREEN", "BLUE"])
        self.assertEqual(native_json["shapes"], {"a": "SQUARE", "b": "TRIANGLE"})

        # Test Pydantic serialization
        pydantic_json = json.loads(TypeAdapter(DrawingStruct).dump_json(drawing))
        self.assertEqual(pydantic_json, native_json)  # Should be identical for enums

        # Test round-trip through both methods
        native_restored = DrawingStruct.from_dict(json.loads(drawing.to_json()))
        pydantic_restored = TypeAdapter(DrawingStruct).validate_json(TypeAdapter(DrawingStruct).dump_json(drawing))

        self.assertEqual(native_restored, drawing)
        self.assertEqual(pydantic_restored, drawing)

    def test_pydantic_serialization_vs_native(self):
        """Test that Pydantic serialization matches CSP native serialization for basic types"""
        from pydantic.functional_serializers import PlainSerializer

        class MyEnum(csp.Enum):
            OPTION1 = csp.Enum.auto()
            OPTION2 = csp.Enum.auto()

        # Define custom datetime serialization
        # This is so that pydantic serializes datetime with the same precision as csp natively does
        SimpleDatetime = Annotated[
            datetime,
            PlainSerializer(lambda dt: dt.strftime("%Y-%m-%dT%H:%M:%S.%f+00:00"), return_type=str, when_used="json"),
        ]

        class SimpleStruct(csp.Struct):
            i: int = 123
            f: float = 3.14
            s: str = "test"
            b: bool = True
            # dt: datetime = datetime(2023, 1, 1)
            dt: SimpleDatetime = datetime(2023, 1, 1)
            l: List[int] = [1, 2, 3]
            d: Dict[str, float] = {"a": 1.1, "b": 2.2}
            e: MyEnum

        # Test with default values
        s1 = SimpleStruct()
        json_native = s1.to_json()
        json_pydantic = TypeAdapter(SimpleStruct).dump_json(s1).decode()
        self.assertEqual(json.loads(json_native), json.loads(json_pydantic))
        python_native = s1.to_dict()
        python_pydantic = TypeAdapter(SimpleStruct).dump_python(s1)
        self.assertEqual(python_native, python_pydantic)
        # unset variables with no default do not get encoded
        self.assertTrue("e" not in python_native)
        self.assertTrue("e" not in python_pydantic)

        # Test with custom values
        s2 = SimpleStruct(
            i=456,
            f=2.718,
            s="custom",
            b=False,
            dt=datetime(2024, 1, 1, tzinfo=pytz.UTC),
            l=[4, 5, 6],
            d={"x": 9.9, "y": 8.8},
            e=MyEnum.OPTION2,
        )
        python_native = s2.to_dict()
        python_pydantic = TypeAdapter(SimpleStruct).dump_python(s2)
        # NOTE: csp, when running 'to_dict'
        # converts csp Enums to str
        # The pydantic version maintains them as csp Enums, which is arguably more correct
        enum_as_str = python_native.pop("e")
        enum_as_enum = python_pydantic.pop("e")
        self.assertEqual(python_native, python_pydantic)
        self.assertEqual(enum_as_enum.name, enum_as_str)

        json_native = s2.to_json()
        json_pydantic = TypeAdapter(SimpleStruct).dump_json(s2).decode()
        self.assertEqual(json.loads(json_native), json.loads(json_pydantic))

        # Test with nested structs
        class NestedStruct(csp.Struct):
            name: str
            simple: SimpleStruct
            simples: List[SimpleStruct]

        nested = NestedStruct(name="test", simple=s1, simples=[s1, s2])

        python_native = nested.to_dict()
        python_pydantic = TypeAdapter(NestedStruct).dump_python(nested)
        enum_as_str = python_native["simples"][1].pop("e")
        enum_as_enum = python_pydantic["simples"][1].pop("e")
        self.assertEqual(python_native, python_pydantic)
        self.assertEqual(enum_as_enum.name, enum_as_str)

        self.assertEqual(
            nested, NestedStruct.from_dict(TypeAdapter(NestedStruct).dump_python(nested), use_pydantic=True)
        )

        json_native = nested.to_json()
        json_pydantic = TypeAdapter(NestedStruct).dump_json(nested).decode()
        self.assertEqual(json.loads(json_native), json.loads(json_pydantic))

    def test_pydantic_np_arr(self):
        class NPStruct(csp.Struct):
            arr: Numpy1DArray[float] = np.array([])

        val = NPStruct(arr=np.array([1, 2]))
        json_val = TypeAdapter(NPStruct).dump_json(val)
        # We serialize as a list
        self.assertEqual(json.loads(json_val), dict(arr=[1, 2]))
        revived_val = TypeAdapter(NPStruct).validate_json(json_val)
        np.all(val.arr == revived_val)

        NPStruct(arr=np.array([1, 3, "ab"]))  # No error, even though the types are wrong
        with self.assertRaises(ValidationError) as exc_info:
            NPStruct.from_dict(dict(arr=[1, 3, "ab"]), use_pydantic=True)
        self.assertIn("could not convert string to float", str(exc_info.exception))
        # We should be able to generate the json_schema
        TypeAdapter(NPStruct).json_schema()

    def test_struct_with_private_fields(self):
        """Test CSP Struct with private (_) fields to ensure they're validated but excluded from serialization"""

        class BaseMetric(csp.Struct):
            _base_id: str
            value: float

        class MetricMetadata(BaseMetric):
            _internal_id: str
            public_tag: str
            _inherited: bool = True

        class MetricStruct(csp.Struct):
            value: float
            _confidence: float
            metadata: MetricMetadata

        class EventStruct(csp.Struct):
            name: str
            timestamp: datetime
            _source: str = "system"

        class DataPoint(csp.Struct):
            id: str
            data: Union[MetricStruct, EventStruct]
            _last_updated: datetime

        # Test validation with private fields
        metric_data = {
            "id": "metric-1",
            "_last_updated": datetime(2023, 1, 1, 12, 0),  # not validated
            "data": {
                "value": 42.5,
                "_confidence": 0.95,
                "metadata": {
                    "_base_id": "base123",
                    "value": 99.9,
                    "_internal_id": "internal123",
                    "public_tag": "temperature",
                    "_inherited": False,
                },
            },
        }

        result = DataPoint.from_dict(metric_data, use_pydantic=True)

        # Verify private fields are properly set including inherited ones
        self.assertEqual(result._last_updated, datetime(2023, 1, 1, 12, 0))
        self.assertEqual(result.data._confidence, 0.95)
        self.assertEqual(result.data.metadata._base_id, "base123")
        self.assertEqual(result.data.metadata._internal_id, "internal123")
        self.assertEqual(result.data.metadata._inherited, False)
        self.assertEqual(result.data.metadata.value, 99.9)

        # Test serialization - private fields should be excluded, including inherited ones
        serialized = TypeAdapter(DataPoint).dump_python(result)
        self.assertNotIn("_last_updated", serialized)
        self.assertNotIn("_confidence", serialized["data"])
        self.assertNotIn("_base_id", serialized["data"]["metadata"])
        self.assertNotIn("_internal_id", serialized["data"]["metadata"])
        self.assertNotIn("_inherited", serialized["data"]["metadata"])
        self.assertEqual(serialized["data"]["metadata"]["value"], 99.9)

        # Verify JSON serialization also excludes private fields
        json_data = json.loads(TypeAdapter(DataPoint).dump_json(result))
        self.assertNotIn("_last_updated", json_data)
        self.assertNotIn("_confidence", json_data["data"])
        self.assertNotIn("_base_id", json_data["data"]["metadata"])
        self.assertNotIn("_internal_id", json_data["data"]["metadata"])
        self.assertNotIn("_inherited", json_data["data"]["metadata"])
        self.assertEqual(json_data["data"]["metadata"]["value"], 99.9)

        # Test that public fields are still included
        self.assertEqual(json_data["data"]["metadata"]["public_tag"], "temperature")

        # Test with event data
        event_data = {
            "id": "event-1",
            "_last_updated": datetime(2023, 1, 1, 12, 0),  # not validated
            "data": {
                "name": "system_start",
                "timestamp": "2023-01-01T12:00:00",  # validated
                "_source": "automated_test",
            },
        }

        result = DataPoint.from_dict(event_data, use_pydantic=True)

        # Verify private fields are set but excluded from serialization
        self.assertEqual(result._last_updated, datetime(2023, 1, 1, 12, 0))
        self.assertEqual(result.data._source, "automated_test")

        json_data = json.loads(TypeAdapter(DataPoint).dump_json(result))
        self.assertNotIn("_last_updated", json_data)
        self.assertNotIn("_source", json_data["data"])

    def test_literal_types_validation(self):
        """Test that Literal type annotations correctly validate input values in CSP Structs"""

        # Define a simple class with various Literal types
        class StructWithLiterals(csp.Struct):
            # String literals
            color: Literal["red", "green", "blue"]
            # Integer literals
            size: Literal[1, 2, 3]
            # Mixed type literals
            status: Literal["on", "off", 0, 1, True, False]
            # Optional literal with default
            mode: Optional[Literal["fast", "slow"]] = "fast"

        # Test valid assignments
        s1 = StructWithLiterals(color="red", size=2, status="on")
        self.assertEqual(s1.color, "red")
        self.assertEqual(s1.size, 2)
        self.assertEqual(s1.status, "on")
        self.assertEqual(s1.mode, "fast")  # Default value

        s2 = StructWithLiterals.from_dict(dict(color="blue", size=1, status=True, mode="slow"))
        s2_dump = s2.to_json()
        s2_looped = TypeAdapter(StructWithLiterals).validate_json(s2_dump)
        self.assertEqual(s2, s2_looped)
        s2_dict = s2.to_dict()
        s2_looped_dict = s2.from_dict(s2_dict)
        self.assertEqual(s2_looped_dict, s2)

        # Invalid color, but from_dict still accepts
        StructWithLiterals.from_dict(dict(color="yellow", size=1, status="on"))

        # Invalid size but from_dict still accepts
        StructWithLiterals.from_dict(dict(color="red", size=4, status="on"))

        # Invalid status but from_dict still accepts
        StructWithLiterals.from_dict(dict(color="red", size=1, status="standby"))

        # Invalid mode but from_dict still accepts
        StructWithLiterals.from_dict(dict(color="red", size=1, mode=12))

        # Invalid size and since the literals are all the same type
        # If we give an incorrect type, we catch the error
        with self.assertRaises(ValueError) as exc_info:
            StructWithLiterals.from_dict(dict(color="red", size="adasd", mode=12))
        self.assertIn("Expected type <class 'int'> received <class 'str'>", str(exc_info.exception))

        # Test valid values
        result = TypeAdapter(StructWithLiterals).validate_python({"color": "green", "size": 3, "status": 0})
        self.assertEqual(result.color, "green")
        self.assertEqual(result.size, 3)
        self.assertEqual(result.status, 0)

        # Test invalid color with Pydantic validation
        with self.assertRaises(ValidationError) as exc_info:
            TypeAdapter(StructWithLiterals).validate_python({"color": "yellow", "size": 1, "status": "on"})
        self.assertIn("1 validation error for", str(exc_info.exception))
        self.assertIn("color", str(exc_info.exception))

        # Test invalid size with Pydantic validation
        with self.assertRaises(ValidationError) as exc_info:
            TypeAdapter(StructWithLiterals).validate_python({"color": "red", "size": 4, "status": "on"})
        self.assertIn("1 validation error for", str(exc_info.exception))
        self.assertIn("size", str(exc_info.exception))

        # Test invalid status with Pydantic validation
        with self.assertRaises(ValidationError) as exc_info:
            TypeAdapter(StructWithLiterals).validate_python({"color": "red", "size": 1, "status": "standby"})
        self.assertIn("1 validation error for", str(exc_info.exception))
        self.assertIn("status", str(exc_info.exception))

        # Test invalid mode with Pydantic validation
        with self.assertRaises(ValidationError) as exc_info:
            TypeAdapter(StructWithLiterals).validate_python(
                {"color": "red", "size": 1, "status": "on", "mode": "medium"}
            )
        self.assertIn("1 validation error for", str(exc_info.exception))
        self.assertIn("mode", str(exc_info.exception))

    def test_pipe_operator_types(self):
        """Test using the pipe operator for union types in Python 3.10+"""

        # Define a class using various pipe operator combinations
        class PipeTypesConfig(csp.Struct):
            # Basic primitive types with pipe
            id_field: str | int
            # Pipe with None (similar to Optional)
            description: str | None = None
            # Multiple types with pipe
            value: str | int | float | bool
            # Container with pipe
            tags: List[str] | Dict[str, str] | None = None
            # Pipe with literal for comparison
            status: Literal["active", "inactive"] | None = "active"

        # Test all valid types
        valid_cases = [
            {"id_field": "string_id", "value": "string_value"},
            {"id_field": 42, "value": 123},
            {"id_field": "mixed", "value": 3.14},
            {"id_field": 999, "value": True},
            {"id_field": "with_desc", "value": 1, "description": "Description"},
            {"id_field": "with_dict", "value": 1, "tags": None},
        ]

        for case in valid_cases:
            result = PipeTypesConfig.from_dict(case)
            # use the other route to get back the result
            result_to_dict_loop = TypeAdapter(PipeTypesConfig).validate_python(result.to_dict())
            self.assertEqual(result, result_to_dict_loop)

        # Test invalid values
        invalid_cases = [
            {"id_field": 3.14, "value": 1},  # Float for id_field
            {"id_field": None, "value": 1},  # None for required id_field
            {"id_field": "test", "value": {}},  # Dict for value
            {"id_field": "test", "value": None},  # None for required value
            {"id_field": "test", "value": 1, "status": "unknown"},  # Invalid literal
        ]
        for case in invalid_cases:
            with self.assertRaises(ValidationError):
                TypeAdapter(PipeTypesConfig).validate_python(case)

    def test__metadata_info(self):
        class MyStruct(DerivedMixed):
            typed: BaseNative
            generic: csp.Struct

        metadata_info = MyStruct._metadata_info()
        self.assertEqual(metadata_info["is_native"], False)
        typed_field = [f for f in metadata_info["fields"] if f["fieldname"] == "typed"][0]
        generic_field = [f for f in metadata_info["fields"] if f["fieldname"] == "generic"][0]
        self.assertEqual(typed_field["type"]["pytype"], BaseNative)
        self.assertEqual(generic_field["type"]["pytype"], None)

    def test_type_adapter_inherited(self):
        class MyStruct(csp.Struct):
            x: int

        class MyStructB(MyStruct):
            y: str

        validated_struct = MyStruct.type_adapter().validate_python(dict(x=11))
        self.assertEqual(validated_struct, MyStruct(x=11))

        validated_child_struct = MyStructB.type_adapter().validate_python(dict(y="a"))
        self.assertEqual(validated_child_struct, MyStructB(y="a"))

    def test_struct_type_lifetime(self):
        """Was a crashing bug: https://github.com/Point72/csp/issues/579"""
        import gc

        class T(csp.Struct):
            s: csp.Struct

        def foo2():
            meta = {"A": str, "B": int}
            DynStruct = defineStruct("DynStruct", meta)
            t = T(s=DynStruct(A="testing"))
            del DynStruct
            gc.collect(0)
            return t

        s = foo2()
        self.assertEqual(s.s.A, "testing")
        del s.s
        del s
        gc.collect(0)


if __name__ == "__main__":
    unittest.main()
