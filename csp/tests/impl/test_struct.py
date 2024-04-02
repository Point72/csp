import json
import numpy as np
import pytz
import typing
import unittest
from datetime import date, datetime, time, timedelta

import csp
from csp.impl.struct import defineStruct


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
    a1: [int]
    a2: [str]
    a3: [object]
    a4: [bytes]


class StructWithDefaults(csp.Struct):
    b: bool
    i: int = 123
    f: float
    s: str = "456"
    e: MyEnum = MyEnum.FOO
    o: object
    a1: [int] = [1, 2, 3]
    a2: [str] = ["1", "2", "3"]
    a3: [object] = ["hey", 123, (1, 2, 3)]
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
    a1: [int]
    a2: [str]
    a3: [object]


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
    # native_list: [int]
    struct_list: [BaseNative]
    dialect_generic_list: [list]


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
    arr: [int] = [1, 2, 3]
    o: object = {"k": "v"}


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
            b: [FOO]

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

        o = StructWithLists(struct_list=[BaseNative(i=123)], dialect_generic_list=[{"a": 1}])

        o_deepcopy = o.deepcopy()
        o.struct_list[0].i = -1
        o.dialect_generic_list[0]["b"] = 2

        self.assertEqual(o.struct_list[0].i, -1)
        self.assertEqual(o.dialect_generic_list[0], {"a": 1, "b": 2})
        self.assertEqual(o_deepcopy.struct_list[0].i, 123)
        self.assertEqual(o_deepcopy.dialect_generic_list[0], {"a": 1})

        # TODO struct deepcopy doesnt actually account for this case right now, which relies on memo passing
        # deepcopy supports ensuring that object instances that appear multiple times in a container will remain
        # the same ( copied ) instance in the copy.  Uncomment final assert if/when this is fixed
        class Inner(csp.Struct):
            v: int

        class Outer(csp.Struct):
            a3: [object]

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

    def test_deepcopy_from(self):
        source = StructWithLists(struct_list=[BaseNative(i=123)], dialect_generic_list=[{"a": 1}])

        blank = StructWithLists()
        blank.deepcopy_from(source)

        source.struct_list[0].i = -1
        source.dialect_generic_list[0]["b"] = 2

        self.assertEqual(source.struct_list[0].i, -1)
        self.assertEqual(source.dialect_generic_list[0], {"a": 1, "b": 2})
        self.assertEqual(blank.struct_list[0].i, 123)
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
        from csp.impl.struct import defineStruct

        BigStruct = defineStruct("BigStruct", {k: float for k in "abcdefghijklmnopqrdtuvwxyz"})

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

    def test_from_dict_loop_no_defaults(self):
        looped = StructNoDefaults.from_dict(StructNoDefaults(a1=[9, 10]).to_dict())
        self.assertEqual(looped, StructNoDefaults(a1=[9, 10]))

    def test_from_dict_loop_with_defaults(self):
        looped = StructWithDefaults.from_dict(StructWithDefaults().to_dict())
        # Note that we cant compare numpy arrays, so we check them independently
        comp = StructWithDefaults()
        self.assertTrue(np.array_equal(looped.np_arr, comp.np_arr))

        del looped.np_arr
        del comp.np_arr
        self.assertEqual(looped, comp)

    def test_from_dict_loop_with_generic_typing(self):
        class MyStruct(csp.Struct):
            foo: typing.Set[int]
            bar: typing.Tuple[str]
            np_arr: csp.typing.NumpyNDArray[float]

        looped = MyStruct.from_dict(MyStruct(foo=set((9, 10)), bar=("a", "b"), np_arr=np.array([1, 3])).to_dict())
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
            value: typing.Tuple[int]
            set_value: typing.Set[str]

        class S(csp.Struct):
            d: typing.Dict[str, S1]
            ls: typing.List[int]
            lc: typing.List[S2]

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
            l: [object]

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

    def test_defineNestedStruct(self):
        from csp.impl.struct import defineNestedStruct

        metadata = {
            "a": float,
            "b": int,
            "c": {
                "x": MyEnum,
                "y": [int],
            },
            "d": {"s": object, "t": [object]},
        }
        TestStruct = defineNestedStruct("TestStruct", metadata)
        self.assertEqual(TestStruct.__name__, "TestStruct")
        self.assertEqual(list(TestStruct.metadata().keys()), ["a", "b", "c", "d"])
        self.assertEqual(TestStruct.metadata()["a"], float)
        self.assertEqual(TestStruct.metadata()["b"], int)
        c = TestStruct.metadata()["c"]
        self.assertTrue(issubclass(c, csp.Struct))
        self.assertEqual(c.__name__, "TestStruct_c")
        self.assertEqual(c.metadata(), metadata["c"])
        d = TestStruct.metadata()["d"]
        self.assertTrue(issubclass(d, csp.Struct))
        self.assertEqual(d.__name__, "TestStruct_d")
        self.assertEqual(d.metadata(), metadata["d"])

        defaults = {"a": 0.0, "c": {"y": []}, "d": {}}
        TestStruct2 = defineNestedStruct("TestStruct2", metadata, defaults)
        s = TestStruct2()
        self.assertEqual(s.a, 0.0)
        self.assertEqual(s.c, s.metadata()["c"]())
        self.assertEqual(s.c.y, [])
        self.assertEqual(s.d, s.metadata()["d"]())

    def test_all_fields_set(self):
        from csp.impl.struct import defineStruct

        types = [int, bool, list, str]
        for num_fields in range(1, 25):
            meta = {chr(ord("a") + x): types[x % len(types)] for x in range(num_fields)}
            stype = defineStruct("foo", meta)
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
            stype2 = defineStruct("foo", meta2, base=stype)
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
        from csp.impl.struct import defineStruct

        for i in range(1000):
            name = f"struct_{i}"
            fieldname = f"field{i}"
            S = defineStruct(name, {fieldname: int})
            s = S()
            setattr(s, fieldname, i)
            ts = getattr(csp.const(s), fieldname)
            csp.run(ts, starttime=datetime.utcnow(), endtime=timedelta())

    def test_struct_printing(self):
        # simple test
        class StructA(csp.Struct):
            a: int
            b: str
            c: typing.List[int]

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
        exp_repr_s2 = f"StructB( a={repr(timedelta(1))}, b={repr(datetime(2020,1,1))}, c=False, d={repr(f1)}, e=ClassA(a=1), f=StructA( a=1, b=b, c=[1, 2] ), g=<EnumA.RED: 1>, h={repr(f2)}, i={repr(f3)}, j={repr(f4)}, k={repr(f5)} )"
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
            b: typing.List[StructC]

        s5 = StructE(
            a=StructA(a=1, b="b", c=[1, 2]),
            b=[StructC(a=2, b="b", c=[3, 4], d=2, e="e"), StructC(a=3, b="b", c=[5, 6], d=3, e="e")],
        )
        exp_repr_s5 = "StructE( a=StructA( a=1, b=b, c=[1, 2] ), b=[StructC( a=2, b=b, c=[3, 4], d=2, e=e ), StructC( a=3, b=b, c=[5, 6], d=3, e=e )] )"
        self.assertEqual(repr(s5), exp_repr_s5)
        self.assertEqual(str(s5), exp_repr_s5)

        # test array fields
        class StructF(csp.Struct):
            a: typing.List[int]
            b: typing.List[bool]
            c: typing.List[typing.List[float]]
            d: typing.List[ClassA]
            e: typing.List[EnumA]
            f: typing.List[StructC]  # leave unset for test

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
            b: typing.List[StructA]
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
            sType = defineStruct("foo", {"a": dict})
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
            l: [bool]

        raw = [True, False, True]
        a = A(l=raw)
        self.assertTrue(all(a.l[i] is raw[i] for i in range(3)))

        r = repr(a)
        self.assertTrue(repr(raw) in r)

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
            l_i: typing.List[int]
            l_b: typing.List[bool]
            l_dt: typing.List[datetime]
            l_l_i: typing.List[typing.List[int]]
            l_tuple: typing.Tuple[int, float, str]
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

        l_any = [[1, 2], "hello", [4, 3.2, [6, [7], (8, True, 10.5, (11, [12, False]))]]]
        l_any_result = [[1, 2], "hello", [4, 3.2, [6, [7], [8, True, 10.5, [11, [12, False]]]]]]
        test_struct = MyStruct(i=456, l_any=l_any)
        result_dict = {"i": 456, "l_any": l_any_result}
        self.assertEqual(json.loads(test_struct.to_json()), result_dict)

    def test_to_json_dict(self):
        class MyStruct(csp.Struct):
            i: int = 123
            d_i: typing.Dict[int, int]
            d_dt: typing.Dict[str, datetime]
            d_d_s: typing.Dict[str, typing.Dict[str, str]]
            d_any: dict

        test_struct = MyStruct()
        result_dict = {"i": 123}
        self.assertEqual(json.loads(test_struct.to_json()), result_dict)

        d_i = {1: 2, 3: 4, 5: 6}
        d_i_res = {str(k): v for k, v in d_i.items()}
        test_struct = MyStruct(i=456, d_i=d_i)
        result_dict = {"i": 456, "d_i": d_i_res}
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

        dt = datetime.now(tz=pytz.utc)
        d_dt = {"d1": dt, "d2": dt}
        test_struct = MyStruct(i=456, d_any=d_dt)
        result_dict = json.loads(test_struct.to_json())
        self.assertEqual({k: datetime.fromisoformat(d) for k, d in result_dict["d_any"].items()}, d_dt)

        d_any = {"b1": {1: "k1", "d2": {4: 5.5}}, "b2": {"d3": {}, "d4": {"d5": {"d6": {"d7": {}}}}}}
        d_any_res = {"b1": {"1": "k1", "d2": {"4": 5.5}}, "b2": {"d3": {}, "d4": {"d5": {"d6": {"d7": {}}}}}}
        test_struct = MyStruct(i=456, d_any=d_any)
        result_dict = {"i": 456, "d_any": d_any_res}
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
            l_ncsp: typing.List[NonCspStruct]
            py_l_ncsp: list

        class MyStruct(csp.Struct):
            i: int = 789
            s: str = "MyStruct"
            ts: datetime
            l_mss: typing.List[MySubStruct]
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


if __name__ == "__main__":
    unittest.main()
