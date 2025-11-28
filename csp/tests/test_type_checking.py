import os
import pickle
import re
import sys
import typing
import unittest
import warnings
from datetime import datetime, time, timedelta
from typing import Callable, Dict, List, Optional, Union

import numpy as np

import csp
import csp.impl.types.instantiation_type_resolver as type_resolver
from csp import ts
from csp.impl.types.typing_utils import CspTypingUtils
from csp.impl.wiring.runtime import build_graph
from csp.typing import Numpy1DArray

USE_PYDANTIC = os.environ.get("CSP_PYDANTIC", True)


class TestTypeChecking(unittest.TestCase):
    class Dummy:
        pass

    class Dummy2(Dummy):
        pass

    class Dummy3:
        pass

    def test_graph_build_type_checking(self):
        @csp.node
        def typed_ts(x: ts[int]):
            if csp.ticked(x):
                pass

        @csp.node
        def typed_scalar(x: ts[int], y: str):
            if csp.ticked(x):
                pass

        def graph():
            i = csp.const(5)
            typed_ts(i)

            typed_scalar(i, "xyz")

            if USE_PYDANTIC:
                msg = "(?s)1 validation error for typed_ts.*" + re.escape(
                    "cannot validate ts[str] as ts[int]: <class 'str'> is not a subclass of <class 'int'>"
                )
            else:
                msg = "Expected ts\\[int\\] for argument 'x', got ts\\[str\\]"
            with self.assertRaisesRegex(TypeError, msg):
                s = csp.const("xyz")
                ## THIS SHOULD RAISE, passing ts[str] but typed takes ts[int]
                typed_ts(s)

            if USE_PYDANTIC:
                msg = "(?s)1 validation error for typed_scalar.*y.*Input should be a valid string"
            else:
                msg = "Expected str for argument 'y', got 123 \\(int\\)"
            with self.assertRaisesRegex(TypeError, msg):
                ## THIS SHOULD RAISE, passing int instead of str
                typed_scalar(i, 123)

        csp.run(graph, starttime=datetime(2020, 2, 7, 9), endtime=datetime(2020, 2, 7, 9, 1))

    def test_const_eq_scalar_no_warning(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _ = csp.const(1) == 1
            self.assertEqual(len(caught), 0)

    def test_runtime_type_check(self):
        ## native output type
        @csp.node
        def typed_int(x: ts["T"]) -> ts[int]:
            if csp.ticked(x):
                return x

        # python object output type
        @csp.node
        def typed_list(x: ts["T"]) -> ts[list]:
            if csp.ticked(x):
                return x

        @csp.node
        def typed_alarm(v: "~T", alarm_type: "V") -> ts["V"]:
            with csp.alarms():
                alarm = csp.alarm("V")
            with csp.start():
                csp.schedule_alarm(alarm, timedelta(), v)

            if csp.ticked(alarm):
                return alarm

        # Valid
        csp.run(typed_int, csp.const(5), starttime=datetime(2020, 2, 7))

        # Invalid
        with self.assertRaisesRegex(
            TypeError, '"typed_int" node expected output type on output #0 to be of type "int" got type "str"'
        ):
            csp.run(typed_int, csp.const("5"), starttime=datetime(2020, 2, 7))

        # valid
        csp.run(typed_list, csp.const([1, 2, 3]), starttime=datetime(2020, 2, 7))

        # Invalid
        with self.assertRaisesRegex(
            TypeError, '"typed_list" node expected output type on output #0 to be of type "list" got type "str"'
        ):
            csp.run(typed_list, csp.const("5"), starttime=datetime(2020, 2, 7))

        # valid
        csp.run(typed_alarm, 5, int, starttime=datetime(2020, 2, 7))
        csp.run(typed_alarm, 5, object, starttime=datetime(2020, 2, 7))
        csp.run(typed_alarm, [1, 2, 3], [int], starttime=datetime(2020, 2, 7))

        # Invalid
        with self.assertRaisesRegex(
            TypeError, '"typed_alarm" node expected output type on output #0 to be of type "str" got type "int"'
        ):
            csp.run(typed_alarm, 5, str, starttime=datetime(2020, 2, 7))

        with self.assertRaisesRegex(
            TypeError, '"typed_alarm" node expected output type on output #0 to be of type "bool" got type "int"'
        ):
            csp.run(typed_alarm, 5, bool, starttime=datetime(2020, 2, 7))

        with self.assertRaisesRegex(
            TypeError, '"typed_alarm" node expected output type on output #0 to be of type "str" got type "list"'
        ):
            csp.run(typed_alarm, [1, 2, 3], str, starttime=datetime(2020, 2, 7))

    def test_primitive_to_obj_casting(self):
        @csp.node
        def typed_ts_int(x: ts[int]):
            pass

        @csp.node
        def typed_ts_float(x: ts[float]):
            pass

        @csp.node
        def typed_ts_object(x: ts[object]):
            pass

        @csp.node
        def typed_ts_dummy(x: ts[TestTypeChecking.Dummy]):
            pass

        @csp.node
        def typed_scalar(t: "V", x: ts["V"], y: "~V"):
            pass

        @csp.node
        def typed_scalar_two_args(t: "T", x: ts["T"]):
            pass

        @csp.node
        def str_typed_scalar(x: ts["T"], y: str):
            pass

        @csp.node
        def float_typed_scalar(x: ts["T"], y: float):
            pass

        def graph():
            i = csp.const(5)
            f = csp.const(5.0)
            o = csp.const(object())
            d = csp.const(TestTypeChecking.Dummy())
            typed_ts_int(i)
            typed_ts_object(i)
            typed_ts_object(f)
            typed_ts_object(o)
            typed_ts_float(i)
            typed_ts_float(f)
            typed_ts_dummy(d)

            typed_scalar(int, i, 1)
            typed_scalar(float, f, 1.0)
            typed_scalar(object, o, object())
            typed_scalar(float, i, 1)
            typed_scalar(object, i, 1)

            # T resolved to float - OK
            typed_scalar(int, i, 1.0)

            # T resolved to  object - OK
            typed_scalar(int, i, object())

            # T resolved to  object - OK
            typed_scalar(TestTypeChecking.Dummy, o, object())

            # Weirdly ok, T is resolved to object, and all are objects
            typed_scalar(TestTypeChecking.Dummy, o, 1)

            # # Weirdly ok, T is resolved to object, and all are objects
            typed_scalar(TestTypeChecking.Dummy, i, object())

            # # Weirdly ok, T is resolved to object, and all are objects
            typed_scalar(TestTypeChecking.Dummy, i, object())
            # # Weirdly ok, T is resolved to object, and all are objects
            typed_scalar.using(V=object)(TestTypeChecking.Dummy, i, object())

            typed_scalar_two_args(TestTypeChecking.Dummy, o)
            typed_scalar_two_args(int, o)

            # OK, resolved to Dummy
            typed_scalar_two_args(TestTypeChecking.Dummy2, d)

            with self.assertRaisesRegex(TypeError, "Conflicting type resolution for V.*"):
                typed_scalar(int, i, TestTypeChecking.Dummy())

            with self.assertRaisesRegex(
                TypeError,
                "Conflicting type resolution for T.*",
            ):
                typed_scalar_two_args(TestTypeChecking.Dummy, i)

            if USE_PYDANTIC:
                msg = "(?s)1 validation error for typed_ts_int.*" + re.escape(
                    "cannot validate ts[str] as ts[int]: <class 'str'> is not a subclass of <class 'int'>"
                )
            else:
                msg = "Expected ts\\[int\\] for argument 'x', got ts\\[str\\]"
            with self.assertRaisesRegex(TypeError, msg):
                s = csp.const("xyz")
                typed_ts_int(s)

            if USE_PYDANTIC:
                msg = "(?s)1 validation error for str_typed_scalar.*Input should be a valid string"
            else:
                msg = "Expected str for argument 'y', got 123 \\(int\\)"
            with self.assertRaisesRegex(TypeError, msg):
                ## THIS SHOULD RAISE, passing int instead of str
                str_typed_scalar(i, 123)

            if USE_PYDANTIC:
                msg = "(?s)1 validation error for typed_scalar.*Input should be a valid integer"
            else:
                msg = r"Expected ~V for argument 't', got .*Dummy.*\(V=int\)"
            with self.assertRaisesRegex(TypeError, msg):
                typed_scalar.using(V=int)(TestTypeChecking.Dummy, i, object())

        csp.run(graph, starttime=datetime(2020, 2, 7, 9), endtime=datetime(2020, 2, 7, 9, 1))

    def test_dict_type_resolutions(self):
        @csp.node
        def typed_dict_int_int(x: {int: int}):
            pass

        @csp.node
        def typed_dict_int_int2(x: typing.Dict[int, int]):
            pass

        @csp.node
        def typed_dict_int_float(x: {int: int}):
            pass

        @csp.node
        def typed_dict_float_float(x: {float: float}):
            pass

        @csp.node
        def typed_dict(x: {"T": "V"}):
            pass

        @csp.node
        def typed_ts_and_scalar(x: ts[{int: int}], y: {int: int}):
            pass

        @csp.node
        def typed_ts_and_scalar_generic(x: ts[{"T": "T"}], y: {"T": "T"}, z: "~T"):
            pass

        @csp.node
        def deep_nested_generic_resolution(x: "T1", y: "T2", z: {"T1": {"T2": [{"T1"}]}}):
            pass

        @csp.node
        def deep_nested_generic_resolution2(
            x: "T1", y: "T2", z: typing.Dict["T1", typing.Dict["T2", typing.List[typing.Set["T1"]]]]
        ):
            pass

        def graph():
            d_i_i = csp.const({1: 2, 3: 4})

            # Ok int dict expected
            typed_dict_int_int({1: 2, 3: 4})

            # Ok int dict expected
            typed_dict_int_int2({1: 2, 3: 4})

            typed_dict_float_float({1: 2})
            typed_dict_float_float({1.0: 2})
            typed_dict_float_float({})

            typed_ts_and_scalar(d_i_i, {1: 2})
            typed_ts_and_scalar_generic(d_i_i, {1: 2.0}, 1)

            for f in (deep_nested_generic_resolution, deep_nested_generic_resolution2):
                f(
                    TestTypeChecking.Dummy,
                    TestTypeChecking.Dummy2,
                    {TestTypeChecking.Dummy(): {TestTypeChecking.Dummy2(): [{TestTypeChecking.Dummy()}, set()]}},
                )
                # Internal sets are Dummy and Dummy2, since Dummy2 inherits from Dummy, it's ok, it's in fact Dummy, so we are good
                f(
                    TestTypeChecking.Dummy,
                    TestTypeChecking.Dummy2,
                    {
                        TestTypeChecking.Dummy(): {
                            TestTypeChecking.Dummy2(): [{TestTypeChecking.Dummy()}, {TestTypeChecking.Dummy2()}]
                        }
                    },
                )

            if USE_PYDANTIC:
                msg = "(?s)1 validation error for typed_dict_int_int2.*Input should be a valid integer"
            else:
                msg = r"Expected typing.Dict\[int, int\] for argument 'x', got .*"
            with self.assertRaisesRegex(TypeError, msg):
                # Passing a float value instead of expected ints
                typed_dict_int_int2({1: 2, 3: 4.1})

            if USE_PYDANTIC:
                msg = "(?s)1 validation error for typed_dict_float_float.*Input should be a valid number"
            else:
                msg = r"Expected typing.Dict\[float, float\] for argument 'x', got .*"
            with self.assertRaisesRegex(TypeError, msg):
                # Passing a Dummy value instead of expected float
                typed_dict_float_float({1.0: TestTypeChecking.Dummy()})

            if USE_PYDANTIC:
                msg = "(?s)1 validation error for typed_ts_and_scalar_generic.*Conflicting type resolution for T"
            else:
                msg = "Conflicting type resolution for T when calling to typed_ts_and_scalar_generic .*"
            with self.assertRaisesRegex(TypeError, msg):
                # Passing a Dummy value instead of expected float
                typed_ts_and_scalar_generic(d_i_i, {1: 2.0}, TestTypeChecking.Dummy())

            if USE_PYDANTIC:
                msg = "(?s)1 validation error for deep_nested_generic_resolution.*Conflicting type resolution for T1"
            else:
                msg = r"Conflicting type resolution for T1 when calling to deep_nested_generic_resolution : " ".*"
            with self.assertRaisesRegex(TypeError, msg):
                # Here for inernal sets we pass Dummy and Dummy3 - they result in conflicting type resolution for T1
                deep_nested_generic_resolution(
                    TestTypeChecking.Dummy,
                    TestTypeChecking.Dummy2,
                    {
                        TestTypeChecking.Dummy(): {
                            TestTypeChecking.Dummy2(): [{TestTypeChecking.Dummy()}, {TestTypeChecking.Dummy3()}]
                        }
                    },
                )
            l_good = csp.const.using(T={int: float})({})
            l_also_good = csp.const({})
            self.assertEqual(l_also_good.tstype.typ, dict)

            l_good = csp.const.using(T={int: float})({2: 1})
            l_good = csp.const.using(T={int: float})({2: 1.0})
            with self.assertRaises(TypeError):
                # passing float to int
                l_bad = csp.const.using(T={int: float})({2.0: 1})

        csp.run(graph, starttime=datetime(2020, 2, 7, 9), endtime=datetime(2020, 2, 7, 9, 1))

    def test_list_type_resolutions(self):
        @csp.node
        def typed_list_int(x: [int]):
            pass

        @csp.node
        def typed_list_int2(x: typing.List[int]):
            pass

        @csp.node
        def typed_list_float(x: [float]):
            pass

        @csp.node
        def typed_ts_and_scalar(x: ts[[int]], y: [int]):
            pass

        @csp.node
        def typed_ts_and_scalar_generic(x: ts[["T"]], y: ["T"], z: "~T"):
            pass

        def graph():
            l_i = csp.const([1, 2, 3, 4])

            typed_list_int([])
            typed_list_int([1, 2, 3])
            typed_list_int2([1, 2, 3])
            typed_list_float([1, 2, 3])
            typed_list_float([1, 2, 3.0])

            typed_ts_and_scalar(l_i, [1, 2, 3])
            typed_ts_and_scalar_generic(l_i, [1, 2, 3], 1)

            if USE_PYDANTIC:
                msg = "(?s)1 validation error for typed_list_int.*x.*Input should be a valid integer"
            else:
                msg = r"Expected typing.List\[int\] for argument 'x', got .*"
            with self.assertRaisesRegex(TypeError, msg):
                # Passing a float value instead of expected ints
                typed_list_int([1, 2, 3.1])

            if USE_PYDANTIC:
                msg = "(?s)1 validation error for typed_list_float.*Input should be a valid number"
            else:
                msg = r"Expected typing.List\[float\] for argument 'x', got .*"
            with self.assertRaisesRegex(TypeError, msg):
                # Passing a Dummy value instead of expected float
                typed_list_float([TestTypeChecking.Dummy()])

            if USE_PYDANTIC:
                msg = "(?s)1 validation error for typed_ts_and_scalar_generic.*Conflicting type resolution for T"
            else:
                msg = "Conflicting type resolution for T when calling to typed_ts_and_scalar_generic .*"
            with self.assertRaisesRegex(TypeError, msg):
                # Passing a Dummy value instead of expected float
                typed_ts_and_scalar_generic(l_i, [1, 2], TestTypeChecking.Dummy())

            l_good = csp.const.using(T=[int])([])
            l_also_good = csp.const([])
            self.assertEqual(l_also_good.tstype.typ, list)

        csp.run(graph, starttime=datetime(2020, 2, 7, 9), endtime=datetime(2020, 2, 7, 9, 1))

    def test_set_type_resolutions(self):
        @csp.node
        def typed_set_int(x: {int}):
            pass

        @csp.node
        def typed_set_int2(x: typing.Set[int]):
            pass

        @csp.node
        def typed_set_float(x: {float}):
            pass

        @csp.node
        def typed_ts_and_scalar(x: ts[{int}], y: {int}):
            pass

        @csp.node
        def typed_ts_and_scalar_generic(x: ts[{"T"}], y: {"T"}, z: "~T"):
            pass

        def graph():
            l_i = csp.const({1, 2, 3, 4})

            typed_set_int(set())
            typed_set_int({1, 2, 3})
            typed_set_int2({1, 2, 3})
            typed_set_float({1, 2, 3})
            typed_set_float({1, 2, 3.0})

            typed_ts_and_scalar(l_i, {1, 2, 3})
            typed_ts_and_scalar_generic(l_i, {1, 2, 3}, 1)

            if USE_PYDANTIC:
                msg = "(?s)1 validation error for typed_set_int.*Input should be a valid integer"
            else:
                msg = r"Expected typing.Set\[int\] for argument 'x', got .*"
            with self.assertRaisesRegex(TypeError, msg):
                # Passing a float value instead of expected ints
                typed_set_int({1, 2, 3.1})

            if USE_PYDANTIC:
                msg = "(?s)1 validation error for typed_set_float.*Input should be a valid number"
            else:
                msg = r"Expected typing.Set\[float\] for argument 'x', got .*"
            with self.assertRaisesRegex(TypeError, msg):
                # Passing a Dummy value instead of expected float
                typed_set_float({TestTypeChecking.Dummy()})

            if USE_PYDANTIC:
                msg = "(?s)1 validation error for typed_ts_and_scalar_generic.*Conflicting type resolution for T"
            else:
                msg = "Conflicting type resolution for T when calling to typed_ts_and_scalar_generic .*"
            with self.assertRaisesRegex(TypeError, msg):
                # Passing a Dummy value instead of expected float
                typed_ts_and_scalar_generic(l_i, {1, 2}, TestTypeChecking.Dummy())

            l_good = csp.const.using(T={int})(set())
            l_also_good = csp.const(set())
            self.assertEqual(l_also_good.tstype.typ, set)

        csp.run(graph, starttime=datetime(2020, 2, 7, 9), endtime=datetime(2020, 2, 7, 9, 1))

    def test_graph_output_type_checking(self):
        with self.assertRaises(TypeError):

            @csp.graph
            def sub_graph() -> csp.OutputBasket(typing.Dict[str, ts[int]]):
                return csp.output({"x": csp.const(5), "y": csp.const(6.0)})

            def graph():
                sub_graph()

            build_graph(graph)

        with self.assertRaises(TypeError):

            @csp.graph
            def sub_graph() -> csp.OutputBasket(typing.List[ts[int]]):
                return csp.output([csp.const(5), csp.const(6.0)])

            def graph():
                sub_graph()

            build_graph(graph)

        with self.assertRaises(TypeError):

            @csp.graph
            def sub_graph() -> ts[int]:
                return csp.output(csp.const(6.0))

            def graph():
                sub_graph()

            build_graph(graph)

        with self.assertRaises(TypeError):

            @csp.graph
            def sub_graph() -> csp.Outputs(x=ts[int]):
                return csp.output(csp.const(6.0))

            def graph():
                sub_graph()

            build_graph(graph)

        with self.assertRaises(TypeError):

            @csp.graph
            def sub_graph() -> csp.Outputs(x=ts[int], y=ts[float]):
                return csp.output(x=csp.const(6.0), y=csp.const(7.0))

            def graph():
                sub_graph()

            build_graph(graph)

        @csp.graph
        def sub_graph() -> csp.OutputBasket(typing.Dict[str, ts[int]]):
            return csp.output({"x": csp.const(5), "y": csp.const(6)})

        def graph():
            sub_graph()

        build_graph(graph)

        @csp.graph
        def sub_graph() -> csp.OutputBasket(typing.List[ts[int]]):
            return csp.output([csp.const(5), csp.const(6)])

        def graph():
            sub_graph()

        build_graph(graph)

        @csp.graph
        def sub_graph() -> ts[int]:
            return csp.output(csp.const(6))

        def graph():
            sub_graph()

        build_graph(graph)

        @csp.graph
        def sub_graph() -> csp.Outputs(x=ts[int]):
            return csp.output(x=csp.const(6))

        def graph():
            sub_graph()

        build_graph(graph)

        @csp.graph
        def sub_graph() -> csp.Outputs(x=ts[float], y=ts[float]):
            return csp.output(x=csp.const(6.0), y=csp.const(7.0))

        def graph():
            sub_graph()

        build_graph(graph)

    def test_basket_type_check_bug(self):
        # Tests a bug that wasn't covered in the initial implementation. The code below was crashing on _ForwardRef before the fix
        @csp.node
        def dummy(x: csp.ts[typing.List["T"]]):
            pass

        def g():
            dummy(csp.const([1]))

        csp.run(g, starttime=datetime(2020, 2, 7, 9), endtime=datetime(2020, 2, 7, 9, 1))

    def test_graph_return_type_checking_bug(self):
        # There was a big where return order in the __return__ mattered, this tests that this bug is addressed
        @csp.graph
        def foo() -> csp.Outputs(x=ts[int], y=ts[str]):
            return csp.output(y=csp.const("hey"), x=csp.const(1))

        csp.run(foo, starttime=datetime.utcnow(), endtime=timedelta())

    def test_typed_to_untyped_container(self):
        @csp.graph
        def g(d: csp.ts[dict], s: csp.ts[set], l: csp.ts[list]):
            pass

        def main():
            g(
                d=csp.const.using(T=typing.Dict[int, int])({}),
                s=csp.const.using(T=typing.Set[int])(set()),
                l=csp.const.using(T=typing.List[int])([]),
            )

        csp.run(main, starttime=datetime.utcnow(), endtime=timedelta())

    def test_typed_to_untyped_container_wrong(self):
        @csp.graph
        def g1(d: csp.ts[dict]):
            pass

        @csp.graph
        def g2(d: csp.ts[set]):
            pass

        @csp.graph
        def g3(d: csp.ts[list]):
            pass

        def main():
            # This should fail - wrong key type in Dict
            if USE_PYDANTIC:
                msg = "(?s)1 validation error for csp.const.*Input should be a valid integer \\[type=int_type"
            else:
                msg = "In function csp\\.const: Expected ~T for argument 'value', got .* \\(dict\\)\\(T=typing\\.Dict\\[int, int\\]\\)"
            with self.assertRaisesRegex(TypeError, msg):
                g1(d=csp.const.using(T=typing.Dict[int, int])({"a": 10}))

            # This should fail - wrong element type in Set
            if USE_PYDANTIC:
                msg = "(?s)1 validation error for csp.const.*Input should be a valid integer \\[type=int_type"
            else:
                msg = "In function csp\\.const: Expected ~T for argument 'value', got .* \\(set\\)\\(T=typing\\.Set\\[int\\]\\)"
            with self.assertRaisesRegex(TypeError, msg):
                g2(d=csp.const.using(T=typing.Set[int])(set(["z"])))

            # This should fail - wrong element type in List
            if USE_PYDANTIC:
                msg = "(?s)1 validation error for csp.const.*Input should be a valid integer \\[type=int_type"
            else:
                msg = "In function csp\\.const: Expected ~T for argument 'value', got .* \\(list\\)\\(T=typing\\.List\\[int\\]\\)"
            with self.assertRaisesRegex(TypeError, msg):
                g3(d=csp.const.using(T=typing.List[int])(["d"]))

        csp.run(main, starttime=datetime.utcnow(), endtime=timedelta())

    def test_time_tzinfo(self):
        import pytz

        timetz = time(1, 2, 3, tzinfo=pytz.timezone("EST"))
        with self.assertRaisesRegex(TypeError, "csp time type does not support timezones"):
            # Now that Time is a native type it no longer supports ticking with tzinfo
            csp.run(csp.const, timetz, starttime=datetime.utcnow(), endtime=timedelta())

        res = csp.run(csp.const.using(T=object), timetz, starttime=datetime.utcnow(), endtime=timedelta())[0][0][1]
        self.assertEqual(res, timetz)

    def test_np_ndarray_ts_arg(self):
        @csp.node
        def foo(arr: csp.ts[np.ndarray]) -> csp.ts[np.ndarray]:
            return arr

        inp_arr = np.zeros(shape=(2, 2))
        st = datetime(2020, 2, 7, 9)
        res = csp.run(foo(csp.const(inp_arr)), starttime=st, endtime=datetime(2020, 2, 7, 9, 1))  # should not raise
        self.assertEqual(res[0], [(st, inp_arr)])

    def test_pickle_type_resolver_errors(self):
        errors = [
            type_resolver.ContainerTypeVarResolutionError("g", "T", "NotT"),
            type_resolver.ArgTypeMismatchError("g", "T", "NotT", "Var", {"field": 1}),
            type_resolver.ArgContainerMismatchError("g", "T", "NotT", "Var"),
            type_resolver.TSArgTypeMismatchError("g", "T", "NotT", "Var"),
            type_resolver.TSDictBasketKeyMismatchError("g", "T", "Var"),
        ]

        for err in errors:
            pickled = pickle.loads(pickle.dumps(err))
            self.assertEqual(str(err), str(pickled))

    def test_empty_containers(self):
        def g():
            x = csp.const([])
            y = csp.const(set())
            z = csp.const(dict())

            csp.add_graph_output("x", x)
            csp.add_graph_output("y", y)
            csp.add_graph_output("z", z)

        res = csp.run(g, starttime=datetime(2020, 1, 1), endtime=timedelta())
        self.assertEqual(res["x"][0][1], [])
        self.assertEqual(res["y"][0][1], set())
        self.assertEqual(res["z"][0][1], {})

    def test_callable_type_checking(self):
        @csp.node
        def node_callable_typed(x: ts[int], my_data: Callable[[int], int]) -> ts[int]:
            if csp.ticked(x):
                if my_data:
                    return my_data(x) if callable(my_data) else 12

        @csp.node
        def node_callable_untyped(x: ts[int], my_data: Callable) -> ts[int]:
            if csp.ticked(x):
                if my_data:
                    return my_data(x) if callable(my_data) else 12

        def graph():
            # These should work
            node_callable_untyped(csp.const(10), lambda x: 2 * x)
            node_callable_typed(csp.const(10), lambda x: x + 1)

            # We intentionally allow setting None to be allowed
            node_callable_typed(csp.const(10), None)
            node_callable_untyped(csp.const(10), None)

            # Here the Callable's type hints don't match the signature
            # but we allow anyways, both with the pydantic version and without
            node_callable_typed(csp.const(10), lambda x, y: "a")
            node_callable_untyped(csp.const(10), lambda x, y: "a")

            # This should fail - passing non-callable
            if USE_PYDANTIC:
                msg = "(?s)1 validation error for node_callable_untyped.*my_data.*Input should be callable \\[type=callable_type"
            else:
                msg = "In function node_callable_untyped: Expected typing\\.Callable for argument 'my_data', got 11 \\(int\\)"
            with self.assertRaisesRegex(TypeError, msg):
                node_callable_untyped(csp.const(10), 11)

        csp.run(graph, starttime=datetime(2020, 2, 7, 9), endtime=datetime(2020, 2, 7, 9, 1))

    def test_optional_type_checking(self):
        for use_dict in [True, False]:
            if use_dict:

                @csp.node
                def node_optional_list_typed(x: ts[int], my_data: Optional[Dict[int, int]] = None) -> ts[int]:
                    if csp.ticked(x):
                        return my_data[0] if my_data else x

                @csp.node
                def node_optional_list_untyped(x: ts[int], my_data: Optional[dict] = None) -> ts[int]:
                    if csp.ticked(x):
                        return my_data[0] if my_data else x
            else:

                @csp.node
                def node_optional_list_typed(x: ts[int], my_data: Optional[List[int]] = None) -> ts[int]:
                    if csp.ticked(x):
                        return my_data[0] if my_data else x

                @csp.node
                def node_optional_list_untyped(x: ts[int], my_data: Optional[list] = None) -> ts[int]:
                    if csp.ticked(x):
                        return my_data[0] if my_data else x

            def graph():
                # Optional[list] tests - these should work
                node_optional_list_untyped(csp.const(10), {} if use_dict else [])
                node_optional_list_untyped(csp.const(10), None)
                node_optional_list_untyped(csp.const(10), {9: 10} if use_dict else [9])

                # Optional[List[int]] tests
                node_optional_list_typed(csp.const(10), None)
                node_optional_list_typed(csp.const(10), {} if use_dict else [])
                node_optional_list_typed(csp.const(10), {9: 10} if use_dict else [9])

                # Here the List/Dict type hints don't match the signature
                # But, for backwards compatibility (as this was the behavior with Optional in version 0.0.5)
                # The pydantic version of the checks, however, catches this.
                if USE_PYDANTIC:
                    msg = "(?s).*validation error.* for node_optional_list_typed.*my_data.*Input should be a valid integer.*type=int_parsing"
                    with self.assertRaisesRegex(TypeError, msg):
                        node_optional_list_typed(csp.const(10), {"a": "b"} if use_dict else ["a"])
                else:
                    node_optional_list_typed(csp.const(10), {"a": "b"} if use_dict else ["a"])

                # This should fail - type mismatch
                if USE_PYDANTIC:
                    msg = "(?s)1 validation error for node_optional_list_typed.*my_data"
                else:
                    msg = "In function node_optional_list_typed: Expected typing\\.(?:Optional\\[typing|Union\\[typing)\\..*"
                with self.assertRaisesRegex(TypeError, msg):
                    node_optional_list_typed(csp.const(10), [] if use_dict else {})

            csp.run(graph, starttime=datetime(2020, 2, 7, 9), endtime=datetime(2020, 2, 7, 9, 1))

    def test_optional_callable_type_checking(self):
        @csp.node
        def node_optional_callable_typed(x: ts[int], my_data: Optional[Callable[[int], int]] = None) -> ts[int]:
            if csp.ticked(x):
                return my_data(x) if my_data else x

        @csp.node
        def node_optional_callable_untyped(x: ts[int], my_data: Optional[Callable] = None) -> ts[int]:
            if csp.ticked(x):
                return my_data(x) if my_data else x

        def graph():
            # These should work for both typed and untyped
            node_optional_callable_typed(csp.const(10), None)
            node_optional_callable_untyped(csp.const(10), None)

            # These should also work - valid callables
            node_optional_callable_typed(csp.const(10), lambda x: x + 1)
            node_optional_callable_untyped(csp.const(10), lambda x: 2 * x)

            # Here the Callable's type hints don't match the signature
            # but we allow anyways, both with the pydantic version and without
            node_optional_callable_typed(csp.const(10), lambda x, y: "a")
            node_optional_callable_untyped(csp.const(10), lambda x, y: "a")

        # This should fail - passing non-callable to typed version
        if USE_PYDANTIC:
            msg = "(?s)1 validation error for node_optional_callable_typed.*my_data.*Input should be callable \\[type=callable_type"
        else:
            msg = "In function node_optional_callable_typed: Expected typing\\.(?:Optional\\[typing\\.Callable\\[\\[int\\], int\\]\\]|Union\\[typing\\.Callable\\[\\[int\\], int\\], NoneType\\]) for argument 'my_data', got 12 \\(int\\)"
        with self.assertRaisesRegex(TypeError, msg):
            node_optional_callable_typed(csp.const(10), 12)

            # This should fail - passing non-callable to typed version
            if USE_PYDANTIC:
                msg = "(?s)1 validation error for node_optional_callable_typed.*my_data.*Input should be callable \\[type=callable_type"
            else:
                msg = "In function node_optional_callable_typed: Expected typing\\.(?:Optional\\[typing\\.Callable\\[\\[int\\], int\\]\\]|Union\\[typing\\.Callable\\[\\[int\\], int\\], NoneType\\]) for argument 'my_data', got 12 \\(int\\)"
            with self.assertRaisesRegex(TypeError, msg):
                node_optional_callable_typed(csp.const(10), 12)

        csp.run(graph, starttime=datetime(2020, 2, 7, 9), endtime=datetime(2020, 2, 7, 9, 1))

    def test_union_type_checking(self):
        @csp.node
        def node_union_typed(x: ts[int], my_data: Union[int, str]) -> ts[int]:
            if csp.ticked(x):
                return x + int(my_data) if isinstance(my_data, str) else x + my_data

        def graph():
            # These should work - valid int inputs
            node_union_typed(csp.const(10), 5)

            # These should also work - valid str inputs
            node_union_typed(csp.const(10), "123")

            # These should fail - passing float when expecting Union[int, str]
            if USE_PYDANTIC:
                msg = "(?s)2 validation errors for node_union_typed.*my_data\\.int.*Input should be a valid integer, got a number with a fractional part.*my_data\\.str.*Input should be a valid string"
            else:
                msg = "In function node_union_typed: Expected typing\\.Union\\[int, str\\] for argument 'my_data', got 12\\.5 \\(float\\)"
            with self.assertRaisesRegex(TypeError, msg):
                node_union_typed(csp.const(10), 12.5)

        csp.run(graph, starttime=datetime(2020, 2, 7, 9), endtime=datetime(2020, 2, 7, 9, 1))

    def test_union_list_type_checking(self):
        @csp.node
        def node_union_typed(x: ts[int], my_data: Union[List[str], int] = None) -> ts[int]:
            if csp.ticked(x):
                if isinstance(my_data, list):
                    return x + len(my_data)
                return x + my_data

        @csp.node
        def node_union_untyped(x: ts[int], my_data: Union[list, int] = None) -> ts[int]:
            if csp.ticked(x):
                if isinstance(my_data, list):
                    return x + len(my_data)
                return x + my_data

        def graph():
            # These should work - valid int inputs
            node_union_typed(csp.const(10), 5)
            node_union_untyped(csp.const(10), 42)

            # These should work - valid list inputs
            node_union_typed(csp.const(10), ["hello", "world"])
            node_union_untyped(csp.const(10), ["hello", "world"])

            # This should fail - passing float when expecting Union[List[str], int]
            if USE_PYDANTIC:
                msg = "(?s)2 validation errors for node_union_typed.*my_data\\.list.*Input should be a valid list.*my_data\\.int.*Input should be a valid integer, got a number with a fractional part"
            else:
                msg = "In function node_union_typed: Expected typing\\.Union\\[typing\\.List\\[str\\], int\\] for argument 'my_data', got 12\\.5 \\(float\\)"
            with self.assertRaisesRegex(TypeError, msg):
                node_union_typed(csp.const(10), 12.5)

            # This should fail - passing list with wrong element type
            if USE_PYDANTIC:
                msg = "(?s)3 validation errors for node_union_typed.*my_data\\.list\\[str\\]\\.0.*Input should be a valid string.*my_data\\.list\\[str\\]\\.1.*Input should be a valid string.*my_data\\.int.*Input should be a valid integer"
                with self.assertRaisesRegex(TypeError, msg):
                    node_union_typed(csp.const(10), [1, 2])  # List of ints instead of strings
            else:
                # We choose to intentionally not enforce the types provided
                # to maintain previous flexibility when not using pydantic type validation
                node_union_typed(csp.const(10), [1, 2])

            node_union_untyped(csp.const(10), [1, 2])

        csp.run(graph, starttime=datetime(2020, 2, 7, 9), endtime=datetime(2020, 2, 7, 9, 1))

    def test_is_callable(self):
        """Test CspTypingUtils.is_callable with various input types"""
        # Test cases as (input, expected_result) pairs
        test_cases = [
            # Direct Callable types
            (Callable, True),
            (Callable[[int, str], bool], True),
            (Callable[..., None], True),
            (Callable[[int], str], True),
            # optional Callable is not Callable
            (Optional[Callable], False),
            # Typing module types
            (List[int], False),
            (Dict[str, int], False),
            (typing.Set[str], False),
        ]
        for input_type, expected in test_cases:
            result = CspTypingUtils.is_callable(input_type)
            self.assertEqual(result, expected)

    def test_literal_typing(self):
        """Test using Literal types for type checking in CSP nodes."""
        from typing import Literal

        @csp.node
        def node_with_literal(x: ts[int], choice: Literal["a", "b", "c"]) -> ts[str]:
            if csp.ticked(x):
                return str(choice)

        @csp.graph
        def graph_with_literal(choice: Literal["a", "b", "c"]) -> ts[str]:
            return csp.const(str(choice))

        @csp.node
        def dummy_node(x: ts["T"]):  # to avoid pruning
            if csp.ticked(x):
                pass

        def graph():
            # These should work - valid literal values
            dummy_node(node_with_literal(csp.const(10), "a"))
            dummy_node(node_with_literal(csp.const(10), "b"))
            dummy_node(node_with_literal(csp.const(10), "c"))

            graph_with_literal("a")
            graph_with_literal("b")
            graph_with_literal("c")

            # This should fail with invalid literal value
            # But only pydantic type checking catches this
            if USE_PYDANTIC:
                msg = "(?s)1 validation error for node_with_literal.*choice.*"
                with self.assertRaisesRegex(TypeError, msg):
                    dummy_node(node_with_literal(csp.const(10), "d"))

        csp.run(graph, starttime=datetime(2020, 2, 7, 9), endtime=datetime(2020, 2, 7, 9, 1))

        # Test direct graph building
        csp.build_graph(graph_with_literal, "a")

        # This should fail with invalid literal value
        # But only pydantic type checking catches this
        if USE_PYDANTIC:
            msg = "(?s)1 validation error for graph_with_literal.*choice.*"
            with self.assertRaisesRegex(TypeError, msg):
                csp.build_graph(graph_with_literal, "d")

    def test_union_with_pipe_operator(self):
        """Test using the pipe operator for Union types in Python 3.10+."""

        @csp.node
        def node_with_pipe_union(x: ts[int], value: str | int | None) -> ts[str]:
            if csp.ticked(x):
                return str(value) if value is not None else "none"

        @csp.graph
        def graph_with_pipe_union(value: str | int | None) -> ts[str]:
            return csp.const(str(value) if value is not None else "none")

        @csp.node
        def dummy_node(x: ts["T"]):  # to avoid pruning
            if csp.ticked(x):
                pass

        def graph():
            # These should work - valid union types (str, int, None)
            dummy_node(node_with_pipe_union(csp.const(10), "hello"))
            dummy_node(node_with_pipe_union(csp.const(10), 42))
            dummy_node(node_with_pipe_union(csp.const(10), None))

            graph_with_pipe_union("world")
            graph_with_pipe_union(123)
            graph_with_pipe_union(None)

            # This should fail - float is not part of the union
            if USE_PYDANTIC:
                # Pydantic provides a structured error message
                msg = "(?s)2 validation errors for node_with_pipe_union.*value.*"
            else:
                # Non-Pydantic error has specific format to match
                msg = r"In function node_with_pipe_union: Expected str \| int \| None for argument 'value', got .* \(float\)"
            with self.assertRaisesRegex(TypeError, msg):
                dummy_node(node_with_pipe_union(csp.const(10), 3.14))

        csp.run(graph, starttime=datetime(2020, 2, 7, 9), endtime=datetime(2020, 2, 7, 9, 1))

        # Test direct graph building
        csp.build_graph(graph_with_pipe_union, "test")
        csp.build_graph(graph_with_pipe_union, 42)
        csp.build_graph(graph_with_pipe_union, None)

        # This should fail - bool is not explicitly included in the union
        if USE_PYDANTIC:
            msg = "(?s)2 validation errors for graph_with_pipe_union.*value.*"
        else:
            msg = r"In function graph_with_pipe_union: Expected str \| int \| None for argument 'value', got .*"
        with self.assertRaisesRegex(TypeError, msg):
            csp.build_graph(graph_with_pipe_union, 3.14)

    def test_generic_annotation(self):
        @csp.node
        def node_with_generic_default(
            x: ts[int], np_float: ts[Numpy1DArray[float]] = csp.null_ts(Numpy1DArray[float])
        ) -> ts[Numpy1DArray[float]]:
            if csp.ticked(x):
                return np.array([3.0])
            if csp.ticked(np_float):
                return np_float

        res = csp.run(
            node_with_generic_default,
            x=csp.const(11),
            starttime=datetime(2020, 2, 7, 9),
            endtime=datetime(2020, 2, 7, 9, 1),
        )
        assert res[0][0][1] == np.array([3.0])

        res = csp.run(
            node_with_generic_default,
            x=csp.null_ts(int),
            np_float=csp.const(np.array([4.0])),
            starttime=datetime(2020, 2, 7, 9),
            endtime=datetime(2020, 2, 7, 9, 1),
        )
        assert res[0][0][1] == np.array([4.0])

    def test_union_ts_with_scalar_type(self):
        """Test that Union types mixing timeseries and non-timeseries types are rejected.

        CSP should not allow a Union that contains both a ts[T] and a non-ts type like:
        Union[ts[int], int] since this mixes event-driven and static types.
        """

        # Different error patterns based on whether we're using Pydantic or not
        error_pattern = "Cannot mix TS and non-TS types in a union"

        # Test 1: Node with mixed Union input type
        with self.assertRaisesRegex(ValueError, error_pattern):

            @csp.node
            def node_mixed_union(x: Union[ts[int], int]) -> ts[int]:
                if csp.ticked(x):
                    return x

        # Test 2: Graph with mixed Union input type
        with self.assertRaisesRegex(ValueError, error_pattern):

            @csp.graph
            def graph_mixed_union(x: Union[ts[int], int]) -> ts[int]:
                # This should fail at definition time
                return x

        # Test 3: More complex Union with multiple mixed types
        with self.assertRaisesRegex(ValueError, error_pattern):

            @csp.node
            def complex_mixed_union(
                x: Union[ts[int], int, ts[str], float],
                y: ts[str],
            ) -> ts[float]:
                if csp.ticked(x):
                    if isinstance(x, ts[int]):
                        return float(x)
                    else:
                        return float(len(x))  # For ts[str]

        # Test 4: Union with nested ts in container types
        with self.assertRaisesRegex(ValueError, error_pattern):

            @csp.node
            def nested_mixed_union(
                x: str,
                y: Union[List[ts[int]], int],
            ) -> ts[int]:
                if csp.ticked(y):
                    return sum(y) if isinstance(y, list) else y

        # Test 5: Optional (should work)
        @csp.node
        def nested_optional_union(
            x: str,
            y: Optional[ts[int]] = None,
        ) -> ts[int]:
            if csp.ticked(y):
                return sum(y) if isinstance(y, list) else y


if __name__ == "__main__":
    unittest.main()
