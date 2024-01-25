import typing
import unittest
from datetime import datetime

import csp
from csp import ts


class TestNumbaTypeChecking(unittest.TestCase):
    @unittest.skip("numba not yet used, tests fail on newer numba we get in our 3.8 build")
    def test_graph_build_type_checking(self):
        @csp.numba_node
        def typed_ts(x: ts[int]):
            if csp.ticked(x):
                pass

        @csp.numba_node
        def typed_scalar(x: ts[int], y: str):
            if csp.ticked(x):
                pass

        @csp.graph
        def graph():
            i = csp.const(5)
            typed_ts(i)

            typed_scalar(i, "xyz")

            with self.assertRaisesRegex(TypeError, "Expected ts\\[int\\] for argument 'x', got ts\\[str\\]"):
                s = csp.const("xyz")
                ## THIS SHOULD RAISE, passing ts[str] but typed takes ts[int]
                typed_ts(s)

            with self.assertRaisesRegex(TypeError, "Expected str for argument 'y', got 123 \\(int\\)"):
                ## THIS SHOULD RAISE, passing int instead of str
                typed_scalar(i, 123)

        csp.run(graph, starttime=datetime(2020, 2, 7, 9), endtime=datetime(2020, 2, 7, 9, 1))

    @unittest.skip("numba not yet used, tests fail on newer numba we get in our 3.8 build")
    def test_runtime_type_check(self):
        ## native output type
        @csp.numba_node
        def typed_int(x: ts["T"]) -> ts[int]:
            if csp.ticked(x):
                return x

        # TODO: Uncomment
        # @csp.numba_node
        # def typed_alarm(v: '~T', alarm_type: 'V') -> outputs(ts['V']):
        #     with csp.alarms():
        #         alarm = csp.alarm( 'V' )
        #     with csp.start():
        #         csp.schedule_alarm(alarm, timedelta(), v)
        #
        #     if csp.ticked(alarm):
        #         return alarm

        # Valid
        csp.run(typed_int, csp.const(5), starttime=datetime(2020, 2, 7))

        # Invalid
        with self.assertRaisesRegex(RuntimeError, "Unable to resolve getter function for type.*"):
            csp.run(typed_int, csp.const("5"), starttime=datetime(2020, 2, 7))

        # TODO: uncomment
        # # valid
        # csp.run(typed_alarm, 5, int, starttime=datetime(2020, 2, 7))
        # csp.run(typed_alarm, 5, object, starttime=datetime(2020, 2, 7))
        # csp.run(typed_alarm, [1, 2, 3], [int], starttime=datetime(2020, 2, 7))
        #
        # # Invalid
        # with self.assertRaisesRegex(TypeError,
        #                             '"typed_alarm" node expected output type on output #0 to be of type "str" got type "int"'):
        #     csp.run(typed_alarm, 5, str, starttime=datetime(2020, 2, 7))
        #
        # with self.assertRaisesRegex(TypeError,
        #                             '"typed_alarm" node expected output type on output #0 to be of type "bool" got type "int"'):
        #     csp.run(typed_alarm, 5, bool, starttime=datetime(2020, 2, 7))
        #
        # with self.assertRaisesRegex(TypeError,
        #                             '"typed_alarm" node expected output type on output #0 to be of type "str" got type "list"'):
        #     csp.run(typed_alarm, [1, 2, 3], str, starttime=datetime(2020, 2, 7))

    @unittest.skip("numba not yet used, tests fail on newer numba we get in our 3.8 build")
    def test_dict_type_resolutions(self):
        @csp.numba_node
        def typed_dict_int_int(x: {int: int}):
            pass

        @csp.numba_node
        def typed_dict_int_int2(x: typing.Dict[int, int]):
            pass

        @csp.numba_node
        def typed_dict_int_float(x: {int: int}):
            pass

        @csp.numba_node
        def typed_dict_float_float(x: {float: float}):
            pass

        @csp.numba_node
        def typed_dict(x: {"T": "V"}):
            pass

        @csp.numba_node
        def deep_nested_generic_resolution(x: "T1", y: "T2", z: {"T1": {"T2": [{"T1"}]}}):
            pass

        @csp.graph
        def graph():
            d_i_i = csp.const({1: 2, 3: 4})
            csp.add_graph_output("o1", d_i_i)

            # Ok int dict expected
            typed_dict_int_int({1: 2, 3: 4})

            # Ok int dict expected
            typed_dict_int_int2({1: 2, 3: 4})

            typed_dict_float_float({1: 2})
            typed_dict_float_float({1.0: 2})
            typed_dict_float_float({})

            with self.assertRaisesRegex(TypeError, r"Expected typing.Dict\[int, int\] for argument 'x', got .*"):
                # Passing a float value instead of expected ints
                typed_dict_int_int2({1: 2, 3: 4.0})

            l_good = csp.const.using(T={int: float})({})
            csp.add_graph_output("o2", l_good)
            l_good = csp.const.using(T={int: float})({2: 1})
            csp.add_graph_output("o3", l_good)
            l_good = csp.const.using(T={int: float})({2: 1.0})
            csp.add_graph_output("o4", l_good)

        csp.run(graph, starttime=datetime(2020, 2, 7, 9), endtime=datetime(2020, 2, 7, 9, 1))

    @unittest.skip("numba not yet used, tests fail on newer numba we get in our 3.8 build")
    def test_list_type_resolutions(self):
        @csp.numba_node
        def typed_list_int(x: [int]):
            pass

        @csp.numba_node
        def typed_list_int2(x: typing.List[int]):
            pass

        @csp.numba_node
        def typed_list_float(x: [float]):
            pass

        def graph():
            l_i = csp.const([1, 2, 3, 4])

            typed_list_int([])
            typed_list_int([1, 2, 3])
            typed_list_int2([1, 2, 3])
            typed_list_float([1, 2, 3])
            typed_list_float([1, 2, 3.0])

            with self.assertRaisesRegex(TypeError, r"Expected typing.List\[int\] for argument 'x', got .*"):
                # Passing a float value instead of expected ints
                typed_list_int([1, 2, 3.0])

        csp.run(graph, starttime=datetime(2020, 2, 7, 9), endtime=datetime(2020, 2, 7, 9, 1))


if __name__ == "__main__":
    unittest.main()
