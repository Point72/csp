import sys
import unittest
from datetime import datetime, timedelta
from typing import Callable, Dict, List

import csp
from csp import CspParseError, OutputBasket, Outputs, OutputTypeError, __outputs__, __return__, ts
from csp.impl.types.instantiation_type_resolver import ArgTypeMismatchError


class TestParsing(unittest.TestCase):
    def test_parse_errors(self):
        # These are roughly in order of exceptions that are thrown in node_parser.py ( as of the time of this writing )

        with self.assertRaisesRegex(CspParseError, "Passing arguments with \\* is unsupported for make_passive"):

            @csp.node
            def foo():
                args = []
                csp.make_passive(*args)

        with self.assertRaisesRegex(CspParseError, "Passing arguments with \\*\\* is unsupported for make_passive"):

            @csp.node
            def foo(x: ts[int]):
                kwargs = {}
                csp.make_passive(x, **kwargs)

        with self.assertRaisesRegex(CspParseError, "make_passive expects a timeseries as first positional argument"):

            @csp.node
            def foo():
                csp.make_passive(x=1)

        with self.assertRaisesRegex(CspParseError, "make_passive expects a timeseries as first positional argument"):

            @csp.node
            def foo():
                csp.make_passive()

        with self.assertRaisesRegex(CspParseError, "value_at\\(\\) got multiple values for argument 'index_or_time'"):

            @csp.node
            def foo(x: ts[int]):
                csp.value_at(x, 1, index_or_time=1)

        with self.assertRaisesRegex(CspParseError, "Invalid use of 'with_shape'"):

            @csp.node
            def foo(x: List[str]):
                __outputs__(List[ts[int]].with_shape(x=1))
                pass

        with self.assertRaisesRegex(CspParseError, "__outputs__ must all be named or be single output, cant be both"):

            @csp.node
            def foo(x: List[str]):
                __outputs__(ts[int], x=ts[bool])
                pass

        with self.assertRaisesRegex(CspParseError, "__outputs__ single unnamed arg only"):

            @csp.node
            def foo(x: List[str]):
                __outputs__(ts[int], ts[bool])
                pass

        with self.assertRaisesRegex(
            OutputTypeError, "Outputs must all be named or be a single unnamed output, cant be both"
        ):

            @csp.node
            def foo(x: List[str]) -> Outputs(ts[int], ts[bool]):
                pass

        with self.assertRaisesRegex(
            OutputTypeError, "Outputs must all be named or be a single unnamed output, cant be both"
        ):

            @csp.node
            def foo(x: List[str]) -> Outputs(ts[int], x=ts[bool]):
                pass

        with self.assertRaisesRegex(
            CspParseError, "csp.node and csp.graph outputs must be via return annotation or __outputs__ call, not both"
        ):

            @csp.node
            def foo(x: List[str]) -> Outputs(ts[int]):
                __outputs__(ts[int])
                pass

        with self.assertRaisesRegex(
            CspParseError,
            "Invalid usage of __outputs__, it should appear at the beginning of the function \\(consult documentation for details\\)",
        ):

            @csp.node
            def foo(x: List[str]):
                x = 1
                __outputs__(ts[int])

        with self.assertRaisesRegex(CspParseError, "\\*args and \\*\\*kwargs arguments are not supported in csp nodes"):

            @csp.node
            def foo(*args):
                __outputs__(ts[int])
                pass

        with self.assertRaisesRegex(CspParseError, "position only arguments are not supported in csp nodes"):

            @csp.node
            def posonly_sample(
                posonlyargs,
                /,
            ):
                __outputs__(ts[int])
                pass

        with self.assertRaisesRegex(CspParseError, "csp.node and csp.graph args must be type annotated"):

            @csp.node
            def foo(x):
                __outputs__(ts[int])
                pass

        with self.assertRaisesRegex(CspParseError, "__alarms__ does not accept positional arguments"):

            @csp.node
            def foo():
                __alarms__(x)
                pass

        with self.assertRaisesRegex(CspParseError, "Alarms must be initialized with csp.alarm in __alarms__ block"):

            @csp.node
            def foo():
                with __alarms__():
                    x = 5
                pass

        with self.assertRaisesRegex(CspParseError, "alarms must be ts types"):

            @csp.node
            def foo():
                __alarms__(x=int)
                pass

        with self.assertRaisesRegex(CspParseError, "Alarms must be initialized with csp.alarm in __alarms__ block"):

            @csp.node
            def foo():
                with __alarms__():
                    x: int = 5
                pass

        with self.assertRaisesRegex(CspParseError, "Alarms must be initialized with csp.alarm in __alarms__ block"):

            @csp.node
            def foo():
                with __alarms__():
                    x: ts[int]
                pass

        with self.assertRaisesRegex(CspParseError, "Alarms must be initialized with csp.alarm in __alarms__ block"):

            @csp.node
            def foo():
                with __alarms__():
                    x: ts[int] = 5
                pass

        with self.assertRaisesRegex(CspParseError, "unrecognized input 'z'"):

            @csp.node
            def foo():
                csp.make_passive(z)

        with self.assertRaisesRegex(CspParseError, "expected 'z' to be a timeseries input"):

            @csp.node
            def foo(z: int):
                csp.make_passive(z)

        with self.assertRaisesRegex(CspParseError, "invalid csp call csp.make_cpassive"):

            @csp.node
            def foo():
                csp.make_cpassive()

        with self.assertRaisesRegex(CspParseError, "returning from within a for or while loop is not supported"):

            @csp.node
            def foo():
                __outputs__(ts[int])
                for _ in range(10):
                    return 1

        with self.assertRaisesRegex(CspParseError, "returning from within a for or while loop is not supported"):

            @csp.node
            def foo() -> Outputs(ts[int]):
                for _ in range(10):
                    return 1

        with self.assertRaisesRegex(
            CspParseError,
            "Returning multiple outputs must use the following syntax: return csp.output\\(out1=val1, \\.\\.\\.\\)",
        ):

            @csp.node
            def foo():
                __outputs__(x=ts[int], y=ts[int])
                return 1

        with self.assertRaisesRegex(
            CspParseError,
            "Returning multiple outputs must use the following syntax: return csp.output\\(out1=val1, \\.\\.\\.\\)",
        ):

            @csp.node
            def foo() -> Outputs(x=ts[int], y=ts[int]):
                return 1

        with self.assertRaisesRegex(CspParseError, ".*single unnamed arg in node returning 2 outputs.*"):

            @csp.node
            def foo():
                __outputs__(x=ts[int], y=ts[int])
                __return__(1)

        with self.assertRaisesRegex(CspParseError, ".*single unnamed arg in node returning 2 outputs.*"):

            @csp.node
            def foo() -> Outputs(x=ts[int], y=ts[int]):
                __return__(1)

        with self.assertRaisesRegex(CspParseError, "returning from node without any outputs defined"):

            @csp.node
            def foo(x: ts[int]):
                return 1

        with self.assertRaisesRegex(CspParseError, "returning from node without any outputs defined"):

            @csp.node
            def foo(x: ts[int]):
                return __return__(1)

        with self.assertRaisesRegex(
            CspParseError,
            "csp.output expects to be called with \\(output, value\\) or \\(output = value, output2 = value2\\)",
        ):

            @csp.node
            def foo(x: ts[int]):
                __outputs__(ts[int])
                csp.output(1, 2, 3)

        with self.assertRaisesRegex(
            CspParseError,
            "csp.output expects to be called with \\(output, value\\) or \\(output = value, output2 = value2\\)",
        ):

            @csp.node
            def foo(x: ts[int]) -> Outputs(ts[int]):
                csp.output(1, 2, 3)

        with self.assertRaisesRegex(
            CspParseError,
            "return expects to be called with \\(output, value\\) or \\(output = value, output2 = value2\\)",
        ):

            @csp.node
            def foo(x: ts[int]):
                __outputs__(ts[int])
                __return__(1, 2, 3)

        with self.assertRaisesRegex(
            CspParseError,
            "return expects to be called with \\(output, value\\) or \\(output = value, output2 = value2\\)",
        ):

            @csp.node
            def foo(x: ts[int]) -> Outputs(ts[int]):
                __return__(1, 2, 3)

        with self.assertRaisesRegex(CspParseError, "cannot csp.output single unnamed arg in node returning 2 outputs"):

            @csp.node
            def foo(x: ts[int]):
                __outputs__(x=ts[int], y=ts[bool])
                csp.output(5)

        with self.assertRaisesRegex(CspParseError, "cannot csp.output single unnamed arg in node returning 2 outputs"):

            @csp.node
            def foo(x: ts[int]) -> Outputs(x=ts[int], y=ts[bool]):
                csp.output(5)

        with self.assertRaisesRegex(CspParseError, "cannot return single unnamed arg in node returning 2 outputs"):

            @csp.node
            def foo(x: ts[int]):
                __outputs__(x=ts[int], y=ts[bool])
                __return__(5)

        with self.assertRaisesRegex(CspParseError, "cannot return single unnamed arg in node returning 2 outputs"):

            @csp.node
            def foo(x: ts[int]) -> Outputs(x=ts[int], y=ts[bool]):
                __return__(5)

        with self.assertRaisesRegex(CspParseError, "returning from node without any outputs defined"):

            @csp.node
            def foo(x: ts[int]):
                csp.output(5)

        with self.assertRaisesRegex(CspParseError, "returning from node without any outputs defined"):

            @csp.node
            def foo(x: ts[int]):
                __return__(5)

        with self.assertRaisesRegex(
            CspParseError, "csp.output\\(x\\[k\\],v\\) syntax can only be used on basket outputs"
        ):

            @csp.node
            def foo(x: ts[int]):
                __outputs__(x=ts[bool])
                csp.output(x[1], 7)

        with self.assertRaisesRegex(
            CspParseError, "csp.output\\(x\\[k\\],v\\) syntax can only be used on basket outputs"
        ):

            @csp.node
            def foo(x: ts[int]) -> Outputs(x=ts[bool]):
                csp.output(x[1], 7)

        with self.assertRaisesRegex(
            CspParseError, "Invalid use of return basket element returns is not possible with return"
        ):

            @csp.node
            def foo(x: ts[int]):
                __outputs__(x=ts[bool])
                __return__(x[1], 7)

        with self.assertRaisesRegex(
            CspParseError, "Invalid use of return basket element returns is not possible with return"
        ):

            @csp.node
            def foo(x: ts[int]) -> Outputs(x=ts[bool]):
                __return__(x[1], 7)

        with self.assertRaisesRegex(CspParseError, "unrecognized output 'x'"):

            @csp.node
            def foo(x: ts[int]):
                __outputs__(z=ts[bool])
                csp.output(x, 7)

        with self.assertRaisesRegex(CspParseError, "unrecognized output 'x'"):

            @csp.node
            def foo(x: ts[int]) -> Outputs(z=ts[bool]):
                csp.output(x, 7)

        with self.assertRaisesRegex(CspParseError, "unrecognized output 'x'"):

            @csp.node
            def foo(x: ts[int]):
                __outputs__(z=ts[bool])
                __return__(x, 7)

        with self.assertRaisesRegex(CspParseError, "unrecognized output 'x'"):

            @csp.node
            def foo(x: ts[int]) -> Outputs(z=ts[bool]):
                __return__(x, 7)

        with self.assertRaisesRegex(CspParseError, "unrecognized output 'x'"):

            @csp.node
            def foo(x: ts[int]):
                __outputs__(z=ts[bool])
                csp.output(x=7)

        with self.assertRaisesRegex(CspParseError, "unrecognized output 'x'"):

            @csp.node
            def foo(x: ts[int]) -> Outputs(z=ts[bool]):
                csp.output(x=7)

        with self.assertRaisesRegex(CspParseError, "unrecognized output 'x'"):

            @csp.node
            def foo(x: ts[int]):
                __outputs__(z=ts[bool])
                __return__(x=7)

        with self.assertRaisesRegex(CspParseError, "unrecognized output 'x'"):

            @csp.node
            def foo(x: ts[int]) -> Outputs(z=ts[bool]):
                __return__(x=7)

        with self.assertRaisesRegex(CspParseError, "csp.now takes no arguments"):

            @csp.node
            def foo(x: ts[int]):
                csp.now(5)

        with self.assertRaisesRegex(CspParseError, "csp.now takes no arguments"):

            @csp.node
            def foo(x: ts[int]):
                csp.now(x=5)

        with self.assertRaisesRegex(CspParseError, "cannot schedule alarm on non-alarm input 'x'"):

            @csp.node
            def foo(x: ts[int]):
                csp.schedule_alarm(x)

        with self.assertRaisesRegex(
            CspParseError, "node has __outputs__ defined but no return or csp.output statements"
        ):

            @csp.node
            def foo(x: ts[int]):
                __outputs__(ts[int])
                if csp.ticked(x):
                    pass

        with self.assertRaisesRegex(CspParseError, "output 'y' is never returned"):

            @csp.node
            def foo(x: ts[int]):
                __outputs__(x=ts[int], y=ts[int])
                if csp.ticked(x):
                    csp.output(x=1)

        with self.assertRaisesRegex(CspParseError, r"outputs must be ts\[\] or basket types, got <class \'dict\'\>"):

            @csp.node
            def foo():
                __outputs__(dict)
                pass

        with self.assertRaisesRegex(
            CspParseError, "output baskets must define shape using with_shape or with_shape_of"
        ):

            @csp.node
            def foo():
                __outputs__(x={str: ts[int]})
                pass

        with self.assertRaisesRegex(
            CspParseError, "output baskets must define shape using with_shape or with_shape_of"
        ):

            @csp.node
            def foo():
                __outputs__(x=[ts[int]])
                pass

    def test_special_block(self):
        with self.assertRaisesRegex(CspParseError, "__outputs__ can not be used in a with statement"):

            @csp.node
            def foo():
                with __outputs__():
                    pass

        with self.assertRaisesRegex(CspParseError, "__start__ must be used in a with statement"):

            @csp.node
            def foo():
                __start__()

        with self.assertRaisesRegex(CspParseError, "__stop__ must be used in a with statement"):

            @csp.node
            def foo():
                __stop__()

        @csp.node
        def foo():
            __alarms__()

        @csp.node
        def foo():
            __state__(x=10)

        @csp.node
        def foo():
            __alarms__()
            __state__(x=10)

        @csp.node
        def foo():
            with __alarms__():
                pass

        @csp.node
        def foo():
            with __alarms__():
                ...

        @csp.node
        def foo():
            with __state__(x=10):
                pass

        @csp.node
        def foo():
            with __state__(x=10):
                ...

        @csp.node
        def foo():
            with __alarms__():
                pass
            with __state__(x=10):
                pass

        @csp.node
        def foo():
            with __alarms__(y=ts[int]):
                pass
            with __state__(x=10):
                pass

        @csp.node
        def foo():
            with __alarms__():
                y: ts[int] = csp.alarm(int)
            with __state__(x=10):
                pass

        @csp.node
        def foo():
            with __alarms__():
                y: ts[bool] = csp.alarm(bool)
            with __state__(x=10):
                pass
            with __start__():
                csp.schedule_alarm(y, 4, True)

        with self.assertRaisesRegex(CspParseError, "__alarms__ does not accept positional arguments"):

            @csp.node
            def foo():
                with __alarms__(ts[int]):
                    pass

        with self.assertRaisesRegex(CspParseError, "__alarms__ must be called, cannot use as a bare name"):

            @csp.node
            def foo():
                with __alarms__:
                    pass

        with self.assertRaisesRegex(CspParseError, "__state__ must be called, cannot use as a bare name"):

            @csp.node
            def foo():
                with __state__:
                    pass

        with self.assertRaisesRegex(CspParseError, "csp.__state__ must be called, cannot use as a bare name"):

            @csp.node
            def foo():
                with csp.__state__:
                    pass

        with self.assertRaisesRegex(CspParseError, "__start__ must be called, cannot use as a bare name"):

            @csp.node
            def foo():
                with __start__:
                    pass

        with self.assertRaisesRegex(CspParseError, "csp.__start__ must be called, cannot use as a bare name"):

            @csp.node
            def foo():
                with csp.__start__:
                    pass

        with self.assertRaisesRegex(CspParseError, "__stop__ must be called, cannot use as a bare name"):

            @csp.node
            def foo():
                with __stop__:
                    pass

        with self.assertRaisesRegex(CspParseError, "csp.__stop__ must be called, cannot use as a bare name"):

            @csp.node
            def foo():
                with csp.__stop__:
                    pass

        @csp.node
        def foo():
            __outputs__(x=ts[int])
            __alarms__(y=ts[int])
            __state__(z=1)
            with __start__():
                pass
            with __stop__():
                pass
            __return__(x=1)

        @csp.node
        def foo():
            __outputs__(x=ts[int])
            __alarms__(y=ts[int])
            __state__(z=1)
            with __start__():
                ...
            with __stop__():
                ...
            __return__(x=1)

        @csp.node
        def foo():
            return

        with self.assertRaisesRegex(CspParseError, "__outputs__ must be declared before __stop__"):

            @csp.node
            def foo():
                __alarms__(y=ts[int])
                __state__(z=1)
                with __start__():
                    pass
                with __stop__():
                    pass
                __outputs__(x=ts[int])

                __return__(x=1)

        with self.assertRaisesRegex(CspParseError, "__alarms__ must be declared before __state__"):

            @csp.node
            def foo():
                __outputs__(x=ts[int])
                __state__(z=1)
                __alarms__(y=ts[int])
                with __start__():
                    pass
                with __stop__():
                    pass

                __return__(x=1)

        with self.assertRaisesRegex(CspParseError, "__state__ must be declared before __start__"):

            @csp.node
            def foo():
                __outputs__(x=ts[int])
                __alarms__(y=ts[int])
                with __start__():
                    pass
                __state__(z=1)
                with __stop__():
                    pass

                __return__(x=1)

        with self.assertRaisesRegex(CspParseError, "__start__ must be declared before __stop__"):

            @csp.node
            def foo():
                __outputs__(x=ts[int])
                __alarms__(y=ts[int])
                __state__(z=1)
                with __stop__():
                    pass
                with __start__():
                    pass

                __return__(x=1)

        with self.assertRaisesRegex(CspParseError, "start must be declared before __stop__"):

            @csp.node
            def foo():
                with __stop__():
                    pass
                with csp.start():
                    pass

        with self.assertRaisesRegex(CspParseError, "__start__ must be declared before stop"):

            @csp.node
            def foo():
                with csp.stop():
                    pass
                with __start__():
                    pass

        #  test with mixed pythonic/dunder syntax
        @csp.node
        def foo():
            with csp.start():
                pass
            with __stop__():
                pass

        @csp.node
        def foo():
            with __start__():
                pass
            with csp.stop():
                pass

        with self.assertRaisesRegex(CspParseError, "Invalid usage of __stop__, .*"):

            @csp.node
            def foo():
                __outputs__(x=ts[int])
                __alarms__(y=ts[int])
                __state__(z=1)
                with __start__():
                    pass
                __return__(x=1)
                with __stop__():
                    pass

    def test_method_parsing(self):
        class C:
            @csp.node
            def f(self, x: int):
                pass

        with self.assertRaises(CspParseError):

            class C:
                @csp.node
                def f(self, x):
                    pass

        with self.assertRaises(CspParseError):

            @csp.node
            def f(self, x: int):
                pass

        class C:
            @csp.graph
            def f(self, x: int):
                pass

        with self.assertRaises(CspParseError):

            class C:
                @csp.graph
                def f(self, x):
                    pass

        with self.assertRaises(CspParseError):

            @csp.graph
            def f(self, x: int):
                pass

    def test_classmethod_parsing(self):
        class C:
            @classmethod
            @csp.node
            def f(cls, x: int):
                pass

        with self.assertRaises(CspParseError):

            class C:
                @csp.node
                def f(cls, x):
                    pass

        with self.assertRaises(CspParseError):

            @csp.node
            def f(cls, x: int):
                pass

        class C:
            @classmethod
            @csp.graph
            def f(cls, x: int):
                pass

        with self.assertRaises(CspParseError):

            class C:
                @csp.graph
                def f(cls, x):
                    pass

        with self.assertRaises(CspParseError):

            @csp.graph
            def f(cls, x: int):
                pass

    def test_graph_return_parsing(self):
        @csp.graph
        def graph():
            __outputs__(ts[int])
            return csp.const(5)

        @csp.graph
        def graph():
            __outputs__(ts[int])
            __return__(csp.const(5))

        @csp.graph
        def graph():
            __outputs__(ts[int])
            __return__(csp.const(5))

        @csp.graph
        def graph():
            __outputs__(x=ts[int])
            return csp.const(5)

        @csp.graph
        def graph():
            __outputs__(x=ts[int])
            __return__(csp.const(5))

        @csp.graph
        def graph():
            __outputs__(x=ts[int])
            __return__(x=csp.const(5))

        @csp.graph
        def graph():
            __outputs__(x=ts[int], y=ts[float])
            __return__(x=csp.const(5), y=csp.const(6.0))

        with self.assertRaisesRegex(CspParseError, "return does not return values with non empty outputs"):

            @csp.graph
            def graph():
                __outputs__(ts[int])
                return

        with self.assertRaisesRegex(CspParseError, "return does not return values with non empty outputs"):

            @csp.graph
            def graph() -> Outputs(ts[int]):
                return

        with self.assertRaisesRegex(CspParseError, "return does not return values with non empty outputs"):

            @csp.graph
            def graph():
                __outputs__(ts[int])
                __return__()

        with self.assertRaisesRegex(CspParseError, "return does not return values with non empty outputs"):

            @csp.graph
            def graph() -> Outputs(ts[int]):
                __return__()

        with self.assertRaisesRegex(
            CspParseError,
            "Returning multiple outputs must use the following syntax: return csp.output\\(out1=val1, \\.\\.\\.\\)",
        ):

            @csp.graph
            def graph():
                __outputs__(x=ts[int], y=ts[int])
                return csp.const(5)

        with self.assertRaisesRegex(
            CspParseError,
            "Returning multiple outputs must use the following syntax: return csp.output\\(out1=val1, \\.\\.\\.\\)",
        ):

            @csp.graph
            def graph() -> Outputs(x=ts[int], y=ts[int]):
                return csp.const(5)

        with self.assertRaisesRegex(CspParseError, "cannot return single unnamed arg in graph returning 2 outputs"):

            @csp.graph
            def graph():
                __outputs__(x=ts[int], y=ts[int])
                __return__(csp.const(5))

        with self.assertRaisesRegex(CspParseError, "cannot return single unnamed arg in graph returning 2 outputs"):

            @csp.graph
            def graph() -> Outputs(x=ts[int], y=ts[int]):
                __return__(csp.const(5))

        with self.assertRaisesRegex(
            CspParseError, "return expects to be called with \\(value\\) or \\(output = value, output2 = value2\\)"
        ):

            @csp.graph
            def graph():
                __outputs__(x=ts[int], y=ts[float])
                __return__(csp.const(5), csp.const(6.0))

        with self.assertRaisesRegex(
            CspParseError, "return expects to be called with \\(value\\) or \\(output = value, output2 = value2\\)"
        ):

            @csp.graph
            def graph() -> Outputs(x=ts[int], y=ts[float]):
                __return__(csp.const(5), csp.const(6.0))

        # Test basket output types
        @csp.graph
        def graph():
            __outputs__({str: ts[int]})
            return {"x": csp.const(5), "y": csp.const(6.0)}

        @csp.graph
        def graph():
            __outputs__([ts[int]])
            return [csp.const(5), csp.const(6.0)]

        @csp.graph
        def graph() -> Outputs({str: ts[int]}):
            return {"x": csp.const(5), "y": csp.const(6.0)}

        @csp.graph
        def graph() -> {str: ts[int]}:
            return {"x": csp.const(5), "y": csp.const(6.0)}

        @csp.graph
        def graph() -> Outputs(out={str: ts[int]}):
            return __return__(out={"x": csp.const(5), "y": csp.const(6.0)})

        @csp.graph
        def graph() -> Outputs([ts[int]]):
            return [csp.const(5), csp.const(6.0)]

        @csp.graph
        def graph() -> [ts[int]]:
            return [csp.const(5), csp.const(6.0)]

        @csp.graph
        def graph() -> Outputs(out=[ts[int]]):
            return __return__(out=[csp.const(5), csp.const(6.0)])

        # basket types with promotion
        @csp.graph
        def graph():
            __outputs__({str: ts[int]})
            __return__({"x": csp.const(5), "y": csp.const(6.0)})

        @csp.graph
        def graph():
            __outputs__([ts[int]])
            __return__([csp.const(5), csp.const(6.0)])

        @csp.graph
        def graph() -> Outputs({str: ts[int]}):
            __return__({"x": csp.const(5), "y": csp.const(6.0)})

        @csp.graph
        def graph() -> {str: ts[int]}:
            __return__({"x": csp.const(5), "y": csp.const(6.0)})

        @csp.graph
        def graph() -> Outputs([ts[int]]):
            __return__([csp.const(5), csp.const(6.0)])

        @csp.graph
        def graph() -> [ts[int]]:
            __return__([csp.const(5), csp.const(6.0)])

    def test_none_output_annotation(self):
        """Test that -> None annotation is properly parsed for nodes and graphs with no outputs."""

        @csp.node
        def node_with_none_output(x: ts[int]) -> None:
            if csp.ticked(x):
                print(x)

        @csp.graph
        def graph_with_none_output() -> None:
            node_with_none_output(csp.const(1))

        # Should parse and run without errors
        csp.run(graph_with_none_output, starttime=datetime(2020, 1, 1), endtime=datetime(2020, 1, 1, 0, 0, 1))

    def test_list_inside_callable(self):
        """was a bug "Empty list inside callable annotation raises exception" """

        @csp.graph
        def graph(v: Dict[str, Callable[[], str]]):
            pass

    def test_list_default_value(self):
        # There was a bug parsing list default value
        @csp.graph
        def g(x: List[int] = [1, 2, 3]):
            pass

    def test_wrong_parse_error(self):
        @csp.graph
        def g(x: {str: csp.ts[float]}):
            pass

        @csp.graph
        def g2():
            __outputs__({str: csp.ts[object]})
            return {"A": csp.null_ts(object)}

        def main():
            g(g2())

        with self.assertRaises(TypeError):
            main()

    def test_bad_parse_message(self):
        with self.assertRaisesRegex(CspParseError, "Invalid use of csp.output please consult documentation"):

            @csp.node
            def x():
                __outputs__(x=ts[int])
                csp.output("x", 1)

    def test_output_annotation_parsing_nodes(self):
        @csp.node
        def my_node(x: ts[int], y: ts[int]) -> Outputs(ts[int]):
            if csp.ticked(x, y):
                __return__(x + y)

        @csp.node
        def my_node2(x: ts[int], y: ts[int]) -> Outputs(my_output=ts[int]):
            if csp.ticked(x, y):
                __return__(my_output=x + y)

        @csp.node
        def my_node3(x: ts[int], y: ts[int]) -> Outputs(x=ts[int], y=ts[int]):
            if csp.ticked(x):
                csp.output(x=x)
            if csp.ticked(y):
                __return__(y=y)

        class MyOutputs(Outputs):
            x: ts[int]
            y: ts[int]

        @csp.node
        def my_node4(x: ts[int], y: ts[int]) -> MyOutputs:
            if csp.ticked(x):
                csp.output(x=x)
            if csp.ticked(y):
                __return__(y=y)

        @csp.node
        def my_node5(x: ts[int], y: ts[int]) -> ts[int]:
            if csp.ticked(x, y):
                return x + y

        # @csp.graph
        # def my_graph():
        #     csp.print("out1", my_node(csp.const(1), csp.const(2)))
        #     csp.print("out2", my_node2(csp.const(1), csp.const(2)))

        #     node3 = my_node3(csp.const(1), csp.const(2))
        #     csp.print("out3.x", node3.x)
        #     csp.print("out3.y", node3.y)

        #     node4 = my_node4(csp.const(1), csp.const(2))
        #     csp.print("out4.x", node4.x)
        #     csp.print("out4.y", node4.y)

        #     csp.print("out5", my_node5(csp.const(1), csp.const(2)))

    def test_output_annotation_parsing_graphs(self):
        @csp.graph
        def graph() -> Outputs({str: ts[int]}):
            return {"x": csp.const(5), "y": csp.const(6.0)}

        @csp.graph
        def graph() -> {str: ts[int]}:
            return {"x": csp.const(5), "y": csp.const(6.0)}

        @csp.graph
        def graph() -> Outputs([ts[int]]):
            return [csp.const(5), csp.const(6.0)]

        @csp.graph
        def graph() -> [ts[int]]:
            return [csp.const(5), csp.const(6.0)]

        @csp.graph
        def graph() -> Outputs({str: ts[int]}):
            __return__({"x": csp.const(5), "y": csp.const(6.0)})

        @csp.graph
        def graph() -> {str: ts[int]}:
            __return__({"x": csp.const(5), "y": csp.const(6.0)})

        @csp.graph
        def graph() -> Outputs([ts[int]]):
            __return__([csp.const(5), csp.const(6.0)])

        @csp.graph
        def graph() -> [ts[int]]:
            __return__([csp.const(5), csp.const(6.0)])

    def test_output_annotation_parsing_baskets(self):
        @csp.node
        def my_node_1_1(x: Dict[str, ts[str]], y: List[str]) -> Outputs(OutputBasket(Dict[str, ts[str]], shape="y")):
            if csp.ticked(x):
                return x

        my_node_1_1({"x": csp.const("x ")}, ["x"])

        @csp.node
        def my_node_1_2(x: Dict[str, ts[str]]) -> Outputs(OutputBasket(Dict[str, ts[str]], shape_of="x")):
            if csp.ticked(x):
                return x

        my_node_1_2({"x": csp.const("x ")})

        @csp.node
        def my_node_2_1(x: Dict[str, ts[str]], y: List[str]) -> Outputs(OutputBasket({str: ts[str]}, shape="y")):
            if csp.ticked(x):
                return x

        my_node_2_1({"x": csp.const("x ")}, ["x"])

        @csp.node
        def my_node_2_2(x: Dict[str, ts[str]]) -> Outputs(OutputBasket({str: ts[str]}, shape_of="x")):
            if csp.ticked(x):
                return x

        my_node_2_2({"x": csp.const("x")})

        @csp.node
        def my_node_3_1(x: List[ts[str]], y: int) -> OutputBasket(List[ts[str]], shape="y"):
            if csp.ticked(x):
                return x

        my_node_3_1([csp.const("x")], 1)

        @csp.node
        def my_node_3_2(x: List[ts[str]]) -> OutputBasket(List[ts[str]], shape_of="x"):
            if csp.ticked(x):
                return x

        my_node_3_2([csp.const("x")])

        @csp.node
        def my_node_4_1(x: List[ts[str]], y: int) -> Outputs(OutputBasket(List[ts[str]], shape="y")):
            if csp.ticked(x):
                return x

        my_node_4_1([csp.const("x")], 1)

        @csp.node
        def my_node_4_2(x: List[ts[str]]) -> Outputs(OutputBasket(List[ts[str]], shape_of="x")):
            if csp.ticked(x):
                return x

        my_node_4_2([csp.const("x")])

        @csp.node
        def my_node_4_3(x: ts[int]) -> Outputs(OutputBasket(Dict[str, ts[str]], shape=["a", "b", "c"])):
            if csp.ticked(x):
                return x

        my_node_4_3(csp.const(1))

        @csp.node
        def my_node_4_4(x: ts[int]) -> Outputs(OutputBasket(List[ts[int]], shape=10)):
            if csp.ticked(x):
                return x

        my_node_4_4(csp.const(1))

        @csp.node
        def my_node_5_1(x: ts[int]) -> {ts[str]: ts[str]}:
            return {}

        my_node_5_1(csp.const(1))

        @csp.node
        def my_node_5_2(x: ts[int]) -> OutputBasket({ts[str]: ts[str]}):
            return {}

        my_node_5_2(csp.const(1))

    def test_output_annotation_parsing_graph_baskets(self):
        @csp.graph
        def my_graph_1_1(x: Dict[str, ts[str]], y: List[str]) -> Outputs(OutputBasket(Dict[str, ts[str]], shape="y")):
            return x

        my_graph_1_1({"x": csp.const("x ")}, ["x"])

        @csp.graph
        def my_graph_1_2(x: Dict[str, ts[str]]) -> Outputs(OutputBasket(Dict[str, ts[str]], shape_of="x")):
            return x

        my_graph_1_2({"x": csp.const("x ")})

        @csp.graph
        def my_graph_1_3(x: Dict["K", ts[str]]) -> OutputBasket({"K": ts[str]}, shape_of="x"):
            return x

        my_graph_1_3({"x": csp.const("x ")})

        @csp.graph
        def my_graph_2_1(x: Dict[str, ts[str]], y: List[str]) -> Outputs(OutputBasket({str: ts[str]}, shape="y")):
            return x

        my_graph_2_1({"x": csp.const("x ")}, ["x"])

        @csp.graph
        def my_graph_2_2(x: Dict[str, ts[str]]) -> Outputs(OutputBasket({str: ts[str]}, shape_of="x")):
            return x

        my_graph_2_2({"x": csp.const("x ")})

        @csp.graph
        def my_graph_3_1(x: List[ts[str]], y: int) -> OutputBasket(List[ts[str]], shape="y"):
            return x

        my_graph_3_1([csp.const("x")], 1)

        @csp.graph
        def my_graph_3_2(x: List[ts[str]]) -> OutputBasket(List[ts[str]], shape_of="x"):
            return x

        my_graph_3_2([csp.const("x")])

        @csp.graph
        def my_graph_3_3(x: List[ts["T"]]) -> OutputBasket(List[ts["T"]], shape_of="x"):
            return x

        my_graph_3_3([csp.const("x")])

        @csp.graph
        def my_graph_4_1(x: List[ts[str]], y: int) -> Outputs(OutputBasket(List[ts[str]], shape="y")):
            return x

        my_graph_4_1([csp.const("x")], 1)

        @csp.graph
        def my_graph_4_2(x: List[ts[str]]) -> Outputs(OutputBasket(List[ts[str]], shape_of="x")):
            return x

        my_graph_4_2([csp.const("x")])

        @csp.graph
        def my_graph_4_3(x: ts[int]) -> Outputs(OutputBasket(Dict[str, ts[str]], shape=["a", "b", "c"])):
            return {k: csp.const(k) for k in "abc"}

        my_graph_4_3(csp.const(1))

        @csp.graph
        def my_graph_4_4(x: ts[int]) -> Outputs(OutputBasket(List[ts[int]], shape=10)):
            return [csp.const(1)] * 10

        my_graph_4_4(csp.const(1))

        @csp.node()
        def _dyn_basket() -> {ts[str]: ts[str]}:
            return {}

        @csp.graph
        def my_graph_5_1(x: ts[int]) -> {ts[str]: ts[str]}:
            return _dyn_basket()

        my_graph_5_1(csp.const(1))

        @csp.graph
        def my_graph_5_2(x: ts[int]) -> OutputBasket({ts[str]: ts[str]}):
            return _dyn_basket()

        my_graph_5_2(csp.const(1))

        # tests bug found with shape arguments + scalar inputs
        @csp.graph
        def my_graph_5_3(x: str) -> OutputBasket({str: ts[str]}, shape=["x", "y"]):
            return {"x": csp.const("a"), "y": csp.const(x)}

        st = datetime(2020, 1, 1)
        res1 = csp.run(my_graph_5_3, "b", starttime=st, endtime=timedelta())
        exp_out = {"x": [(st, "a")], "y": [(st, "b")]}
        self.assertEqual(res1, exp_out)

        # tests bug found with multiple dictionary baskets getting bound to same shape
        @csp.graph
        def g() -> csp.Outputs(
            i=csp.OutputBasket(Dict[str, csp.ts[int]], shape=["V1"]),
            s=csp.OutputBasket(Dict[str, csp.ts[str]], shape=["V2"]),
        ):
            i_v1 = csp.curve(int, [(timedelta(hours=10), 1), (timedelta(hours=30), 1)])
            s_v2 = csp.curve(str, [(timedelta(hours=30), "val1")])
            __return__(i={"V1": i_v1}, s={"V2": s_v2})

        csp.run(g, starttime=datetime(2020, 1, 1), endtime=timedelta())

    def test_pythonic_node_syntax(self):
        # Proper parse errors
        node_specific = [
            ("state", csp.state),
            ("alarms", csp.alarms),
            ("start", csp.start),
            ("stop", csp.stop),
            ("__state__", csp.__state__),
            ("__alarms__", csp.__alarms__),
            ("__start__", csp.__start__),
            ("__stop__", csp.__stop__),
        ]
        for fn, func in node_specific:
            with self.assertRaisesRegex(RuntimeError, f"Unexpected use of {fn}, possibly using outside of @node?"):
                func()
            with self.assertRaisesRegex(RuntimeError, f"Unexpected use of {fn}, possibly using outside of @node?"):

                @csp.graph
                def g():
                    func()

                csp.run(g, starttime=datetime(2020, 1, 1), endtime=timedelta())

        node_or_graph = [("__return__", __return__), ("__outputs__", __outputs__), ("csp.output", csp.output)]
        for fn, func in node_or_graph:
            with self.assertRaisesRegex(
                RuntimeError, f"Unexpected use of {fn}, possibly using outside of @graph and @node?"
            ):
                func()

        with self.assertRaisesRegex(CspParseError, "csp.state must be called, cannot use as a bare name"):

            @csp.node
            def foo():
                with csp.state:
                    pass

        with self.assertRaisesRegex(CspParseError, "csp.alarms must be called, cannot use as a bare name"):

            @csp.node
            def foo():
                with csp.alarms:
                    pass

        with self.assertRaisesRegex(CspParseError, "csp.start must be called, cannot use as a bare name"):

            @csp.node
            def foo():
                with csp.start:
                    pass

        with self.assertRaisesRegex(CspParseError, "csp.stop must be called, cannot use as a bare name"):

            @csp.node
            def foo():
                with csp.stop:
                    pass

        with self.assertRaisesRegex(CspParseError, "state must be used in a with statement"):

            @csp.node
            def foo():
                csp.state()

        with self.assertRaisesRegex(CspParseError, "alarms must be used in a with statement"):

            @csp.node
            def foo():
                csp.alarms()

        with self.assertRaisesRegex(CspParseError, "start must be used in a with statement"):

            @csp.node
            def foo():
                csp.start()

        with self.assertRaisesRegex(CspParseError, "stop must be used in a with statement"):

            @csp.node
            def foo():
                csp.stop()

        with self.assertRaisesRegex(CspParseError, "alarms must be declared before state"):

            @csp.node
            def foo():
                with csp.state():
                    pass
                with csp.alarms():
                    pass

        with self.assertRaisesRegex(CspParseError, "state must be declared before start"):

            @csp.node
            def foo():
                with csp.start():
                    pass
                with csp.state():
                    pass

        with self.assertRaisesRegex(CspParseError, "start must be declared before stop"):

            @csp.node
            def foo():
                with csp.stop():
                    pass
                with csp.start():
                    pass

        with self.assertRaisesRegex(CspParseError, "Invalid usage of stop, .*"):

            @csp.node
            def foo() -> Outputs(x=ts[int]):
                __return__(x=1)
                with csp.stop():
                    pass

        # Verify state works properly
        @csp.node
        def lagger(x: ts[int]) -> ts[int]:
            with csp.state():
                s_x = 0

            past = s_x
            s_x = x
            return past

        @csp.node
        def history(x: ts[int]) -> ts[dict]:
            with csp.state():
                s_i = 0
                s_x = {}

            s_x[s_i] = x
            s_i += 1
            return s_x.copy()

        st = datetime(2020, 1, 1)

        @csp.graph
        def g():
            x = csp.curve(typ=int, data=[(st + timedelta(i), i + 1) for i in range(3)])
            lag = lagger(x)
            hist = history(x)
            csp.add_graph_output("lag", lag)
            csp.add_graph_output("hist", hist)

        res = csp.run(g, starttime=st, endtime=timedelta(4))

        exp_lag = [(st + timedelta(i), i) for i in range(3)]
        exp_hist = [(st, {0: 1}), (st + timedelta(1), {0: 1, 1: 2}), (st + timedelta(2), {0: 1, 1: 2, 2: 3})]
        self.assertEqual(exp_lag, res["lag"])
        self.assertEqual(exp_hist, res["hist"])

        # Verify alarms, start, stop

        class MyClass:
            self.x = 0

        @csp.node
        def n1(my_class: MyClass) -> ts[int]:
            with csp.alarms():
                alarm: ts[bool] = csp.alarm(bool)

            with csp.start():
                csp.schedule_alarm(alarm, timedelta(1), True)

            with csp.stop():
                my_class.x = 1

            if csp.ticked(alarm):
                return 1

        @csp.node
        def n2(x: ts[int]) -> ts[int]:
            with csp.alarms():
                alarm: ts[int] = csp.alarm(int)

            if csp.ticked(x):
                csp.schedule_alarm(alarm, timedelta(1), 1)

            if csp.ticked(alarm):
                return alarm

        my_class = MyClass()

        @csp.graph
        def g():
            x = n1(my_class)
            y = n2(x)
            csp.add_graph_output("x", x)
            csp.add_graph_output("y", y)

        res = csp.run(g, starttime=st, endtime=timedelta(4))

        exp_x = [(st + timedelta(1), 1)]
        exp_y = [(st + timedelta(2), 1)]
        self.assertEqual(exp_x, res["x"])
        self.assertEqual(exp_y, res["y"])
        self.assertEqual(my_class.x, 1)

        # Make sure function call dunders are accounted for
        @csp.node
        def n1():
            csp.__state__(x=5)

        @csp.node
        def n2():
            with csp.__state__(x=5):
                pass

    def test_state_context_logic(self):
        # Enforce that no logic other than variable assignments/inits can be within a state context

        with self.assertWarnsRegex(
            DeprecationWarning,
            "Only variable assignments and declarations should be present in a csp.state block. Any logic should be moved to csp.start",
        ):

            @csp.node
            def n1():
                with csp.state():
                    i = 0
                    for x in range(10):
                        i += x
                    s_y = x

        with self.assertWarnsRegex(
            DeprecationWarning,
            "Only variable assignments and declarations should be present in a csp.state block. Any logic should be moved to csp.start",
        ):

            @csp.node
            def n2():
                with csp.state():
                    i = 0
                    if i > 0:
                        s_y = i

        with self.assertWarnsRegex(
            DeprecationWarning,
            "Only variable assignments and declarations should be present in a csp.state block. Any logic should be moved to csp.start",
        ):

            @csp.node
            def n3():
                with csp.state():
                    i = 10
                    while i > 0:
                        i -= 1
                    s_y = i

        # This is fine, though - just assignments
        @csp.node
        def n4():
            with csp.state():
                x = 2
                y = x
                z = x**2

        # This is also fine - annotated assignments
        class A:
            pass

        @csp.node
        def n4():
            with csp.state():
                x: int = 2
                y: set[bool] = {True, False}
                z: A = A()

    def test_pythonic_alarm_syntax(self):
        st = datetime(2020, 1, 1)

        # test new alarm syntax
        class ClassA:
            def __init__(self):
                self.my_member = 1

        class StructA(csp.Struct):
            a: int
            b: str

        class StructB(csp.Struct):
            c: int
            d: StructA

        @csp.node
        def n5() -> ts[int]:
            with csp.alarms():
                a = csp.alarm(int)
                b = csp.alarm(StructA)
                c = csp.alarm(StructB)
                d = csp.alarm(ClassA)

            with csp.start():
                csp.schedule_alarm(a, timedelta(seconds=1), 1)
                csp.schedule_alarm(b, timedelta(seconds=2), StructA())
                csp.schedule_alarm(c, timedelta(seconds=3), StructB())
                csp.schedule_alarm(d, timedelta(seconds=4), ClassA())

            if csp.ticked(a):
                return 1
            if csp.ticked(b):
                return 2
            if csp.ticked(c):
                return 3
            if csp.ticked(d):
                return 4

        @csp.node
        def n6() -> ts[int]:
            with csp.alarms():
                a: ts[int] = csp.alarm(int)
                b: ts[StructA] = csp.alarm(StructA)
                c: ts[StructB] = csp.alarm(StructB)
                d: ts[ClassA] = csp.alarm(ClassA)

            with csp.start():
                csp.schedule_alarm(a, timedelta(seconds=1), 1)
                csp.schedule_alarm(b, timedelta(seconds=2), StructA())
                csp.schedule_alarm(c, timedelta(seconds=3), StructB())
                csp.schedule_alarm(d, timedelta(seconds=4), ClassA())

            if csp.ticked(a):
                return 1
            if csp.ticked(b):
                return 2
            if csp.ticked(c):
                return 3
            if csp.ticked(d):
                return 4

        @csp.graph
        def g1():
            csp.add_graph_output("x", n5())
            csp.add_graph_output("y", n6())

        res = csp.run(g1, starttime=st, endtime=timedelta(10))
        exp_out = [(st + timedelta(seconds=(i + 1)), i + 1) for i in range(4)]
        self.assertEqual(res["x"], exp_out)
        self.assertEqual(res["y"], exp_out)

        # Now verify a ton of error messages
        with self.assertRaisesRegex(CspParseError, "Alarms must be initialized with csp.alarm in __alarms__ block"):

            @csp.node
            def n():
                with csp.alarms():
                    a: ts[int]

        with self.assertRaisesRegex(CspParseError, "Alarms must be initialized with csp.alarm in __alarms__ block"):

            @csp.node
            def n():
                with csp.alarms():
                    a: ts[int] = csp.foo()

        with self.assertRaisesRegex(TypeError, "function `csp.alarm` does not take keyword arguments"):

            @csp.node
            def n():
                with csp.alarms():
                    a: ts[int] = csp.alarm(typ=int)

        with self.assertRaisesRegex(
            TypeError, "function `csp.alarm` requires a single type argument: 0 arguments given"
        ):

            @csp.node
            def n():
                with csp.alarms():
                    a: ts[int] = csp.alarm()

        with self.assertRaisesRegex(
            TypeError, "function `csp.alarm` requires a single type argument: 2 arguments given"
        ):

            @csp.node
            def n():
                with csp.alarms():
                    a = csp.alarm(int, bool)

        foo = lambda: int

        @csp.node
        def n():
            with csp.alarms():
                a = csp.alarm(foo())

        with self.assertRaisesRegex(
            TypeError, "function `csp.alarm` requires a single type argument: 2 arguments given"
        ):

            @csp.node
            def n():
                with csp.alarms():
                    a = csp.alarm(int, bool)

        # we don't check type annotations
        @csp.node
        def n():
            with csp.alarms():
                a: ts[StructA] = csp.alarm(StructB)

        with self.assertRaisesRegex(CspParseError, "Alarms must be initialized with csp.alarm in __alarms__ block"):

            @csp.node
            def n():
                with csp.alarms():
                    x = 5

        with self.assertRaisesRegex(CspParseError, "Only alarm assignments are allowed in csp.alarms block"):

            @csp.node
            def n():
                with csp.alarms():
                    print()

        with self.assertRaisesRegex(CspParseError, "Exactly one alarm can be assigned per line"):

            @csp.node
            def n():
                with csp.alarms():
                    a, b = csp.alarm(int), csp.alarm(bool)

        # test generic alarms
        @csp.node
        def n_gen(x: ts["T"]) -> ts["T"]:
            with csp.alarms():
                a = csp.alarm("T")
                b: ts["T"] = csp.alarm("T")

            with csp.start():
                csp.schedule_alarm(a, timedelta(seconds=1), 1)
                csp.schedule_alarm(b, timedelta(seconds=2), 2)

            if csp.ticked(a):
                return a

            if csp.ticked(b):
                return b

            if csp.ticked(x):
                return x

        @csp.graph
        def g_gen():
            csp.add_graph_output("z", n_gen(csp.const(0)))

        res = csp.run(g_gen, starttime=st, endtime=timedelta(seconds=10))
        self.assertEqual(res["z"], [(st + timedelta(seconds=i), i) for i in range(3)])

        # test container alarms
        @csp.node
        def n_cont() -> ts[bool]:
            with csp.alarms():
                a: ts[List[bool]] = csp.alarm(List[bool])
                b: ts[List[List[int]]] = csp.alarm(List[List[int]])
                c: ts[Dict[str, int]] = csp.alarm(Dict[str, int])
                d: ts[Dict[str : List[int]]] = csp.alarm(Dict[str, List[int]])  # dict of lists
                e: ts[List[Dict[str, bool]]] = csp.alarm(List[Dict[str, bool]])  # list of dicts

            with csp.start():
                csp.schedule_alarm(a, timedelta(seconds=1), [True])
                csp.schedule_alarm(b, timedelta(seconds=2), [[1]])
                csp.schedule_alarm(c, timedelta(seconds=3), {"a": 1})
                csp.schedule_alarm(d, timedelta(seconds=4), {"a": [1]})
                csp.schedule_alarm(e, timedelta(seconds=5), [{"a": True}])

            return True

        @csp.graph
        def g_cont():
            csp.add_graph_output("u", n_cont())

        res = csp.run(g_cont, starttime=st, endtime=timedelta(seconds=10))
        self.assertEqual(res["u"], [(st + timedelta(seconds=i + 1), True) for i in range(5)])

        with self.assertRaises(TypeError):

            @csp.node
            def n():
                with csp.alarms():
                    a = csp.alarm([bool, int])

        with self.assertRaises(TypeError):

            @csp.node
            def n():
                with csp.alarms():
                    a = csp.alarm([bool, [bool]])

        with self.assertRaises(TypeError):

            @csp.node
            def n():
                with csp.alarms():
                    a = csp.alarm([])

        with self.assertRaises(TypeError):

            @csp.node
            def n():
                with csp.alarms():
                    a = csp.alarm({})

        with self.assertRaises(TypeError):

            @csp.node
            def n():
                with csp.alarms():
                    a = csp.alarm({str: int, StructA: bool})

    def test_return(self):
        # basic test
        @csp.node
        def n(x: ts[bool]) -> csp.Outputs(x=ts[int], y=ts[int]):
            return csp.output(x=1, y=2)

        @csp.graph
        def g() -> csp.Outputs(a=ts[int], b=ts[int]):
            node_out = n(csp.timer(timedelta(seconds=1), True))
            return csp.output(a=node_out.x, b=node_out.y)

        st = datetime(2020, 1, 1)
        res = csp.run(g, starttime=st, endtime=timedelta(seconds=10))
        self.assertEqual(res["a"], [(st + timedelta(seconds=i + 1), 1) for i in range(10)])
        self.assertEqual(res["b"], [(st + timedelta(seconds=i + 1), 2) for i in range(10)])

        # test different tstypes
        @csp.node
        def n1(a: ts[int], b: ts[str]) -> csp.Outputs(x=ts[int], y=ts[str]):
            return csp.output(x=a, y=b)

        @csp.graph
        def g1() -> csp.Outputs(a=ts[int], b=ts[str]):
            int_data = csp.timer(timedelta(seconds=1), 1)
            str_data = csp.timer(timedelta(seconds=1), "a")
            node_out = n1(int_data, str_data)
            return csp.output(a=node_out.x, b=node_out.y)

        res = csp.run(g1, starttime=st, endtime=timedelta(seconds=10))
        self.assertEqual(res["a"], [(st + timedelta(seconds=i + 1), 1) for i in range(10)])
        self.assertEqual(res["b"], [(st + timedelta(seconds=i + 1), "a") for i in range(10)])

        # test graph baskets: lists
        # unnamed single
        @csp.graph
        def g2() -> csp.OutputBasket(List[ts[int]], shape=3):
            x1 = csp.timer(timedelta(seconds=1), 1)
            x2 = csp.timer(timedelta(seconds=1), 2)
            x3 = csp.timer(timedelta(seconds=1), 3)
            return [x1, x2, x3]

        @csp.graph
        def g3() -> csp.OutputBasket(List[ts[int]], shape=3):
            x1 = csp.timer(timedelta(seconds=1), 1)
            x2 = csp.timer(timedelta(seconds=1), 2)
            x3 = csp.timer(timedelta(seconds=1), 3)
            return csp.output([x1, x2, x3])

        r2 = csp.run(g2, starttime=st, endtime=timedelta(seconds=10))
        r3 = csp.run(g3, starttime=st, endtime=timedelta(seconds=10))
        self.assertEqual(r2, r3)

        # named single
        @csp.graph
        def g4() -> csp.Outputs(l=csp.OutputBasket(List[ts[int]], shape=3)):
            x1 = csp.timer(timedelta(seconds=1), 1)
            x2 = csp.timer(timedelta(seconds=1), 2)
            x3 = csp.timer(timedelta(seconds=1), 3)
            return csp.output(l=[x1, x2, x3])

        r4 = csp.run(g4, starttime=st, endtime=timedelta(seconds=10))
        self.assertEqual(r2[0], r4["l[0]"])
        self.assertEqual(r2[1], r4["l[1]"])
        self.assertEqual(r2[2], r4["l[2]"])

        # named multiple
        @csp.graph
        def g5() -> csp.Outputs(l=csp.OutputBasket(List[ts[int]], shape=2), m=csp.OutputBasket(List[ts[int]], shape=2)):
            x1 = csp.timer(timedelta(seconds=1), 1)
            x2 = csp.timer(timedelta(seconds=1), 2)
            x3 = csp.timer(timedelta(seconds=1), 3)
            x4 = csp.timer(timedelta(seconds=1), 4)
            __return__(l=[x1, x2], m=[x3, x4])

        @csp.graph
        def g6() -> csp.Outputs(l=csp.OutputBasket(List[ts[int]], shape=2), m=csp.OutputBasket(List[ts[int]], shape=2)):
            x1 = csp.timer(timedelta(seconds=1), 1)
            x2 = csp.timer(timedelta(seconds=1), 2)
            x3 = csp.timer(timedelta(seconds=1), 3)
            x4 = csp.timer(timedelta(seconds=1), 4)
            return csp.output(l=[x1, x2], m=[x3, x4])

        r5 = csp.run(g5, starttime=st, endtime=timedelta(seconds=10))
        r6 = csp.run(g6, starttime=st, endtime=timedelta(seconds=10))
        self.assertEqual(r5, r6)

        # test graph baskets: dictionaries
        # unnamed single
        @csp.graph
        def g7() -> csp.OutputBasket(Dict[str, ts[int]], shape=["v1", "v2"]):
            x1 = csp.timer(timedelta(seconds=1), 1)
            x2 = csp.timer(timedelta(seconds=1), 2)
            return {"v1": x1, "v2": x2}

        @csp.graph
        def g8() -> csp.OutputBasket(Dict[str, ts[int]], shape=["v1", "v2"]):
            x1 = csp.timer(timedelta(seconds=1), 1)
            x2 = csp.timer(timedelta(seconds=1), 2)
            return csp.output({"v1": x1, "v2": x2})

        r7 = csp.run(g7, starttime=st, endtime=timedelta(seconds=10))
        r8 = csp.run(g8, starttime=st, endtime=timedelta(seconds=10))
        self.assertEqual(r7, r8)

        # named single
        @csp.graph
        def g9() -> csp.Outputs(d=csp.OutputBasket(Dict[str, ts[int]], shape=["v1", "v2"])):
            x1 = csp.timer(timedelta(seconds=1), 1)
            x2 = csp.timer(timedelta(seconds=1), 2)
            __return__(d={"v1": x1, "v2": x2})

        @csp.graph
        def g10() -> csp.Outputs(d=csp.OutputBasket(Dict[str, ts[int]], shape=["v1", "v2"])):
            x1 = csp.timer(timedelta(seconds=1), 1)
            x2 = csp.timer(timedelta(seconds=1), 2)
            return csp.output(d={"v1": x1, "v2": x2})

        r9 = csp.run(g9, starttime=st, endtime=timedelta(seconds=10))
        r10 = csp.run(g10, starttime=st, endtime=timedelta(seconds=10))
        self.assertEqual(r9, r10)

        # named multiple
        @csp.graph
        def g11() -> csp.Outputs(
            d1=csp.OutputBasket(Dict[str, ts[int]], shape=["v1", "v2"]),
            d2=csp.OutputBasket(Dict[str, ts[int]], shape=["v3", "v4"]),
        ):
            x1 = csp.timer(timedelta(seconds=1), 1)
            x2 = csp.timer(timedelta(seconds=1), 2)
            x3 = csp.timer(timedelta(seconds=1), 3)
            x4 = csp.timer(timedelta(seconds=1), 4)
            __return__(d1={"v1": x1, "v2": x2}, d2={"v3": x3, "v4": x4})

        @csp.graph
        def g12() -> csp.Outputs(
            d1=csp.OutputBasket(Dict[str, ts[int]], shape=["v1", "v2"]),
            d2=csp.OutputBasket(Dict[str, ts[int]], shape=["v3", "v4"]),
        ):
            x1 = csp.timer(timedelta(seconds=1), 1)
            x2 = csp.timer(timedelta(seconds=1), 2)
            x3 = csp.timer(timedelta(seconds=1), 3)
            x4 = csp.timer(timedelta(seconds=1), 4)
            return csp.output(d1={"v1": x1, "v2": x2}, d2={"v3": x3, "v4": x4})

        r11 = csp.run(g11, starttime=st, endtime=timedelta(seconds=10))
        r12 = csp.run(g12, starttime=st, endtime=timedelta(seconds=10))
        self.assertEqual(r11, r12)

        # smorgasbord
        @csp.node
        def n2(x: ts["T"]) -> csp.Outputs(
            l=csp.OutputBasket(List[ts[int]], shape=2), d=csp.OutputBasket(Dict[str, ts[int]], shape=["v1", "v2"])
        ):
            csp.output(l[0], 1)
            csp.output(l[1], 2)
            csp.output(d["v1"], 3)
            csp.output(d["v2"], 4)

        @csp.graph
        def g13() -> csp.Outputs(
            l=csp.OutputBasket(List[ts[int]], shape=2),
            d=csp.OutputBasket(Dict[str, ts[int]], shape=["v1", "v2"]),
            s=ts[str],
        ):
            x1 = csp.timer(timedelta(seconds=1), 1)
            x = n2(x1)
            __return__(l=x.l, d=x.d, s=csp.timer(timedelta(seconds=1), "a"))

        @csp.graph
        def g14() -> csp.Outputs(
            l=csp.OutputBasket(List[ts[int]], shape=2),
            d=csp.OutputBasket(Dict[str, ts[int]], shape=["v1", "v2"]),
            s=ts[str],
        ):
            x1 = csp.timer(timedelta(seconds=1), 1)
            x = n2(x1)
            return csp.output(l=x.l, d=x.d, s=csp.timer(timedelta(seconds=1), "a"))

        r13 = csp.run(g13, starttime=st, endtime=timedelta(seconds=10))
        r14 = csp.run(g14, starttime=st, endtime=timedelta(seconds=10))
        self.assertEqual(r13, r14)

        # empty return statements
        @csp.node
        def n():
            return

        @csp.graph
        def g():
            return

        @csp.node
        def n(x: ts[int]) -> ts[int]:
            if x > 0:
                return
            return 1

        @csp.node
        def n(z: ts[int]) -> csp.Outputs(x=ts[int], y=ts[str]):
            if z > 0:
                return
            return csp.output(x=1, y=2)

        with self.assertRaisesRegex(CspParseError, "return does not return values with non empty outputs"):

            @csp.graph
            def g() -> ts[int]:
                return

        # verify error messages
        with self.assertRaisesRegex(CspParseError, "return does not return values with non empty outputs"):

            @csp.graph
            def g() -> csp.Outputs(x=ts[int], y=ts[str]):
                return

        with self.assertRaisesRegex(
            CspParseError,
            "Returning multiple outputs must use the following syntax: return csp.output\\(out1=val1, \\.\\.\\.\\)",
        ):

            @csp.node
            def n() -> csp.Outputs(x=ts[int], y=ts[str]):
                return 1, "a"

        with self.assertRaisesRegex(
            CspParseError,
            "Returning multiple outputs must use the following syntax: return csp.output\\(out1=val1, \\.\\.\\.\\)",
        ):

            @csp.node
            def n() -> csp.Outputs(x=ts[int], y=ts[str]):
                return (1, "a")

        with self.assertRaisesRegex(
            CspParseError,
            "Returning multiple outputs must use the following syntax: return csp.output\\(out1=val1, \\.\\.\\.\\)",
        ):

            @csp.node
            def n() -> csp.Outputs(x=ts[int], y=ts[str]):
                return csp.output()

        with self.assertRaisesRegex(
            CspParseError,
            "Returning multiple outputs must use the following syntax: return csp.output\\(out1=val1, \\.\\.\\.\\)",
        ):

            @csp.node
            def n() -> csp.Outputs(x=ts[int], y=ts[str]):
                return csp.outputs(x=1, y="a")  # note the s

        with self.assertRaisesRegex(
            CspParseError,
            "Returning multiple outputs must use the following syntax: return csp.output\\(out1=val1, \\.\\.\\.\\)",
        ):

            @csp.node
            def n() -> csp.Outputs(x=ts[int], y=ts[str]):
                return output(x=1, y="a")

        # test running a node directly
        @csp.node
        def n() -> csp.Outputs(x=csp.ts[int]):
            with csp.alarms():
                a = csp.alarm(int)
            with csp.start():
                csp.schedule_alarm(a, timedelta(), 0)
            return csp.output(x=a)

        csp.run(n, starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=1))

        # test csp node builtins within the output statement
        @csp.node
        def n() -> csp.Outputs(ts[datetime]):
            with csp.alarms():
                a = csp.alarm(int)
            with csp.start():
                csp.schedule_alarm(a, timedelta(seconds=1), 1)

            if csp.ticked(a):
                return csp.now()

        r15 = csp.run(n, starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=1))

        @csp.node
        def n() -> csp.Outputs(ts[datetime]):
            with csp.alarms():
                a = csp.alarm(int)
            with csp.start():
                csp.schedule_alarm(a, timedelta(seconds=1), 1)

            if csp.ticked(a):
                return csp.output(csp.now())

        r16 = csp.run(n, starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=1))
        self.assertEqual(r15, r16)

        @csp.node
        def n() -> csp.Outputs(c=ts[datetime]):
            with csp.alarms():
                a = csp.alarm(int)
            with csp.start():
                csp.schedule_alarm(a, timedelta(seconds=1), 1)

            if csp.ticked(a):
                return csp.output(c=csp.now())

        csp.run(n, starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=1))

        @csp.node
        def n() -> csp.Outputs(c=ts[datetime], d=ts[datetime]):
            with csp.alarms():
                a = csp.alarm(int)
            with csp.start():
                csp.schedule_alarm(a, timedelta(seconds=1), 1)

            if csp.ticked(a):
                return csp.output(c=csp.now(), d=csp.now())

        csp.run(n, starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=1))

    def test_pythonic_depr_warning(self):
        original_setting = csp.impl.warnings.set_deprecation_warning(True)

        # alarm
        with self.assertWarnsRegex(DeprecationWarning, "Calling __alarms__ is deprecated: *"):

            @csp.node
            def n():
                __alarms__(a=ts[bool])
                pass

        @csp.node
        def n():
            with csp.alarms():
                a = csp.alarm(bool)
            pass

        # state
        with self.assertWarnsRegex(DeprecationWarning, "Calling __state__ is deprecated: *"):

            @csp.node
            def n():
                __state__(a=int)
                pass

        @csp.node
        def n():
            with csp.state():
                a = 1
            pass

        # start
        with self.assertWarnsRegex(DeprecationWarning, "Calling __start__ is deprecated: *"):

            @csp.node
            def n():
                with __start__():
                    pass
                pass

        @csp.node
        def n():
            with csp.start():
                pass
            pass

        # stop
        with self.assertWarnsRegex(DeprecationWarning, "Calling __stop__ is deprecated: *"):

            @csp.node
            def n():
                with __stop__():
                    pass
                pass

        @csp.node
        def n():
            with csp.stop():
                pass
            pass

        # outputs
        with self.assertWarnsRegex(DeprecationWarning, "Declaring __outputs__ is deprecated; *"):

            @csp.node
            def n():
                __outputs__(ts[int])
                return 1

        with self.assertWarnsRegex(DeprecationWarning, "Declaring __outputs__ is deprecated; *"):

            @csp.graph
            def g():
                __outputs__(a=ts[int], b=ts[str])
                pass

        @csp.node
        def n() -> ts[int]:
            return 1

        # return
        with self.assertWarnsRegex(DeprecationWarning, "Calling __return__ is deprecated*"):

            @csp.node
            def n() -> csp.Outputs(x=ts[int]):
                __return__(x=1)

        with self.assertWarnsRegex(DeprecationWarning, "Calling __return__ is deprecated*"):

            @csp.graph
            def g() -> csp.Outputs(x=ts[int], y=ts[int]):
                __return__(x=1, y=2)

        @csp.node
        def n() -> ts[int]:
            return csp.output(1)

        # catch variable declarations within a state/alarm call
        with self.assertWarnsRegex(DeprecationWarning, "Variable declarations within alarms\\(\\) are deprecated*"):

            @csp.node
            def n():
                with csp.alarms(a=ts[bool]):
                    pass

        with self.assertWarnsRegex(DeprecationWarning, "Variable declarations within state\\(\\) are deprecated*"):

            @csp.node
            def n():
                with csp.state(s=0):
                    pass

        # reset the opt-in settings now that the test is done
        csp.impl.warnings.set_deprecation_warning(original_setting)


if __name__ == "__main__":
    unittest.main()
