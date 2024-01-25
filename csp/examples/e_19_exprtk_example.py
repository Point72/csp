import datetime

import csp


def g():
    s = csp.curve(
        str,
        [(datetime.timedelta(microseconds=i), "ADD" if i % 3 else ("MULT" if i % 2 else "TIME")) for i in range(0, 10)],
    )
    x = csp.curve(float, [(datetime.timedelta(microseconds=i), i) for i in range(10)])
    y = csp.curve(float, [(datetime.timedelta(microseconds=i), i / 2.0) for i in range(0, 10, 2)])

    csp.print("python_ticked_values", {"x": x, "y": y, "s": s})

    # this below expression shows the ability to:
    #   - use float and string variables
    #   - declare and access state variables with defaults
    #   - use csp.now()

    expression = """
if (s == 'ADD')
    FOO := FOO + x + y;
else if (s == 'MULT')
    FOO := FOO + (x * y);
else if (s == 'TIME')
    csp.now();
"""

    csp.print("expr_ts_val", csp.exprtk(expression, {"x": x, "y": y, "s": s}, {"FOO": 1000}))


def g2():
    x = csp.curve(float, [(datetime.timedelta(microseconds=i), i) for i in range(10)])
    y = csp.curve(float, [(datetime.timedelta(microseconds=i), i / 2.0) for i in range(0, 10, 2)])
    trigger = csp.curve(bool, [(datetime.timedelta(microseconds=3), True), (datetime.timedelta(microseconds=7), True)])

    csp.print("python_ticked_values", {"x": x, "y": y, "trigger": trigger})

    expression = "x+y"

    csp.print("expr_ts_val", csp.exprtk(expression, {"x": x, "y": y}, trigger=trigger))


def g3():
    x = csp.curve(float, [(datetime.timedelta(microseconds=i), i) for i in range(10)])
    y = csp.const(10.0)
    csp.print("expr_w_fn", csp.exprtk("2 * foo(x,y)", {"x": x, "y": y}, functions={"foo": (("x", "y"), "x / y")}))


def g4():
    x = csp.curve(float, [(datetime.timedelta(microseconds=i), i) for i in range(10)])
    csp.print("expr_array_out", csp.exprtk("return [x, 2*x, 3*x]", {"x": x}, output_ndarray=True))


if __name__ == "__main__":
    print("W/O TRIGGER:")
    csp.run(g, starttime=datetime.datetime(2022, 1, 1), endtime=datetime.timedelta(seconds=10))
    print("WITH TRIGGER:")
    csp.run(g2, starttime=datetime.datetime(2022, 1, 1), endtime=datetime.timedelta(seconds=10))
    print("WITH FUNCTION:")
    csp.run(g3, starttime=datetime.datetime(2022, 1, 1), endtime=datetime.timedelta(seconds=10))
    print("NDARRAY OUTPUT:")
    csp.run(g4, starttime=datetime.datetime(2022, 1, 1), endtime=datetime.timedelta(seconds=10))
