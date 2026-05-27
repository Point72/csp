from datetime import datetime, timedelta

from csp import Outputs, curve, graph, print, run, ts

from . import piglatin


@graph
def my_graph() -> Outputs(ts[str]):
    st = datetime(2020, 1, 1)

    # curve of values
    names = curve(
        str,
        [
            (st + timedelta(seconds=0.5), "pig"),
            (st + timedelta(seconds=1.5), "latin"),
            (st + timedelta(seconds=5), "fun"),
        ],
    )

    # piglatinify
    print("input", names)
    piglatinify = piglatin(names, capitalize=True)
    print("output", piglatinify)
    return piglatinify


if __name__ == "__main__":
    start = datetime(2020, 1, 1)
    run(my_graph, starttime=start)

# Output:
# 2020-01-01 00:00:00.500000 input:pig
# 2020-01-01 00:00:00.500000 output:IGPAY
# 2020-01-01 00:00:01.500000 input:latin
# 2020-01-01 00:00:01.500000 output:ATINLAY
# 2020-01-01 00:00:05 input:fun
# 2020-01-01 00:00:05 output:UNFAY
