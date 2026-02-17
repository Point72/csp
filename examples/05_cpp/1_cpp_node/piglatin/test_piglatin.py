from datetime import datetime

import csp

from .__main__ import my_graph


def test_piglatin():
    start = datetime(2020, 1, 1)
    res = csp.run(my_graph, starttime=start)
    assert res == {
        0: [
            (
                datetime(2020, 1, 1, 0, 0, 0, 500000),
                "IGPAY",
            ),
            (
                datetime(2020, 1, 1, 0, 0, 1, 500000),
                "ATINLAY",
            ),
            (
                datetime(2020, 1, 1, 0, 0, 5),
                "UNFAY",
            ),
        ],
    }
