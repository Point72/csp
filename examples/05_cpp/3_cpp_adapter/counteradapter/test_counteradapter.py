from datetime import timedelta

import csp
from csp.utils.datetime import utc_now

from .__main__ import counter_graph


def test_counteradapter():
    start = utc_now()
    end = start + timedelta(seconds=2)

    result = csp.run(counter_graph, starttime=start, endtime=end, realtime=True)

    # Verify we got counter values
    assert "counter" in result
    assert len(result["counter"]) == 10  # max_count=10 in counter_graph
    # Verify counter values are sequential 1-10
    values = [v for _, v in result["counter"]]
    assert values == list(range(1, 11))
