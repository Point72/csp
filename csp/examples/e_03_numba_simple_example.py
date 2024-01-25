from datetime import datetime, timedelta

import csp
from csp import ts

node_t = csp.numba_node
# node_t = csp.node


@node_t
def add_val(input: ts['T'], val_to_add: '~T') -> ts['T']:
    if csp.ticked(input) and csp.valid(input):
        res = input + val_to_add
        return res


@node_t
def my_prod(input1: ts['T'], input2: ts['T']) -> ts['T']:
    if csp.ticked(input1, input2) and csp.valid(input1, input2):
        return input1 * input2


@node_t(state_types={'accum': 'T'})
# @node_t(debug_print=True)
def cum_sum(input: ts['T']) -> ts['T']:
    with csp.state():
        accum=0

    if csp.ticked(input) and csp.valid(input):
        accum += input // 10000
        return accum


@csp.graph
def my_graph():
    my_ts = csp.timer(timedelta(microseconds=100000), 100)
    my_ts2 = add_val(my_ts, 100)

    accum_prod = cum_sum(my_prod(my_ts2, my_ts2))

    sampled_s = csp.sample(csp.timer(timedelta(seconds=3600)), accum_prod)

    csp.add_graph_output('sampled', sampled_s)


if __name__ == '__main__':
    s = datetime.now()
    g = csp.run(my_graph, starttime=datetime(2020, 3, 1, 9, 30), endtime=timedelta(hours=0, minutes=390))
    e = datetime.now()
    print((e - s).total_seconds())
