import unittest
from datetime import datetime, timedelta

import csp
from csp import ts


class TestNumbaFeaturesSupport(unittest.TestCase):
    def graph_factory(self, use_numba=False):
        """
        The graph here has no special logic, it's pretty much garbage.
        The goal is to cover all features and make sure that the python implementation output matches numba
        :param use_numba:
        :return:
        """

        def node_wrapper(*args, state_types=None, **kwargs):
            if use_numba:
                return csp.numba_node(*args, state_types=state_types, **kwargs)
            else:
                return csp.node(*args, **kwargs)

        @node_wrapper
        def add_val(input: ts["T"], val_to_add: "~T") -> ts["T"]:
            if csp.ticked(input) and csp.valid(input):
                res = input + val_to_add
                return res

        @node_wrapper
        def my_prod(input1: ts["T"], input2: ts["T"]) -> ts["T"]:
            if csp.ticked(input1, input2) and csp.valid(input1, input2):
                return input1 * input2

        @node_wrapper
        def cum_sum(input: ts["T"]) -> ts["T"]:
            with csp.state():
                accum = 0
            if csp.ticked(input) and csp.valid(input):
                accum += input // 10000

                return accum

        @node_wrapper(state_types={"my_value_to_add": "~T"})
        def add_val_once(input: ts["T"], val_to_add: "~T") -> ts["T"]:
            with csp.state():
                my_value_to_add = val_to_add
                empty_return = True
                val_to_add *= val_to_add

            if csp.ticked(input) and csp.valid(input):
                if empty_return:
                    empty_return = False
                    return

                # TODO: when alarms are supported schedule an alarm in the future to make it active
                csp.make_passive(input)
                return input + val_to_add

        @csp.graph
        def graph() -> csp.Outputs(sampled=ts[int], sampled2=ts[int], sampled3=ts[int], sampled4=ts[float]):
            my_ts = csp.timer(timedelta(microseconds=100000000), 100)
            my_ts2 = add_val(my_ts, 100)

            accum_prod = cum_sum(my_prod(my_ts2, my_ts2))
            # accum_prod = cum_sum(accum_prod, 6)
            # accum_prod = cum_sum(accum_prod, 7.0)

            sampled_s = csp.sample(csp.timer(timedelta(seconds=3600)), accum_prod)
            sampled_s2 = add_val_once(sampled_s, 1)
            sampled_s3 = add_val_once(sampled_s, 2)
            sampled_s4 = add_val_once(sampled_s, 3.0)

            return csp.output(sampled=sampled_s, sampled2=sampled_s2, sampled3=sampled_s3, sampled4=sampled_s4)

        return graph

    @unittest.skip("numba not yet used, tests fail on newer numba we get in our 3.8 build")
    def test_numba_python_output_identical(self):
        python_graph = self.graph_factory(use_numba=False)
        numba_graph = self.graph_factory(use_numba=True)

        python_res = csp.run(
            python_graph, starttime=datetime(2020, 3, 1, 9, 30), endtime=timedelta(hours=0, minutes=390)
        )
        numba_res = csp.run(numba_graph, starttime=datetime(2020, 3, 1, 9, 30), endtime=timedelta(hours=0, minutes=390))

        self.assertEqual(python_res, numba_res)


if __name__ == "__main__":
    unittest.main()
