import unittest
from datetime import datetime, timedelta, timezone

import csp
from csp import ts
from csp.impl.wiring.csp_numba import numba_node
from numba_type_utils.numba_config import NumbaList, NumbaDict, create_new_list, create_new_dict


class TestBasicTypes(unittest.TestCase):
    def test_primitives(self):
        @numba_node
        def primitives_node(x: ts[int]) -> ts[float]:
            with csp.state():
                s_count: int = 0
                s_sum: float = 0.0
                s_flag: bool = False

            s_count = s_count + 1
            s_sum = s_sum + x
            s_flag = not s_flag
            # Add 10 on odd ticks (when s_flag is True)
            if s_flag:
                return s_count + s_sum + 10.0
            else:
                return s_count + s_sum

        @csp.graph
        def g():
            # x = 1, 2, 3
            values = csp.curve(int, [(timedelta(seconds=i), i + 1) for i in range(3)])
            result = primitives_node(values)
            csp.add_graph_output("result", result)

        # tick 1: s_count=1, s_sum=1.0, s_flag=True  -> output = 1 + 1.0 + 10 = 12.0
        # tick 2: s_count=2, s_sum=3.0, s_flag=False -> output = 2 + 3.0 = 5.0
        # tick 3: s_count=3, s_sum=6.0, s_flag=True  -> output = 3 + 6.0 + 10 = 19.0
        results = csp.run(g, starttime=datetime(2024, 1, 1), endtime=timedelta(seconds=10))
        actual = [v for _, v in results["result"]]
        self.assertEqual(actual, [12.0, 5.0, 19.0])

    def test_constant_inputs(self):
        @numba_node
        def with_constants(x: ts[int], multiplier: int, offset: float) -> ts[float]:
            return x * multiplier + offset

        @csp.graph
        def g():
            values = csp.curve(int, [(timedelta(seconds=i), i + 1) for i in range(3)])
            result = with_constants(values, 10, 0.5)
            csp.add_graph_output("result", result)

        results = csp.run(g, starttime=datetime(2024, 1, 1), endtime=timedelta(seconds=10))
        actual = [v for _, v in results["result"]]
        self.assertEqual(actual, [10.5, 20.5, 30.5])

    def test_timedelta(self):
        @numba_node
        def timedelta_node(td: ts[timedelta], scale: timedelta) -> ts[timedelta]:
            with csp.state():
                s_total: timedelta = timedelta(seconds=10)

            local_incr = timedelta(seconds=1)
            s_total = s_total + td + local_incr
            return s_total + scale

        @csp.graph
        def g():
            deltas = csp.curve(
                timedelta,
                [
                    (timedelta(seconds=0), timedelta(seconds=5)),
                    (timedelta(seconds=1), timedelta(seconds=10)),
                ],
            )
            result = timedelta_node(deltas, timedelta(seconds=100))
            csp.add_graph_output("result", result)

        results = csp.run(g, starttime=datetime(2024, 1, 1), endtime=timedelta(seconds=10))
        actual = [v for _, v in results["result"]]
        self.assertEqual(actual[0], timedelta(seconds=116))
        self.assertEqual(actual[1], timedelta(seconds=127))

    def test_datetime(self):
        @numba_node
        def datetime_node(dt: ts[datetime], base: datetime) -> ts[datetime]:
            with csp.state():
                s_last: datetime = 0

            with csp.start():
                s_last = base

            # If dt is after threshold, return dt; otherwise return s_last
            local_threshold = datetime(2024, 6, 1, tzinfo=timezone.utc)
            if dt > local_threshold:
                result = dt
            else:
                result = s_last
            s_last = dt
            return result

        @csp.graph
        def g():
            dates = csp.curve(
                datetime,
                [
                    (timedelta(seconds=0), datetime(2024, 1, 1, tzinfo=timezone.utc)),  # before threshold
                    (timedelta(seconds=1), datetime(2024, 7, 1, tzinfo=timezone.utc)),  # after threshold
                ],
            )
            result = datetime_node(dates, datetime(2020, 1, 1, tzinfo=timezone.utc))
            csp.add_graph_output("result", result)

        # tick 1: dt=2024-01-01 < threshold, return s_last=2020-01-01, then s_last=2024-01-01
        # tick 2: dt=2024-07-01 > threshold, return dt=2024-07-01, then s_last=2024-07-01
        results = csp.run(g, starttime=datetime(2024, 1, 1), endtime=timedelta(seconds=10))
        actual = [v for _, v in results["result"]]
        self.assertEqual(
            actual,
            [
                datetime(2020, 1, 1),
                datetime(2024, 7, 1),
            ],
        )

    def test_datetime_timedelta_combined(self):
        @numba_node
        def time_calc(x: ts[int], base: datetime, interval: timedelta) -> ts[datetime]:
            with csp.state():
                s_current: datetime = 0

            with csp.start():
                s_current = base

            s_current = s_current + interval
            return s_current

        @csp.graph
        def g():
            values = csp.curve(int, [(timedelta(seconds=i), i) for i in range(3)])
            result = time_calc(values, datetime(2024, 1, 1, tzinfo=timezone.utc), timedelta(hours=2))
            csp.add_graph_output("result", result)

        # tick 1: s_current = base + 2h = 2024-01-01 02:00
        # tick 2: s_current = base + 4h = 2024-01-01 04:00
        # tick 3: s_current = base + 6h = 2024-01-01 06:00
        results = csp.run(g, starttime=datetime(2024, 1, 1), endtime=timedelta(seconds=10))
        actual = [v for _, v in results["result"]]
        self.assertEqual(
            actual,
            [
                datetime(2024, 1, 1, 2),
                datetime(2024, 1, 1, 4),
                datetime(2024, 1, 1, 6),
            ],
        )

    def test_enum(self):
        class Mode(csp.Enum):
            IDLE = 0
            RUNNING = 1
            PAUSED = 2

        @numba_node
        def enum_node(cmd: ts[int], default_mode: Mode) -> ts[Mode]:
            with csp.state():
                s_mode: Mode = Mode.IDLE

            local_running = Mode.RUNNING
            local_paused = Mode.PAUSED
            if cmd == 1:
                s_mode = local_running
            elif cmd == 2:
                s_mode = local_paused
            else:
                s_mode = default_mode
            return s_mode

        @csp.graph
        def g():
            commands = csp.curve(
                int,
                [
                    (timedelta(seconds=0), 0),
                    (timedelta(seconds=1), 1),
                    (timedelta(seconds=2), 2),
                ],
            )
            result = enum_node(commands, Mode.IDLE)
            csp.add_graph_output("result", result)

        results = csp.run(g, starttime=datetime(2024, 1, 1), endtime=timedelta(seconds=10))
        actual = [v for _, v in results["result"]]
        self.assertEqual(actual, [Mode.IDLE, Mode.RUNNING, Mode.PAUSED])

    def test_unsupported_state_expression(self):
        with self.assertRaisesRegex(ValueError, "Unable to infer type for state variable 'total'"):

            @numba_node
            def bad_state_inference(x: ts[int]) -> ts[int]:
                with csp.state():
                    total = x + 1
                return x


class TestContainerTypes(unittest.TestCase):
    def test_list_basic_operations(self):
        @numba_node
        def list_node(x: ts[int]) -> csp.Outputs(
            length=ts[int], first=ts[int], last=ts[int], popped=ts[int], total=ts[int]
        ):
            with csp.state():
                s_values: NumbaList[int] = create_new_list(int)

            s_values.append(x)
            s_values.append(x * 2)
            csp.output(length=len(s_values))
            csp.output(first=s_values[0])
            csp.output(last=s_values[len(s_values) - 1])
            csp.output(popped=s_values.pop())
            total = 0
            for v in s_values:
                total = total + v
            csp.output(total=total)

        @csp.graph
        def g():
            inputs = csp.curve(int, [(timedelta(seconds=i), i + 1) for i in range(3)])
            result = list_node(inputs)
            csp.add_graph_output("length", result.length)
            csp.add_graph_output("first", result.first)
            csp.add_graph_output("last", result.last)
            csp.add_graph_output("popped", result.popped)
            csp.add_graph_output("total", result.total)

        # tick 1: x=1 -> [1,2], len=2, first=1, last=2, pop->2, total=1
        # tick 2: x=2 -> [1,2,4], len=3, first=1, last=4, pop->4, total=3
        # tick 3: x=3 -> [1,2,3,6], len=4, first=1, last=6, pop->6, total=6
        results = csp.run(g, starttime=datetime(2024, 1, 1), endtime=timedelta(seconds=10))
        self.assertEqual([v for _, v in results["length"]], [2, 3, 4])
        self.assertEqual([v for _, v in results["first"]], [1, 1, 1])
        self.assertEqual([v for _, v in results["last"]], [2, 4, 6])
        self.assertEqual([v for _, v in results["popped"]], [2, 4, 6])
        self.assertEqual([v for _, v in results["total"]], [1, 3, 6])

    def test_list_float(self):
        @numba_node
        def running_avg(x: ts[float], window: int) -> ts[float]:
            with csp.state():
                s_values: NumbaList[float] = create_new_list(float)

            s_values.append(x)
            # Keep only last 'window' values
            while len(s_values) > window:
                # Shift elements left (poor man's deque)
                for i in range(len(s_values) - 1):
                    s_values[i] = s_values[i + 1]
                s_values.pop()

            total = 0.0
            for v in s_values:
                total = total + v
            return total / len(s_values)

        @csp.graph
        def g():
            inputs = csp.curve(float, [(timedelta(seconds=i), float(i + 1)) for i in range(5)])
            result = running_avg(inputs, 3)
            csp.add_graph_output("result", result)

        # window=3: [1]->1, [1,2]->1.5, [1,2,3]->2, [2,3,4]->3, [3,4,5]->4
        results = csp.run(g, starttime=datetime(2024, 1, 1), endtime=timedelta(seconds=10))
        actual = [v for _, v in results["result"]]
        self.assertEqual(actual, [1.0, 1.5, 2.0, 3.0, 4.0])

    def test_list_index_assignment(self):
        @numba_node
        def modify_list(x: ts[int]) -> ts[int]:
            with csp.state():
                s_values: NumbaList[int] = create_new_list(int)

            s_values.append(x)
            if len(s_values) >= 2:
                # Double the first element
                s_values[0] = s_values[0] * 2
            return s_values[0]

        @csp.graph
        def g():
            inputs = csp.curve(int, [(timedelta(seconds=i), 10) for i in range(4)])
            result = modify_list(inputs)
            csp.add_graph_output("result", result)

        # tick 1: [10] -> first=10
        # tick 2: [10,10], first*=2 -> [20,10] -> first=20
        # tick 3: [20,10,10], first*=2 -> [40,10,10] -> first=40
        # tick 4: [40,10,10,10], first*=2 -> [80,10,10,10] -> first=80
        results = csp.run(g, starttime=datetime(2024, 1, 1), endtime=timedelta(seconds=10))
        actual = [v for _, v in results["result"]]
        self.assertEqual(actual, [10, 20, 40, 80])

    def test_list_multiple_lists(self):
        @numba_node
        def dual_lists(x: ts[int]) -> csp.Outputs(evens=ts[int], odds=ts[int]):
            with csp.state():
                s_evens: NumbaList[int] = create_new_list(int)
                s_odds: NumbaList[int] = create_new_list(int)

            if x % 2 == 0:
                s_evens.append(x)
                csp.output(evens=len(s_evens))
            else:
                s_odds.append(x)
                csp.output(odds=len(s_odds))

        @csp.graph
        def g():
            inputs = csp.curve(int, [(timedelta(seconds=i), i + 1) for i in range(6)])
            result = dual_lists(inputs)
            csp.add_graph_output("evens", result.evens)
            csp.add_graph_output("odds", result.odds)

        # 1(odd), 2(even), 3(odd), 4(even), 5(odd), 6(even)
        results = csp.run(g, starttime=datetime(2024, 1, 1), endtime=timedelta(seconds=10))
        self.assertEqual([v for _, v in results["evens"]], [1, 2, 3])
        self.assertEqual([v for _, v in results["odds"]], [1, 2, 3])

    def test_dict_count_occurrences(self):
        @numba_node
        def count_node(x: ts[int]) -> ts[int]:
            with csp.state():
                s_counts: NumbaDict[int, int] = create_new_dict(int, int)

            s_counts[x] += 1
            return s_counts[x]

        @csp.graph
        def g():
            inputs = csp.curve(int, [(timedelta(seconds=i), v) for i, v in enumerate([1, 2, 1, 1, 2])])
            result = count_node(inputs)
            csp.add_graph_output("result", result)

        results = csp.run(g, starttime=datetime(2024, 1, 1), endtime=timedelta(seconds=10))
        actual = [v for _, v in results["result"]]
        self.assertEqual(actual, [1, 1, 2, 3, 2])

    def test_dict_contains(self):
        @numba_node
        def contains_node(x: ts[int]) -> csp.Outputs(found=ts[int], not_found=ts[int]):
            with csp.state():
                s_seen: NumbaDict[int, int] = create_new_dict(int, int)

            if x in s_seen:
                csp.output(found=s_seen[x])
            else:
                csp.output(not_found=x)
            s_seen[x] = x * 10

        @csp.graph
        def g():
            # 1, 2, 1, 3, 2
            inputs = csp.curve(int, [(timedelta(seconds=i), v) for i, v in enumerate([1, 2, 1, 3, 2])])
            result = contains_node(inputs)
            csp.add_graph_output("found", result.found)
            csp.add_graph_output("not_found", result.not_found)

        # t=0: x=1 not in {} -> not_found=1, then s_seen={1:10}
        # t=1: x=2 not in {1:10} -> not_found=2, then s_seen={1:10,2:20}
        # t=2: x=1 in {1:10,2:20} -> found=10, then s_seen={1:10,2:20}
        # t=3: x=3 not in {1:10,2:20} -> not_found=3, then s_seen={1:10,2:20,3:30}
        # t=4: x=2 in {1:10,2:20,3:30} -> found=20
        results = csp.run(g, starttime=datetime(2024, 1, 1), endtime=timedelta(seconds=10))
        self.assertEqual([v for _, v in results["not_found"]], [1, 2, 3])
        self.assertEqual([v for _, v in results["found"]], [10, 20])

    def test_dict_float_values(self):
        @numba_node
        def sum_by_key(key: ts[int], val: ts[float]) -> ts[float]:
            with csp.state():
                s_sums: NumbaDict[int, float] = create_new_dict(int, float)

            if csp.ticked(key) and csp.ticked(val):
                s_sums[key] += val
                return s_sums[key]

        @csp.graph
        def g():
            keys = csp.curve(int, [(timedelta(seconds=i), k) for i, k in enumerate([1, 2, 1, 2, 1])])
            vals = csp.curve(float, [(timedelta(seconds=i), float(i + 1) * 10) for i in range(5)])
            result = sum_by_key(keys, vals)
            csp.add_graph_output("result", result)

        # key=1,val=10 -> {1:10} -> 10
        # key=2,val=20 -> {1:10,2:20} -> 20
        # key=1,val=30 -> {1:40,2:20} -> 40
        # key=2,val=40 -> {1:40,2:60} -> 60
        # key=1,val=50 -> {1:90,2:60} -> 90
        results = csp.run(g, starttime=datetime(2024, 1, 1), endtime=timedelta(seconds=10))
        actual = [v for _, v in results["result"]]
        self.assertEqual(actual, [10.0, 20.0, 40.0, 60.0, 90.0])

    def test_list_state_with_basket_inputs(self):
        @numba_node
        def basket_aggregator(xs: list[ts[float]]) -> csp.Outputs(avg=ts[float], delta=ts[float]):
            with csp.state():
                s_history: NumbaList[float] = create_new_list(float)

            total = 0.0
            count = 0
            for i in range(len(xs)):
                if xs[i].ticked():
                    total = total + xs[i]
                    count = count + 1

            if count > 0:
                avg = total / count
                s_history.append(avg)
                csp.output(avg=avg)
                if len(s_history) >= 2:
                    prev = s_history[len(s_history) - 2]
                    csp.output(delta=avg - prev)

        @csp.graph
        def g():
            a = csp.curve(float, [(timedelta(seconds=0), 10.0), (timedelta(seconds=1), 20.0)])
            b = csp.curve(float, [(timedelta(seconds=0), 100.0), (timedelta(seconds=2), 300.0)])
            result = basket_aggregator([a, b])
            csp.add_graph_output("avg", result.avg)
            csp.add_graph_output("delta", result.delta)

        # t=0: a=10, b=100 tick -> avg=(10+100)/2=55
        # t=1: a=20 ticks -> avg=20, delta=20-55=-35
        # t=2: b=300 ticks -> avg=300, delta=300-20=280
        results = csp.run(g, starttime=datetime(2024, 1, 1), endtime=timedelta(seconds=10))
        self.assertEqual([v for _, v in results["avg"]], [55.0, 20.0, 300.0])
        self.assertEqual([v for _, v in results["delta"]], [-35.0, 280.0])

    def test_list_with_enum_mode_state(self):
        """Test list state coexisting with enum state and windowed processing."""

        class Mode(csp.Enum):
            SUM = 0
            AVG = 1

        @numba_node
        def windowed_processor(x: ts[float], mode: ts[Mode], window_size: int) -> ts[float]:
            with csp.state():
                s_values: NumbaList[float] = create_new_list(float)
                s_mode: Mode = Mode.SUM

            if csp.ticked(mode):
                s_mode = mode

            if csp.ticked(x):
                s_values.append(x)

                # Calculate over last window_size values
                start_idx = 0
                if len(s_values) > window_size:
                    start_idx = len(s_values) - window_size

                total = 0.0
                count = 0
                for i in range(start_idx, len(s_values)):
                    total = total + s_values[i]
                    count = count + 1

                if s_mode == Mode.SUM:
                    return total
                return total / count

        @csp.graph
        def g():
            values = csp.curve(float, [(timedelta(seconds=i), float(i + 1)) for i in range(6)])
            # Start with SUM, switch to AVG at t=3
            mode = csp.curve(
                Mode,
                [
                    (timedelta(seconds=0), Mode.SUM),
                    (timedelta(seconds=3), Mode.AVG),
                ],
            )
            result = windowed_processor(values, mode, 3)
            csp.add_graph_output("result", result)

        # window_size=3
        # t=0: x=1, window=[1], mode=SUM -> 1.0
        # t=1: x=2, window=[1,2], mode=SUM -> 3.0
        # t=2: x=3, window=[1,2,3], mode=SUM -> 6.0
        # t=3: x=4, window=[2,3,4], mode=AVG -> 9/3 = 3.0
        # t=4: x=5, window=[3,4,5], mode=AVG -> 12/3 = 4.0
        # t=5: x=6, window=[4,5,6], mode=AVG -> 15/3 = 5.0
        results = csp.run(g, starttime=datetime(2024, 1, 1), endtime=timedelta(seconds=10))
        actual = [v for _, v in results["result"]]
        self.assertEqual(actual, [1.0, 3.0, 6.0, 3.0, 4.0, 5.0])


class TestFeatures(unittest.TestCase):
    def test_lifecycle_start(self):
        @numba_node
        def with_start(x: ts[int]) -> ts[int]:
            with csp.state():
                s_mult = 0
                s_base = 100

            with csp.start():
                s_mult = 10
                s_base = 500

            s_base = s_base + x * s_mult
            return s_base

        @csp.graph
        def g():
            values = csp.curve(int, [(timedelta(seconds=i), i + 1) for i in range(3)])
            result = with_start(values)
            csp.add_graph_output("result", result)

        results = csp.run(g, starttime=datetime(2024, 1, 1), endtime=timedelta(seconds=10))
        actual = [v for _, v in results["result"]]
        self.assertEqual(actual, [510, 530, 560])

    def test_ticked_and_valid_methods(self):
        @numba_node
        def check_signals(a: ts[int], b: ts[int]) -> csp.Outputs(
            a_count=ts[int],
            b_count=ts[int],
            any_ticked=ts[int],
            a_valid=ts[int],
            b_valid=ts[int],
            both_valid=ts[int],
        ):
            with csp.state():
                s_a_count: int = 0
                s_b_count: int = 0

            if csp.ticked(a):
                s_a_count = s_a_count + 1
                csp.output(a_count=s_a_count)
            if csp.ticked(b):
                s_b_count = s_b_count + 1
                csp.output(b_count=s_b_count)

            if csp.ticked(a, b):
                csp.output(any_ticked=s_a_count * 10 + s_b_count)

            if csp.valid(a):
                csp.output(a_valid=a)
            if csp.valid(b):
                csp.output(b_valid=b)
            if csp.valid(a, b):
                csp.output(both_valid=a + b)

        @csp.graph
        def g():
            a = csp.curve(int, [(timedelta(seconds=0), 1), (timedelta(seconds=2), 2)])
            b = csp.curve(int, [(timedelta(seconds=1), 5), (timedelta(seconds=2), 15)])
            result = check_signals(a, b)
            csp.add_graph_output("a_count", result.a_count)
            csp.add_graph_output("b_count", result.b_count)
            csp.add_graph_output("any_ticked", result.any_ticked)
            csp.add_graph_output("a_valid", result.a_valid)
            csp.add_graph_output("b_valid", result.b_valid)
            csp.add_graph_output("both_valid", result.both_valid)

        # t=0: a ticks, only a is valid
        # t=1: b ticks, both a and b are valid
        # t=2: a and b both tick, both are valid
        results = csp.run(g, starttime=datetime(2024, 1, 1), endtime=timedelta(seconds=10))
        self.assertEqual([v for _, v in results["a_count"]], [1, 2])
        self.assertEqual([v for _, v in results["b_count"]], [1, 2])
        self.assertEqual([v for _, v in results["any_ticked"]], [10, 11, 22])
        self.assertEqual([v for _, v in results["a_valid"]], [1, 1, 2])
        self.assertEqual([v for _, v in results["b_valid"]], [5, 15])
        self.assertEqual([v for _, v in results["both_valid"]], [6, 17])

    def test_list_basket_input(self):
        @numba_node
        def basket_stats(xs: [ts[int]]) -> csp.Outputs(ticked_count=ts[int], ticked_sum=ts[int]):
            ticked_count = 0
            ticked_sum = 0
            for i in range(len(xs)):
                if xs[i].ticked():
                    ticked_count = ticked_count + 1
                    ticked_sum = ticked_sum + xs[i]
            csp.output(ticked_count=ticked_count)
            csp.output(ticked_sum=ticked_sum)

        @csp.graph
        def g():
            a = csp.curve(int, [(timedelta(seconds=0), 10), (timedelta(seconds=2), 20)])
            b = csp.curve(int, [(timedelta(seconds=1), 100)])
            c = csp.const(1000)
            result = basket_stats([a, b, c])
            csp.add_graph_output("ticked_count", result.ticked_count)
            csp.add_graph_output("ticked_sum", result.ticked_sum)

        # t=0: a=10, c=1000 tick -> ticked: a,c
        # t=1: b=100 ticks -> ticked: b
        # t=2: a=20 ticks -> ticked: a
        results = csp.run(g, starttime=datetime(2024, 1, 1), endtime=timedelta(seconds=10))
        self.assertEqual([v for _, v in results["ticked_count"]], [2, 1, 1])
        self.assertEqual([v for _, v in results["ticked_sum"]], [1010, 100, 20])

    def test_list_basket_input_modern_annotation(self):
        @numba_node
        def basket_stats(xs: list[ts[int]]) -> csp.Outputs(ticked_count=ts[int], ticked_sum=ts[int]):
            ticked_count = 0
            ticked_sum = 0
            for i in range(len(xs)):
                if xs[i].ticked():
                    ticked_count = ticked_count + 1
                    ticked_sum = ticked_sum + xs[i]
            csp.output(ticked_count=ticked_count)
            csp.output(ticked_sum=ticked_sum)

        @csp.graph
        def g():
            a = csp.curve(int, [(timedelta(seconds=0), 10), (timedelta(seconds=2), 20)])
            b = csp.curve(int, [(timedelta(seconds=1), 100)])
            c = csp.const(1000)
            result = basket_stats([a, b, c])
            csp.add_graph_output("ticked_count", result.ticked_count)
            csp.add_graph_output("ticked_sum", result.ticked_sum)

        results = csp.run(g, starttime=datetime(2024, 1, 1), endtime=timedelta(seconds=10))
        self.assertEqual([v for _, v in results["ticked_count"]], [2, 1, 1])
        self.assertEqual([v for _, v in results["ticked_sum"]], [1010, 100, 20])

    def test_dict_basket_input(self):
        @numba_node
        def basket_stats(prices: {str: ts[float]}) -> csp.Outputs(ticked_sum=ts[float], valid_sum=ts[float]):
            ticked_sum = 0.0
            valid_sum = 0.0
            for key in prices.tickedkeys():
                ticked_sum = ticked_sum + prices[key]
            for key in prices.validkeys():
                valid_sum = valid_sum + prices[key]
            csp.output(ticked_sum=ticked_sum)
            csp.output(valid_sum=valid_sum)

        @csp.graph
        def g():
            a = csp.curve(float, [(timedelta(seconds=0), 100.0), (timedelta(seconds=2), 200.0)])
            b = csp.curve(float, [(timedelta(seconds=1), 50.0)])
            result = basket_stats({"a": a, "b": b})
            csp.add_graph_output("ticked_sum", result.ticked_sum)
            csp.add_graph_output("valid_sum", result.valid_sum)

        # t=0: a=100 ticks -> ticked: a; valid: a
        # t=1: b=50 ticks -> ticked: b; valid: a,b
        # t=2: a=200 ticks -> ticked: a; valid: a,b
        results = csp.run(g, starttime=datetime(2024, 1, 1), endtime=timedelta(seconds=10))
        self.assertEqual([v for _, v in results["ticked_sum"]], [100.0, 50.0, 200.0])
        self.assertEqual([v for _, v in results["valid_sum"]], [100.0, 150.0, 250.0])

    def test_dict_basket_input_modern_annotation(self):
        @numba_node
        def basket_stats(prices: dict[str, ts[float]]) -> csp.Outputs(ticked_sum=ts[float], valid_sum=ts[float]):
            ticked_sum = 0.0
            valid_sum = 0.0
            for key in prices.tickedkeys():
                ticked_sum = ticked_sum + prices[key]
            for key in prices.validkeys():
                valid_sum = valid_sum + prices[key]
            csp.output(ticked_sum=ticked_sum)
            csp.output(valid_sum=valid_sum)

        @csp.graph
        def g():
            a = csp.curve(float, [(timedelta(seconds=0), 100.0), (timedelta(seconds=2), 200.0)])
            b = csp.curve(float, [(timedelta(seconds=1), 50.0)])
            result = basket_stats({"a": a, "b": b})
            csp.add_graph_output("ticked_sum", result.ticked_sum)
            csp.add_graph_output("valid_sum", result.valid_sum)

        results = csp.run(g, starttime=datetime(2024, 1, 1), endtime=timedelta(seconds=10))
        self.assertEqual([v for _, v in results["ticked_sum"]], [100.0, 50.0, 200.0])
        self.assertEqual([v for _, v in results["valid_sum"]], [100.0, 150.0, 250.0])

    def test_basket_ticked_and_valid(self):
        @numba_node
        def check_basket(xs: list[ts[int]]) -> csp.Outputs(tick_count=ts[int], all_valid=ts[int]):
            with csp.state():
                s_count: int = 0

            s_count = s_count + 1
            csp.output(tick_count=s_count)
            if csp.valid(xs):
                csp.output(all_valid=1)
            else:
                csp.output(all_valid=0)

        @csp.graph
        def g():
            a = csp.curve(int, [(timedelta(seconds=0), 1)])
            b = csp.curve(int, [(timedelta(seconds=1), 2)])
            result = check_basket([a, b])
            csp.add_graph_output("tick_count", result.tick_count)
            csp.add_graph_output("all_valid", result.all_valid)

        # t=0: a ticks, b not valid -> tick_count=1, all_valid=0
        # t=1: b ticks, both valid -> tick_count=2, all_valid=1
        results = csp.run(g, starttime=datetime(2024, 1, 1), endtime=timedelta(seconds=10))
        self.assertEqual([v for _, v in results["tick_count"]], [1, 2])
        self.assertEqual([v for _, v in results["all_valid"]], [0, 1])

    def test_multiple_outputs(self):
        @numba_node
        def multi_out(x: ts[int]) -> csp.Outputs(doubled=ts[int], squared=ts[int], positive=ts[int]):
            csp.output(doubled=x * 2)
            csp.output(squared=x * x)
            if x > 0:
                csp.output(positive=x)

        @csp.graph
        def g():
            values = csp.curve(
                int,
                [
                    (timedelta(seconds=0), -2),
                    (timedelta(seconds=1), 3),
                    (timedelta(seconds=2), 0),
                ],
            )
            outputs = multi_out(values)
            csp.add_graph_output("doubled", outputs.doubled)
            csp.add_graph_output("squared", outputs.squared)
            csp.add_graph_output("positive", outputs.positive)

        results = csp.run(g, starttime=datetime(2024, 1, 1), endtime=timedelta(seconds=10))
        self.assertEqual([v for _, v in results["doubled"]], [-4, 6, 0])
        self.assertEqual([v for _, v in results["squared"]], [4, 9, 0])
        self.assertEqual([v for _, v in results["positive"]], [3])

    def test_conditional_output(self):
        @numba_node
        def emit_every_n(x: ts[int]) -> ts[int]:
            with csp.state():
                s_count = 0

            s_count = s_count + 1
            if s_count % 3 == 0:
                return x

        @csp.graph
        def g():
            values = csp.curve(int, [(timedelta(seconds=i), (i + 1) * 10) for i in range(9)])
            result = emit_every_n(values)
            csp.add_graph_output("result", result)

        results = csp.run(g, starttime=datetime(2024, 1, 1), endtime=timedelta(seconds=10))
        actual = [v for _, v in results["result"]]
        self.assertEqual(actual, [30, 60, 90])

    def test_dict_basket_with_enum_output(self):
        class Trend(csp.Enum):
            DOWN = -1
            FLAT = 0
            UP = 1

        @numba_node
        def trend_analyzer(prices: {str: ts[float]}) -> ts[Trend]:
            with csp.state():
                s_prev = 0.0
                s_has_prev = 0

            total = 0.0
            for key in prices.validkeys():
                total = total + prices[key]

            if s_has_prev == 0:
                s_prev = total
                s_has_prev = 1
                return Trend.FLAT

            diff = total - s_prev
            s_prev = total
            if diff > 0.01:
                return Trend.UP
            elif diff < -0.01:
                return Trend.DOWN
            return Trend.FLAT

        @csp.graph
        def g():
            a = csp.curve(float, [(timedelta(seconds=0), 100.0), (timedelta(seconds=1), 110.0)])
            b = csp.curve(float, [(timedelta(seconds=0), 200.0), (timedelta(seconds=1), 180.0)])
            result = trend_analyzer({"a": a, "b": b})
            csp.add_graph_output("result", result)

        results = csp.run(g, starttime=datetime(2024, 1, 1), endtime=timedelta(seconds=10))
        actual = [v for _, v in results["result"]]
        self.assertEqual(actual, [Trend.FLAT, Trend.DOWN])

    def test_running_statistics(self):
        @numba_node
        def running_stats(x: ts[float]) -> csp.Outputs(mean=ts[float], variance=ts[float]):
            with csp.state():
                s_count = 0
                s_mean = 0.0
                s_m2 = 0.0

            s_count = s_count + 1
            delta = x - s_mean
            s_mean = s_mean + delta / s_count
            delta2 = x - s_mean
            s_m2 = s_m2 + delta * delta2

            csp.output(mean=s_mean)
            if s_count > 1:
                var = s_m2 / (s_count - 1)
                csp.output(variance=var)

        @csp.graph
        def g():
            values = csp.curve(
                float, [(timedelta(milliseconds=i), v) for i, v in enumerate([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])]
            )
            outputs = running_stats(values)
            csp.add_graph_output("mean", outputs.mean)
            csp.add_graph_output("variance", outputs.variance)

        results = csp.run(g, starttime=datetime(2024, 1, 1), endtime=timedelta(seconds=1))
        means = [v for _, v in results["mean"]]
        variances = [v for _, v in results["variance"]]
        self.assertAlmostEqual(means[-1], 5.0, places=5)
        self.assertAlmostEqual(variances[-1], 32 / 7, places=5)


class TestStructSupport(unittest.TestCase):
    def test_invalid_struct_output(self):
        class Point(csp.Struct):
            x: float
            y: float

        @numba_node
        def emit_struct(x: ts[float]) -> ts[Point]:
            with csp.state():
                s_point: Point = Point(x=1.0, y=2.0)

            s_point.x = s_point.x + x
            return s_point

        @csp.graph
        def g():
            result = emit_struct(csp.const(1.0))
            csp.add_graph_output("result", result)

        with self.assertRaisesRegex(TypeError, r"Unsupported type '.*Point.*' for output\[0\]"):
            csp.build_graph(g)

    def test_invalid_struct_var(self):
        class NonNativePoint(csp.Struct):
            x: float
            label: str

        @numba_node
        def non_native_struct_state(x: ts[float]) -> ts[float]:
            with csp.state():
                s_point: NonNativePoint = NonNativePoint(x=0.0, label="start")

            s_point.x = s_point.x + x
            return s_point.x

        @csp.graph
        def g():
            result = non_native_struct_state(csp.const(1.0))
            csp.add_graph_output("result", result)

        with self.assertRaisesRegex(
            TypeError, r"numba_node only supports native csp\.Struct types; 'NonNativePoint' is non-native"
        ):
            csp.build_graph(g)

    def test_struct_state_multiple_structs(self):
        """Test multiple struct state variables."""

        class Direction(csp.Enum):
            UP = 1
            DOWN = -1

        class Position(csp.Struct):
            x: float
            steps: int
            active: bool

        class Velocity(csp.Struct):
            vx: float
            step_mult: int
            direction: Direction

        @numba_node
        def physics_node(dt: ts[float]) -> ts[float]:
            with csp.state():
                s_pos: Position = Position(x=0.0, steps=0, active=False)
                s_vel: Velocity = Velocity(vx=1.0, step_mult=2, direction=Direction.UP)

            s_pos.steps = s_pos.steps + 1
            s_pos.active = not s_pos.active

            if s_pos.active:
                delta = s_vel.vx * dt * s_vel.step_mult
                if s_vel.direction == Direction.UP:
                    s_pos.x = s_pos.x + delta
                else:
                    s_pos.x = s_pos.x - delta

            return s_pos.x + s_pos.steps

        @csp.graph
        def g():
            deltas = csp.curve(float, [(timedelta(seconds=i), 0.5) for i in range(4)])
            result = physics_node(deltas)
            csp.add_graph_output("result", result)

        results = csp.run(g, starttime=datetime(2024, 1, 1), endtime=timedelta(seconds=10))
        actual = [v for _, v in results["result"]]
        # steps: 1, 2, 3, 4
        # active toggles: True, False, True, False
        # x updates only on active ticks: +1.0, +0.0, +1.0, +0.0
        # result = x + steps -> 2.0, 3.0, 5.0, 6.0
        self.assertEqual(actual, [2.0, 3.0, 5.0, 6.0])

    def test_struct_state_conditional_field_update(self):
        class Accumulator(csp.Struct):
            positive_sum: float
            negative_sum: float

        @numba_node
        def accumulate_node(x: ts[float]) -> csp.Outputs(positive=ts[float], negative=ts[float]):
            with csp.state():
                s_acc: Accumulator = Accumulator(positive_sum=0.0, negative_sum=0.0)

            if x > 0:
                s_acc.positive_sum = s_acc.positive_sum + x
                csp.output(positive=s_acc.positive_sum)
            else:
                s_acc.negative_sum = s_acc.negative_sum + x
                csp.output(negative=s_acc.negative_sum)

        @csp.graph
        def g():
            values = csp.curve(
                float,
                [
                    (timedelta(seconds=0), 10.0),
                    (timedelta(seconds=1), -5.0),
                    (timedelta(seconds=2), 20.0),
                    (timedelta(seconds=3), -10.0),
                ],
            )
            result = accumulate_node(values)
            csp.add_graph_output("positive", result.positive)
            csp.add_graph_output("negative", result.negative)

        results = csp.run(g, starttime=datetime(2024, 1, 1), endtime=timedelta(seconds=10))
        self.assertEqual([v for _, v in results["positive"]], [10.0, 30.0])
        self.assertEqual([v for _, v in results["negative"]], [-5.0, -15.0])

    def test_iterate_basket_of_structs(self):
        class Point(csp.Struct):
            x: float
            y: float

        @numba_node
        def sum_x_from_basket(points: list[ts[Point]]) -> ts[float]:
            total = 0.0
            for i in range(len(points)):
                if points[i].ticked():
                    total = total + points[i].x
            return total

        @csp.graph
        def g():
            # Create individual Point signals (basket of 3 struct signals)
            p1 = csp.curve(
                Point,
                [
                    (timedelta(seconds=0), Point(x=1.0, y=10.0)),
                    (timedelta(seconds=2), Point(x=10.0, y=100.0)),
                ],
            )
            p2 = csp.curve(
                Point,
                [
                    (timedelta(seconds=0), Point(x=2.0, y=20.0)),
                    (timedelta(seconds=1), Point(x=5.0, y=50.0)),
                ],
            )
            p3 = csp.curve(
                Point,
                [
                    (timedelta(seconds=0), Point(x=3.0, y=30.0)),
                ],
            )

            # Pass as a basket
            result = sum_x_from_basket([p1, p2, p3])
            csp.add_graph_output("result", result)

        results = csp.run(g, starttime=datetime(2024, 1, 1), endtime=timedelta(seconds=10))
        actual = [v for _, v in results["result"]]
        # t=0: all 3 tick: 1.0 + 2.0 + 3.0 = 6.0
        # t=1: only p2 ticks: 5.0
        # t=2: only p1 ticks: 10.0
        self.assertEqual(actual, [6.0, 5.0, 10.0])


if __name__ == "__main__":
    unittest.main()
