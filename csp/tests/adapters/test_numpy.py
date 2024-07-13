import unittest
from datetime import datetime, timedelta
from typing import List

import numpy as np

import csp


class Foo:
    def __init__(self, x):
        self.x = x


@csp.graph(memoize=False)
def g(typ: object, dts: np.ndarray, values: np.ndarray):
    x = csp.curve(typ=typ, data=(dts, values))
    csp.add_graph_output("out", x)


test_dts = [datetime(2000, 1, 1, 1), datetime(2000, 1, 2, 1), datetime(2000, 1, 2, 2)]
test_dts_ndarray = np.array(test_dts, dtype="datetime64[ns]")
test_starttime = datetime(2000, 1, 1)


class TestNumpyAdapter(unittest.TestCase):
    def test_int(self):
        raw_vals = [7, -13, 21]
        res = csp.run(g, typ=int, values=np.array(raw_vals), dts=test_dts_ndarray, starttime=test_starttime)
        self.assertEqual(res["out"], list(zip(test_dts, raw_vals)))

        for dtype in ("b", "h", "i", "l", "object"):
            res = csp.run(
                g, typ=int, values=np.array(raw_vals, dtype=dtype), dts=test_dts_ndarray, starttime=test_starttime
            )
            self.assertEqual(res["out"], list(zip(test_dts, raw_vals)))

        unsigned_raw_vals = [2, 6, 13]
        for dtype in ("B", "H", "I", "object"):
            res = csp.run(
                g,
                typ=int,
                values=np.array(unsigned_raw_vals, dtype=dtype),
                dts=test_dts_ndarray,
                starttime=test_starttime,
            )
            self.assertEqual(res["out"], list(zip(test_dts, unsigned_raw_vals)))

    def test_float(self):
        raw_vals = [7.7, 13.13, -21.21]
        res = csp.run(g, typ=float, values=np.array(raw_vals), dts=test_dts_ndarray, starttime=test_starttime)
        self.assertEqual(res["out"], list(zip(test_dts, raw_vals)))

        for dtype in ("f", "d", "object"):
            res = csp.run(
                g, typ=float, values=np.array(raw_vals, dtype=dtype), dts=test_dts_ndarray, starttime=test_starttime
            )
            for x in zip(res["out"], raw_vals):
                self.assertAlmostEqual(x[0][1], x[1], places=5)

    def test_bool(self):
        raw_vals = [False, True, True]
        res = csp.run(g, typ=bool, values=np.array(raw_vals), dts=test_dts_ndarray, starttime=test_starttime)
        self.assertEqual(res["out"], list(zip(test_dts, raw_vals)))

        res = csp.run(
            g, typ=bool, values=np.array(raw_vals, dtype="object"), dts=test_dts_ndarray, starttime=test_starttime
        )
        self.assertEqual(res["out"], list(zip(test_dts, raw_vals)))

    def test_obj(self):
        raw_vals = [Foo(7), Foo(11), Foo(-5)]
        res = csp.run(g, typ=Foo, values=np.array(raw_vals), dts=test_dts_ndarray, starttime=test_starttime)
        self.assertEqual(res["out"], list(zip(test_dts, raw_vals)))

    def test_complex(self):
        raw_vals = [complex(0, -1), complex(2, 4), complex(1.5, -7)]
        res = csp.run(
            g, typ=Foo, values=np.array(raw_vals, dtype="object"), dts=test_dts_ndarray, starttime=test_starttime
        )
        self.assertEqual(res["out"], list(zip(test_dts, raw_vals)))

        with self.assertRaisesRegex(ValueError, "numpy complex type only supported with dtype='object'"):
            csp.run(g, typ=complex, values=np.array(raw_vals), dts=test_dts_ndarray, starttime=test_starttime)

    def test_string(self):
        raw_vals = ["spam", "ff", "pancakes"]
        res = csp.run(g, typ=str, values=np.array(raw_vals), dts=test_dts_ndarray, starttime=test_starttime)
        self.assertEqual(res["out"], list(zip(test_dts, raw_vals)))

        res = csp.run(
            g, typ=str, values=np.array(raw_vals, dtype="object"), dts=test_dts_ndarray, starttime=test_starttime
        )
        self.assertEqual(res["out"], list(zip(test_dts, raw_vals)))

    def test_old_string(self):
        raw_vals = ["spam", "woodchuck", "eggs"]
        res = csp.run(g, typ=str, values=np.array(raw_vals, dtype="S"), dts=test_dts_ndarray, starttime=test_starttime)
        self.assertEqual(res["out"], list(zip(test_dts, raw_vals)))

    def test_char(self):
        raw_vals = ["a", "b", "c"]
        res = csp.run(g, typ=str, values=np.array(raw_vals, dtype="c"), dts=test_dts_ndarray, starttime=test_starttime)
        self.assertEqual(res["out"], list(zip(test_dts, raw_vals)))

        res = csp.run(
            g, typ=str, values=np.array(raw_vals, dtype="object"), dts=test_dts_ndarray, starttime=test_starttime
        )
        self.assertEqual(res["out"], list(zip(test_dts, raw_vals)))

    def test_datetime(self):
        raw_vals = [datetime(1980, 1, 1), datetime(1980, 1, 1, 0, 0, 1), datetime(1980, 1, 1, 0, 0, 0, 1)]
        res = csp.run(
            g,
            typ=datetime,
            values=np.array(raw_vals, dtype="datetime64[ns]"),
            dts=test_dts_ndarray,
            starttime=test_starttime,
        )
        self.assertEqual(res["out"], list(zip(test_dts, raw_vals)))

        res = csp.run(
            g,
            typ=datetime,
            values=np.array(raw_vals, dtype="datetime64[us]"),
            dts=test_dts_ndarray,
            starttime=test_starttime,
        )
        self.assertEqual(res["out"], list(zip(test_dts, raw_vals)))

        res = csp.run(
            g, typ=datetime, values=np.array(raw_vals, dtype="object"), dts=test_dts_ndarray, starttime=test_starttime
        )
        self.assertEqual(res["out"], list(zip(test_dts, raw_vals)))

        raw_vals = [datetime(1980, 1, 1), datetime(1980, 1, 2), datetime(1980, 1, 3)]
        res = csp.run(
            g,
            typ=datetime,
            values=np.array(raw_vals, dtype="datetime64[s]"),
            dts=test_dts_ndarray,
            starttime=test_starttime,
        )
        self.assertEqual(res["out"], list(zip(test_dts, raw_vals)))

        res = csp.run(
            g,
            typ=datetime,
            values=np.array(raw_vals, dtype="datetime64[D]"),
            dts=test_dts_ndarray,
            starttime=test_starttime,
        )
        self.assertEqual(res["out"], list(zip(test_dts, raw_vals)))

    def test_timedelta(self):
        raw_vals = [timedelta(seconds=3), timedelta(days=5), timedelta(milliseconds=7)]
        res = csp.run(
            g,
            typ=timedelta,
            values=np.array(raw_vals, dtype="timedelta64[ns]"),
            dts=test_dts_ndarray,
            starttime=test_starttime,
        )
        self.assertEqual(res["out"], list(zip(test_dts, raw_vals)))

        res = csp.run(
            g,
            typ=timedelta,
            values=np.array(raw_vals, dtype="timedelta64[ms]"),
            dts=test_dts_ndarray,
            starttime=test_starttime,
        )
        self.assertEqual(res["out"], list(zip(test_dts, raw_vals)))

        res = csp.run(
            g, typ=timedelta, values=np.array(raw_vals, dtype="object"), dts=test_dts_ndarray, starttime=test_starttime
        )
        self.assertEqual(res["out"], list(zip(test_dts, raw_vals)))

        raw_vals = [timedelta(seconds=3), timedelta(minutes=5), timedelta(hours=7)]
        res = csp.run(
            g,
            typ=timedelta,
            values=np.array(raw_vals, dtype="timedelta64[s]"),
            dts=test_dts_ndarray,
            starttime=test_starttime,
        )
        self.assertEqual(res["out"], list(zip(test_dts, raw_vals)))

        raw_vals = [timedelta(days=3), timedelta(hours=15), timedelta(days=7)]
        res = csp.run(
            g,
            typ=timedelta,
            values=np.array(raw_vals, dtype="timedelta64[h]"),
            dts=test_dts_ndarray,
            starttime=test_starttime,
        )
        self.assertEqual(res["out"], list(zip(test_dts, raw_vals)))

        raw_vals = [timedelta(days=3), timedelta(days=15), timedelta(days=1)]
        res = csp.run(
            g,
            typ=timedelta,
            values=np.array(raw_vals, dtype="timedelta64[D]"),
            dts=test_dts_ndarray,
            starttime=test_starttime,
        )
        self.assertEqual(res["out"], list(zip(test_dts, raw_vals)))

    def test_timestamps(self):
        starttime = datetime(1990, 1, 1)
        raw_vals = [-6, 4, 25]
        dts = [datetime(1990, 6, 1), datetime(1990, 8, 2, 0, 0, 0, 1), datetime(1992, 5, 1, 1, 1, 1, 1)]
        dts_ndarray = np.array(dts, dtype="datetime64[ns]")
        res = csp.run(g, typ=int, values=np.array(raw_vals), dts=dts_ndarray, starttime=starttime)
        self.assertEqual(res["out"], list(zip(dts, raw_vals)))

        dts_ndarray = np.array(dts, dtype="object")
        res = csp.run(g, typ=int, values=np.array(raw_vals), dts=dts_ndarray, starttime=starttime)
        self.assertEqual(res["out"], list(zip(dts, raw_vals)))

        with self.assertRaisesRegex(ValueError, "timestamps ndarray must be dtype of datetime64 or object"):
            res = csp.run(g, typ=int, values=np.array(raw_vals), dts=np.array(raw_vals), starttime=starttime)
            self.assertEqual(res["out"], list(zip(dts, raw_vals)))

        dts = [datetime(1990, 6, 1), datetime(1990, 8, 2, 0, 0, 1), datetime(1992, 5, 1, 0, 1, 5)]
        dts_ndarray = np.array(dts, dtype="datetime64[ms]")
        res = csp.run(g, typ=int, values=np.array(raw_vals), dts=dts_ndarray, starttime=starttime)
        self.assertEqual(res["out"], list(zip(dts, raw_vals)))

        dts = [datetime(1990, 6, 1), datetime(1991, 8, 3, 5, 4, 3), datetime(1995, 9, 2, 0, 0, 12)]
        dts_ndarray = np.array(dts, dtype="datetime64[s]")
        res = csp.run(g, typ=int, values=np.array(raw_vals), dts=dts_ndarray, starttime=starttime)
        self.assertEqual(res["out"], list(zip(dts, raw_vals)))

        dts = [datetime(1990, 6, 1), datetime(1991, 8, 3), datetime(1995, 9, 2)]
        dts_ndarray = np.array(dts, dtype="datetime64[D]")
        res = csp.run(g, typ=int, values=np.array(raw_vals), dts=dts_ndarray, starttime=starttime)
        self.assertEqual(res["out"], list(zip(dts, raw_vals)))

    def test_array(self):
        raw_vals = [[1, 2], [3], [4, 5, 6]]
        res = csp.run(
            g, typ=List[int], values=np.array(raw_vals, dtype="object"), dts=test_dts_ndarray, starttime=test_starttime
        )
        self.assertEqual(res["out"], list(zip(test_dts, raw_vals)))

        raw_vals = [["hello", "world"], ["hows"], ["it", "going"]]
        res = csp.run(
            g, typ=List[str], values=np.array(raw_vals, dtype="object"), dts=test_dts_ndarray, starttime=test_starttime
        )
        self.assertEqual(res["out"], list(zip(test_dts, raw_vals)))

        # res = csp.run(g, typ=str, values=np.array(raw_vals, dtype='object'), dts=test_dts_ndarray, starttime=test_starttime)
        # self.assertEqual(res['out'], list(zip(test_dts, raw_vals)))

    def test_numpy_arrays(self):
        # 1D array
        # float
        raw_vals = [np.array([1.0, 2.0]), np.array([3.0, 4.0]), np.array([5.0, 6.0])]
        for dtype in ("f", "d", "object"):
            res = csp.run(
                g,
                typ=csp.typing.Numpy1DArray[float],
                values=np.array(raw_vals, dtype=dtype),
                dts=test_dts_ndarray,
                starttime=test_starttime,
            )
            np.testing.assert_equal(res["out"], list(zip(test_dts, raw_vals)))

        # int
        raw_vals = [np.array([1]), np.array([2]), np.array([3])]
        for dtype in ("b", "h", "i", "l", "object"):
            res = csp.run(
                g,
                typ=csp.typing.Numpy1DArray[int],
                values=np.array(raw_vals, dtype=dtype),
                dts=test_dts_ndarray,
                starttime=test_starttime,
            )
            np.testing.assert_equal(res["out"], list(zip(test_dts, raw_vals)))

        # str
        raw_vals = [np.array(["spam", "ff"]), np.array(["eggs", "bacon"]), np.array(["pancakes", "toast"])]
        for dtype in ("U", "U10", "object"):
            res = csp.run(
                g,
                typ=csp.typing.Numpy1DArray[str],
                values=np.array(raw_vals, dtype=dtype),
                dts=test_dts_ndarray,
                starttime=test_starttime,
            )
            np.testing.assert_equal(res["out"], list(zip(test_dts, raw_vals)))

        # object
        raw_vals = [np.array(Foo(7)), np.array(Foo(11)), np.array(Foo(-5))]
        res = csp.run(
            g,
            typ=csp.typing.Numpy1DArray[Foo],
            values=np.array(raw_vals, dtype="object"),
            dts=test_dts_ndarray,
            starttime=test_starttime,
        )
        self.assertEqual(res["out"], list(zip(test_dts, raw_vals)))

        # datetimes
        raw_vals = [np.array(datetime(1980, 1, 1)), np.array(datetime(1980, 1, 2)), np.array(datetime(1980, 1, 3))]
        for dtype in ("datetime64[ns]", "datetime64[s]", "datetime64[D]"):
            res = csp.run(
                g, typ=datetime, values=np.array(raw_vals, dtype=dtype), dts=test_dts_ndarray, starttime=test_starttime
            )
            self.assertEqual(res["out"], list(zip(test_dts, raw_vals)))

        # ND array
        raw_vals = [
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            np.array([[5.0, 6.0], [7.0, 8.0]]),
            np.array([[9.0, 10.0], [11.0, 12.0]]),
        ]
        for dtype in ("f", "d", "object"):
            res = csp.run(
                g,
                typ=csp.typing.NumpyNDArray[float],
                values=np.array(raw_vals, dtype=dtype),
                dts=test_dts_ndarray,
                starttime=test_starttime,
            )
            np.testing.assert_equal(res["out"], list(zip(test_dts, raw_vals)))

        raw_vals = [np.array([[1], [2], [3]]), np.array([[4], [5], [6]]), np.array([[7], [8], [9]])]
        for dtype in ("b", "h", "i", "l", "object"):
            res = csp.run(
                g,
                typ=csp.typing.NumpyNDArray[int],
                values=np.array(raw_vals, dtype=dtype),
                dts=test_dts_ndarray,
                starttime=test_starttime,
            )
            np.testing.assert_equal(res["out"], list(zip(test_dts, raw_vals)))

        # views and striding: transpositions, reshape, flips, broadcast to zero-stride dimensions
        raw_vals = np.array([np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0]), np.array([7.0, 8.0, 9.0])])
        transposed = raw_vals.T  # change from C-contig to F-contig
        res = csp.run(
            g, typ=csp.typing.NumpyNDArray[float], values=transposed, dts=test_dts_ndarray, starttime=test_starttime
        )
        np.testing.assert_equal(res["out"], list(zip(test_dts, transposed)))

        raw_vals = np.array(["a", "b", "c", "d", "e", "f"])
        reshaped = np.reshape(raw_vals, (3, 2))
        res = csp.run(
            g, typ=csp.typing.NumpyNDArray[str], values=reshaped, dts=test_dts_ndarray, starttime=test_starttime
        )
        np.testing.assert_equal(res["out"], list(zip(test_dts, reshaped)))

        raw_vals = np.array([np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0]), np.array([7.0, 8.0, 9.0])])
        flipped = np.flip(np.flip(raw_vals, axis=0), axis=1)  # create negative strides
        res = csp.run(
            g, typ=csp.typing.NumpyNDArray[float], values=flipped, dts=test_dts_ndarray, starttime=test_starttime
        )
        np.testing.assert_equal(res["out"], list(zip(test_dts, flipped)))

        raw_vals = np.array([np.array([1.0, 2.0, 3.0])])
        broadcast = np.broadcast_to(raw_vals, (3, 3))  # create 0-stride in first dimension
        res = csp.run(
            g, typ=csp.typing.NumpyNDArray[float], values=broadcast, dts=test_dts_ndarray, starttime=test_starttime
        )
        np.testing.assert_equal(res["out"], list(zip(test_dts, broadcast)))

        # ensure 1-D irregular stride arrays also work (they don't use the NumpyCurveAccessor)
        raw_vals = np.array([1.0, 2.0, 3.0])
        flipped_1D = np.flip(raw_vals, axis=0)
        res = csp.run(g, typ=float, values=flipped_1D, dts=test_dts_ndarray, starttime=test_starttime)
        np.testing.assert_equal(res["out"], list(zip(test_dts, flipped_1D)))

    def test_forward_past_start(self):
        raw_vals = [-6, 29, 1121]

        # 2nd tick after starttime
        starttime = datetime(2000, 1, 2)
        res = csp.run(g, typ=int, values=np.array(raw_vals), dts=test_dts_ndarray, starttime=starttime)
        self.assertEqual(res["out"], list(zip(test_dts[1:], raw_vals[1:])))

        # 2nd tick at starttime
        starttime = datetime(2000, 1, 2, 1)
        res = csp.run(g, typ=int, values=np.array(raw_vals), dts=test_dts_ndarray, starttime=starttime)
        self.assertEqual(res["out"], list(zip(test_dts[1:], raw_vals[1:])))

        # 3rd tick after starttime
        starttime = datetime(2000, 1, 2, 1, 10)
        res = csp.run(g, typ=int, values=np.array(raw_vals), dts=test_dts_ndarray, starttime=starttime)
        self.assertEqual(res["out"], list(zip(test_dts[2:], raw_vals[2:])))

        # 3rd tick at starttime
        starttime = datetime(2000, 1, 2, 2)
        res = csp.run(g, typ=int, values=np.array(raw_vals), dts=test_dts_ndarray, starttime=starttime)
        self.assertEqual(res["out"], list(zip(test_dts[2:], raw_vals[2:])))

        # all ticks before starttime
        starttime = datetime(2000, 1, 3)
        res = csp.run(g, typ=int, values=np.array(raw_vals), dts=test_dts_ndarray, starttime=starttime)
        self.assertEqual(res["out"], list())

    def test_ndarray_len_match(self):
        with self.assertRaisesRegex(ValueError, "ndarrays passed to csp.curve must be of equal length"):
            csp.run(g, typ=int, values=np.array([1, 2]), dts=test_dts_ndarray, starttime=test_starttime)

    def test_type_match(self):
        with self.assertRaisesRegex(ValueError, "numpy type .* requires float output type"):
            csp.run(g, typ=int, values=np.array([1.1, 2.2, 3.3]), dts=test_dts_ndarray, starttime=test_starttime)
        with self.assertRaisesRegex(ValueError, "numpy type .* requires int output type"):
            csp.run(g, typ=float, values=np.array([1, 2, 3]), dts=test_dts_ndarray, starttime=test_starttime)
        with self.assertRaisesRegex(ValueError, "numpy type .* requires int output type"):
            csp.run(g, typ=str, values=np.array([1, 2, 3]), dts=test_dts_ndarray, starttime=test_starttime)
        with self.assertRaisesRegex(ValueError, "numpy type .* requires bool output type"):
            csp.run(g, typ=int, values=np.array([True, False, True]), dts=test_dts_ndarray, starttime=test_starttime)
        with self.assertRaisesRegex(ValueError, "numpy type .* requires int output type"):
            csp.run(g, typ=bool, values=np.array([1, 2, 3]), dts=test_dts_ndarray, starttime=test_starttime)
        with self.assertRaisesRegex(ValueError, "numpy type .* requires string output type"):
            csp.run(g, typ=float, values=np.array(["aa", "bb", "cc"]), dts=test_dts_ndarray, starttime=test_starttime)
        with self.assertRaisesRegex(ValueError, "numpy type .* requires datetime output type"):
            csp.run(g, typ=int, values=test_dts_ndarray, dts=test_dts_ndarray, starttime=test_starttime)
        with self.assertRaisesRegex(ValueError, "numpy type .* requires timedelta output type"):
            raw_vals = [timedelta(seconds=3), timedelta(days=5), timedelta(milliseconds=7)]
            csp.run(
                g,
                typ=int,
                values=np.array(raw_vals, dtype="timedelta64[ns]"),
                dts=test_dts_ndarray,
                starttime=test_starttime,
            )


if __name__ == "__main__":
    unittest.main()
