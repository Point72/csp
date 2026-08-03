import os
import unittest

from datetime import datetime

import csp

from csp.adapters.csv import CsvAdapterManager


# Current adapter only supports string fields
class PriceQuantity(csp.Struct):
    PRICE: str
    SIZE: str
    SIDE: str
    SYMBOL: str


class TestCSVReader(unittest.TestCase):
    def setUp(self):
<<<<<<< HEAD
<<<<<<< HEAD
        self._filename = os.path.join(os.path.dirname(__file__), "csv_test_data.csv")
=======
        self._filename = os.path.join(
            os.path.dirname(__file__),
            "csv_test_data.csv"
        )
>>>>>>> e329ef2 (Update python side and add tests)
=======
        self._filename = os.path.join(os.path.dirname(__file__), "csv_test_data.csv")
>>>>>>> eff75eb (Fix linter issue)

        self.reader = CsvAdapterManager(
            self._filename,
            time_column="TIME",
            symbol_column="SYMBOL",
            delimiter="|",
        )
<<<<<<< HEAD
=======

<<<<<<< HEAD
>>>>>>> e329ef2 (Update python side and add tests)

=======
>>>>>>> eff75eb (Fix linter issue)
    def test_basic(self):
        def graph():
<<<<<<< HEAD
<<<<<<< HEAD
            # Subscribe AAPL
            aapl = self.reader.subscribe(PriceQuantity, symbol="AAPL")

            # Subscribe IBM
            ibm = self.reader.subscribe(PriceQuantity, symbol="IBM")

            # Specific field (string only)
            aapl_price = self.reader.subscribe(str, symbol="AAPL", field_map="PRICE")

            # Subscribe all symbols
            all_data = self.reader.subscribe(PriceQuantity)
=======

=======
>>>>>>> eff75eb (Fix linter issue)
            # Subscribe AAPL
            aapl = self.reader.subscribe(PriceQuantity, symbol="AAPL")

            # Subscribe IBM
            ibm = self.reader.subscribe(PriceQuantity, symbol="IBM")

            # Specific field (string only)
            aapl_price = self.reader.subscribe(str, symbol="AAPL", field_map="PRICE")

            # Subscribe all symbols
<<<<<<< HEAD
            all_data = self.reader.subscribe(
                PriceQuantity
            )

>>>>>>> e329ef2 (Update python side and add tests)
=======
            all_data = self.reader.subscribe(PriceQuantity)
>>>>>>> eff75eb (Fix linter issue)

            csp.add_graph_output("aapl", aapl)
            csp.add_graph_output("ibm", ibm)
            csp.add_graph_output("aapl_price", aapl_price)
            csp.add_graph_output("all", all_data)

<<<<<<< HEAD
<<<<<<< HEAD
        result = csp.run(graph, starttime=datetime(2020, 3, 3, 9, 30))

        # AAPL
        self.assertEqual(len(result["aapl"]), 4)

        self.assertTrue(all(v[1].SYMBOL == "AAPL" for v in result["aapl"]))
=======

        result = csp.run(
            graph,
            starttime=datetime(2020, 3, 3, 9, 30)
        )

=======
        result = csp.run(graph, starttime=datetime(2020, 3, 3, 9, 30))
>>>>>>> eff75eb (Fix linter issue)

        # AAPL
        self.assertEqual(len(result["aapl"]), 4)

<<<<<<< HEAD
>>>>>>> e329ef2 (Update python side and add tests)
=======
        self.assertTrue(all(v[1].SYMBOL == "AAPL" for v in result["aapl"]))
>>>>>>> eff75eb (Fix linter issue)

        self.assertEqual(
            [v[1] for v in result["aapl"]],
            [
                PriceQuantity(
                    PRICE="500.00",
                    SIZE="100",
                    SIDE="BUY",
                    SYMBOL="AAPL",
                ),
                PriceQuantity(
                    PRICE="400.00",
                    SIZE="100",
                    SIDE="BUY",
                    SYMBOL="AAPL",
                ),
                PriceQuantity(
                    PRICE="300.00",
                    SIZE="200",
                    SIDE="SELL",
                    SYMBOL="AAPL",
                ),
                PriceQuantity(
                    PRICE="200.00",
                    SIZE="400",
                    SIDE="BUY",
                    SYMBOL="AAPL",
                ),
            ],
<<<<<<< HEAD
        )

        # IBM
        self.assertEqual(len(result["ibm"]), 2)

        self.assertTrue(all(v[1].SYMBOL == "IBM" for v in result["ibm"]))

        # Single field
        self.assertEqual(
            [v[1] for v in result["aapl_price"]],
            [
                "500.00",
                "400.00",
                "300.00",
                "200.00",
            ],
        )

        # Subscribe all
        self.assertEqual(len(result["all"]), 7)

    def test_starttime(self):
        aapl = self.reader.subscribe(str, symbol="AAPL", field_map="PRICE")

        # Exact hit
        res = csp.run(aapl, starttime=datetime(2020, 3, 3, 9, 30, 4))[0]

        self.assertEqual(len(res), 2)

        self.assertEqual(res[0][0], datetime(2020, 3, 3, 9, 30, 4))

        # Missed timestamp:
        # should start from first available tick
        res = csp.run(aapl, starttime=datetime(2020, 3, 3, 9, 30, 3, 2))[0]

        self.assertEqual(len(res), 2)

        self.assertEqual(res[0][0], datetime(2020, 3, 3, 9, 30, 4))
=======
        )

        # IBM
        self.assertEqual(len(result["ibm"]), 2)

        self.assertTrue(all(v[1].SYMBOL == "IBM" for v in result["ibm"]))

        # Single field
        self.assertEqual(
            [v[1] for v in result["aapl_price"]],
            [
                "500.00",
                "400.00",
                "300.00",
                "200.00",
            ],
        )

        # Subscribe all
        self.assertEqual(len(result["all"]), 7)

    def test_starttime(self):
        aapl = self.reader.subscribe(str, symbol="AAPL", field_map="PRICE")

        # Exact hit
        res = csp.run(aapl, starttime=datetime(2020, 3, 3, 9, 30, 4))[0]

        self.assertEqual(len(res), 2)

        self.assertEqual(res[0][0], datetime(2020, 3, 3, 9, 30, 4))

        # Missed timestamp:
        # should start from first available tick
        res = csp.run(aapl, starttime=datetime(2020, 3, 3, 9, 30, 3, 2))[0]

        self.assertEqual(len(res), 2)

<<<<<<< HEAD
        self.assertEqual(
            len(res),
            2
        )

        self.assertEqual(
            res[0][0],
            datetime(2020, 3, 3, 9, 30, 4)
        )
>>>>>>> e329ef2 (Update python side and add tests)
=======
        self.assertEqual(res[0][0], datetime(2020, 3, 3, 9, 30, 4))
>>>>>>> eff75eb (Fix linter issue)


if __name__ == "__main__":
    unittest.main()
