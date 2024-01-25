import os
import unittest
from datetime import datetime, timedelta

import csp
from csp import ts
from csp.adapters.csv import CSVReader, YYYYMMDD_TIME_formatter


class PriceQuantity(csp.Struct):
    PRICE: float
    SIZE: int
    SIDE: str
    SYMBOL: str


class PriceQuantity2(csp.Struct):
    price: float
    quantity: int
    side: str


class TestCSVReader(unittest.TestCase):
    def setUp(self):
        self._filename = os.path.join(os.path.dirname(__file__), "csv_test_data.csv")
        self._time_formatter = YYYYMMDD_TIME_formatter("TIME")

    def test_basic(self):
        def graph():
            reader = CSVReader(self._filename, self._time_formatter, symbol_column="SYMBOL", delimiter="|")

            # Struct
            aapl = reader.subscribe("AAPL", PriceQuantity)
            ibm = reader.subscribe("IBM", PriceQuantity)

            # Struct with fieldMapping
            aapl2 = reader.subscribe(
                "AAPL", PriceQuantity2, field_map={"PRICE": "price", "SIZE": "quantity", "SIDE": "side"}
            )

            # specific field
            aapl_price = reader.subscribe("AAPL", float, field_map="PRICE")

            # all data
            all = reader.subscribe_all(PriceQuantity)

            csp.add_graph_output("aapl", aapl)
            csp.add_graph_output("ibm", ibm)
            csp.add_graph_output("aapl2", aapl2)
            csp.add_graph_output("aapl_price", aapl_price)
            csp.add_graph_output("all", all)

        result = csp.run(graph, starttime=datetime(2020, 3, 3, 9, 30))
        self.assertEqual(len(result["aapl"]), 4)
        self.assertTrue(all(v[1].SYMBOL == "AAPL" for v in result["aapl"]))

        self.assertEqual(len(result["ibm"]), 2)
        self.assertTrue(all(v[1].SYMBOL == "IBM" for v in result["ibm"]))

        self.assertEqual(
            [v[1] for v in result["aapl"]],
            [
                PriceQuantity(PRICE=500.0, SIZE=100, SIDE="BUY", SYMBOL="AAPL"),
                PriceQuantity(PRICE=400.0, SIZE=100, SIDE="BUY", SYMBOL="AAPL"),
                PriceQuantity(PRICE=300.0, SIZE=200, SIDE="SELL", SYMBOL="AAPL"),
                PriceQuantity(PRICE=200.0, SIZE=400, SIDE="BUY", SYMBOL="AAPL"),
            ],
        )

        self.assertEqual(
            [v[1] for v in result["aapl2"]],
            [
                PriceQuantity2(price=500.0, quantity=100, side="BUY"),
                PriceQuantity2(price=400.0, quantity=100, side="BUY"),
                PriceQuantity2(
                    price=300.0,
                    quantity=200,
                    side="SELL",
                ),
                PriceQuantity2(price=200.0, quantity=400, side="BUY"),
            ],
        )

        self.assertEqual([v[1] for v in result["aapl_price"]], [500.0, 400.0, 300.0, 200.0])
        self.assertEqual(len(result["all"]), 7)

    def test_starttime(self):
        reader = CSVReader(self._filename, self._time_formatter, symbol_column="SYMBOL", delimiter="|")
        aapl = reader.subscribe("AAPL", float, "PRICE")

        # Exact hit
        res = csp.run(aapl, starttime=datetime(2020, 3, 3, 9, 30, 4))[0]
        self.assertEqual(len(res), 2)
        self.assertEqual(res[0][0], datetime(2020, 3, 3, 9, 30, 4))

        # Missed, should start with first found tick
        res = csp.run(aapl, starttime=datetime(2020, 3, 3, 9, 30, 3, 2))[0]
        self.assertEqual(len(res), 2)
        self.assertEqual(res[0][0], datetime(2020, 3, 3, 9, 30, 4))

        # TBD snapshoting


if __name__ == "__main__":
    unittest.main()
