import os
import unittest
from datetime import date, datetime, time

import pytz
import sqlalchemy as db

import csp
from csp.adapters.db import _SQLALCHEMY_2, DateTimeAccessor, DBReader, EngineStartTimeAccessor, TimestampAccessor


class PriceQuantity(csp.Struct):
    PRICE: float
    SIZE: int
    SIDE: str
    SYMBOL: str


class PriceQuantity2(csp.Struct):
    price: float
    quantity: int
    side: str


def execute_with_commit(engine, query, values):
    if _SQLALCHEMY_2:
        with engine.connect() as conn:
            conn.execute(query, values)
            conn.commit()
    else:
        engine.execute(query, values)


class TestDBReader(unittest.TestCase):
    def _prepopulate_in_mem_engine(self):
        engine = db.create_engine("sqlite:///:memory:")  # in-memory sqlite db
        metadata = db.MetaData()
        emp = db.Table(
            "test",
            metadata,
            db.Column("TIME", db.DateTime(timezone=True)),
            db.Column("SYMBOL", db.String(255)),
            db.Column("PRICE", db.Float()),
            db.Column("SIZE", db.Integer()),
            db.Column("SIDE", db.String(255)),
        )
        metadata.create_all(engine)
        query = db.insert(emp)
        starttime = datetime(year=2020, month=3, day=3, hour=9, minute=30, second=0)
        values_list = [
            {"TIME": starttime, "SYMBOL": "AAPL", "PRICE": 500.0, "SIZE": 100, "SIDE": "BUY"},
            {"TIME": starttime.replace(second=1), "SYMBOL": "IBM", "PRICE": 100.0, "SIZE": 200, "SIDE": "BUY"},
            {"TIME": starttime.replace(second=2), "SYMBOL": "AAPL", "PRICE": 400.0, "SIZE": 100, "SIDE": "BUY"},
            {"TIME": starttime.replace(second=3), "SYMBOL": "IBM", "PRICE": 200.0, "SIZE": 300, "SIDE": "SELL"},
            {"TIME": starttime.replace(second=4), "SYMBOL": "AAPL", "PRICE": 300.0, "SIZE": 200, "SIDE": "SELL"},
            {"TIME": starttime.replace(second=5), "SYMBOL": "AAPL", "PRICE": 200.0, "SIZE": 400, "SIDE": "BUY"},
            {"TIME": starttime.replace(second=6), "SYMBOL": "GM", "PRICE": 2.0, "SIZE": 1, "SIDE": "BUY"},
        ]
        execute_with_commit(engine, query, values_list)
        return engine

    def test_sqlite_basic(self):
        engine = self._prepopulate_in_mem_engine()
        for time_accessor in (
            EngineStartTimeAccessor(),
            TimestampAccessor(time_column="TIME", tz=pytz.timezone("US/Eastern")),
        ):

            def graph():
                reader = DBReader.create_from_connection(
                    connection=engine, time_accessor=time_accessor, table_name="test", symbol_column="SYMBOL"
                )

                # Struct
                aapl = reader.subscribe("AAPL", PriceQuantity)
                ibm = reader.subscribe("IBM", PriceQuantity)

                # Struct with fieldMapping
                aapl2 = reader.subscribe(
                    "AAPL", PriceQuantity2, field_map={"PRICE": "price", "SIZE": "quantity", "SIDE": "side"}
                )

                # specific field
                aapl_price = reader.subscribe("AAPL", float, field_map="PRICE")

                # Dynamic struct (and field_map to limit fields)
                aapl_dyn = reader.subscribe("AAPL", None, field_map={"PRICE": "PRICE", "SIZE": "SIZE", "SIDE": "SIDE"})
                aapl_dyn2 = reader.subscribe(
                    "AAPL", reader.schema_struct(), field_map={"PRICE": "PRICE", "SIZE": "SIZE", "SIDE": "SIDE"}
                )

                # all data
                all = reader.subscribe_all(PriceQuantity)

                csp.add_graph_output("aapl", aapl)
                csp.add_graph_output("ibm", ibm)
                csp.add_graph_output("aapl2", aapl2)
                csp.add_graph_output("aapl_price", aapl_price)
                csp.add_graph_output("aapl_dyn", aapl_dyn)
                csp.add_graph_output("aapl_dyn2", aapl_dyn2)
                csp.add_graph_output("all", all)

            # UTC
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

            self.assertEqual(result["aapl_dyn"], result["aapl_dyn2"])

            # Retrieve dynamic struct based on name of table and name of schema
            dyn_struct = csp.adapters.db.DBDynStruct_test_
            self.assertEqual(
                [v[1] for v in result["aapl_dyn"]],
                [
                    dyn_struct(PRICE=500.0, SIZE=100, SIDE="BUY"),
                    dyn_struct(PRICE=400.0, SIZE=100, SIDE="BUY"),
                    dyn_struct(PRICE=300.0, SIZE=200, SIDE="SELL"),
                    dyn_struct(PRICE=200.0, SIZE=400, SIDE="BUY"),
                ],
            )

    def test_sqlite_constraints(self):
        engine = db.create_engine("sqlite:///:memory:")  # in-memory sqlite db
        metadata = db.MetaData()

        emp = db.Table(
            "test",
            metadata,
            db.Column("DATE", db.Date()),
            db.Column("TIME", db.Time()),
            db.Column("SYMBOL", db.String(255)),
            db.Column("PRICE", db.Float()),
            db.Column("SIZE", db.Integer()),
            db.Column("SIDE", db.String(255)),
        )

        metadata.create_all(engine)

        query = db.insert(emp)
        startdate = date(year=2020, month=3, day=3)
        starttime = time(hour=9, minute=30, second=0)
        values_list = [
            {"DATE": startdate, "TIME": starttime, "SYMBOL": "AAPL", "PRICE": 500.0, "SIZE": 100, "SIDE": "BUY"},
            {
                "DATE": startdate,
                "TIME": starttime.replace(second=1),
                "SYMBOL": "IBM",
                "PRICE": 100.0,
                "SIZE": 200,
                "SIDE": "BUY",
            },
            {
                "DATE": startdate,
                "TIME": starttime.replace(second=2),
                "SYMBOL": "AAPL",
                "PRICE": 400.0,
                "SIZE": 100,
                "SIDE": "BUY",
            },
            {
                "DATE": startdate,
                "TIME": starttime.replace(second=3),
                "SYMBOL": "IBM",
                "PRICE": 200.0,
                "SIZE": 300,
                "SIDE": "SELL",
            },
            {
                "DATE": startdate,
                "TIME": starttime.replace(second=4),
                "SYMBOL": "AAPL",
                "PRICE": 300.0,
                "SIZE": 200,
                "SIDE": "SELL",
            },
            {
                "DATE": startdate,
                "TIME": starttime.replace(second=5),
                "SYMBOL": "AAPL",
                "PRICE": 200.0,
                "SIZE": 400,
                "SIDE": "BUY",
            },
            {
                "DATE": startdate,
                "TIME": starttime.replace(second=6),
                "SYMBOL": "GM",
                "PRICE": 2.0,
                "SIZE": 1,
                "SIDE": "BUY",
            },
        ]

        execute_with_commit(engine, query, values_list)

        def graph():
            time_accessor = DateTimeAccessor(date_column="DATE", time_column="TIME", tz=pytz.timezone("US/Eastern"))
            reader = DBReader.create_from_connection(
                connection=engine,
                table_name="test",
                time_accessor=time_accessor,
                symbol_column="SYMBOL",
                constraint=db.text("PRICE>:price").bindparams(price=100.0),
            )

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

        # NYC
        result = csp.run(graph, starttime=pytz.timezone("US/Eastern").localize(datetime(2020, 3, 3, 9, 30)))

        self.assertEqual(len(result["aapl"]), 4)
        self.assertTrue(all(v[1].SYMBOL == "AAPL" for v in result["aapl"]))

        self.assertEqual(len(result["ibm"]), 1)
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
        self.assertEqual(len(result["all"]), 5)

    def test_join_query(self):
        engine = db.create_engine("sqlite:///:memory:")  # in-memory sqlite db
        metadata = db.MetaData()
        test1 = db.Table(
            "test1",
            metadata,
            db.Column("TIME", db.DateTime(timezone=True)),
            db.Column("SYMBOL", db.String(255)),
            db.Column("PRICE", db.Float()),
        )
        test2 = db.Table(
            "test2",
            metadata,
            db.Column("TIME", db.DateTime(timezone=True)),
            db.Column("SIZE", db.Integer()),
            db.Column("SIDE", db.String(255)),
        )
        metadata.create_all(engine)

        query = db.insert(test1)
        starttime = datetime(year=2020, month=3, day=3, hour=9, minute=30, second=0)
        values_list1 = [
            {"TIME": starttime, "SYMBOL": "AAPL", "PRICE": 500.0},
            {"TIME": starttime.replace(second=1), "SYMBOL": "IBM", "PRICE": 100.0},
            {"TIME": starttime.replace(second=2), "SYMBOL": "AAPL", "PRICE": 400.0},
            {"TIME": starttime.replace(second=3), "SYMBOL": "IBM", "PRICE": 200.0},
            {"TIME": starttime.replace(second=4), "SYMBOL": "AAPL", "PRICE": 300.0},
            {"TIME": starttime.replace(second=5), "SYMBOL": "AAPL", "PRICE": 200.0},
            {"TIME": starttime.replace(second=6), "SYMBOL": "GM", "PRICE": 2.0},
        ]
        execute_with_commit(engine, query, values_list1)

        query = db.insert(test2)
        values_list2 = [
            {"TIME": starttime, "SIZE": 100, "SIDE": "BUY"},
            {"TIME": starttime.replace(second=1), "SIZE": 200, "SIDE": "BUY"},
            {"TIME": starttime.replace(second=2), "SIZE": 100, "SIDE": "BUY"},
            # { 'TIME': starttime.replace( second = 3 ), 'SIZE': 300, 'SIDE': 'SELL' },
            # { 'TIME': starttime.replace( second = 4 ), 'SIZE': 200, 'SIDE': 'SELL' },
            # { 'TIME': starttime.replace( second = 5 ), 'SIZE': 400, 'SIDE': 'BUY' },
            {"TIME": starttime.replace(second=6), "SIZE": 1, "SIDE": "BUY"},
        ]
        execute_with_commit(engine, query, values_list2)

        metadata.create_all(engine)

        def graph():
            time_accessor = TimestampAccessor(time_column="TIME", tz=pytz.timezone("US/Eastern"))
            query = "select * from test1 inner join test2 on test2.TIME=test1.TIME"
            reader = DBReader.create_from_connection(
                connection=engine, query=query, time_accessor=time_accessor, symbol_column="SYMBOL"
            )

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

        # NYC
        starttime = pytz.timezone("US/Eastern").localize(datetime(2020, 3, 3, 9, 30))
        result = csp.run(graph, starttime=starttime)

        # result = csp.run(graph, starttime=datetime(2020, 3, 3, 9, 30))
        self.assertEqual(len(result["aapl"]), 2)
        self.assertTrue(all(v[1].SYMBOL == "AAPL" for v in result["aapl"]))

        self.assertEqual(len(result["ibm"]), 1)
        self.assertTrue(all(v[1].SYMBOL == "IBM" for v in result["ibm"]))

        self.assertEqual(
            [v[1] for v in result["aapl"]],
            [
                PriceQuantity(PRICE=500.0, SIZE=100, SIDE="BUY", SYMBOL="AAPL"),
                PriceQuantity(PRICE=400.0, SIZE=100, SIDE="BUY", SYMBOL="AAPL"),
                # PriceQuantity(PRICE=300.0, SIZE=200, SIDE='SELL', SYMBOL='AAPL'),
                # PriceQuantity(PRICE=200.0, SIZE=400, SIDE='BUY', SYMBOL='AAPL'),
            ],
        )

        self.assertEqual(
            [v[1] for v in result["aapl2"]],
            [
                PriceQuantity2(price=500.0, quantity=100, side="BUY"),
                PriceQuantity2(price=400.0, quantity=100, side="BUY"),
                # PriceQuantity2(price=300.0, quantity=200, side='SELL', ),
                # PriceQuantity2(price=200.0, quantity=400, side='BUY'),
            ],
        )

        self.assertEqual([v[1] for v in result["aapl_price"]], [500.0, 400.0])
        self.assertEqual(len(result["all"]), 4)

    def test_DateTimeAccessor(self):
        engine = db.create_engine("sqlite:///:memory:")  # in-memory sqlite db
        metadata = db.MetaData()

        emp = db.Table(
            "test",
            metadata,
            db.Column("DATE", db.Date()),
            db.Column("TIME", db.Time()),
            db.Column("SYMBOL", db.String(255)),
            db.Column("PRICE", db.Float()),
        )

        metadata.create_all(engine)

        query = db.insert(emp)
        values = [
            (datetime(2020, 3, 3, 0), 100.0),
            (datetime(2020, 3, 3, 12), 200.0),
            (datetime(2020, 3, 4, 0), 300.0),
            (datetime(2020, 3, 4, 8), 400.0),
            (datetime(2020, 3, 4, 16), 500.0),
            (datetime(2020, 3, 5, 0), 600.0),
            (datetime(2020, 3, 5, 12), 700.0),
        ]
        values_list = [{"DATE": v[0].date(), "TIME": v[0].time(), "SYMBOL": "AAPL", "PRICE": v[1]} for v in values]

        execute_with_commit(engine, query, values_list)

        def graph():
            time_accessor = DateTimeAccessor(date_column="DATE", time_column="TIME", tz=pytz.timezone("US/Eastern"))
            reader = DBReader.create_from_connection(
                connection=engine, table_name="test", time_accessor=time_accessor, symbol_column="SYMBOL"
            )

            # specific field
            aapl_price = reader.subscribe("AAPL", float, field_map="PRICE")
            csp.add_graph_output("aapl_price", aapl_price)

        # NYC
        tz = pytz.timezone("US/Eastern")
        periods = [
            (datetime(2020, 3, 3), datetime(2020, 3, 4)),
            (datetime(2020, 3, 3), datetime(2020, 3, 5)),
            (datetime(2020, 3, 4), datetime(2020, 3, 4)),
            (datetime(2020, 3, 4), datetime(2020, 3, 4, 16)),
            (datetime(2020, 3, 4, 13), datetime(2020, 3, 4, 14)),
        ]
        for starttime, endtime in periods:
            result = csp.run(graph, starttime=tz.localize(starttime), endtime=tz.localize(endtime))
            target = [
                (tz.localize(v[0]).astimezone(pytz.UTC).replace(tzinfo=None), v[1])
                for v in values
                if v[0] >= starttime and v[0] <= endtime
            ]
            self.assertListEqual(result["aapl_price"], target)


if __name__ == "__main__":
    unittest.main()
