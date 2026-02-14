from abc import ABC, abstractmethod
from datetime import datetime

import csp

try:
    import zoneinfo
except ImportError:
    from backports import zoneinfo

from importlib.metadata import PackageNotFoundError, version as get_package_version

import pytz
from packaging import version

from csp import PushMode, ts
from csp.impl.adaptermanager import AdapterManagerImpl, ManagedSimInputAdapter
from csp.impl.wiring import py_managed_adapter_def

UTC = zoneinfo.ZoneInfo("UTC")

try:
    if version.parse(get_package_version("sqlalchemy")) >= version.parse("2"):
        _SQLALCHEMY_2 = True
    else:
        _SQLALCHEMY_2 = False

    import sqlalchemy as db

    _HAS_SQLALCHEMY = True
except (PackageNotFoundError, ValueError, TypeError, ImportError):
    _HAS_SQLALCHEMY = False
    db = None


class TimeAccessor(ABC):
    @abstractmethod
    def get_time_columns(self, engine):
        raise NotImplementedError

    @abstractmethod
    def get_time_constraint(self, starttime, endtime):
        raise NotImplementedError

    @abstractmethod
    def get_order_by_columns(self):
        raise NotImplementedError

    @abstractmethod
    def get_time(self, row):
        raise NotImplementedError


class EngineStartTimeAccessor(TimeAccessor):
    """A constant time accessor for data that needs to be ingested once at engine start time"""

    def get_time_columns(self, engine):
        return None

    def get_time_constraint(self, starttime, endtime):
        return None

    def get_order_by_columns(self):
        return None

    def get_time(self, row):
        return csp.engine_start_time().replace(tzinfo=UTC)


class TimestampAccessor(TimeAccessor):
    def __init__(self, time_column, tz=None):
        """
        :param time_column: (string) name of the db column containing the timestamp
        :param tz: timezone if the db timestamp doesn't have timezone
        """
        self._time_column = time_column
        if isinstance(tz, pytz.BaseTzInfo):
            tz = zoneinfo.ZoneInfo(tz.zone)

        self._tz = tz
        self._db_has_tz = None

    def get_time_columns(self, engine):
        # a workaround for the fact that sqlite does not store timezone info.
        if engine.dialect.name == "sqlite":
            return [(self._time_column, db.DateTime)]
        else:
            return [self._time_column]

    def get_time_constraint(self, starttime, endtime):
        if self._tz:
            starttime = starttime.astimezone(self._tz)
            endtime = endtime.astimezone(self._tz)
        return db.sql.column(self._time_column).between(starttime, endtime)

    def get_order_by_columns(self):
        return [db.sql.column(self._time_column)]

    def get_time(self, row):
        timestamp = row[self._time_column]

        if self._db_has_tz is None:
            self._db_has_tz = timestamp.tzinfo is not None
            if self._db_has_tz and self._tz and self._tz != timestamp.tzinfo:
                raise ValueError("Specified timezone: %s, but database has timezone: %s" % (self._tz, timestamp.tzinfo))
            if not self._db_has_tz and not self._tz:
                raise ValueError("No timezone specified and no timezone in database")

        if self._tz:
            return timestamp.replace(tzinfo=self._tz)

        return timestamp


class DateTimeAccessor(TimeAccessor):
    def __init__(self, date_column, time_column, tz=None):
        """
        :param date_column: (string) name of the db column containing the date
        :param time_column: (string) name of the db colummn containing the time
        :param tz: timezone if the db time does not have timezone
        """
        self._date_column = date_column
        self._time_column = time_column

        if isinstance(tz, pytz.BaseTzInfo):
            tz = zoneinfo.ZoneInfo(tz.zone)
        self._tz = tz
        self._db_has_tz = None

    def get_time_columns(self, engine):
        if engine.dialect.name == "sqlite":
            return [(self._date_column, db.Date), (self._time_column, db.Time)]
        else:
            return [self._date_column, self._time_column]

    def get_time_constraint(self, starttime, endtime):
        if self._tz:
            starttime = starttime.astimezone(self._tz)
            endtime = endtime.astimezone(self._tz)
        if starttime.date() == endtime.date():
            return db.and_(
                db.sql.column(self._date_column) == starttime.date(),
                db.sql.column(self._time_column).between(starttime.time(), endtime.time()),
            )
        else:
            return db.and_(
                db.sql.column(self._date_column).between(starttime.date(), endtime.date()),
                db.or_(
                    db.sql.column(self._date_column) > starttime.date(),
                    db.sql.column(self._time_column) >= starttime.time(),
                ),
                db.or_(
                    db.sql.column(self._date_column) < endtime.date(),
                    db.sql.column(self._time_column) <= endtime.time(),
                ),
            )

    def get_order_by_columns(self):
        return [db.sql.column(self._date_column), db.sql.column(self._time_column)]

    def get_time(self, row):
        timestamp = datetime.combine(date=row[self._date_column], time=row[self._time_column])

        if self._db_has_tz is None:
            self._db_has_tz = timestamp.tzinfo is not None
            if self._db_has_tz and self._tz and self._tz != timestamp.tzinfo:
                raise ValueError("Specified timezone: %s, but database has timezone: %s" % (self._tz, timestamp.tzinfo))
            if not self._db_has_tz and not self._tz:
                raise ValueError("No timezone specified and no timezone in database")

        if self._tz:
            return timestamp.replace(tzinfo=self._tz)

        return timestamp


# GRAPH TIME
class DBReader:
    def __init__(
        self,
        connection,
        time_accessor,
        table_name=None,
        schema_name=None,
        query=None,
        symbol_column=None,
        constraint=None,
        log_query=False,
        use_raw_user_query=False,
    ):
        """
        :param connection: sqlalchemy engine or (already connected) connection object.
        :param time_accessor: TimeAccessor object
        :param table_name: name of table in database as a string (table only, do not include database and schema)
        :param schema_name: name of the schema to use (for databases like SQL Server that use schema as part of table location, may not
                                                       be needed if schema name is "dbo")
        :param query: either string query or sqlalchemy query object. Ex: "select * from users"
        :param symbol_column: name of symbol column in table as a string
        :param constraint: additional sqlalchemy constraints for query. Ex: constraint = db.text( 'PRICE>:price' ).bindparams( price = 100.0 )
        :param log_query: set to True to see what query was generated to access the data
        :param use_raw_user_query: Don't do any alteration to user query, assume it contains all the needed columns and sorting
        """
        if not _HAS_SQLALCHEMY:
            raise RuntimeError("Could not find SQLAlchemy installation")
        self._connection = connection
        self._table_name = table_name
        self._schema_name = schema_name
        self._query = query
        self._time_accessor = time_accessor
        self._symbol_column = symbol_column
        self._constraint = constraint
        self._log_query = log_query
        self._use_raw_user_query = use_raw_user_query
        if use_raw_user_query and not query:
            raise RuntimeError("use_raw_user_queries True but no query provided")

        if bool(table_name) == bool(query):
            raise RuntimeError("Must specify table name or query")

        if bool(schema_name) and not bool(table_name):
            raise RuntimeError("Cannot specify schema name without table name")

        self._requested_cols = set()
        time_columns = time_accessor.get_time_columns(connection)
        if time_columns:
            for col in time_accessor.get_time_columns(connection):
                if isinstance(col, tuple):
                    self._requested_cols.add(col[0])
                else:
                    self._requested_cols.add(col)

        if symbol_column is not None:
            self._requested_cols.add(symbol_column)

    def subscribe(self, symbol, typ, field_map=None, push_mode=PushMode.NON_COLLAPSING):
        """If typ=None, then a struct will be dynamically created that reflects the schema of the table.
        In this case, the database will be queried at graph building time, and the resulting struct can
        be independently accessed via .schema_struct().
        """
        if self._symbol_column is None:
            raise RuntimeError("Attempted to subscribe to symbol %s but no symbol column was passed" % symbol)
        return self._subscribe(symbol, typ, field_map, push_mode)

    def subscribe_all(self, typ, field_map=None, push_mode=PushMode.NON_COLLAPSING):
        return self._subscribe("", typ, field_map, push_mode)

    def _subscribe(self, symbol, typ, field_map, push_mode):
        if typ is None:
            typ = self.schema_struct()
        if isinstance(field_map, dict):
            self._requested_cols.update(field_map.keys())
        elif isinstance(field_map, str):
            self._requested_cols.add(field_map)
        else:
            self._requested_cols.update(typ.metadata().keys())
        return DBReadAdapter(self, symbol, typ, field_map, push_mode=push_mode)

    def schema_struct(self):
        """Return a struct that represents the schema of the underlying table.
        Will be returned if typ=None is passed to subscribe.
        Also useful to inspect manually when defining field_map."""
        # Name the struct by the name of the table/schema
        # Note that two tables in different databases with the same name but different schemas will cause problems!
        # Including a hash of engine.url/conn.engine.url in the name may be safer (though less user-friendly)
        name = "DBDynStruct_{table}_{schema}".format(table=self._table_name or "", schema=self._schema_name or "")
        if name not in globals():
            db_metadata = db.MetaData(schema=self._schema_name)
            table = db.Table(self._table_name, db_metadata, autoload_with=self._connection)
            struct_metadata = {col: col_obj.type.python_type for col, col_obj in table.columns.items()}

            from csp.impl.struct import define_struct

            typ = define_struct(name, struct_metadata)
            globals()[name] = typ
        return globals()[name]

    def _create(self, engine, memo):
        return DBReaderImpl(engine, self)

    @classmethod
    def create_from_connection(
        cls,
        connection,
        time_accessor,
        table_name=None,
        schema_name=None,
        query=None,
        symbol_column=None,
        constraint=None,
        log_query=False,
    ):
        return cls(connection, time_accessor, table_name, schema_name, query, symbol_column, constraint, log_query)

    @classmethod
    def create_from_url(
        cls,
        url,
        time_accessor,
        table_name=None,
        schema_name=None,
        query=None,
        symbol_column=None,
        constraint=None,
        log_query=False,
    ):
        return cls(
            db.create_engine(url), time_accessor, table_name, schema_name, query, symbol_column, constraint, log_query
        )


# RUN TIME
class DBReaderImpl(AdapterManagerImpl):
    def __init__(self, engine, adapterRep):
        super().__init__(engine)
        self._rep = adapterRep
        self._inputs = {}
        self._prev_time = None
        self._row = None

    def start(self, starttime, endtime):
        self._query = self.build_query(starttime, endtime)
        if self._rep._log_query:
            import logging

            logging.info("DBReader query: %s", self._query)
        if _SQLALCHEMY_2:
            self._data_yielder = self._data_yielder_function()
        else:
            self._q = self._rep._connection.execute(self._query)

    def _data_yielder_function(self):
        # Connection yielder for SQLAlchemy 2
        with self._rep._connection.connect() as conn:
            for result in conn.execute(self._query).mappings():
                yield result
        # Signify the end
        yield None

    def build_query(self, starttime, endtime):
        if self._rep._table_name:
            metadata = db.MetaData(schema=self._rep._schema_name)

            if _SQLALCHEMY_2:
                table = db.Table(self._rep._table_name, metadata, autoload_with=self._rep._connection)
                cols = [table.c[colname] for colname in self._rep._requested_cols]
                q = db.select(*cols)
            else:
                table = db.Table(self._rep._table_name, metadata, autoload=True, autoload_with=self._rep._connection)
                cols = [table.c[colname] for colname in self._rep._requested_cols]
                q = db.select(cols)

        elif self._rep._use_raw_user_query:
            return db.text(self._rep._query)
        else:  # self._rep._query
            from_obj = db.text(f"({self._rep._query}) AS user_query")

            time_columns = self._rep._time_accessor.get_time_columns(self._rep._connection)
            if time_columns:
                if isinstance(time_columns[0], tuple):
                    time_select = [db.column(col[0], col[1]) for col in time_columns]
                    time_columns = [col[0] for col in time_columns]
                else:
                    time_select = [db.column(col) for col in time_columns]
            else:
                time_columns = []
                time_select = []
            select_cols = [db.column(colname) for colname in self._rep._requested_cols.difference(set(time_columns))]

            if _SQLALCHEMY_2:
                q = db.select(*(select_cols + time_select)).select_from(from_obj)
            else:
                q = db.select(select_cols + time_select, from_obj=from_obj)

        cond = self._rep._time_accessor.get_time_constraint(starttime.replace(tzinfo=UTC), endtime.replace(tzinfo=UTC))

        if "" not in self._inputs:
            symbol_col = db.sql.column(self._rep._symbol_column)
            cond = db.and_(cond, symbol_col.in_(self._inputs))

        if self._rep._constraint is not None:
            cond = db.and_(cond, self._rep._constraint)

        query = q
        if cond is not None:
            query = query.where(cond)

        order_by_columns = self._rep._time_accessor.get_order_by_columns()
        if order_by_columns:
            query = query.order_by(*order_by_columns)

        return query

    def stop(self):
        pass

    def register_input_adapter(self, symbol, adapter):
        if symbol not in self._inputs:
            self._inputs[symbol] = []
        self._inputs[symbol].append(adapter)

    def process_next_sim_timeslice(self, now):
        if self._row is None:
            if _SQLALCHEMY_2:
                self._row = next(self._data_yielder)
            else:
                self._row = self._q.fetchone()

        now = now.replace(tzinfo=UTC)
        while self._row is not None:
            time = self._rep._time_accessor.get_time(self._row)
            if time > now:
                return time
            self.process_row(self._row)
            if _SQLALCHEMY_2:
                self._row = next(self._data_yielder)
            else:
                self._row = self._q.fetchone()
        return None

    def process_row(self, row):
        if self._rep._symbol_column is not None:
            symbol = row[self._rep._symbol_column]

            for input in self._inputs.get(symbol, []):
                input.process_dict(row, keep_none=False)

        # subscribeAll
        for input in self._inputs.get("", []):
            input.process_dict(row, keep_none=False)


class DBReadAdapterImpl(ManagedSimInputAdapter):
    def __init__(self, managerImpl, symbol, typ, field_map):
        managerImpl.register_input_adapter(symbol, self)
        super().__init__(typ, field_map)


DBReadAdapter = py_managed_adapter_def(
    "dbadapter", DBReadAdapterImpl, ts["T"], DBReader, symbol=object, typ="T", fieldMap=(object, None)
)
