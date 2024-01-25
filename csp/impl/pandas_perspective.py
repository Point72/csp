import pandas as pd
import pytz
from datetime import datetime, timedelta
from pandas.compat import set_function_name
from typing import Optional

import csp
import csp.impl.pandas_accessor  # To ensure that the csp accessors are registered
from csp.impl.pandas_ext_type import is_csp_type

_ = csp.impl.pandas_accessor

try:
    import perspective
except ImportError:
    raise ImportError(
        "perspective must be installed to use this module. " "To install, run 'pip install perspective-python'."
    )


@csp.node
def _apply_updates(
    table: perspective.Table,
    data: {object: csp.ts[object]},
    index_col: str,
    time_col: str,
    throttle: timedelta,
    localize: bool,
    static_records: dict = None,
):
    with csp.alarms():
        alarm = csp.alarm(bool)

    with csp.state():
        s_buffer = []
        s_has_time_col = False
        s_datetime_cols = set()

    with csp.start():
        if throttle > timedelta(0):
            csp.schedule_alarm(alarm, throttle, True)
        s_has_time_col = time_col and time_col not in data.keys()
        s_datetime_cols = set([c for c, t in table.schema().items() if t == datetime])

    with csp.stop():
        try:
            # TODO: Remove when __stop__ can be called on a node without __start__ having been called
            # If there is an exception during one node's __start__, it can lead to __stop__ being called on a node that has never had its __start__ called. To repro:
            # import csp
            # @csp.node
            # def foo():
            #     with __start__():
            #         print("foo start")
            #         raise
            # @csp.node
            # def bar():
            #     with __start__():
            #         print("bar start")
            #     with __stop__():
            #         print("bar stop")
            # @csp.graph
            # def my_graph():
            #     foo()
            #     bar()
            # def main():
            #     csp.run(my_graph, realtime=True)
            # if __name__ == '__main__':
            #     main()
            if len(s_buffer) > 0:
                table.update(s_buffer)
        except BaseException:
            pass

    if csp.ticked(data):
        new_rows = {}
        for (idx, col), value in data.tickeditems():
            if idx not in new_rows:
                row = {}
                new_rows[idx] = row
                if index_col:
                    row[index_col] = idx
                if s_has_time_col:
                    if localize:
                        row[time_col] = pytz.utc.localize(csp.now())
                    else:
                        row[time_col] = csp.now()
            else:
                row = new_rows[idx]

            if localize and col in s_datetime_cols and value.tzinfo is None:
                row[col] = pytz.utc.localize(value)
            else:
                row[col] = value

        if static_records:
            for idx, row in new_rows.items():
                row.update(static_records[idx])

        if throttle == timedelta(0):
            table.update(list(new_rows.values()))
        else:
            s_buffer.extend(new_rows.values())

    if csp.ticked(alarm):
        if len(s_buffer) > 0:
            table.update(s_buffer)
            s_buffer = []

        csp.schedule_alarm(alarm, throttle, True)


def _frame_to_basket(df):
    df = df.csp.ts_frame()
    basket = {}
    for col, series in df.items():
        for idx, edge in series.dropna().items():
            basket[(idx, col)] = edge
    return basket


class CspPerspectiveTable:
    def __init__(
        self,
        data: pd.DataFrame,
        index_col: str = "index",
        time_col: Optional[str] = "timestamp",
        throttle: Optional[timedelta] = timedelta(seconds=0.5),
        keep_history: bool = True,
        limit: int = None,
        localize: bool = False,
    ):
        """

        :param data: A data frame containing regular columns as well as TsDtype columns
        :param index_col: The name of the column in perspective which will hold the index of the data frame.
            If the index of the data frame already has a name, this name will be used instead.
        :param time_col: The name of the column to hold timestamps. If not provided, no timestamp column will be added.
        :param throttle: The rate at which we update the perspective table from the buffer
        :param keep_history: Whether to keep the history of all updates to the table. If false, will update in place.
            time_col must be supplied if keep_history is True.
        :param limit: The maximum number of (most recent) rows the perspective table will hold.
            Only works when keep_history=True.
        :param localize: Whether to localize time_col to local timestamps in the perspective view.
            If False, will display utc time.
        """

        if data.index.nlevels > 1:
            raise ValueError("Perspective does not support multi-indices for rows")
        if data.columns.nlevels > 1:
            raise ValueError("Perspective does not support multi-indices for columns")
        if not time_col and keep_history:
            raise ValueError("time_col must be supplied if keep_history is True")
        if limit and not keep_history:
            raise ValueError("Limit only works when keep_history is True")
        self._data = data
        self._index_col = index_col
        self._time_col = time_col
        self._throttle = throttle
        self._keep_history = keep_history
        self._limit = limit
        self._localize = localize

        self._basket = _frame_to_basket(data)
        self._static_frame = data.csp.static_frame()
        self._static_table = perspective.Table(self._static_frame)
        static_schema = self._static_table.schema()
        # Since the index will be accounted for separately, remove the index from the static table schema,
        # and re-enter it under index_col
        raw_index_name = self._static_frame.index.name or "index"
        index_type = static_schema.pop(raw_index_name)
        schema = {index_col: index_type}
        if time_col:
            schema[time_col] = datetime
        for col, series in data.items():
            if is_csp_type(series):
                schema[col] = series.dtype.subtype
            else:
                schema[col] = static_schema[col]

        if self._keep_history:
            self._table = perspective.Table(schema, index=None, limit=limit)
            self._static_records = self._static_frame.to_dict(orient="index")
        else:
            self._table = perspective.Table(schema, index=self._index_col)
            self._static_frame.index = self._static_frame.index.rename(self._index_col)
            self._table.update(self._static_frame)
            self._static_records = None  # No need to update dynamically

        self._runner = None

    def clear(self):
        """Resets the table to it's original state."""
        self._table.clear()
        if not self._keep_history:
            self._table.update(self._static_frame)

    def graph(self):
        """The csp graph that populates the table with ticking data"""
        if self._basket:
            _apply_updates(
                self._table,
                self._basket,
                self._index_col,
                self._time_col,
                self._throttle,
                self._localize,
                self._static_records,
            )

    def run_historical(self, starttime, endtime):
        """Runs the dataframe explicitly upfront, then creates a perspective table out of it in one go.
        This can be significantly faster than incrementally updating the table from a historical graph.
        """
        df = self._data.csp.run(starttime, endtime)
        df.index.set_names([self._index_col, self._time_col], inplace=True)
        if not self._time_col:
            df = df.droplevel(-1)
        df = df.reset_index()
        if self._keep_history:
            index = None
        else:
            index = self._index_col
        if self._limit:
            df = df.sort_values(self._time_col).tail(self._limit).reset_index(drop=True)
        return perspective.Table(df.to_dict("series"), index=index)

    def run(self, starttime=None, endtime=timedelta(seconds=60), realtime=True, clear=False):
        """Run a graph that sends data to the table on the current thread.
        Normally, this is only useful for debugging.
        """
        starttime = starttime or datetime.utcnow()
        if clear:
            self.clear()
        csp.run(self.graph, starttime=starttime, endtime=endtime, realtime=realtime)

    def start(self, starttime=None, endtime=timedelta(seconds=60), *, realtime=True, clear=True, auto_shutdown=True):
        """Start a graph that sends data to the table on a csp engine thread.
        If clear=True, will clear any data from the data before writing to it.
        If auto_shutdown=True, will stop the engine if the table gets garbage collected.
        """
        starttime = starttime or datetime.utcnow()
        if clear:
            self.clear()
        self._runner = csp.run_on_thread(
            self.graph, starttime=starttime, endtime=endtime, realtime=realtime, auto_shutdown=auto_shutdown
        )

    def is_running(self):
        """Return whether the graph is currently running."""
        if self._runner is None:
            return False
        return self._runner.is_alive()

    def stop(self):
        """Stop the running csp engine thread"""
        if self._runner is None:
            raise ValueError("No active runner to stop")
        self._runner.stop_engine()
        self.join()

    def join(self):
        """Block until the csp engine thread is done"""
        if self._runner is None:
            raise ValueError("No active runner to join")
        self._runner.join()

    @property
    def table(self):
        """Return the underlying perspective table"""
        return self._table

    def get_widget(self, **override_kwargs):
        """Create a Jupyter widget with some sensible defaults, and accepting as overrides any of the
        arguments to perspective.PerspectiveWidget."""
        if self._keep_history:
            kwargs = {
                "columns": list(self._data.columns),
                "group_by": [self._index_col, self._time_col],
                "aggregates": {k: "last by index" for k in list(self._data.columns)},
                "sort": [[self._time_col, "desc"]],
            }
        else:
            kwargs = {"columns": list(self._table.schema())}
        kwargs.update(override_kwargs)
        return perspective.PerspectiveWidget(self._table, **kwargs)

    @classmethod
    def _create_view_method(cls, method):
        def _method(self, **options):
            return method(self._table.view(), **options)

        set_function_name(_method, method.__name__, cls)
        return _method

    @classmethod
    def _add_view_methods(cls):
        cls.to_df = cls._create_view_method(perspective.View.to_df)
        cls.to_dict = cls._create_view_method(perspective.View.to_dict)
        cls.to_json = cls._create_view_method(perspective.View.to_json)
        cls.to_csv = cls._create_view_method(perspective.View.to_csv)
        cls.to_numpy = cls._create_view_method(perspective.View.to_numpy)
        cls.to_columns = cls._create_view_method(perspective.View.to_columns)
        cls.to_arrow = cls._create_view_method(perspective.View.to_arrow)


CspPerspectiveTable._add_view_methods()


def _start_many_graph(tables: [CspPerspectiveTable]):
    for table in tables:
        table.graph()


class CspPerspectiveMultiTable:
    """A class to hold multiple CspPerspective tables and coordinate their running/stopping."""

    def __init__(self, tables: {str: CspPerspectiveTable}):
        self._tables = tables
        self._runner = None

    @property
    def tables(self):
        """Return the collection of tables."""
        return self._tables.copy()

    def start(self, starttime=None, endtime=timedelta(seconds=60), *, realtime=True, clear=True, auto_shutdown=True):
        """Start a graph that sends data to the table on a csp engine thread.
        If clear=True, will clear any data from the data before writing to it.
        If auto_shutdown=True, will stop the engine if the table gets garbage collected.
        """
        if clear:
            for table in self._tables.values():
                table.clear()
        self._runner = csp.run_on_thread(
            _start_many_graph,
            set(self.tables.values()),
            starttime=starttime,
            endtime=endtime,
            realtime=realtime,
            auto_shutdown=auto_shutdown,
        )

    def is_running(self):
        """Return whether the graph is currently running."""
        if self._runner is None:
            return False
        return self._runner.is_alive()

    def stop(self):
        """Stop the running csp engine thread"""
        if self._runner is None:
            raise ValueError("No active runner to stop")
        self._runner.stop_engine()
        self.join()

    def join(self):
        """Block until the csp engine thread is done"""
        if self._runner is None:
            raise ValueError("No active runner to join")
        self._runner.join()

    def __getitem__(self, name):
        return self._tables[name]

    def get_widget(self, widget="Tab", config=None):
        """Create a widget from the multi-table.
        :param widget: A string representing the type of widget to use, i.e. "Tab" or "Accordion".
            The type of widget must be a valid ipywidgets type that supports "children" and optionally "titles"
        :param config: An optional dictionary, keyed with the names of tables. The values of the dictionary are
            kwargs to pass to the CspPerspectiveTable.get_widget call
        """
        import ipywidgets

        config = config or {}
        # Create the children
        children = []
        titles = []
        for title, table in self._tables.items():
            if not config or title in config:
                kwargs = config.get(title, {})
                children.append(table.get_widget(**kwargs))
                titles.append(title)

        w = getattr(ipywidgets, widget)(children=children)
        if hasattr(w, "set_title"):
            for idx, title in enumerate(titles):
                w.set_title(idx, title)

        return w
