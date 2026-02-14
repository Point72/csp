from datetime import date, datetime, timedelta
from typing import Dict, Optional

from packaging import version

import csp.baselib
from csp.impl.wiring.edge import Edge

# Lazy declaration below to avoid perspective import
RealtimePerspectiveWidget = None


class DataFrame:
    def __init__(self, data: Optional[Dict] = None):
        self._data = data or {}
        self._columns = list(self._data.keys())
        self._psp_client = None

    @property
    def columns(self):
        return self._columns

    def _filter(self, filter_ts):
        data = {}
        for col, v1 in zip(self._columns, self._data.values()):
            data[col] = csp.baselib.filter(filter_ts, v1)
        return DataFrame(data)

    def __getattr__(self, column):
        try:
            return self.__getitem__(column)
        except KeyError:
            raise AttributeError

    def __getitem__(self, columns):
        if isinstance(columns, Edge):
            if columns.tstype.typ is not bool:
                raise KeyError(
                    "csp.DataFrame access by edge expected ts[bool] got ts[" + columns.tstype.typ.__name__ + "]"
                )
            return self._filter(columns)

        was_list = True
        if not isinstance(columns, list):
            columns = [columns]
            was_list = False

        data = {}
        for col in columns:
            v = self._data.get(col, None)
            if v is None:
                raise KeyError(f"Unrecognized column: '{col}'")
            data[col] = v

        return DataFrame(data) if was_list else next(iter(data.values()))

    def __setitem__(self, columns, values):
        if not isinstance(columns, list):
            columns = [columns]

        if isinstance(values, DataFrame):
            values = list(values._data.values())
        elif not isinstance(values, list):
            values = [values]

        if len(values) != len(columns):
            raise ValueError(f"Expected {len(columns)} values got {len(values)}")

        self._data.update(zip(columns, values))
        self._columns = list(self._data.keys())

    def _apply_binary_op(self, other, method):
        if isinstance(other, DataFrame):
            values = []
            for col in self._columns:
                rhs = other._data.get(col, None)
                if rhs is None:
                    raise ValueError(f"Shape mismatch, missing column {col}")
                values.append(rhs)
        elif not isinstance(other, (tuple, list)):
            values = [other] * len(self._data)
        else:
            values = other

        if len(values) != len(self._columns):
            raise ValueError(f"Shape mismatch, expected {len(self._columns)} columns got {len(values)}")

        data = {}
        for col, v1, v2 in zip(self._columns, self._data.values(), values):
            data[col] = method(v1, v2)

        return DataFrame(data)

    def __add__(self, other):
        return self._apply_binary_op(other, lambda x, y: x + y)

    def __sub__(self, other):
        return self._apply_binary_op(other, lambda x, y: x - y)

    def __mul__(self, other):
        return self._apply_binary_op(other, lambda x, y: x * y)

    def __truediv__(self, other):
        return self._apply_binary_op(other, lambda x, y: x / y)

    def __floordiv__(self, other):
        return self._apply_binary_op(other, lambda x, y: x // y)

    def __pow__(self, other):
        return self._apply_binary_op(other, lambda x, y: x**y)

    def __gt__(self, other):
        return self._apply_binary_op(other, lambda x, y: x > y)

    def __ge__(self, other):
        return self._apply_binary_op(other, lambda x, y: x >= y)

    def __lt__(self, other):
        return self._apply_binary_op(other, lambda x, y: x < y)

    def __le__(self, other):
        return self._apply_binary_op(other, lambda x, y: x <= y)

    def __eq__(self, other):
        return self._apply_binary_op(other, lambda x, y: x == y)

    def __ne__(self, other):
        return self._apply_binary_op(other, lambda x, y: x != y)

    def __str__(self):
        return "\n".join(f"{k} : ts[{v.tstype.typ.__name__}]" for k, v in self._data.items())

    def __repr__(self):
        return "csp.DataFrame( %s )" % (", ".join(f"{k} = ts[{v.tstype.typ.__name__}]" for k, v in self._data.items()))

    # Evaluation methods
    def _eval_graph(self):
        import csp

        for k, v in self._data.items():
            csp.add_graph_output(k, v)

    def _eval(self, starttime: datetime, endtime: datetime = None, realtime: bool = False):
        import csp

        return csp.run(self._eval_graph, starttime=starttime, endtime=endtime, realtime=realtime)

    def show_graph(self):
        from PIL import Image

        import csp.showgraph

        buffer = csp.showgraph.generate_graph(self._eval_graph)
        return Image.open(buffer)

    def to_pandas(self, starttime: datetime, endtime: datetime):
        import pandas

        results = self._eval(starttime, endtime, False)
        data = {}
        for k, s in results.items():
            times = (v[0] for v in s)

            data[k] = pandas.Series((v[1] for v in s), index=times)
        return pandas.DataFrame(data)

    def to_pandas_ts(self, trigger, window, tindex=None, wait_all_valid=True):
        """
        :param trigger: The trigger for generation and output of the DataFrame.
                        A new DataFrame will be produced each time trigger ticks.
                        Can be passed a string referring to a column of the DataFrame
        :param window: An integer or timedelta representing the maximum window size (i.e. scope of the frame index).
                        If an integer, note that:
                        a) DataFrames may have fewer than 'window' rows if the data has not ticked enough
                        b) If tindex is not provided, DataFrames may have more than 'window' rows to account for
                            mis-alignment of underlying time series (i.e. there will be 'window' valid values in
                            each column, but not all at the same times).
                        Lastly, if there is no data in a window, an empty frame (with the right columns and schema)
                        will be returned.
        :param tindex: An (optional) time series on which to sample data before generating the frame.
                        The index of the returned DataFrames will contain the tick times of tindex.
                        This aligns the indices of the columns, improving performance when the columns are combined
                        into a single frame
                        Can be passed a string referring to a column of the DataFrame.
        :param wait_all_valid: Whether to wait for all columns to be valid before including a row in the data set.
                If 'tindex' is provided, and wait_all_valid is True, the DataFrame can be constructed from the buffer numpy
                arrays with no copying. If it is False, any columns that tick after the first tick of 'tindex' (and
                are still included in the window) will be copied each time the trigger causes a new DataFrame to be generated.
        :returns: A time series of pandas DataFrames
        """
        if trigger in self._data:
            trigger = self._data[trigger]
        elif not isinstance(trigger, Edge):
            raise ValueError(f"trigger must be a column in the frame or an Edge, received {trigger}")
        if tindex in self._data:
            tindex = self._data[tindex]
        elif not (tindex is None or isinstance(tindex, Edge)):
            raise ValueError(f"tindex must be a column in the frame or an Edge, received {tindex}")

        from csp.impl.pandas import make_pandas

        return make_pandas(trigger, self._data, window, tindex, wait_all_valid)

    def to_perspective(self, starttime: datetime, endtime: datetime = None, realtime: bool = False):
        import csp

        try:
            import perspective

            if version.parse(perspective.__version__) >= version.parse("3"):
                _PERSPECTIVE_3 = True
                from perspective.widget import PerspectiveWidget
            else:
                _PERSPECTIVE_3 = False
                from perspective import PerspectiveWidget

            global RealtimePerspectiveWidget
            if RealtimePerspectiveWidget is None:

                class RealtimePerspectiveWidget(PerspectiveWidget):
                    def __init__(self, engine_runner, *args, **kwargs):
                        super().__init__(*args, **kwargs)
                        self._runner = engine_runner

                    def stop(self):
                        """stop the running csp engine thread"""
                        self._runner.stop_engine()
                        self.join()

                    def join(self):
                        """block until the csp engine thread is done"""
                        self._runner.join()

        except ImportError:
            raise ImportError("to_perspective requires perspective-python installed")

        if not realtime:
            df = self.to_pandas(starttime, endtime)
            return PerspectiveWidget(df.ffill(), plugin="Y Line", columns=self._columns, group_by="index")

        @csp.node
        def apply_updates(table: object, data: Dict[str, csp.ts[object]], timecol: str, throttle: timedelta):
            with csp.alarms():
                alarm = csp.alarm(bool)
            with csp.state():
                s_buffer = []

            with csp.start():
                csp.schedule_alarm(alarm, throttle, True)

            if csp.ticked(data):
                s_buffer.append(dict(data.tickeditems()))
                if _PERSPECTIVE_3:
                    s_buffer[-1][timecol] = int(csp.now().timestamp() * 1000)
                else:
                    s_buffer[-1][timecol] = csp.now()

            if csp.ticked(alarm):
                if len(s_buffer) > 0:
                    table.update(s_buffer)
                    s_buffer = []

                csp.schedule_alarm(alarm, throttle, True)

        timecol = "time"
        schema = {k: v.tstype.typ for k, v in self._data.items()}
        schema[timecol] = datetime
        if _PERSPECTIVE_3:
            perspective_type_map = {
                str: "string",
                float: "float",
                int: "integer",
                date: "date",
                datetime: "datetime",
                bool: "boolean",
            }
            schema = {col: perspective_type_map[typ] for col, typ in schema.items()}
            if self._psp_client is None:
                self._psp_client = perspective.Server().new_local_client()
            table = self._psp_client.table(schema)
        else:
            table = perspective.Table(schema)
        runner = csp.run_on_thread(
            apply_updates,
            table,
            self._data,
            timecol,
            timedelta(seconds=0.5),
            starttime=starttime,
            endtime=endtime,
            realtime=True,
        )
        widget = RealtimePerspectiveWidget(
            runner,
            table,
            plugin="Y Line",
            columns=self._columns,
            group_by=timecol,
            aggregates={k: "last by index" for k in self._columns},
        )
        return widget
