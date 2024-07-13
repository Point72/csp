from datetime import datetime, timedelta
from typing import Dict, List, TypeVar, Union

import numpy as np
import pandas as pd
from pandas.api.extensions import register_dataframe_accessor, register_series_accessor
from pandas.core.arrays import ExtensionArray

import csp
from csp import ts
from csp.impl.pandas_ext_type import TsDtype, is_csp_type
from csp.impl.struct import define_nested_struct
from csp.impl.wiring.edge import Edge

T = TypeVar("T")


@csp.node
def _basket_valid(xs: List[ts[object]]) -> ts[bool]:
    if csp.valid(xs):
        csp.make_passive(xs)
        return True


@csp.node
def _basket_synchronize(xs: List[ts["T"]], threshold: timedelta) -> csp.OutputBasket(List[ts["T"]], shape_of="xs"):
    with csp.alarms():
        a_end = csp.alarm(bool)

    with csp.state():
        s_current = {}
        s_alarm_handle = None

    if csp.ticked(xs):
        if not s_alarm_handle:
            s_alarm_handle = csp.schedule_alarm(a_end, threshold, True)
        s_current.update(xs.tickeditems())

    if csp.ticked(a_end) or len(s_current) == len(xs):
        csp.output(s_current)
        if s_alarm_handle:
            csp.cancel_alarm(a_end, s_alarm_handle)
            s_alarm_handle = None
        s_current = {}


def _series_to_python_type(series: pd.Series):
    # What a mess!
    if pd.api.types.is_datetime64_any_dtype(series.dtype):
        return datetime
    elif pd.api.types.is_integer_dtype(series.dtype):
        return int
    elif pd.api.types.is_float_dtype(series.dtype):
        return float
    elif pd.api.types.is_bool_dtype(series.dtype):
        return bool
    else:

        def find_valid_index(values: pd.Series, *, how: str):
            assert how in ["first", "last"]
            if len(values) == 0:  # early stop
                return None

            is_valid = ~pd.isna(values)

            if values.ndim == 2:
                is_valid = is_valid.any(axis=1)  # reduce axis 1

            if how == "first":
                idxpos = is_valid[::].argmax()
            elif how == "last":
                idxpos = len(values) - 1 - is_valid[::-1].argmax()

            chk_notna = is_valid.iloc[idxpos]

            if not chk_notna:
                return None
            return idxpos

        first_idx = find_valid_index(series, how="first")
        last_idx = find_valid_index(series, how="last")

        # Note we don't do a full check of uniqueness of all types in the series for performance reasons
        if (first_idx is not None or last_idx is not None) and (
            type(series.iloc[first_idx]) == type(series.iloc[last_idx])  # noqa: E721
        ):
            typ = type(series.iloc[first_idx])
            if typ == pd.Timestamp:
                return datetime
            return typ
        return object


def _from_series(series: pd.Series, drop_na: bool = False) -> Edge:
    """Convert a series to an Edge.
    Note: drop_na only applies to float and object types. For all other types, they are automatically dropped.
    """
    typ = _series_to_python_type(series)
    if drop_na or typ not in (float, object):
        series = series.dropna()

    if series.index.nlevels > 1:
        idx = series.index.get_level_values(level=-1)
    else:
        idx = series.index
    if not isinstance(idx, pd.DatetimeIndex):
        raise TypeError("Index must be a pd.DatetimeIndex")

    if isinstance(series.values, ExtensionArray):
        values = series.values.to_numpy()
    else:
        values = series.values
    if len(series) == 0:
        return csp.null_ts(typ)
    else:
        return csp.curve(typ, (idx.values, values))


def _run(
    eval_graph,
    starttime: datetime,
    endtime: datetime,
    realtime: bool = False,
    tick_count: int = -1,
    tick_history: timedelta = timedelta(),
    snap: bool = False,
) -> Dict:
    results = csp.run(
        eval_graph,
        tick_count,
        tick_history,
        snap,
        starttime=starttime,
        endtime=endtime,
        realtime=realtime,
        output_numpy=True,
    )
    results = results or {}
    output = {}
    for (col, rowidx), s in results.items():
        output.setdefault(col, []).append((rowidx, s))
    return output


def _empty_index_like(orig_index, snap):
    """Creates a new empty index based on an existing (possibly non-empty index).
    If snap is true, the output index shape is just like the input. If not, an extra datetime index is appended
    at the end."""
    # Empty out original index
    orig_index = orig_index[[]]
    if snap:
        return orig_index
    else:
        if isinstance(orig_index, pd.MultiIndex):
            levels = orig_index.levels + [pd.DatetimeIndex([])]
        else:
            levels = [orig_index, pd.DatetimeIndex([])]
        return pd.MultiIndex(levels, codes=[[]] * len(levels))


@register_series_accessor("csp")
class CspSeriesAccessor(object):
    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        if not is_csp_type(obj):
            raise AttributeError("Cannot use 'csp' accessor on objects of dtype '{}'.".format(obj.dtype))

    def _infer_dtype(self, data):
        try:
            return TsDtype(next(e.tstype.typ for e in data if isinstance(e, Edge)))
        except StopIteration:
            return self._obj.dtype

    def apply(self, func, *args, **kwargs):
        """Call Edge.apply to apply a python function to each non-null element of a series, generating a new series.
        :param func: A scalar function that will be applied on each value of each Edge. If a different output type
            is returned, pass a tuple (f, typ), where typ is the output type of f.
        :param args: Positional arguments passed into func
        :param kwargs: Dictionary of keyword arguments passed into func
        :return: A TsDtype series containing the output of Edge.apply
        """

        def apply_series(edge):
            if pd.isnull(edge):
                return edge
            else:
                return edge.apply(func, *args, **kwargs)

        s = self._obj.apply(apply_series)
        return pd.Series(s, dtype=self._infer_dtype(s), index=self._obj.index)

    def pipe(self, node, *args, **kwargs):
        """Call Edge.pipe to apply a csp node to each non-null element of a series, generating a new series.

        :param node: A graph node that will be applied to each Edge, which is passed into node as the first argument.
            Alternatively, a (node, edge_keyword) tuple where edge_keyword is a string indicating the keyword of node
            that expects the edge.
        :param args: Positional arguments passed into node
        :param kwargs: Dictionary of keyword arguments passed into node
        :return: A TsDtype series containing the output of Edge.pipe
        """

        def pipe_series(edge):
            if pd.isnull(edge):
                return edge
            else:
                return edge.pipe(node, *args, **kwargs)

        s = self._obj.apply(pipe_series)
        return pd.Series(s, dtype=self._infer_dtype(s), index=self._obj.index)

    def binop(self, node, other, *args, **kwargs):
        """Call a node of two ts arguments to each pair of edges in the current series and another series.

        :param node: A graph node that takes (at least) two inputs and will be applied to each pair of elements in
            self and other
        :param other: A pd.Series containing the second argument to be passed pairwise to node
        :param args: Positional (non-ts) arguments passed into node
        :param kwargs: Dictionary of (non-ts) keyword arguments passed into node
        :return: A TsDtype series containing the outputs of the pairwise application of op

        :example: series1.csp.binop(filter, series2)
        """

        def binop_series(x1, x2):
            if pd.isnull(x1) or pd.isnull(x2):
                return TsDtype.na_value
            else:
                return node(x1, x2, *args, **kwargs)

        if not self._obj.index.equals(other.index):
            raise ValueError("Series indices must be equal for binop")
        out = [binop_series(e1, e2) for e1, e2 in zip(self._obj.values, other.values)]
        return pd.Series(out, dtype=self._infer_dtype(out), index=self._obj.index)

    def sample(self, trigger: Union[timedelta, np.timedelta64, pd.Timedelta, ts[object]]):
        """Sample all non-null elements of the series at a regular timedelta interval or at some custom trigger

        :param trigger: Either a sampling interval or a time series representing the trigger ticks
        :return: A TsDtype series containing the outputs of csp.sample applied to each non-null element of the series
        """
        if isinstance(trigger, np.timedelta64):
            trigger = pd.Timedelta(trigger)
        if isinstance(trigger, pd.Timedelta):
            trigger = trigger.to_pytimedelta()
        if isinstance(trigger, timedelta):
            trigger = csp.timer(trigger, True)
        return self.pipe((csp.sample, "x"), trigger=trigger)

    @staticmethod
    def _flatten(series, columns, prepend_name, delim, recursive):
        struct = series.dtype.subtype
        if not issubclass(struct, csp.Struct):
            raise TypeError("Series must contain ts[Struct]")
        data = {}
        meta = struct.metadata()
        columns = columns or meta.keys()
        for col in columns:
            if prepend_name and series.name is not None:
                col_name = f"{series.name}{delim}{col}"
            else:
                col_name = col
            col_series = series.astype(object).apply(getattr, args=(col,)).astype(TsDtype(meta[col]))
            col_series.name = col_name
            if recursive and issubclass(meta[col], csp.Struct):
                data.update(CspSeriesAccessor._flatten(col_series, None, prepend_name, delim, recursive))
            else:
                data[col_name] = col_series
        return data

    def flatten(self, columns=None, prepend_name=True, delim=" ", recursive=True):
        """Expands a series of ts of structs into a frame of columns

        :param columns: An optional subset of columns to create from the struct metadata. If not provided, all columns
            will be expanded
        :param prepend_name: Whether to prepend the series name to the names of the columns that are generated.
            Optional, default True
        :param delim: The delimiter between series name and column names, used if prepend_name is True.
            Optional, default is a space as it won't occur in a field name, unlike underscore
        :param recursive: Whether to recursively expand struct fields that themselves contain structs
        :returns: A pd.DataFrame of TsDtype columns corresponding to the Struct fields.
        """
        data = self._flatten(self._obj, columns, prepend_name, delim, recursive)
        return pd.DataFrame(data)

    @csp.graph
    def valid(self) -> ts[bool]:
        """Ticks true when all items in the series are valid"""
        return _basket_valid(self._obj.dropna().to_list())

    def synchronize(self, threshold: timedelta):
        """Synchronizes the series to "bucket" ticks together.
        When one element ticks, will wait until all other elements in the series tick before returning, or until
        the threshold has passed. This is particularly useful when doing cross-sectional analysis on data
        that is generated syncrhonously, but published asynchronously by symbol (i.e. bars)."""
        raw = self._obj.dropna()
        sync = _basket_synchronize(raw.to_list(), threshold)
        data = dict(zip(raw.index, sync))
        return pd.Series(data, index=self._obj.index, dtype=self._obj.dtype)

    def _eval_graph(self, tick_count: int = -1, tick_history: timedelta = timedelta(), snap: bool = False):
        for rowidx, edge in self._obj.items():
            if isinstance(edge, Edge):
                csp.add_graph_output((self._obj.name, rowidx), edge, tick_count=tick_count, tick_history=tick_history)
        if snap:
            csp.stop_engine(self.valid())

    def _parse_results(self, results, name=None, snap=False):
        repeats = pd.Series(0, index=self._obj.index)
        times = []
        values = []
        for rowidx, (time_array, value_array) in results:
            values.append(value_array)
            times.append(time_array)
            repeats[rowidx] = len(time_array)

        values = np.hstack(values)
        times = np.hstack(times)
        if len(values):
            dtype = None
            idx = self._obj.index.repeat(repeats)
            if not snap:  # Join the times into the index
                idx = pd.DataFrame({None: times}, index=idx, dtype="datetime64[ns]").set_index(None, append=True).index
        else:
            dtype = self._obj.dtype.subtype
            idx = _empty_index_like(self._obj.index, snap)

        return pd.Series(values, index=idx, dtype=dtype, name=name)

    def run(
        self,
        starttime: datetime,
        endtime: datetime,
        realtime: bool = False,
        tick_count: int = -1,
        tick_history: timedelta = timedelta(),
        snap: bool = False,
    ):
        """Run a graph containing the edges in the series.

        :param starttime: The start time for the graph
        :param endtime: The start time for the graph
        :param realtime: Whether to use the realtime csp engine
        :param tick_count: The maximum number of ticks to return. Default is -1 (all ticks)
        :param tick_history: The maximum tick history to return. Default is timedelta() (all ticks)
        :param snap: Whether to run in "snap" mode, i.e. return the first tick of every series and stop once all
            series have ticked
        :returns: A new Series of type corresponding to the underlying edge type (TsDtype.subtype),
            with an additional index level containing the datetimes of the ticked values.
        """
        if self._obj.empty:  # Nothing to run
            dtype = self._obj.dtype.subtype
            idx = _empty_index_like(self._obj.index, snap)
            return pd.Series([], dtype=dtype, index=idx)
        else:
            data = _run(self._eval_graph, starttime, endtime, realtime, tick_count, tick_history, snap)
            return self._parse_results(data[self._obj.name], self._obj.name, snap)

    def snap(self, timeout: timedelta = timedelta(seconds=10), starttime: datetime = None):
        """ "Snap" a graph containing the edges in the series.

        :param timeout: How long to wait for all the elements to tick before stopping the circuit
        :param starttime: The start-time for the snap. Could be a time in the past (for snapping historical data) or
            a time in the future (to schedule a snap at, i.e. the next minute)
        :returns: A new Series of type corresponding to the underlying edge type (TsDtype.subtype),
            with the same index as the original series, containing the snapped value of each Edge.
        """
        starttime = starttime or datetime.utcnow()
        return self.run(starttime, timeout, True, tick_count=1, snap=True)

    def show_graph(self):
        """Show the graph corresponding to the evaluation of all the edges.
        For large series, this may be very large, so it may be helpful to call .head() first.
        """
        from PIL import Image

        import csp.showgraph

        buffer = csp.showgraph.generate_graph(self._eval_graph, "png")
        return Image.open(buffer)


@register_series_accessor("to_csp")
class ToCspSeriesAccessor(object):
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def __call__(self, drop_na: bool = False):
        """Return an Edge from the series
        :param drop_na: If True, will drop na types from series of float and object types. For all other series types,
            na's are dropped by default for csp's type consistency.
        """
        nlevels = self._obj.index.nlevels
        if nlevels == 1:
            return _from_series(self._obj, drop_na)
        else:
            idx = self._obj.index.levels[-1]
            if not isinstance(idx, pd.DatetimeIndex):
                raise TypeError("Last (innermost) index must be of type DatetimeIndex")
            levels = list(range(nlevels - 1))
            typ = _series_to_python_type(self._obj)
            return (
                self._obj.groupby(level=levels, group_keys=False)
                .apply(lambda s: _from_series(s, drop_na))
                .astype(TsDtype(typ))
            )

    def _static_agg(self, agg: str = "last"):
        """Helper function that does a "static aggregation" across the time dimension."""
        nlevels = self._obj.index.nlevels
        if nlevels == 1:
            if not isinstance(self._obj.index, pd.DatetimeIndex):
                raise TypeError("Index must be of type DatetimeIndex")
            zero = np.zeros(len(self._obj))
            return self._obj.groupby(zero).agg(agg).loc[0]
        else:
            idx = self._obj.index.levels[-1]
            if not isinstance(idx, pd.DatetimeIndex):
                raise TypeError("Last (innermost) index must be of type DatetimeIndex")
            levels = list(range(nlevels - 1))
            return self._obj.groupby(level=levels, group_keys=False).agg(agg)


@register_dataframe_accessor("csp")
class CspDataFrameAccessor(object):
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def static_frame(self):
        """Return the static part of the data frame (columns not of dtype TsDtype)"""
        cols = [col for col, series in self._obj.items() if not is_csp_type(series)]
        return self._obj[cols]

    def ts_frame(self):
        """Return the csp part of the data frame (columns of type TsDtype)"""
        cols = [col for col, series in self._obj.items() if is_csp_type(series)]
        return self._obj[cols]

    def _eval_graph(self, tick_count: int = -1, tick_history: timedelta = timedelta(), snap: bool = False):
        df_ts = self.ts_frame()
        if df_ts.index.has_duplicates:
            raise ValueError("Cannot run csp on a frame with duplicate row indices.")
        if df_ts.columns.has_duplicates:
            raise ValueError("Cannot run csp on a frame with duplicate column indices.")
        for col, series in df_ts.items():
            series.csp._eval_graph(tick_count, tick_history, snap)

    def run(
        self,
        starttime: datetime,
        endtime: datetime,
        realtime: bool = False,
        tick_count: int = -1,
        tick_history: timedelta = timedelta(),
        snap: bool = False,
        collect: bool = True,
    ):
        """ "Run" a graph containing the edges in all the TsDtype series.

        :param starttime: The start time for the graph
        :param endtime: The start time for the graph
        :param realtime: Whether to use the realtime csp engine
        :param tick_count: The maximum number of ticks to return. Default is -1 (all ticks)
        :param tick_history: The maximum tick history to return. Default is timedelta() (all ticks)
        :param snap: Whether to run in "snap" mode, i.e. return the first tick of every series and stop once all
            series have ticked
        :param collect: Whether to collect all the data in a single series before publishing.
            Improves the performance for publishing asynchronous time series, and can handle the case of duplicate
            timestamps. However, it is slower for synchronous time series.
        :returns: A new DataFrame with an additional index level containing the datetimes of the ticked values.
            Columns of type TsDtype are replaced by their ticked values. Values from static columns (not of TsDtype) are
            repeated for each timestamp.
        """
        if collect:
            orig_frame = self.ts_frame().copy()
            if orig_frame.empty:
                idx = _empty_index_like(self._obj.index, snap)
                df_ts = pd.DataFrame({}, index=idx)
            else:
                for col, s in orig_frame.items():
                    orig_frame[col] = s.fillna(csp.null_ts(s.dtype.subtype))

                def _collect(data):
                    return _collect_numpy(data.tolist(), dim=len(orig_frame.columns))

                collected = orig_frame.apply(_collect, axis=1).astype(TsDtype(object))
                series = collected.csp.run(starttime, endtime, realtime, tick_count, tick_history, snap)
                df_ts = pd.DataFrame(series.values.tolist(), columns=orig_frame.columns, index=series.index)
                for col, s in df_ts.items():
                    df_ts[col] = s.astype(np.dtype(self._obj[col].dtype.subtype))
        else:
            rawdata = _run(self._eval_graph, starttime, endtime, realtime, tick_count, tick_history, snap)
            data = {}
            for col, coldata in rawdata.items():
                data[col] = self._obj[col].csp._parse_results(coldata, col, snap)
            if data:
                idx = None
            else:
                idx = _empty_index_like(self._obj.index, snap)
            duplicate_cols = [col for col, series in data.items() if series.index.has_duplicates]
            try:
                df_ts = pd.DataFrame(data, index=idx)
            except ValueError as e:
                if duplicate_cols and "non-unique" in str(e):
                    raise ValueError(
                        f"Columns {duplicate_cols} with different indices and duplicate timestamps cannot be "
                        f"combined into a single data frame. Try running with collect=True."
                    )
                # Re-raise other exceptions
                raise e

        # Join in static data again.
        if not snap:
            df_ts.index = df_ts.index.rename("_timestamp", level=-1)
            df_ts = df_ts.reset_index(-1)
        df_static = self.static_frame()
        df = df_ts.join(df_static, how="left")

        if not snap:
            # Dtypes can get lost if empty, so handle this case explicitly
            if df.empty:
                idx = _empty_index_like(df.index, snap)
                df.index = idx
            else:
                df = df.set_index("_timestamp", append=True)
                df.index = df.index.rename(None, level=-1)
        else:
            # Fix lost Dtypes from empty join
            if df.empty:
                idx = _empty_index_like(self._obj.index, snap)
                df.index = idx

        # Impose column order of original frame
        df = df.reindex(columns=self._obj.columns)
        return df

    def snap(self, timeout: timedelta = timedelta(seconds=10), starttime: datetime = None):
        """ "Snap" a graph containing the edges in the series.

        :param timeout: How long to wait for all the elements to tick before stopping the circuit
        :param starttime: The start-time for the snap. Could be a time in the past (for snapping historical data) or
            a time in the future (to schedule a snap at, i.e. the next minute)
        :returns: A new DataFrame with the same indices where every Edge is replaced by it's snapped value.
        """
        starttime = starttime or datetime.utcnow()
        return self.run(starttime, timeout, True, tick_count=1, snap=True)

    def sample(self, trigger: Union[timedelta, np.timedelta64, pd.Timedelta, ts[object]], inplace: bool = False):
        """Sample all non-null elements of the series at a regular timedelta interval or at some custom trigger

        :param trigger: Either a sampling interval or a time series representing the trigger ticks
        :param inplace: Whether the sample is applied to the series in place, or whether a new frame is returned
        :return: A TsDtype series containing the outputs of csp.sample applied to each non-null element of the series
        """
        data = {}
        for col, series in self._obj.items():
            if is_csp_type(series):
                data[col] = series.csp.sample(trigger)

        if inplace:
            df = self._obj
        else:
            df = self._obj.copy()

        for col, series in data.items():
            df[col] = series

        if not inplace:
            return df

    def _collect(self, data, struct_type):
        data = data.copy()
        for field, typ in struct_type.metadata().items():
            if issubclass(typ, csp.Struct):
                data[field] = self._collect(data[field], typ)
        # Now apply collect to data
        df = pd.DataFrame(data)

        def row_collect(row):
            # Need to convert the "row" (ndarray of objects) into a dict (while dropping missing values)
            data = {k: v for k, v in row.items() if isinstance(v, Edge)}
            if not data:
                return csp.null_ts(struct_type)
            return struct_type.collectts(**data)

        return df.apply(row_collect, axis=1).astype(TsDtype(struct_type))

    def collect(self, columns=None, struct_type=None, delim=" "):
        """Collects multiple ts columns of a frame into a series of ts or structs.

        :param columns: An optional subset of columns to map to the struct. If not provided, all columns
            will be mapped
        :param struct_type: A struct type to map the columns to
        :param delim: The delimiter between field names on nested fields
        :returns: A pd.DataFrame of TsDtype columns corresponding to the Struct fields.
        """
        metadata = {}
        data = {}
        defaults = {}
        columns = columns or self.ts_frame().columns
        for col in columns:
            parts = col.split(delim)
            metatree = metadata
            datatree = data
            defaultstree = defaults
            for part in parts[:-1]:
                metatree = metatree.setdefault(part, {})
                datatree = datatree.setdefault(part, {})
                defaultstree = defaultstree.setdefault(part, {})
            else:
                metatree[parts[-1]] = self._obj[col].dtype.subtype
            datatree[parts[-1]] = self._obj[col]

        if not struct_type:
            struct_type = define_nested_struct("_C", metadata, defaults)

        if not data:
            return csp.null_ts(struct_type)

        return self._collect(data, struct_type)

    def show_graph(self):
        """Show the graph corresponding to the evaluation of all the edges.
        For large series, this may be very large, so it may be helpful to call .head() first.
        """
        from PIL import Image

        import csp.showgraph

        buffer = csp.showgraph.generate_graph(self._eval_graph, "png")
        return Image.open(buffer)


@register_dataframe_accessor("to_csp")
class ToCspFrameAccessor(object):
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def __call__(self, columns=None, agg="last", drop_na=False):
        """Return an Edge from the series
        :param columns: The set of columns to turn into csp edges. If none are provided, all will be. Other columns
            in the frame are aggregated with a groupby
        :param agg: The type of aggregation to use for static columns. Can be a string or a function, as long as it works
            with pd.DataFrameGroupBy.agg (i.e. pd.Series.mode)
        :param drop_na: If True, will drop na types from series of float and object types. For all other series types,
            na's are dropped by default for csp's type consistency.
        """

        outputs = {}
        for col, series in self._obj.items():
            if columns is None or col in columns:
                outputs[col] = series.to_csp(drop_na)
            else:
                outputs[col] = series.to_csp._static_agg(agg)
        if self._obj.index.nlevels == 1:
            return outputs
        else:
            return pd.DataFrame(outputs, columns=self._obj.columns)


@csp.node
def _collect_numpy(x: List[ts[object]], dim: int) -> ts[object]:
    with csp.state():
        s_array = np.array([np.nan for _ in range(dim)], dtype=object)

    if csp.ticked(x):
        out = s_array.copy()
        for idx, v in x.tickeditems():
            out[idx] = v
        return out
