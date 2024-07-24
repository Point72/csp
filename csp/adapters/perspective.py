import threading
from datetime import timedelta
from typing import Dict, Optional, Union

import csp
from csp import ts
from csp.impl.wiring.delayed_node import DelayedNodeWrapperDef

try:
    import tornado.ioloop
    import tornado.web
    import tornado.websocket
except ImportError:
    raise ImportError("perspective adapter requires tornado package")


try:
    from perspective import PerspectiveManager, Table as Table_, View as View_, __version__, set_threadpool_size

    MAJOR, MINOR, PATCH = map(int, __version__.split("."))
    if (MAJOR, MINOR, PATCH) < (0, 6, 2):
        raise ImportError("perspective adapter requires 0.6.2 or greater of the perspective-python package")
except ImportError:
    raise ImportError("perspective adapter requires 0.6.2 or greater of the perspective-python package")


# Run perspective update in a separate tornado loop
def perspective_thread(manager):
    loop = tornado.ioloop.IOLoop()
    manager.set_loop_callback(loop.add_callback)
    loop.start()


@csp.node
def _apply_updates(table: object, data: {str: ts[object]}, throttle: timedelta):
    with csp.alarms():
        alarm = csp.alarm(bool)

    with csp.state():
        s_buffer = []

    with csp.start():
        csp.schedule_alarm(alarm, throttle, True)

    if csp.ticked(data):
        s_buffer.append(dict(data.tickeditems()))

    if csp.ticked(alarm):
        if len(s_buffer) > 0:
            table.update(s_buffer)
            s_buffer = []

        csp.schedule_alarm(alarm, throttle, True)


@csp.node
def _launch_application(port: int, manager: object, stub: ts[object]):
    with csp.state():
        s_app = None
        s_ioloop = None
        s_iothread = None

    with csp.start():
        from perspective import PerspectiveTornadoHandler

        s_app = tornado.web.Application(
            [
                # create a websocket endpoint that the client Javascript can access
                (r"/websocket", PerspectiveTornadoHandler, {"manager": manager, "check_origin": True})
            ],
            websocket_ping_interval=15,
        )
        s_app.listen(port)
        s_ioloop = tornado.ioloop.IOLoop.current()
        s_iothread = threading.Thread(target=s_ioloop.start)
        s_iothread.start()

    with csp.stop():
        if s_ioloop:
            s_ioloop.add_callback(s_ioloop.stop)
            if s_iothread:
                s_iothread.join()


class View(View_):
    def __init__(self, Table, **kwargs):
        self._start_row = -1
        self._end_row = -1
        super().__init__(Table, **kwargs)

    def to_arrow(self, **kwargs):
        """Override parent class to_arrow to cache requested bounds"""
        self._start_row = kwargs.get("start_row", -1)
        self._end_row = kwargs.get("end_row", -1)
        return super().to_arrow(**kwargs)

    def _get_row_delta(self):
        """Override parent to send back full data-range being viewed"""
        return self.to_arrow(start_row=self._start_row, end_row=self._end_row)


class Table(Table_):
    def view(
        self,
        columns=None,
        group_by=None,
        split_by=None,
        aggregates=None,
        sort=None,
        filter=None,
        computed_columns=None,
    ):
        self._state_manager.call_process(self._table.get_id())

        config = {}
        if columns is None:
            config["columns"] = self.columns()
            if computed_columns is not None:
                # append all computed columns if columns are not specified
                for col in computed_columns:
                    config["columns"].append(col["column"])
        else:
            config["columns"] = columns
        if group_by is not None:
            config["group_by"] = group_by
        if split_by is not None:
            config["split_by"] = split_by
        if aggregates is not None:
            config["aggregates"] = aggregates
        if sort is not None:
            config["sort"] = sort
        if filter is not None:
            config["filter"] = filter
        if computed_columns is not None:
            config["computed_columns"] = computed_columns

        view = View(self, **config)
        self._views.append(view._name)
        return view


class PerspectiveTableAdapter:
    """dont create these directly, use PerspectiveAdapter"""

    def __init__(self, name, limit, index):
        self.name = name
        self.limit = limit
        self.index = index
        self.columns = {}

    def publish(self, value: ts[object], field_map: Union[Dict[str, str], str, None] = None):
        """
        :param value - timeseries to publish onto this table
        :param field_map: if publishing structs, a dictionary of struct field -> perspective fieldname ( if None will pass struct fields as is )
                          if publishing a single field, then string name of the destination column
        """
        if issubclass(value.tstype.typ, csp.Struct):
            self._publish_struct(value, field_map)
        else:
            if not isinstance(field_map, str):
                raise TypeError("Expected type str for field_map on single column publish, got %s" % type(field_map))
            self._publish_field(value, field_map)

    def _publish_struct(self, value: ts[csp.Struct], field_map: Optional[Dict[str, str]]):
        field_map = field_map or {k: k for k in value.tstype.typ.metadata()}
        for k, v in field_map.items():
            self._publish_field(getattr(value, k), v)

    def _publish_field(self, value: ts[object], column_name: str):
        if column_name in self.columns:
            raise KeyError(f"Trying to add column {column_name} more than once")
        if issubclass(value.tstype.typ, csp.Struct):
            raise NotImplementedError(f"Publishing Struct field {value.tstype.typ} in perspective is not yet supported")
        else:
            self.columns[column_name] = value


class PerspectiveAdapter(DelayedNodeWrapperDef):
    def __init__(self, port, threadpool_size=2, throttle=timedelta(seconds=1)):
        super().__init__()
        self._port = port
        self._threadpool_size = threadpool_size
        self._throttle = throttle
        self._tables = {}

    def copy(self):
        res = PerspectiveAdapter(self._port, self._threadpool_size, self._throttle)
        res._tables.update(self._tables)
        return res

    def create_table(self, name, limit=None, index=None):
        if name in self._tables:
            raise ValueError(f"Table {name} already exists")

        table = self._tables[name] = PerspectiveTableAdapter(name, limit, index)
        return table

    def _instantiate(self):
        set_threadpool_size(self._threadpool_size)

        manager = PerspectiveManager()

        thread = threading.Thread(target=perspective_thread, kwargs=dict(manager=manager))
        thread.daemon = True
        thread.start()

        for table_name, table in self._tables.items():
            schema = {
                k: v.tstype.typ if not issubclass(v.tstype.typ, csp.Enum) else str for k, v in table.columns.items()
            }
            ptable = Table(schema, limit=table.limit, index=table.index)
            manager.host_table(table_name, ptable)

            _apply_updates(ptable, table.columns, self._throttle)

        _launch_application(self._port, manager, csp.const("stub"))
