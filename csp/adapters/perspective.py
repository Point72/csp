import threading
from datetime import timedelta
from typing import Dict, Optional, Union

from perspective import Table as Table_, View as View_

import csp
from csp import ts
from csp.impl.perspective_common import (
    date_to_perspective,
    datetime_to_perspective,
    is_perspective3,
    perspective_type_map,
)
from csp.impl.wiring.delayed_node import DelayedNodeWrapperDef

try:
    import tornado.ioloop
    import tornado.web
    import tornado.websocket
except ImportError:
    raise ImportError("perspective adapter requires tornado package")


_PERSPECTIVE_3 = is_perspective3()
if _PERSPECTIVE_3:
    from perspective import Server
else:
    from perspective import PerspectiveManager


# Run perspective update in a separate tornado loop
def perspective_thread(client):
    loop = tornado.ioloop.IOLoop()
    client.set_loop_callback(loop.add_callback)
    loop.start()


@csp.node
def _apply_updates(table: object, data: {str: ts[object]}, throttle: timedelta):
    with csp.alarms():
        alarm = csp.alarm(bool)

    with csp.state():
        s_buffer = []
        s_datetime_cols = set()
        s_date_cols = set()

    with csp.start():
        csp.schedule_alarm(alarm, throttle, True)
        if _PERSPECTIVE_3:
            s_datetime_cols = set([c for c, t in table.schema().items() if t == "datetime"])
            s_date_cols = set([c for c, t in table.schema().items() if t == "date"])

    if csp.ticked(data):
        row = dict(data.tickeditems())
        if _PERSPECTIVE_3:
            for col, value in row.items():
                if col in s_datetime_cols:
                    row[col] = datetime_to_perspective(row[col])
                if col in s_date_cols:
                    row[col] = date_to_perspective(row[col])

        s_buffer.append(row)

    if csp.ticked(alarm):
        if len(s_buffer) > 0:
            table.update(s_buffer)
            s_buffer = []

        csp.schedule_alarm(alarm, throttle, True)


@csp.node
def _launch_application(port: int, server: object, stub: ts[object]):
    with csp.state():
        s_app = None
        s_ioloop = None
        s_iothread = None

    with csp.start():
        if _PERSPECTIVE_3:
            from perspective.handlers.tornado import PerspectiveTornadoHandler

            handler_args = {"perspective_server": server, "check_origin": True}
        else:
            from perspective import PerspectiveTornadoHandler

            handler_args = {"manager": server, "check_origin": True}
        s_app = tornado.web.Application(
            [
                # create a websocket endpoint that the client Javascript can access
                (r"/websocket", PerspectiveTornadoHandler, handler_args)
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
        if _PERSPECTIVE_3:
            server = Server()
            client = server.new_local_client()
            thread = threading.Thread(target=perspective_thread, kwargs=dict(client=client))
        else:
            from perspective import set_threadpool_size

            set_threadpool_size(self._threadpool_size)
            manager = PerspectiveManager()
            thread = threading.Thread(target=perspective_thread, kwargs=dict(manager=manager))
        thread.daemon = True
        thread.start()

        for table_name, table in self._tables.items():
            schema = {
                k: v.tstype.typ if not issubclass(v.tstype.typ, csp.Enum) else str for k, v in table.columns.items()
            }
            if _PERSPECTIVE_3:
                psp_type_map = perspective_type_map()
                schema = {col: psp_type_map.get(typ, typ) for col, typ in schema.items()}
                ptable = client.table(schema, name=table_name, limit=table.limit, index=table.index)
            else:
                ptable = Table(schema, limit=table.limit, index=table.index)
                manager.host_table(table_name, ptable)

            _apply_updates(ptable, table.columns, self._throttle)

        if _PERSPECTIVE_3:
            _launch_application(self._port, server, csp.const("stub"))
        else:
            _launch_application(self._port, manager, csp.const("stub"))
