import os
import pytest
import pytz
import threading
import tornado.ioloop
import tornado.web
import tornado.websocket
from contextlib import contextmanager
from datetime import datetime, timedelta
from tornado.testing import bind_unused_port
from typing import List

import csp
from csp import ts
from csp.adapters.websocket import (
    ActionType,
    ConnectionRequest,
    JSONTextMessageMapper,
    RawTextMessageMapper,
    Status,
    WebsocketAdapterManager,
    WebsocketHeaderUpdate,
    WebsocketStatus,
)


class EchoWebsocketHandler(tornado.websocket.WebSocketHandler):
    def on_message(self, msg):
        # Carve-out to allow inspecting the headers
        if msg == "header1":
            msg = self.request.headers.get(msg, "")
        elif not isinstance(msg, str) and msg.decode("utf-8") == "header1":
            # Need this for bytes
            msg = self.request.headers.get("header1", "")
        return self.write_message(msg)


@contextmanager
def create_tornado_server(port: int = None):
    """Base context manager for creating a Tornado server in a thread

    Args:
        port: Optional port number. If None, an unused port will be chosen.

    Returns:
        Tuple containing (io_loop, app, io_thread, port)
    """
    ready_event = threading.Event()
    io_loop = None
    app = None
    io_thread = None

    # Get an unused port if none specified
    if port is None:
        sock, port = bind_unused_port()
        sock.close()

    def run_io_loop():
        nonlocal io_loop, app
        io_loop = tornado.ioloop.IOLoop()
        app = tornado.web.Application([(r"/", EchoWebsocketHandler)])
        try:
            app.listen(port)
            ready_event.set()
            io_loop.start()
        except Exception as e:
            ready_event.set()  # Ensure we don't hang in case of error
            raise

    io_thread = threading.Thread(target=run_io_loop)
    io_thread.start()
    ready_event.wait()

    try:
        yield io_loop, app, io_thread, port
    finally:
        io_loop.add_callback(io_loop.stop)
        if io_thread:
            io_thread.join(timeout=5)
            if io_thread.is_alive():
                raise RuntimeError("IOLoop failed to stop")


@contextmanager
def tornado_server():
    """Simplified context manager that uses the base implementation with dynamic port"""
    with create_tornado_server() as (_io_loop, _app, _io_thread, port):
        yield port


class TestWebsocket:
    @pytest.fixture(scope="class", autouse=True)
    def setup_tornado(self, request):
        with create_tornado_server() as (io_loop, app, io_thread, port):
            request.cls.io_loop = io_loop
            request.cls.app = app
            request.cls.io_thread = io_thread
            request.cls.port = port  # Make the port available to tests
            yield

    def test_send_recv_msg(self):
        @csp.node
        def send_msg_on_open(status: ts[Status]) -> ts[str]:
            if csp.ticked(status):
                return "Hello, World!"

        @csp.graph
        def g():
            ws = WebsocketAdapterManager(f"ws://localhost:{self.port}/")
            status = ws.status()
            ws.send(send_msg_on_open(status))
            recv = ws.subscribe(str, RawTextMessageMapper())

            csp.add_graph_output("recv", recv)
            csp.stop_engine(recv)

        msgs = csp.run(g, starttime=datetime.now(pytz.UTC), realtime=True)
        assert msgs["recv"][0][1] == "Hello, World!"

    @pytest.mark.parametrize("binary", [False, True])
    def test_headers(self, binary):
        @csp.graph
        def g(dynamic: bool):
            if dynamic:
                ws = WebsocketAdapterManager(dynamic=True, binary=binary)
                conn_request1 = csp.const(
                    [
                        ConnectionRequest(
                            uri=f"ws://localhost:{self.port}/",
                            on_connect_payload="header1",
                            headers={"header1": "value1"},
                        )
                    ]
                )
                conn_request2 = csp.const(
                    [ConnectionRequest(uri=f"ws://localhost:{self.port}/", action=ActionType.DISCONNECT)],
                    delay=timedelta(milliseconds=100),
                )
                conn_request3 = csp.const(
                    [
                        ConnectionRequest(
                            uri=f"ws://localhost:{self.port}/",
                            on_connect_payload="header1",
                            headers={"header1": "value2"},
                        )
                    ],
                    delay=timedelta(milliseconds=150),
                )
                conn_request4 = csp.const(
                    [ConnectionRequest(uri=f"ws://localhost:{self.port}/", action=ActionType.PING)],
                    delay=timedelta(milliseconds=151),
                )
                conn_request5 = csp.const(
                    [
                        ConnectionRequest(
                            uri=f"ws://localhost:{self.port}/",
                            on_connect_payload="header1",
                            headers={"header1": "value2"},
                        )
                    ],
                    delay=timedelta(milliseconds=200),
                )
                conn_req = csp.flatten([conn_request1, conn_request2, conn_request3, conn_request4, conn_request5])
                status = ws.status()
                csp.add_graph_output("status", status)
                recv = ws.subscribe(str, RawTextMessageMapper(), connection_request=conn_req)
                csp.add_graph_output("recv", recv)
                stop = csp.filter(csp.count(recv) == 2, recv)
                csp.stop_engine(stop)

            if not dynamic:
                ws = WebsocketAdapterManager(f"ws://localhost:{self.port}/", headers={"header1": "value1"})
                status = ws.status()
                send_msg = csp.sample(status, csp.const("header1"))
                to_send = csp.merge(send_msg, csp.const("header1", delay=timedelta(milliseconds=100)))
                ws.send(to_send)
                recv = ws.subscribe(str, RawTextMessageMapper())

                header_update = csp.const(
                    [WebsocketHeaderUpdate(key="header1", value="value2")], delay=timedelta(milliseconds=50)
                )
                ws.update_headers(header_update)
                status = ws.status()
                csp.add_graph_output("status", status)

                csp.add_graph_output("recv", recv)
                csp.stop_engine(recv)

    @pytest.mark.parametrize("send_payload_subscribe", [True, False])
    def test_send_recv_json_dynamic_on_connect_payload(self, send_payload_subscribe):
        class MsgStruct(csp.Struct):
            a: int
            b: str

        @csp.graph
        def g():
            ws = WebsocketAdapterManager(dynamic=True)
            conn_request = ConnectionRequest(
                uri=f"ws://localhost:{self.port}/",
                action=ActionType.CONNECT,
                on_connect_payload=MsgStruct(a=1234, b="im a string").to_json(),
            )
            if not send_payload_subscribe:
                # We send payload via the dummy send function
                # The 'on_connect_payload sends the result
                ws.send(csp.null_ts(object), connection_request=csp.const([conn_request]))
            subscribe_connection_request = (
                [ConnectionRequest(uri=f"ws://localhost:{self.port}/", action=ActionType.CONNECT)]
                if not send_payload_subscribe
                else [conn_request]
            )
            recv = ws.subscribe(
                MsgStruct, JSONTextMessageMapper(), connection_request=csp.const(subscribe_connection_request)
            )

            csp.add_graph_output("recv", recv)
            csp.stop_engine(recv)

        msgs = csp.run(g, starttime=datetime.now(pytz.UTC), realtime=True)
        obj = msgs["recv"][0][1]
        assert obj.uri == f"ws://localhost:{self.port}/"
        true_obj = obj.msg
        assert isinstance(true_obj, MsgStruct)
        assert true_obj.a == 1234
        assert true_obj.b == "im a string"

    def test_send_recv_json(self):
        class MsgStruct(csp.Struct):
            a: int
            b: str

        @csp.node
        def send_msg_on_open(status: ts[Status]) -> ts[str]:
            if csp.ticked(status):
                return MsgStruct(a=1234, b="im a string").to_json()

        @csp.graph
        def g():
            ws = WebsocketAdapterManager(f"ws://localhost:{self.port}/")
            status = ws.status()
            ws.send(send_msg_on_open(status))
            recv = ws.subscribe(MsgStruct, JSONTextMessageMapper())

            csp.add_graph_output("recv", recv)
            csp.stop_engine(recv)

        msgs = csp.run(g, starttime=datetime.now(pytz.UTC), realtime=True)
        obj = msgs["recv"][0][1]
        assert isinstance(obj, MsgStruct)
        assert obj.a == 1234
        assert obj.b == "im a string"

    def test_send_multiple_and_recv_msgs(self):
        @csp.node
        def send_msg_on_open(status: ts[Status], idx: int) -> ts[str]:
            if csp.ticked(status):
                return f"Hello, World! {idx}"

        @csp.node
        def stop_on_all_or_timeout(msgs: ts[str], l: int = 50) -> ts[bool]:
            with csp.alarms():
                a_timeout: ts[bool] = csp.alarm(bool)

            with csp.state():
                s_ctr = 0

            with csp.start():
                csp.schedule_alarm(a_timeout, timedelta(seconds=5), False)

            if csp.ticked(msgs):
                s_ctr += 1

            if csp.ticked(a_timeout) or (csp.ticked(msgs) and s_ctr == l):
                return True

        @csp.graph
        def g(n: int):
            ws = WebsocketAdapterManager(f"ws://localhost:{self.port}/")
            status = ws.status()
            ws.send(csp.flatten([send_msg_on_open(status, i) for i in range(n)]))
            recv = ws.subscribe(str, RawTextMessageMapper())

            csp.add_graph_output("recv", recv)
            csp.stop_engine(stop_on_all_or_timeout(recv, n))

        n = 100
        msgs = csp.run(g, n, starttime=datetime.now(pytz.UTC), realtime=True)
        assert len(msgs["recv"]) == n
        assert msgs["recv"][0][1] != msgs["recv"][-1][1]

    def test_send_multiple_and_recv_msgs_dynamic(self):
        @csp.graph
        def g():
            ws = WebsocketAdapterManager(dynamic=True)
            conn_request = csp.const(
                [
                    ConnectionRequest(
                        uri=f"ws://localhost:{self.port}/",
                        action=ActionType.CONNECT,
                    )
                ]
            )
            val = csp.curve(int, [(timedelta(milliseconds=100), 0), (timedelta(milliseconds=500), 1)])
            hello = csp.apply(val, lambda x: f"hi world{x}", str)
            delayed_conn_req = csp.delay(conn_request, delay=timedelta(milliseconds=300))

            # We connect immediately and send out the hello message
            ws.send(hello, connection_request=conn_request)

            recv = ws.subscribe(str, RawTextMessageMapper(), connection_request=delayed_conn_req)
            # This call connects first
            recv2 = ws.subscribe(str, RawTextMessageMapper(), connection_request=conn_request)

            merged = csp.flatten([recv, recv2])
            csp.add_graph_output("recv", merged.msg)

            stop = csp.filter(csp.count(merged) == 3, merged)
            csp.stop_engine(stop)

        msgs = csp.run(g, starttime=datetime.now(pytz.UTC), endtime=timedelta(seconds=1), realtime=True)
        assert len(msgs["recv"]) == 3
        # the first message sent out, only the second subscribe call picks this up
        assert msgs["recv"][0][1] == "hi world0"
        # Both the subscribe calls receive this message
        assert msgs["recv"][1][1] == "hi world1"
        assert msgs["recv"][2][1] == "hi world1"

    @pytest.mark.parametrize("reconnect_immeditately", [False, True])
    def test_dynamic_disconnect_connect_pruned_subscribe(self, reconnect_immeditately):
        @csp.graph
        def g():
            ws = WebsocketAdapterManager(dynamic=True)

            if reconnect_immeditately:
                disconnect_reqs = [
                    ConnectionRequest(uri=f"ws://localhost:{self.port}/", action=ActionType.DISCONNECT),
                    ConnectionRequest(uri=f"ws://localhost:{self.port}/"),
                ]
            else:
                disconnect_reqs = [ConnectionRequest(uri=f"ws://localhost:{self.port}/", action=ActionType.DISCONNECT)]
            conn_request = csp.curve(
                List[ConnectionRequest],
                [
                    (timedelta(), [ConnectionRequest(uri=f"ws://localhost:{self.port}/")]),
                    (
                        timedelta(milliseconds=100),
                        disconnect_reqs,
                    ),
                    (
                        timedelta(milliseconds=700),
                        [
                            ConnectionRequest(
                                uri=f"ws://localhost:{self.port}/",
                                headers={"dummy_key": "dummy_value"},
                            ),
                        ],
                    ),
                ],
            )
            const_conn_request = csp.const([ConnectionRequest(uri=f"ws://localhost:{self.port}/")])
            val = csp.curve(int, [(timedelta(milliseconds=300, microseconds=1), 0), (timedelta(milliseconds=900), 1)])
            hello = csp.apply(val, lambda x: f"hi world{x}", str)

            # We connect immediately and send out the hello message
            ws.send(hello, connection_request=const_conn_request)

            recv = ws.subscribe(str, RawTextMessageMapper(), connection_request=conn_request)
            # This gets pruned by csp
            recv2 = ws.subscribe(str, RawTextMessageMapper(), connection_request=conn_request)
            recv3 = ws.subscribe(str, RawTextMessageMapper(), connection_request=const_conn_request)

            no_persist_conn = ConnectionRequest(
                uri=f"ws://localhost:{self.port}/", persistent=False, on_connect_payload="hi non-persistent world!"
            )
            recv4 = ws.subscribe(
                str,
                RawTextMessageMapper(),
                connection_request=csp.const([no_persist_conn], delay=timedelta(milliseconds=500)),
            )

            csp.add_graph_output("recv", recv)
            csp.add_graph_output("recv3", recv3)
            csp.add_graph_output("recv4", recv4)
            end = csp.filter(csp.count(recv3) == 3, recv3)
            csp.stop_engine(end)

        msgs = csp.run(g, starttime=datetime.now(pytz.UTC), endtime=timedelta(seconds=2), realtime=True)
        # Did not persist, so did not receive any messages
        assert len(msgs["recv4"]) == 0
        # Only the second message is received, since we disonnect before the first one is sent
        if not reconnect_immeditately:
            assert len(msgs["recv"]) == 1
            assert msgs["recv"][0][1].msg == "hi world1"
            assert msgs["recv"][0][1].uri == f"ws://localhost:{self.port}/"
        else:
            assert len(msgs["recv"]) == 3
            assert msgs["recv"][0][1].msg == "hi world0"
            assert msgs["recv"][0][1].uri == f"ws://localhost:{self.port}/"
            assert msgs["recv"][1][1].msg == "hi non-persistent world!"
            assert msgs["recv"][1][1].uri == f"ws://localhost:{self.port}/"
            assert msgs["recv"][2][1].msg == "hi world1"
            assert msgs["recv"][2][1].uri == f"ws://localhost:{self.port}/"

        # This subscribe call received all the messages
        assert len(msgs["recv3"]) == 3
        assert msgs["recv3"][0][1].msg == "hi world0"
        assert msgs["recv3"][0][1].uri == f"ws://localhost:{self.port}/"
        assert msgs["recv3"][1][1].msg == "hi non-persistent world!"
        assert msgs["recv3"][1][1].uri == f"ws://localhost:{self.port}/"
        assert msgs["recv3"][2][1].msg == "hi world1"
        assert msgs["recv3"][2][1].uri == f"ws://localhost:{self.port}/"

    def test_dynamic_pruned_subscribe(self):
        @csp.graph
        def g():
            ws = WebsocketAdapterManager(dynamic=True)
            conn_request = csp.const(
                [
                    ConnectionRequest(
                        uri=f"ws://localhost:{self.port}/",
                        action=ActionType.CONNECT,
                    )
                ]
            )
            val = csp.curve(int, [(timedelta(milliseconds=100), 0), (timedelta(milliseconds=600), 1)])
            hello = csp.apply(val, lambda x: f"hi world{x}", str)
            delayed_conn_req = csp.delay(conn_request, delay=timedelta(milliseconds=400))

            # We connect immediately and send out the hello message
            ws.send(hello, connection_request=conn_request)

            recv = ws.subscribe(str, RawTextMessageMapper(), connection_request=delayed_conn_req)
            # This gets pruned by csp
            recv2 = ws.subscribe(str, RawTextMessageMapper(), connection_request=conn_request)

            csp.add_graph_output("recv", recv)
            csp.stop_engine(recv)

        msgs = csp.run(g, starttime=datetime.now(pytz.UTC), endtime=timedelta(seconds=2), realtime=True)
        assert len(msgs["recv"]) == 1
        # Only the second message is received
        assert msgs["recv"][0][1].msg == "hi world1"
        assert msgs["recv"][0][1].uri == f"ws://localhost:{self.port}/"

    def test_dynamic_multiple_subscribers(self):
        @csp.node
        def send_on_status(status: ts[Status], uri: str, val: str) -> ts[str]:
            if csp.ticked(status):
                if uri in status.msg and status.status_code == WebsocketStatus.ACTIVE.value:
                    return val

        with tornado_server() as port2:  # Get a second dynamic port

            @csp.graph
            def g(use_on_connect_payload: bool):
                ws = WebsocketAdapterManager(dynamic=True)
                if use_on_connect_payload:
                    conn_request1 = csp.const(
                        [
                            ConnectionRequest(
                                uri=f"ws://localhost:{self.port}/", on_connect_payload="hey world from main"
                            )
                        ]
                    )
                    conn_request2 = csp.const(
                        [ConnectionRequest(uri=f"ws://localhost:{port2}/", on_connect_payload="hey world from second")]
                    )
                else:
                    conn_request1 = csp.const([ConnectionRequest(uri=f"ws://localhost:{self.port}/")])
                    conn_request2 = csp.const([ConnectionRequest(uri=f"ws://localhost:{port2}/")])
                    status = ws.status()
                    to_send = send_on_status(status, f"ws://localhost:{self.port}/", "hey world from main")
                    to_send2 = send_on_status(status, f"ws://localhost:{port2}/", "hey world from second")
                    ws.send(to_send, connection_request=conn_request1)
                    ws.send(to_send2, connection_request=conn_request2)

                recv = ws.subscribe(str, RawTextMessageMapper(), connection_request=conn_request1)
                recv2 = ws.subscribe(str, RawTextMessageMapper(), connection_request=conn_request2)

                csp.add_graph_output("recv", recv)
                csp.add_graph_output("recv2", recv2)

                merged = csp.flatten([recv, recv2])
                stop = csp.filter(csp.count(merged) == 2, merged)
                csp.stop_engine(stop)

            for use_on_connect_payload in [True, False]:
                msgs = csp.run(
                    g,
                    use_on_connect_payload,
                    starttime=datetime.now(pytz.UTC),
                    endtime=timedelta(seconds=5),
                    realtime=True,
                )
                assert len(msgs["recv"]) == 1
                assert msgs["recv"][0][1].msg == "hey world from main"
                assert msgs["recv"][0][1].uri == f"ws://localhost:{self.port}/"
                assert len(msgs["recv2"]) == 1
                assert msgs["recv2"][0][1].msg == "hey world from second"
                assert msgs["recv2"][0][1].uri == f"ws://localhost:{port2}/"

    @pytest.mark.parametrize("dynamic", [False, True])
    def test_send_recv_burst_json(self, dynamic):
        class MsgStruct(csp.Struct):
            a: int
            b: str

        @csp.node
        def my_edge_that_handles_burst(objs: ts[List[MsgStruct]]):
            if csp.ticked(objs):
                # Does nothing but makes sure it's not pruned
                ...

        @csp.graph
        def g():
            if dynamic:
                ws = WebsocketAdapterManager(dynamic=True)
                wrapped_recv = ws.subscribe(
                    MsgStruct,
                    JSONTextMessageMapper(),
                    push_mode=csp.PushMode.BURST,
                    connection_request=csp.const(
                        [
                            ConnectionRequest(
                                uri=f"ws://localhost:{self.port}/",
                                on_connect_payload=MsgStruct(a=1234, b="im a string").to_json(),
                            )
                        ]
                    ),
                )
                recv = csp.apply(wrapped_recv, lambda vals: [v.msg for v in vals], List[MsgStruct])
            else:
                ws = WebsocketAdapterManager(f"ws://localhost:{self.port}/")
                status = ws.status()
                ws.send(csp.apply(status, lambda _x: MsgStruct(a=1234, b="im a string").to_json(), str))
                recv = ws.subscribe(MsgStruct, JSONTextMessageMapper(), push_mode=csp.PushMode.BURST)

            my_edge_that_handles_burst(recv)
            csp.add_graph_output("recv", recv)
            csp.stop_engine(recv)

        msgs = csp.run(g, starttime=datetime.now(pytz.UTC), realtime=True)
        obj = msgs["recv"][0][1]
        assert isinstance(obj, list)
        innerObj = obj[0]
        assert innerObj.a == 1234
        assert innerObj.b == "im a string"

    def test_unkown_host_graceful_shutdown(self):
        @csp.graph
        def g():
            ws = WebsocketAdapterManager("wss://localhost/")
            # We need this since without any input or output
            # adapters, the websocket connection is not actually made.
            ws.send(csp.null_ts(str))
            assert ws._properties["port"] == "443"
            csp.stop_engine(ws.status())

        csp.run(g, starttime=datetime.now(pytz.UTC), realtime=True)

    def test_unkown_host_graceful_shutdown_slow(self):
        @csp.graph
        def g():
            ws = WebsocketAdapterManager("wss://localhost/")
            # We need this since without any input or output
            # adapters, the websocket connection is not actually made.
            ws.send(csp.null_ts(str))
            assert ws._properties["port"] == "443"
            stop_flag = csp.filter(csp.count(ws.status()) == 2, ws.status())
            csp.stop_engine(stop_flag)

        csp.run(g, starttime=datetime.now(pytz.UTC), realtime=True)
