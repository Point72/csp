import os
import threading
import unittest
from datetime import datetime, timedelta
from typing import List

import pytz
import tornado.ioloop
import tornado.web
import tornado.websocket
from tornado.testing import bind_unused_port

import csp
from csp import ts
from csp.adapters.websocket import JSONTextMessageMapper, RawTextMessageMapper, Status, WebsocketAdapterManager


class EchoWebsocketHandler(tornado.websocket.WebSocketHandler):
    def on_message(self, msg):
        return self.write_message(msg)


class TestWebsocket(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = tornado.web.Application([(r"/", EchoWebsocketHandler)])
        sock, port = bind_unused_port()
        sock.close()
        cls.port = port
        cls.app.listen(port)
        cls.io_loop = tornado.ioloop.IOLoop.current()
        cls.io_thread = threading.Thread(target=cls.io_loop.start)
        cls.io_thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.io_loop.add_callback(cls.io_loop.stop)
        if cls.io_thread:
            cls.io_thread.join()

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

    def test_unkown_host_graceful_shutdown(self):
        @csp.graph
        def g():
            ws = WebsocketAdapterManager("wss://localhost/")
            assert ws._properties["port"] == "443"
            csp.stop_engine(ws.status())

        csp.run(g, starttime=datetime.now(pytz.UTC), realtime=True)

    def test_send_recv_burst_json(self):
        class MsgStruct(csp.Struct):
            a: int
            b: str

        @csp.node
        def send_msg_on_open(status: ts[Status]) -> ts[str]:
            if csp.ticked(status):
                return MsgStruct(a=1234, b="im a string").to_json()

        @csp.node
        def my_edge_that_handles_burst(objs: ts[List[MsgStruct]]) -> ts[bool]:
            if csp.ticked(objs):
                return True

        @csp.graph
        def g():
            ws = WebsocketAdapterManager(f"ws://localhost:{self.port}/")
            status = ws.status()
            ws.send(send_msg_on_open(status))
            recv = ws.subscribe(MsgStruct, JSONTextMessageMapper(), push_mode=csp.PushMode.BURST)
            _ = my_edge_that_handles_burst(recv)
            csp.add_graph_output("recv", recv)
            csp.stop_engine(recv)

        msgs = csp.run(g, starttime=datetime.now(pytz.UTC), realtime=True)
        obj = msgs["recv"][0][1]
        assert isinstance(obj, list)
        innerObj = obj[0]
        assert innerObj.a == 1234
        assert innerObj.b == "im a string"
