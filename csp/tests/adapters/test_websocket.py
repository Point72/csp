import os
import pytz
import threading
import unittest
from datetime import datetime

import csp
from csp import ts

if os.environ.get("CSP_TEST_WEBSOCKET"):
    import tornado.ioloop
    import tornado.web
    import tornado.websocket

    from csp.adapters.websocket import JSONTextMessageMapper, RawTextMessageMapper, Status, WebsocketAdapterManager

    class EchoWebsocketHandler(tornado.websocket.WebSocketHandler):
        def on_message(self, msg):
            return self.write_message(msg)


@unittest.skipIf(not os.environ.get("CSP_TEST_WEBSOCKET"), "Skipping websocket adapter tests")
class TestWebsocket(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = tornado.web.Application([(r"/ws", EchoWebsocketHandler)])
        cls.app.listen(8000)
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
            ws = WebsocketAdapterManager("ws://localhost:8000/ws")
            status = ws.status()
            ws.send(send_msg_on_open(status))
            recv = ws.subscribe(str, RawTextMessageMapper())

            csp.add_graph_output("recv", recv)
            csp.stop_engine(recv)

        msgs = csp.run(g, starttime=datetime.now(pytz.UTC), realtime=True)
        assert len(msgs) == 1
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
            ws = WebsocketAdapterManager("ws://localhost:8000/ws")
            status = ws.status()
            ws.send(send_msg_on_open(status))
            recv = ws.subscribe(MsgStruct, JSONTextMessageMapper())

            csp.add_graph_output("recv", recv)
            csp.stop_engine(recv)

        msgs = csp.run(g, starttime=datetime.now(pytz.UTC), realtime=True)
        assert len(msgs) == 1
        obj = msgs["recv"][0][1]
        assert isinstance(obj, MsgStruct)
        assert obj.a == 1234
        assert obj.b == "im a string"
