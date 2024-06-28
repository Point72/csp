import sys
from datetime import datetime, timedelta

import csp
from csp.adapters.websocket import RawTextMessageMapper, Status, WebsocketAdapterManager


@csp.node
def send_message_on_connection(s: csp.ts[Status], msg: str) -> csp.ts[str]:
    # once the websocket adapter has connected, send a message to the server
    if csp.ticked(s) and s.status_code == 0:
        return msg


@csp.graph
def g(uri: str):
    print("Trying to connect to", uri)
    ws = WebsocketAdapterManager(uri)
    msgs = ws.subscribe(str, RawTextMessageMapper())
    status = ws.status()
    ws.send(send_message_on_connection(status, "Hello, World!"))
    csp.print("status", status)
    csp.print("received", msgs)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: e1_websocket_client <uri>")
        sys.exit(1)

    csp.run(
        g,
        starttime=datetime.utcnow(),
        endtime=timedelta(minutes=1),
        realtime=True,
        uri=sys.argv[1],
    )
