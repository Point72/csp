from datetime import datetime

import csp
from csp.adapters.status import Status
from csp.adapters.utils import RawTextMessageMapper
from csp.adapters.websocket_client import WSClientAdapterManager


@csp.node
def on_active(s: csp.ts[Status]) -> csp.ts[str]:  # or csp.ts[bytes]
    if csp.ticked(s) and s.status_code == 0:
        return "my message"


@csp.graph
def g():
    adapter = WSClientAdapterManager(
        "ws://localhost:9001/",
        headers={
            "sec-websocket-key": "my-key",
        },
    )
    out = adapter.subscribe(str, RawTextMessageMapper())
    csp.print("ws_event", out)
    csp.print("status", adapter.status())

    # waits for the first status message then starts sending messages from the client to the server
    # will need to collect multiple sends if you have multiple output edges
    adapter.send(
        csp.unroll(
            csp.collect([on_active(adapter.status()), on_active(adapter.status())])
        )
    )


if __name__ == "__main__":
    csp.run(g, starttime=datetime.utcnow(), realtime=True)
