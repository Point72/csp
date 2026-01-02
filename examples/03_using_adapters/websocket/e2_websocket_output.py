import math
from datetime import datetime, timedelta

import csp
from csp import ts
from csp.adapters.websocket import WebsocketTableAdapter
from csp.utils.datetime import utc_now

""" To view the output see sample html code below """


class MyData(csp.Struct):
    key: int
    angle: float
    radians: float
    sin: float
    timestamp: datetime


@csp.node
def sin(radians: ts[float]) -> ts[float]:
    if csp.ticked(radians):
        return math.sin(radians)


@csp.node
def times(timer: ts[bool]) -> ts[datetime]:
    if csp.ticked(timer):
        return csp.now()


@csp.graph
def my_graph(port: int, num_keys: int):
    snap = csp.timer(timedelta(seconds=0.25))
    angle = csp.count(snap)

    all_structs = []
    for key in range(1, num_keys + 1):
        delay = 10.0 * (key / float(num_keys))
        delayed_angle = csp.delay(angle, timedelta(seconds=delay))
        r = delayed_angle / math.pi
        s = sin(r)

        data = MyData.fromts(
            key=csp.const(key), angle=csp.cast_int_to_float(angle), radians=r, sin=s, timestamp=times(snap)
        )
        all_structs.append(data)

    data = csp.flatten(all_structs)
    adapter = WebsocketTableAdapter(port)

    table = adapter.create_table("table", index="key")
    table.publish(data)

    csp.print("data", data)


port = 7677
num_keys = 10


def main():
    csp.run(my_graph, port, num_keys, starttime=utc_now(), endtime=timedelta(seconds=360), realtime=True)


""" Sample html to view the data.  Note to put your machine name on the websocket line below
<html>
<head></head>
<body>
  <script>
    async function main() {

      let response = await fetch("http://server:7677/tables");
      data = await response.json();

      let table = data.tables[0];
      let ws = new WebSocket(table.sub);

      ws.onmessage = (event) => {
          let msg = JSON.parse(event.data);
          console.log('msg', msg.data);
      }
    }

    main();

  </script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
