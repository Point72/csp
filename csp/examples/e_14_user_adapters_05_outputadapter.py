"""
This is a simple example to demonstrate a basic output adapter. It also shows the (better) way to implement this
functionality with a single node.
"""

from csp.impl.outputadapter import OutputAdapter
from csp.impl.wiring import py_output_adapter_def
from csp import ts
import csp
from json import dumps
from datetime import datetime, timedelta


class MyBufferWriterAdapterImpl(OutputAdapter):
    def __init__(self, output_buffer):
        super().__init__()
        self.input_buffer = []
        self.output_buffer = output_buffer

    def start(self):
        # do this in the `start` to demonstrate opening
        # access to a resource at graph start
        self.output_buffer.clear()

    def stop(self):
        # do this in the `end` to demonstrate closing
        # access to a resource at graph stop
        data = dumps(self.input_buffer)
        self.output_buffer.append(data)

    def on_tick(self, time, value):
        self.input_buffer.append(value)


MyBufferWriterAdapter = py_output_adapter_def(
    name='MyBufferWriterAdapter',
    adapterimpl=MyBufferWriterAdapterImpl,
    input=ts['T'],
    output_buffer=list,
)


output_buffer = []


@csp.graph
def my_graph():
    data = [
        {"a": 1, "b": 2, "c": 3},
        {"a": 1, "b": 2, "c": 3},
        {"a": 1, "b": 2, "c": 3},
    ]

    curve = csp.curve(data=[(timedelta(seconds=1), d) for d in data], typ=object)

    csp.print("writing data to buffer", curve)

    MyBufferWriterAdapter(curve, output_buffer=output_buffer)


if __name__ == "__main__":
    csp.run(
        my_graph,
        starttime=datetime.utcnow(),
        endtime=timedelta(seconds=3),
        realtime=True,
    )
    print("output buffer: {}".format(output_buffer))
