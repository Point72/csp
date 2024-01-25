from collections import deque, namedtuple
from io import BytesIO

from csp.impl.wiring.runtime import build_graph

NODE = namedtuple("NODE", ["name", "label", "color", "shape"])
EDGE = namedtuple("EDGE", ["start", "end"])


def _build_graphviz_graph(graph_func, *args, **kwargs):
    from graphviz import Digraph

    graph = build_graph(graph_func, *args, **kwargs)
    digraph = Digraph(strict=True)
    digraph.attr(rankdir="LR", size="150,150")

    rootnames = set()
    q = deque()
    for nodedef in graph.roots:
        q.append(nodedef)
        rootnames.add(str(id(nodedef)))

    nodes = []
    edges = []
    visited = set()

    while q:
        nodedef = q.popleft()
        name = str(id(nodedef))
        visited.add(nodedef)
        if name in rootnames:  # output node
            color = "red"
            shape = "rarrow"
        elif not sum(1 for _ in nodedef.ts_inputs()):  # input node
            color = "cadetblue1"
            shape = "rarrow"
        else:
            color = "white"
            shape = "box"

        label = nodedef.__name__ if hasattr(nodedef, "__name__") else type(nodedef).__name__
        nodes.append(NODE(name=name, label=label, color=color, shape=shape))

        for input in nodedef.ts_inputs():
            if input[1].nodedef not in visited:
                q.append(input[1].nodedef)
            edges.append(EDGE(start=str(id(input[1].nodedef)), end=name))

    for node in nodes:
        digraph.node(
            node.name,
            node.label,
            style="filled",
            fillcolor=node.color,
            shape=node.shape,
        )
    for edge in edges:
        digraph.edge(edge.start, edge.end)

    return digraph


def generate_graph(graph_func, *args, image_format="png", **kwargs):
    """Generate a BytesIO image representation of the given graph"""
    digraph = _build_graphviz_graph(graph_func, *args, **kwargs)
    digraph.format = image_format
    buffer = BytesIO()
    buffer.write(digraph.pipe())
    buffer.seek(0)
    return buffer


def show_graph(graph_func, *args, graph_filename=None, **kwargs):
    image_format = graph_filename.split(".")[-1] if graph_filename else "png"
    buffer = generate_graph(graph_func, *args, image_format=image_format, **kwargs)

    if graph_filename:
        with open(graph_filename, "wb") as f:
            f.write(buffer.read())
    else:
        from PIL import Image

        image = Image.open(buffer)
        image.show()
