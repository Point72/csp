import csp
from csp.showgraph import _build_graphviz_graph


def test_showgraph_names():
    # Simple test to assert that node names
    # are properly propagated into graph viewer
    x = csp.const(5)
    y = csp.const(6)

    x.nodedef.__name__ = "x"
    y.nodedef.__name__ = "y"

    @csp.graph
    def graph():
        csp.print("x", x)
        csp.print("y", y)

    g = _build_graphviz_graph(graph)
    print(g.source)
    assert "{} [label=x".format(id(x.nodedef)) in g.source
    assert "{} [label=y".format(id(y.nodedef)) in g.source
    assert "[label=print " in g.source
