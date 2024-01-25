import csp
from csp import ts


def test_node_custom_name():
    @csp.node(name="blerg")
    def _other_name(x: ts[int]) -> ts[int]:
        if csp.ticked(x):
            return x

    assert _other_name.__name__ == "blerg"
