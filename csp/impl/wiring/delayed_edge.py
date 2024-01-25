import typing

from csp.impl.types.tstype import ts

from .edge import Edge

# relative imports to avoid cycles
from .graph import graph

T = typing.TypeVar("T")


class _UnsetNodedef:
    @classmethod
    def _create(self, *args, **kwargs):
        raise RuntimeError("Encountered unbound DelayedEdge")


class DelayedEdge(Edge):
    def __init__(self, tstype, default_to_null: bool = False):
        super().__init__(tstype=tstype, nodedef=_UnsetNodedef, output_idx=-1)
        self._null_node = _UnsetNodedef
        if default_to_null:
            import csp.baselib

            self.bind(csp.baselib.null_ts(tstype.typ))
            self._null_node = self.nodedef

    @graph(memoize=False)
    def _bind(self, edge: ts["T"]):
        if self.is_bound():
            raise RuntimeError(
                f'Attempted to bind DelayedEdge multiple times, previously bound to output from node "{self.nodedef._signature._name}"'
            )

        self.nodedef = edge.nodedef
        self.output_idx = edge.output_idx
        self.basket_idx = edge.basket_idx

    def bind(self, edge: Edge):
        self._bind.using(T=self.tstype.typ)(self, edge)

    def is_bound(self):
        return self.nodedef != self._null_node
