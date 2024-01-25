from abc import ABCMeta, abstractmethod

from csp.impl.wiring import Context


class DelayedNodeWrapperDef(metaclass=ABCMeta):
    """Base utility class that should be used for wiring time "delayed" instantiation of output nodes"""

    def __init__(self):
        self._nodedef = None
        if not hasattr(Context.TLS, "instance"):
            raise RuntimeError("Delayed node must be created under a wiring context")
        Context.TLS.instance.delayed_nodes.append(self)

    @abstractmethod
    def copy(self):
        raise NotImplementedError()

    @abstractmethod
    def _instantiate(self):
        raise NotImplementedError()
