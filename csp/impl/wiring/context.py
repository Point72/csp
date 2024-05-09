import threading
from datetime import datetime

from csp.impl.mem_cache import CspGraphObjectsMemCache


class Context:
    TLS = threading.local()

    def __init__(
        self,
        start_time: datetime = None,
        end_time: datetime = None,
        is_global_instance: bool = False,
    ):
        self.roots = []
        self.start_time = start_time
        self.end_time = end_time
        self.mem_cache = None
        self.delayed_nodes = []

        self._is_global_instance = is_global_instance
        if hasattr(self.TLS, "instance") and self.TLS.instance._is_global_instance:
            prev = self.TLS.instance
            with self:
                # Note we don't copy everything here from the global context since it may cause undesired behaviors.
                # roots - We want to accumulate roots that are only relevant for the current run, not all the roots in the global context.
                # start_time, end_time - not even set in the global context
                # mem_cache - can cause issues with dynamic graph nodes
                for delayed_node in prev.delayed_nodes:
                    # The copy of the delayed node will add all the new delayed nodes to the current context
                    delayed_node.copy()

    @classmethod
    def instance(cls):
        return cls.TLS.instance

    @classmethod
    def new_global_context(cls, enable=True):
        if hasattr(cls.TLS, "instance"):
            raise NotImplementedError("Setting of global context while thread local instance exists is not supported")
        instance = Context(is_global_instance=True)
        if enable:
            instance.__enter__()
        return instance

    @classmethod
    def clear_global_context(cls):
        if not hasattr(cls.TLS, "instance"):
            return
        if not cls.TLS.instance._is_global_instance:
            return
        cls.TLS.instance.__exit__(None, None, None)

    def __enter__(self):
        self._prevstate = self.TLS.instance if hasattr(self.TLS, "instance") else None
        self.TLS.instance = self
        self.mem_cache = CspGraphObjectsMemCache.new_context()
        self.mem_cache.__enter__()

        return self

    def __exit__(self, type, value, traceback):
        del self.TLS.instance
        if self._prevstate is not None:
            self.TLS.instance = self._prevstate
        return self.mem_cache.__exit__(type, value, traceback)


def new_global_context(enable=True):
    """
    Example 1:
    new_global_context() # at this point the context is visible to all subsequent csp calls

    Example 2:
    context = new_global_context(False) # the context is not visible yet in the CSP lib
    with context:
        # The global context is visible only inside the with statement
        ...
    :param enable: Whether the context shoudl be entered. If False provided then the new context should be used in a "with" statement
    :return: The new context
    """
    return Context.new_global_context(enable)


def clear_global_context():
    """Clear the global context that was previously set by call to new_global_context"""
    return Context.clear_global_context()
