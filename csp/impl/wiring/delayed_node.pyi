"""Type stubs for delayed node wrapper."""

class DelayedNodeWrapperDef:
    """
    Base class for delayed node wrappers.

    Delayed nodes are instantiated after all other graph construction
    is complete, allowing for deferred binding patterns.
    """

    def __init__(self) -> None: ...
    def copy(self) -> "DelayedNodeWrapperDef":
        """Create a copy of this delayed node wrapper."""
        ...

    def _instantiate(self) -> None:
        """Instantiate the delayed node. Called internally by csp."""
        ...
