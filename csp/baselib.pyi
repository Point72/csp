"""Type stubs for csp baselib functions."""

from datetime import datetime, timedelta
from logging import Logger
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union

import numpy as np
import numpy.typing as npt

from csp.impl.types.common_definitions import Outputs
from csp.impl.types.tstype import TsType as ts
from csp.impl.wiring.delayed_node import DelayedNodeWrapperDef

T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")
U = TypeVar("U")

__all__: List[str]

# Logging
class LogSettings:
    """Settings for csp logging."""

    @classmethod
    def set(cls, logger_name: str = ..., logging_tz: Optional[Any] = ...) -> None:
        """Set the logger name and timezone."""
        ...

    @classmethod
    def set_queue(cls) -> Any:
        """Set up a logging queue for thread-safe logging."""
        ...

    @classmethod
    def get_queue(cls) -> Any:
        """Get the current logging queue."""
        ...

    @classmethod
    def has_queue(cls) -> bool:
        """Check if a logging queue is set."""
        ...

    @classmethod
    def join_queue(cls) -> None:
        """Wait for the logging queue to complete."""
        ...

    @classmethod
    def with_set(
        cls,
        logger_name: str = ...,
        logging_tz: Optional[Any] = ...,
    ) -> Any:
        """Context manager for temporarily setting log settings."""
        ...

    @classmethod
    def with_set_instance(cls, instance: "LogSettings") -> Any:
        """Context manager for temporarily using a LogSettings instance."""
        ...

    @classmethod
    def get_instance(cls) -> "LogSettings":
        """Get the current LogSettings instance."""
        ...

# Core input adapters
def const(value: T, delay: timedelta = ...) -> ts[T]:
    """
    Create a constant timeseries that ticks once with the given value.

    Args:
        value: The value to emit
        delay: Optional delay before emitting the value

    Returns:
        A time series that ticks once with the given value
    """
    ...

def timer(interval: timedelta, value: T = ..., allow_deviation: bool = ...) -> ts[T]:
    """
    Create a timeseries that ticks at regular intervals.

    Args:
        interval: Time between ticks (must be > 0)
        value: Value to emit on each tick (default True)
        allow_deviation: Allow timing deviations in realtime mode

    Returns:
        A time series that ticks at the specified interval
    """
    ...

# Printing and logging
def print(tag: str, x: ts[T]) -> None:
    """
    Print a timeseries value with a tag.

    Args:
        tag: Label for the printed value
        x: Time series to print
    """
    ...

def log(
    level: int,
    tag: str,
    x: ts[T],
    logger: Optional[Logger] = ...,
    logger_tz: Optional[Any] = ...,
    use_thread: bool = ...,
) -> None:
    """
    Log a timeseries value.

    Args:
        level: Log level (e.g., logging.INFO)
        tag: Label for the logged value
        x: Time series to log
        logger: Optional logger to use
        logger_tz: Optional timezone for timestamps
        use_thread: Run logging in a separate thread
    """
    ...

# Sampling and filtering
def sample(trigger: ts[Any], x: ts[T]) -> ts[T]:
    """Return current value of x on trigger ticks."""
    ...

def firstN(x: ts[T], N: int) -> ts[T]:
    """Return first N ticks of input and then stop."""
    ...

def count(x: ts[T]) -> ts[int]:
    """Return count of ticks of input."""
    ...

def delay(x: ts[T], delay: Union[timedelta, int]) -> ts[T]:
    """
    Delay input ticks by given delay.

    Args:
        x: Input time series
        delay: Delay as timedelta or number of ticks
    """
    ...

def diff(x: ts[T], lag: Union[timedelta, int]) -> ts[T]:
    """Diff x against itself lag time/ticks ago."""
    ...

def merge(x: ts[T], y: ts[T]) -> ts[T]:
    """
    Merge two timeseries into one.

    If both tick at the same time, left side (x) wins.
    """
    ...

def split(flag: ts[bool], x: ts[T]) -> Outputs:
    """
    Split input based on flag to true/false outputs.

    Returns Outputs with 'true' and 'false' fields.
    """
    ...

def cast_int_to_float(x: ts[int]) -> ts[float]:
    """Cast an int timeseries to float."""
    ...

def apply(x: ts[T], f: Callable[[T], U], result_type: Type[U]) -> ts[U]:
    """
    Apply a scalar function to each value of x.

    Args:
        x: Input time series
        f: Function to apply to each value
        result_type: Type of the result values
    """
    ...

def filter(flag: ts[bool], x: ts[T]) -> ts[T]:
    """Only tick out input if flag is true."""
    ...

def drop_dups(x: ts[T], eps: Optional[float] = ...) -> ts[T]:
    """
    Remove consecutive duplicates from the input series.

    Args:
        x: Input time series
        eps: For float series, tolerance for equality comparison
    """
    ...

def drop_nans(x: ts[float]) -> ts[float]:
    """Remove any nan values from the input series."""
    ...

def unroll(x: ts[List[T]]) -> ts[T]:
    """Unroll timeseries of lists into individual ticks."""
    ...

def collect(x: List[ts[T]]) -> ts[List[T]]:
    """Convert basket of timeseries into timeseries of list of ticked values."""
    ...

def flatten(x: List[ts[T]]) -> ts[T]:
    """Flatten a basket of inputs into ts[T]."""
    ...

def gate(x: ts[T], release: ts[bool], release_on_tick: bool = ...) -> ts[List[T]]:
    """
    Gate the input.

    If release is false, input will be held until release is true.
    When release ticks true, all gated inputs tick in one shot.
    """
    ...

def default(x: ts[T], default: T, delay: timedelta = ...) -> ts[T]:
    """
    Default a timeseries with a constant value.

    Default will tick at start of engine, unless the input has a valid startup tick.
    """
    ...

def stop_engine(x: ts[Any], dynamic: bool = ...) -> None:
    """
    Stop engine on tick of x.

    Args:
        x: Trigger to stop
        dynamic: If True and in a dynamic graph, only shutdown the sub-graph
    """
    ...

def null_ts(typ: Type[T]) -> ts[T]:
    """
    Create an empty time series that never ticks.

    Useful as a stub argument for nodes/graphs that expect ts of a given type.
    """
    ...

# Multiplexing and demultiplexing
def multiplex(
    x: Dict[K, ts[T]],
    key: ts[K],
    tick_on_index: bool = ...,
    raise_on_bad_key: bool = ...,
) -> ts[T]:
    """
    Multiplex from a basket of timeseries based on a key.

    Args:
        x: Basket of time series indexed by key
        key: Key selecting which series to output
        tick_on_index: Tick when key changes (even if value unchanged)
        raise_on_bad_key: Raise error if key not in basket
    """
    ...

def demultiplex(
    x: ts[T],
    key: ts[K],
    keys: List[K],
    raise_on_bad_key: bool = ...,
) -> Dict[K, ts[T]]:
    """
    Demultiplex input to appropriate basket output based on key.

    Args:
        x: Input time series
        key: Key determining which output to use
        keys: List of valid keys
        raise_on_bad_key: Raise error if key not in keys
    """
    ...

def dynamic_demultiplex(x: ts[T], key: ts[K]) -> Dict[ts[K], ts[T]]:
    """Demultiplex with dynamic keys."""
    ...

def dynamic_collect(data: Dict[ts[K], ts[V]]) -> ts[Dict[K, V]]:
    """Collect ticked key-value pairs from dynamic basket into a dictionary."""
    ...

def accum(x: ts[T], start: T = ...) -> ts[T]:
    """
    Accumulate values of x starting from start.

    Args:
        x: Input time series
        start: Initial accumulator value (default 0)
    """
    ...

def exprtk(
    expression_str: str,
    inputs: Dict[str, ts[Any]],
    state_vars: Dict[str, Any] = ...,
    trigger: Optional[ts[Any]] = ...,
    functions: Dict[str, Any] = ...,
    constants: Dict[str, Any] = ...,
    output_ndarray: bool = ...,
) -> ts[Union[float, npt.NDArray[np.floating[Any]]]]:
    """
    Evaluate a mathematical expression on ticking inputs using ExprTk.

    Args:
        expression_str: Mathematical expression (ExprTk syntax)
        inputs: Dict basket of input time series
        state_vars: Variables to maintain state between executions
        trigger: Optional trigger controlling when to calculate
        functions: User-defined functions for use in expression
        constants: Constant values for use in expression
        output_ndarray: If True, output ndarray instead of float
    """
    ...

# Struct operations
def struct_field(x: ts[T], field: str, fieldType: Type[U]) -> ts[U]:
    """
    Extract a field from a ticking Struct timeseries.

    Args:
        x: Time series of structs
        field: Name of the field to extract
        fieldType: Type of the field
    """
    ...

def struct_fromts(
    cls: Type[T],
    inputs: Dict[str, ts[Any]],
    trigger: Optional[ts[Any]] = ...,
) -> ts[T]:
    """
    Construct a ticking Struct from given timeseries basket.

    Note: Structs are created from all valid (not just ticked) items.

    Args:
        cls: Struct class to create
        inputs: Dict mapping field names to time series
        trigger: Optional trigger controlling when to create structs
    """
    ...

def struct_collectts(cls: Type[T], inputs: Dict[str, ts[Any]]) -> ts[T]:
    """
    Construct a ticking Struct from all ticked inputs.

    Unlike struct_fromts, only includes fields that ticked.
    """
    ...

def wrap_feedback(i: ts[T]) -> ts[T]:
    """
    Wrap the given time series as a feedback.

    Convenience function for creating feedback loops.
    """
    ...

def schedule_on_engine_stop(f: Callable[[], None]) -> None:
    """
    Schedule a function to be called on engine stop.

    Useful for cleanup operations.
    """
    ...

def times(x: ts[Any]) -> ts[datetime]:
    """Return a time-series of datetimes at which x ticks."""
    ...

def times_ns(x: ts[Any]) -> ts[int]:
    """Return a time-series of epoch time (nanoseconds) at which x ticks."""
    ...

def static_cast(x: ts[T], outType: Type[U]) -> ts[U]:
    """
    Static cast a timeseries type to another type.

    No runtime type checking - use only when conversion is guaranteed valid.
    """
    ...

def dynamic_cast(x: ts[T], outType: Type[U]) -> ts[U]:
    """
    Dynamic cast with runtime type checking.

    Safer but slower than static_cast.
    """
    ...

def get_basket_field(
    dict_basket: Dict[K, ts[V]],
    field_name: str,
) -> Dict[K, ts[Any]]:
    """Get a dict basket of a given field from a dict basket of Structs."""
    ...

# Delayed nodes
class DelayedCollect:
    """
    Delayed collect for adding inputs at graph build time.

    Useful for APIs that have publish calls from multiple places
    feeding into a single sink.
    """

    def __init__(self, ts_type: Type[T], default_to_null: bool = ...) -> None: ...
    def copy(self) -> "DelayedCollect": ...
    def add_input(self, x: ts[T]) -> None:
        """Add an input to this collector."""
        ...

    def output(self) -> ts[List[T]]:
        """Get the collected inputs as a list time series."""
        ...

class DelayedDemultiplex(DelayedNodeWrapperDef):
    """
    Delayed demultiplex for subscribing to keys at graph build time.

    Useful for APIs that subscribe to a fat pipe but want to demux
    based on graph-time requests for keys.
    """

    def __init__(
        self,
        x: ts[T],
        key: ts[K],
        raise_on_bad_key: bool = ...,
    ) -> None: ...
    def copy(self) -> "DelayedDemultiplex": ...
    def demultiplex(self, key: K) -> ts[T]:
        """Get the time series for a specific key."""
        ...
