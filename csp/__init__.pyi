"""Type stubs for csp - a high performance reactive stream processing library."""

from datetime import datetime, timedelta
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import numpy as np
import numpy.typing as npt

from csp.curve import curve as curve
from csp.impl.constants import UNSET as UNSET

# Re-exports from impl modules
from csp.impl.enum import DynamicEnum as DynamicEnum, Enum as Enum
from csp.impl.error_handling import set_print_full_exception_stack as set_print_full_exception_stack
from csp.impl.genericpushadapter import GenericPushAdapter as GenericPushAdapter
from csp.impl.mem_cache import csp_memoized as csp_memoized, memoize as memoize
from csp.impl.struct import Struct as Struct
from csp.impl.types.common_definitions import OutputBasket as OutputBasket, Outputs as Outputs, PushMode as PushMode
from csp.impl.types.tstype import DynamicBasket as DynamicBasket, TsType as TsType, ts as ts
from csp.impl.wiring import (
    DelayedEdge as DelayedEdge,
    dynamic as dynamic,
    feedback as feedback,
    graph as graph,
    node as node,
    numba_node as numba_node,
)
from csp.impl.wiring.context import (
    clear_global_context as clear_global_context,
    new_global_context as new_global_context,
)
from csp.impl.wiring.edge import Edge as Edge
from csp.showgraph import show_graph as show_graph

# Type variables
T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")
U = TypeVar("U")

# Version
__version__: str

# Helper functions
def get_include_path() -> str: ...
def get_lib_path() -> str: ...

# Core functions re-exported from baselib

def const(value: T, delay: timedelta = ...) -> ts[T]:
    """Create a constant timeseries that ticks once at the start (or after delay) with the given value."""
    ...

def timer(interval: timedelta, value: T = ..., allow_deviation: bool = ...) -> ts[T]:
    """Create a timeseries that ticks at regular intervals."""
    ...

def merge(x: ts[T], y: ts[T]) -> ts[T]:
    """Merge two timeseries. If both tick at the same time, left side (x) wins."""
    ...

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
    """Delay input ticks by given delay (timedelta or number of ticks)."""
    ...

def diff(x: ts[T], lag: Union[timedelta, int]) -> ts[T]:
    """Diff x against itself lag time/ticks ago."""
    ...

def split(flag: ts[bool], x: ts[T]) -> Outputs[ts[T], ts[T]]:
    """Split input based on flag to true/false outputs."""
    ...

def filter(flag: ts[bool], x: ts[T]) -> ts[T]:
    """Only tick out input if flag is true."""
    ...

def drop_dups(x: ts[T], eps: Optional[float] = ...) -> ts[T]:
    """Remove consecutive duplicates from the input series."""
    ...

def drop_nans(x: ts[float]) -> ts[float]:
    """Remove any nan values from the input series."""
    ...

def unroll(x: ts[List[T]]) -> ts[T]:
    """Unroll timeseries of lists of type T into individual ticks of type T."""
    ...

def collect(x: List[ts[T]]) -> ts[List[T]]:
    """Convert basket of timeseries into timeseries of list of ticked values."""
    ...

def flatten(x: List[ts[T]]) -> ts[T]:
    """Flatten a basket of inputs into ts[T]."""
    ...

def gate(x: ts[T], release: ts[bool], release_on_tick: bool = ...) -> ts[List[T]]:
    """Gate the input. If release is false, input will be held until release is true."""
    ...

def default(x: ts[T], default: T, delay: timedelta = ...) -> ts[T]:
    """Default a timeseries with a constant value."""
    ...

def apply(x: ts[T], f: Callable[[T], U], result_type: Type[U]) -> ts[U]:
    """Apply a scalar function to each value of x."""
    ...

def null_ts(typ: Type[T]) -> ts[T]:
    """An empty time series that is guaranteed to never tick."""
    ...

def multiplex(
    x: Dict[K, ts[T]],
    key: ts[K],
    tick_on_index: bool = ...,
    raise_on_bad_key: bool = ...,
) -> ts[T]:
    """Multiplex from a basket of timeseries based on a key."""
    ...

def demultiplex(
    x: ts[T],
    key: ts[K],
    keys: List[K],
    raise_on_bad_key: bool = ...,
) -> Dict[K, ts[T]]:
    """Demultiplex input to appropriate basket output based on key."""
    ...

def dynamic_demultiplex(x: ts[T], key: ts[K]) -> Dict[ts[K], ts[T]]:
    """Demultiplex with dynamic keys."""
    ...

def dynamic_collect(data: Dict[ts[K], ts[V]]) -> ts[Dict[K, V]]:
    """Collect ticked key-value pairs from dynamic basket into a dictionary."""
    ...

def accum(x: ts[T], start: T = ...) -> ts[T]:
    """Accumulate values of x starting from start."""
    ...

def struct_field(x: ts[T], field: str, fieldType: Type[U]) -> ts[U]:
    """Extract a field from a ticking Struct timeseries."""
    ...

def struct_fromts(
    cls: Type[T],
    inputs: Dict[str, ts[Any]],
    trigger: Optional[ts[Any]] = ...,
) -> ts[T]:
    """Construct a ticking Struct from given timeseries basket."""
    ...

def struct_collectts(cls: Type[T], inputs: Dict[str, ts[Any]]) -> ts[T]:
    """Construct a ticking Struct from all ticked inputs."""
    ...

def wrap_feedback(i: ts[T]) -> ts[T]:
    """Wrap the given time series as a feedback."""
    ...

def schedule_on_engine_stop(f: Callable[[], None]) -> None:
    """Schedule a function to be called on engine stop."""
    ...

def times(x: ts[Any]) -> ts[datetime]:
    """Return a time-series of datetimes at which x ticks."""
    ...

def times_ns(x: ts[Any]) -> ts[int]:
    """Return a time-series of epoch time (nanoseconds) at which x ticks."""
    ...

def static_cast(x: ts[T], outType: Type[U]) -> ts[U]:
    """Static cast a timeseries type to another type (no runtime checking)."""
    ...

def dynamic_cast(x: ts[T], outType: Type[U]) -> ts[U]:
    """Dynamic cast with runtime type checking."""
    ...

def stop_engine(x: ts[Any], dynamic: bool = ...) -> None:  # noqa: F811
    """Stop engine on tick of x."""
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
    """Evaluate a mathematical expression on ticking inputs using ExprTk."""
    ...

def get_basket_field(
    dict_basket: Dict[K, ts[V]],
    field_name: str,
) -> Dict[K, ts[Any]]:
    """Get a dict basket of a given field from a dict basket of Structs."""
    ...

# Print and log
def print(tag: str, x: ts[Any]) -> None:
    """Print a timeseries value with a tag."""
    ...

def log(
    level: int,
    tag: str,
    x: ts[Any],
    logger: Optional[Any] = ...,
    logger_tz: Optional[Any] = ...,
    use_thread: bool = ...,
) -> None:
    """Log a timeseries value."""
    ...

# Builtin functions (only usable inside nodes)
def ticked(*ts_or_basket: ts[T]) -> bool:
    """Check if the given ts/basket ticked. Only usable inside a node."""
    ...

def valid(*ts_or_basket: ts[T]) -> bool:
    """Check if the given ts/basket have valid values. Only usable inside a node."""
    ...

def make_passive(ts_or_basket: ts[T]) -> None:
    """Make the given ts or basket passive. Only usable inside a node."""
    ...

def make_active(ts_or_basket: ts[T]) -> None:
    """Make the given ts or basket active. Only usable inside a node."""
    ...

def num_ticks(input: ts[T]) -> int:
    """Get number of ticks of the input. Only usable inside a node."""
    ...

def value_at(
    series: ts[T],
    index_or_time: Union[int, timedelta, datetime, None] = ...,
    duplicate_policy: int = ...,
    default: Any = ...,
) -> T:
    """Get value at given index or time. Only usable inside a node."""
    ...

def time_at(
    series: ts[T],
    index_or_time: Union[int, timedelta, datetime, None] = ...,
    duplicate_policy: int = ...,
    default: Any = ...,
) -> datetime:
    """Get timestamp at given index or time. Only usable inside a node."""
    ...

def item_at(
    series: ts[T],
    index_or_time: Union[int, timedelta, datetime, None] = ...,
    duplicate_policy: int = ...,
    default: Any = ...,
) -> Tuple[datetime, T]:
    """Get (timestamp, value) tuple at given index or time. Only usable inside a node."""
    ...

def values_at(
    series: ts[T],
    start_index_or_time: Union[int, timedelta, datetime, None] = ...,
    end_index_or_time: Union[int, timedelta, datetime, None] = ...,
    start_index_policy: Any = ...,
    end_index_policy: Any = ...,
) -> npt.NDArray[Any]:
    """Get values between indices or times. Only usable inside a node."""
    ...

def times_at(
    series: ts[T],
    start_index_or_time: Union[int, timedelta, datetime, None] = ...,
    end_index_or_time: Union[int, timedelta, datetime, None] = ...,
    start_index_policy: Any = ...,
    end_index_policy: Any = ...,
) -> npt.NDArray[Any]:
    """Get timestamps between indices or times. Only usable inside a node."""
    ...

def items_at(
    series: ts[T],
    start_index_or_time: Union[int, timedelta, datetime, None] = ...,
    end_index_or_time: Union[int, timedelta, datetime, None] = ...,
    start_index_policy: Any = ...,
    end_index_policy: Any = ...,
) -> Tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    """Get (timestamps, values) between indices or times. Only usable inside a node."""
    ...

def set_buffering_policy(
    ts_or_basket: ts[T],
    tick_count: Optional[int] = ...,
    tick_history: Optional[timedelta] = ...,
) -> None:
    """Set the buffering window on the given timeseries. Only usable inside a node."""
    ...

def alarm(typ: Type[T]) -> ts[T]:
    """Initialize an alarm event. Only usable inside a node."""
    ...

def schedule_alarm(
    series: ts[T],
    when: Union[datetime, timedelta],
    value: T,
) -> None:
    """Schedule an alarm event. Only usable inside a node."""
    ...

def cancel_alarm(series: ts[T], when: Optional[Union[datetime, timedelta]] = ...) -> None:
    """Cancel a scheduled alarm. Only usable inside a node."""
    ...

def now() -> datetime:
    """Return the current engine time. Only usable inside a node."""
    ...

def remove_dynamic_key(basket: ts[T], key: Any) -> None:
    """Remove a key from a dynamic basket output. Only usable inside a node."""
    ...

def in_realtime() -> bool:
    """Return whether the engine is in realtime or sim mode. Only usable inside a node."""
    ...

def engine_start_time() -> datetime:
    """Return the engine run start time."""
    ...

def engine_end_time() -> datetime:
    """Return the engine run end time."""
    ...

def is_configured_realtime() -> bool:
    """Return whether the graph is configured to run in realtime mode."""
    ...

def set_capture_cpp_backtrace(value: bool = ...) -> None:
    """Set whether C++ exceptions should capture the C++ backtrace."""
    ...

def output(*args: Any, **kwargs: Any) -> None:
    """Output values from a node. Only usable inside a node."""
    ...

def engine_stats() -> Any:
    """Get engine statistics. Only usable inside a node."""
    ...

# Context managers for node definition
class __state__:
    """State context manager for nodes."""
    def __enter__(self) -> None: ...
    def __exit__(self, *args: Any) -> None: ...

class __alarms__:
    """Alarms context manager for nodes."""
    def __enter__(self) -> None: ...
    def __exit__(self, *args: Any) -> None: ...

class __start__:
    """Start context manager for nodes."""
    def __enter__(self) -> None: ...
    def __exit__(self, *args: Any) -> None: ...

class __stop__:
    """Stop context manager for nodes."""
    def __enter__(self) -> None: ...
    def __exit__(self, *args: Any) -> None: ...

class __outputs__:
    """Outputs context manager for nodes."""
    def __enter__(self) -> None: ...
    def __exit__(self, *args: Any) -> None: ...

class __return__:
    """Return context manager for nodes."""
    def __enter__(self) -> None: ...
    def __exit__(self, *args: Any) -> None: ...

# Aliases for context managers
state = __state__
alarms = __alarms__
start = __start__
stop = __stop__

# Snap and attach types
class snap(Generic[T]):
    """Snap type for capturing current value at graph build time."""
    def __init__(self, edge: ts[T]) -> None: ...

class snapkey:
    """Snapkey type for capturing dynamic basket keys."""
    def __init__(self) -> None: ...

class attach(Generic[T]):
    """Attach type for dynamic basket attachment."""
    def __init__(self) -> None: ...

# Run function
def run(
    g: Callable[..., Any],
    *args: Any,
    starttime: Optional[datetime] = ...,
    endtime: Union[datetime, timedelta] = ...,
    queue_wait_time: Optional[timedelta] = ...,
    realtime: bool = ...,
    output_numpy: bool = ...,
    **kwargs: Any,
) -> Dict[str, List[Tuple[datetime, Any]]]:
    """
    Run a csp graph.

    Args:
        g: The graph function to run
        starttime: Start time for the simulation
        endtime: End time for the simulation (datetime or timedelta from starttime)
        queue_wait_time: Time to wait for realtime events
        realtime: Whether to run in realtime mode
        output_numpy: Whether to return outputs as numpy arrays

    Returns:
        Dictionary mapping output names to list of (timestamp, value) tuples
    """
    ...

def build_graph(
    f: Callable[..., Any],
    *args: Any,
    starttime: Optional[datetime] = ...,
    endtime: Optional[Union[datetime, timedelta]] = ...,
    realtime: bool = ...,
    **kwargs: Any,
) -> Any:
    """Build a graph without running it."""
    ...

def add_graph_output(name: str, edge: ts[T]) -> None:
    """Add a named output to the graph."""
    ...

def run_on_thread(
    g: Callable[..., Any],
    *args: Any,
    starttime: Optional[datetime] = ...,
    endtime: Union[datetime, timedelta] = ...,
    queue_wait_time: Optional[timedelta] = ...,
    realtime: bool = ...,
    output_numpy: bool = ...,
    **kwargs: Any,
) -> Any:
    """Run a graph on a separate thread."""
    ...

# Logging settings
class LogSettings:
    """Settings for csp logging."""
    @classmethod
    def set(cls, logger_name: str = ..., logging_tz: Optional[Any] = ...) -> None: ...
    @classmethod
    def get_instance(cls) -> "LogSettings": ...

# Delayed nodes
class DelayedCollect:
    """Delayed collect for adding inputs at graph build time."""
    def __init__(self, ts_type: Type[T], default_to_null: bool = ...) -> None: ...
    def add_input(self, x: ts[T]) -> None: ...
    def output(self) -> ts[List[T]]: ...

class DelayedDemultiplex(Generic[T, K]):
    """Delayed demultiplex for subscribing to keys at graph build time."""
    def __init__(
        self,
        x: ts[T],
        key: ts[K],
        raise_on_bad_key: bool = ...,
    ) -> None: ...
    def demultiplex(self, key: K) -> ts[T]: ...

# Cast int to float
def cast_int_to_float(x: ts[int]) -> ts[float]:
    """Cast an int timeseries to float."""
    ...
