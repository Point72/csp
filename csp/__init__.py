import os

from csp.baselib import *
from csp.curve import curve
from csp.dataframe import DataFrame
from csp.impl.builtin_functions import *
from csp.impl.constants import UNSET
from csp.impl.enum import DynamicEnum, Enum
from csp.impl.error_handling import set_print_full_exception_stack
from csp.impl.genericpushadapter import GenericPushAdapter
from csp.impl.mem_cache import csp_memoized, memoize
from csp.impl.struct import Struct
from csp.impl.types.common_definitions import OutputBasket, Outputs, OutputTypeError, PushMode
from csp.impl.types.tstype import AttachType as attach, DynamicBasket, SnapKeyType as snapkey, SnapType as snap, ts
from csp.impl.wiring import (
    CspParseError,
    DelayedEdge,
    add_graph_output,
    build_graph,
    dynamic,
    feedback,
    graph,
    node,
    numba_node,
    run,
    run_on_thread,
)
from csp.impl.wiring.context import clear_global_context, new_global_context
from csp.math import *
from csp.showgraph import show_graph

from . import stats

__version__ = "0.13.2"


def get_include_path():
    return os.path.join(os.path.dirname(__file__), "include")


def get_lib_path():
    return os.path.join(os.path.dirname(__file__), "lib")


if os.environ.get("CSP_PRINT_FULL_EXCEPTION_STACK", "0").lower() in ("1", "on", "true"):
    # Print full stack trace if CSP_PRINT_FULL_EXCEPTION_STACK is set in the environment
    set_print_full_exception_stack(True)
