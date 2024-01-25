import os

from csp.impl.enum import DynamicEnum,Enum
from csp.impl.error_handling import set_print_full_exception_stack
from csp.impl.struct import Struct
from csp.impl.types.common_definitions import PushMode, Outputs, OutputBasket, OutputTypeError
from csp.impl.types.tstype import ts, DynamicBasket, SnapType as snap, SnapKeyType as snapkey, AttachType as attach
from csp.impl.wiring import build_graph, node, numba_node, graph, run, run_on_thread, add_graph_output, CspParseError, feedback, dynamic, DelayedEdge
from csp.impl.constants import UNSET
from csp.baselib     import *
from csp.dataframe   import DataFrame
from csp.curve       import curve
from csp.showgraph   import show_graph
from csp.impl.builtin_functions import *
from csp.impl.mem_cache import memoize, csp_memoized
from csp.impl.config import Config
from csp.impl.wiring.context import new_global_context, clear_global_context
from csp.impl.genericpushadapter import GenericPushAdapter
from csp import cache_support


__version__ = "0.1.0"


def get_include_path():
    return os.path.join(os.path.dirname(__file__), 'include')


def get_lib_path():
    return os.path.join(os.path.dirname(__file__), 'lib')
