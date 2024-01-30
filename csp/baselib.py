import builtins
import contextlib
import csp
import logging
import math
import numpy as np
import pytz
import queue
import threading
import typing

from csp import UNSET, ts
from csp.impl.__cspimpl import _cspimpl
from csp.impl.wiring import Edge, DelayedEdge, OutputsContainer, input_adapter_def
from csp.impl.wiring.delayed_node import DelayedNodeWrapperDef
from csp.lib import _cspbaselibimpl
from csp.typing import Numpy1DArray, NumpyNDArray
from datetime import datetime, timedelta
from functools import lru_cache


__all__ = ['get_basket_field', 'timer', 'const', 'print', 'log', 'LogSettings', 'firstN', 'count', 'delay', 'diff', 'merge', 'sample', 'filter',
           'default', 'accum', 'add', 'sub', 'multiply', 'divide', 'max', 'min', 'gate', 'floordiv', 'pow', 'ln', 'exp', 'abs', 'unroll', 'collect', 'flatten', 'split',
           'cast_int_to_float', 'drop_dups', 'drop_nans', 'apply', 'stop_engine', 'not_', 'bitwise_not', 'and_', 'or_', 'gt', 'ge', 'lt', 'le', 'eq', 'ne', 'exprtk','struct_field', 'struct_fromts',
           'struct_collectts', 'null_ts', 'multiplex', 'demultiplex', 'dynamic_demultiplex', 'dynamic_collect', 'wrap_feedback', 'schedule_on_engine_stop', 'times', 'times_ns',
           'static_cast', 'dynamic_cast',
           'DelayedDemultiplex', 'DelayedCollect' ]

const = input_adapter_def('csp.const', _cspimpl._const, ts['T'], value='~T', delay=(timedelta, timedelta()))
_timer = input_adapter_def('csp.timer', _cspimpl._timer, ts['T'], interval=timedelta, value=('~T', True), allow_deviation=(bool, False))

@csp.graph
def timer( interval : timedelta, value: '~T' = True, allow_deviation : bool = False ) -> ts['T']:
    if interval <= timedelta():
        raise ValueError( 'csp.timer interval must be > 0')
    return _timer(interval, value, allow_deviation)

class LogSettings:
    TLS = threading.local()

    def __init__(self, logger_name, logging_tz):
        self.logger = logging.getLogger(logger_name)
        self.logging_tz = logging_tz

    @classmethod
    def set(cls, logger_name='csp', logging_tz=None):
        cls.TLS.instance = LogSettings(logger_name=logger_name, logging_tz=logging_tz)

    @classmethod
    def set_queue(cls):
        instance = cls.TLS.instance
        instance.queue = queue.Queue()
        instance.thread = threading.Thread(target=_log_thread, args=(instance.queue, ))
        instance.thread.start()
        return instance.queue

    @classmethod
    def get_queue(cls):
        instance = cls.get_instance()
        res = getattr(instance, 'queue', None)
        if res is not None:
            return res
        return cls.set_queue()

    @classmethod
    def has_queue(cls):
        instance = cls.get_instance()
        return hasattr(instance, 'queue')

    @classmethod
    def join_queue(cls):
        cls.TLS.instance.queue.join()
        cls.TLS.instance.thread.join() # must exist and be done execution
        del cls.TLS.instance.queue, cls.TLS.instance.thread

    @classmethod
    @contextlib.contextmanager
    def with_set(cls, logger_name='csp', logging_tz=None):
        prev_instance = getattr(cls.TLS, 'instance', None)
        cls.set(logger_name=logger_name, logging_tz=logging_tz)
        try:
            yield cls.get_instance()
        finally:
            if prev_instance is not None:
                cls.TLS.instance = prev_instance
            else:
                del cls.TLS.instance

    @classmethod
    @contextlib.contextmanager
    def with_set_instance(cls, instance):
        prev_instance = getattr(cls.TLS, 'instance', None)
        cls.TLS.instance = instance
        try:
            yield instance
        finally:
            if prev_instance is not None:
                cls.TLS.instance = prev_instance
            else:
                del cls.TLS.instance

    @classmethod
    def get_instance(cls):
        res = getattr(cls.TLS, 'instance', None)
        if res is not None:
            return res
        cls.set()
        return cls.TLS.instance


@csp.node
def _list_basket_to_string_ts(x: [ts['T']]) -> ts[str]:
    value = ','.join([str(x[i]) if csp.ticked(x[i]) else '' for i in range(len(x))])
    return f'[{value}]'


@csp.node
def _dict_basket_to_string_ts(x: {'K': ts[object]}) -> ts[str]:
    return str({k: x[k] for k in x.tickedkeys()})


def _convert_ts_object_for_print(x):
    if isinstance(x, OutputsContainer):
        return _dict_basket_to_string_ts({k: _convert_ts_object_for_print(v) for k, v in x._items()})
    elif isinstance(x, list):
        return _list_basket_to_string_ts(x)
    elif isinstance(x, dict):
        return _dict_basket_to_string_ts(x)
    else:
        return x


@csp.node(name="print")
def _print_ts(tag: str, x: ts['T']):
    with csp.state():
        s_log_tz=LogSettings.get_instance().logging_tz

    if csp.ticked(x):
        t = csp.now()
        if s_log_tz is not None:
            t = pytz.UTC.localize(t).astimezone(s_log_tz)
        builtins.print('%s %s:%s' % (t, tag, x))


def print(tag: str, x):
    return _print_ts(tag, _convert_ts_object_for_print(x))

def log(level: int, tag: str, x,
        logger: typing.Optional[logging.Logger] = None,
        logger_tz: object = None,
        use_thread: bool = False
    ):
    '''
    Logs a time-series value during the execution of a csp graph

    Arguments:
    level: log level to use e.g. CRITICAL
    tag: label for the logged entry e.g. "x"
    x: the time-series to log
    logger: optional, the logger to use. If not set, the default csp logger is used
    logger_tz: optional, time-zone that the logger is set to.
    use_thread: whether or not to run the log node in a separate thread
        - This feature is useful for expensive log calls that will be printing large objects, such as structs
        - NOTE: If use_thread=True and a time-series value is modified after log is called, it may log the modified version.
        In this case, which is rare, you must pass a copy of the time-series to the log node to ensure correct behavior.
    '''
    return _log_ts(level, tag, _convert_ts_object_for_print(x), logger=logger, logger_tz=logger_tz, use_thread=use_thread)

# consumer function used to log to logger
def _log_thread(q: queue.Queue):
    while True:
        logger, level, tag, t, x = q.get()
        if logger is None: # sentinel
            break
        logger.log(level, '%s %s:%s', t, tag, x)
        q.task_done()
    q.task_done()

@csp.node(name="log")
def _log_ts(level: int, tag: str, x: ts['T'],
        logger: typing.Optional[logging.Logger] = None,
        logger_tz: object = None,
        use_thread: bool = False
    ):

    with csp.state():
        s_logger=LogSettings.get_instance().logger
        s_log_tz=LogSettings.get_instance().logging_tz
        s_queue = None

    with csp.start():
        s_logger = logger if logger else LogSettings.get_instance().logger
        s_log_tz = logger_tz if logger_tz else LogSettings.get_instance().logging_tz
        s_queue = LogSettings.get_queue() if use_thread else None

    with csp.stop():
        if use_thread and LogSettings.has_queue():
            s_queue.put((None, None, None, None, None)) # sentinel
            LogSettings.join_queue()

    if csp.ticked(x):
        t = csp.now()
        if s_log_tz is not None:
            t = pytz.UTC.localize(t).astimezone(s_log_tz)

        if use_thread:
            s_queue.put((s_logger, level, tag, t, x))
        else:
            s_logger.log(level, '%s %s:%s', t, tag, x)

@csp.graph
def get_basket_field(dict_basket: {'K': csp.ts['V']}, field_name: str) -> csp.OutputBasket( { 'K': ts[object]}, shape_of='dict_basket'):
    """Given a dict basket of Struct objects, get a dict basket of the given field of struct for the matching key

    :param dict_basket:
    :param field_name:
    :return:
    """
    if isinstance(dict_basket, csp.impl.wiring.cache_support.graph_building.WrappedCachedStructBasket):
        return dict_basket.get_basket_field(field_name)
    else:
        return {k: getattr(v, field_name) for k, v in dict_basket.items()}


@csp.node(cppimpl=_cspbaselibimpl.sample)
def sample(trigger: ts['Y'], x: ts['T']) -> ts['T']:
    ''' will return current value of x on trigger ticks '''

    with csp.start():
        csp.make_passive(x)

    if csp.ticked(trigger) and csp.valid(x):
        return x


@csp.node(cppimpl=_cspbaselibimpl.firstN)
def firstN(x: ts['T'], N: int) -> ts['T']:
    ''' return first N ticks of input and then stop '''
    with csp.state():
        s_count=0
    with csp.start():
        if N <= 0:
            csp.make_passive(x)

    if csp.ticked(x):
        s_count += 1
        if s_count == N:
            csp.make_passive(x)
        return x

import typing

_TypeVar = typing.TypeVar('T')

@csp.node(cppimpl=_cspbaselibimpl.count)
def count(x: ts['T']) -> ts[int]:
    ''' return count of ticks of input '''
    if csp.ticked(x):
        return csp.num_ticks(x)


@csp.node(cppimpl=_cspbaselibimpl._delay_by_timedelta)
def _delay_by_timedelta(x: ts['T'], delay: timedelta) -> ts['T']:
    with csp.alarms():
        alarm = csp.alarm('T')

    if csp.ticked(x):
        csp.schedule_alarm(alarm, delay, x)

    if csp.ticked(alarm):
        return alarm


@csp.node(cppimpl=_cspbaselibimpl._delay_by_ticks)
def _delay_by_ticks(x: ts['T'], delay: int) -> ts['T']:
    with csp.start():
        assert delay > 0
        csp.set_buffering_policy(x, tick_count=delay + 1)

    if csp.ticked(x) and csp.num_ticks(x) > delay:
        return csp.value_at(x, -delay)


@csp.graph
def delay(x: ts['T'], delay: typing.Union[timedelta, int]) -> ts['T']:
    ''' delay input ticks by given delay '''
    if isinstance(delay, int):
        return _delay_by_ticks(x, delay)
    else:
        return _delay_by_timedelta(x, delay)


@csp.graph
def _lag(x: ts['T'], lag: typing.Union[timedelta, int]) -> ts['T']:
    ''' ticks when input ticks, but with lagged value of input '''
    if isinstance(lag, int):
        return _delay_by_ticks(x, lag)
    else:
        return csp.sample(x, csp.delay(x, lag))


@csp.graph
def diff(x: ts['T'], lag: typing.Union[timedelta, int]) -> ts['T']:
    ''' diff x against itself lag time/ticks ago '''
    return x - _lag(x, lag)


@csp.node(cppimpl=_cspbaselibimpl.merge)
def merge(x: ts['T'], y: ts['T']) -> ts['T']:
    ''' merge two timeseries into one.  If both tick at the same time, left side wins'''
    if csp.ticked(x):
        return x

    return y

@csp.node(cppimpl=_cspbaselibimpl.split)
def split(flag: ts[bool], x: ts['T']) -> csp.Outputs(false=ts['T'], true=ts['T']):
    ''' based on flag tick input on true/false outputs '''
    with csp.start():
        csp.make_passive(flag)

    if csp.ticked(x) and csp.valid(flag):
        if flag:
            csp.output(true=x)
        else:
            csp.output(false=x)


@csp.node(cppimpl=_cspbaselibimpl.cast_int_to_float)
def cast_int_to_float(x: csp.ts[int]) -> ts[float]:
    if csp.ticked(x):
        # Will be properly converted on the c++ side
        return x

@csp.node(cppimpl=_cspbaselibimpl.bitwise_not)
def bitwise_not(x: csp.ts[int]) -> ts[int]:
    return ~x

@csp.node()
def apply(x: csp.ts['T'], f: object, result_type: 'O') -> ts['O']:
    """
    :param x: The time series on which the function should be applied
    :param f: A scalar function that will be applied on each value of x
    :param result_type: The type of the values in the resulting time series
    :return: A time series that ticks on each tick of x. Each item in the result time series is the return value of f applied on x. All values should match
    the specified result_type
    """
    if csp.ticked(x):
        return f(x)


@csp.node(cppimpl=_cspbaselibimpl.filter)
def filter(flag: ts[bool], x: ts['T']) -> ts['T']:
    ''' only ticks out input if flag is true '''
    with csp.start():
        csp.make_passive(flag)

    if csp.ticked(x) and csp.valid(flag):
        if flag:
            return x

# TODO cppimpl
@csp.node
def _drop_dups(x:ts['T']) -> ts['T']:
    with csp.start():
        s_prev = csp.impl.constants.UNSET

    if csp.ticked(x) and x != s_prev:
        s_prev = x
        return x


# TODO cppimpl
@csp.node
def _drop_dups_float(x: ts[float]) -> ts[float]:
    with csp.start():
        s_prev = csp.impl.constants.UNSET

    if csp.ticked(x):
        if math.isnan(x):
            if s_prev is csp.impl.constants.UNSET or not math.isnan(s_prev):
                s_prev = x
                return x
        elif x != s_prev:
            s_prev = x
            return x


@csp.graph
def drop_dups(x:ts['T']) -> ts['T']:
    if x.tstype.typ is float:
        return _drop_dups_float(x)
    return _drop_dups(x)


@csp.node(cppimpl=_cspbaselibimpl.drop_nans)
def drop_nans(x: ts[float]) -> ts[float]:
    '''removes any nan values from the input series'''
    if not math.isnan(x):
        return x


@csp.node(cppimpl=_cspbaselibimpl.unroll)
def unroll(x: ts[['T']]) -> ts['T']:
    ''' "unrolls" timeseries of lists of type 'T' into individual ticks of type 'T' '''
    with csp.alarms():
        alarm = csp.alarm('T')
    with csp.state():
        s_pending=0

    if csp.ticked(x):
        start = 0
        if not s_pending and x:
            csp.output(x[0])
            start = 1

        for v in x[start:]:
            s_pending += 1
            csp.schedule_alarm(alarm, timedelta(), v)

    if csp.ticked(alarm):
        s_pending -= 1
        return alarm


@csp.node(cppimpl=_cspbaselibimpl.collect)
def collect(x: [ts['T']]) -> ts[['T']]:
    ''' convert basket of timeseries into timeseries of list of ticked values '''
    if csp.ticked(x):
        return list(x.tickedvalues())


@csp.graph
def flatten(x: [ts['T']]) -> ts['T']:
    ''' flatten a basket of inputs into ts[ 'T' ] '''
    # Minor optimization, if we have a list with just
    # a single ts, then just emit it as-is. Otherwise,
    # collect and unroll the full list
    if len(x) == 1:
        return x[0]
    # If x is empty, we let it take this path as well
    return unroll(collect(x))

# TODO cppimpl
@csp.node
def gate(x: ts['T'], release: ts[bool]) -> ts[['T']]:
    ''' "gate" the input.
        if release is false, input will be held until release is true.
        when release ticks true, all gated inputs will tick in one shot '''
    with csp.state():
        s_pending=[]

    if csp.ticked(x):
        s_pending.append(x)

    if csp.valid(release) and release and len(s_pending):
        out = s_pending
        s_pending = []
        return out


@csp.graph
def default(x: ts['T'], default: '~T', delay: timedelta = timedelta()) -> ts['T']:
    ''' default a timeseries with a constant value.  Default will tick at start of engine, unless the input has
        a valid startup tick '''
    default = csp.const(default, delay=delay)
    if delay != timedelta():
        default = csp.firstN(csp.merge(x, default), 1)
    return csp.merge(x, default)


@csp.node
def stop_engine(x: ts['T'], dynamic : bool = False ):
    ''' stop engine on tick of x
     :param dynamic: if True, and this is called within a dynamic graph, only shutdown the dynamic sub-graph
    '''
    if csp.ticked(x):
        csp.stop_engine( dynamic )


@csp.node
def null_ts(typ: 'T') -> ts['T']:
    """An empty time series that is guaranteed to never provide any data, can be connected as stub argument of nodes/graphs that expect ts of
    a given type
    :param typ: The type of the time series
    """
    if False:
        return None


@csp.node(cppimpl=_cspbaselibimpl.multiplex)
def multiplex(x: {'K': ts['T']}, key: ts['K'], tick_on_index: bool = False, raise_on_bad_key: bool = False) -> ts['T']:
    """
    :param x: The basket of time series to multiplex
    :param key: A
    :param tick_on_index:
    :param raise_on_bad_key:
    :return:
    """
    with csp.state():
        s_key_valid = False

    if csp.ticked(key):
        csp.make_passive(x)
        if key in x:
            csp.make_active(x[key])
            s_key_valid=True
        else:
            if raise_on_bad_key:
                raise ValueError('key %s not in input basket' % key)
            s_key_valid = False

    if s_key_valid:
        if csp.ticked( x[key] ) or \
           (tick_on_index and csp.ticked(key) and csp.valid(x[key])):
            csp.output(x[key])



@csp.node(cppimpl=_cspbaselibimpl.demultiplex)
def demultiplex(x: ts['T'], key: ts['K'], keys: ['K'],
                raise_on_bad_key: bool = False) -> csp.OutputBasket( { 'K' : ts['T']}, shape='keys'):
    '''whenever the timeseries input ticks, output a tick on the appropriate basket output'''
    with csp.state():
        s_keys=set(keys)

    if csp.ticked(x) and csp.valid(key):
        if key in s_keys:
            csp.output({key: x})
        elif raise_on_bad_key:
            raise ValueError('key %s not in keys' % key)


# TODO - looks like output annotations arent working for dynamic baskets, needs to be fixed
# @csp.node(cppimpl=_cspbaselibimpl.dynamic_demultiplex)
@csp.node
def dynamic_demultiplex(x: ts['T'], key: ts['K']) -> {ts['K']: ts['T']}:
    '''whenever the timeseries input ticks, output a tick on the appropriate dynamic basket output'''
    if csp.ticked(x) and csp.valid(key):
        csp.output({key: x})

# @csp.node(cppimpl=_cspbaselibimpl.dynamic_collect)
@csp.node
def dynamic_collect(data: {ts['K']: ts['V']}) -> ts[{'K': 'V'}]:
    '''whenever any input of the dynamic basket ticks, output the key-value pairs in a dictionary'''
    if csp.ticked(data):
        return dict(data.tickeditems())

# May want to move these into separate math lib
@csp.node(cppimpl=_cspbaselibimpl.not_, name="not_")
def not_(x: ts[bool]) -> ts[bool]:
    ''' boolean not '''
    if csp.ticked(x):
        return not x


@csp.node
def andnode(x: [ts[bool]]) -> ts[bool]:
    if csp.valid(x):
        return all(x.validvalues())


def and_(*inputs):
    ''' binary and of basket of ts[ bool ]. Note that all inputs must be valid
        before any value is returned '''
    return andnode(list(inputs))


@csp.node
def ornode(x: [ts[bool]]) -> ts[bool]:
    if csp.valid(x):
        return any(x.validvalues())


def or_(*inputs):
    ''' binary or of ts[ bool ] inputs.  Note that all inputs must be valid
        before any value is returned '''
    return ornode(list(inputs))


# Math/comparison binary operators are supported in C++ only for (int,int) and
# (float, float) arguments. For all other types, the Python implementation is used.

MATH_OPS = ['add', 'sub', 'multiply', 'divide', 'pow', 'max', 'min']

COMP_OPS = ['eq', 'ne', 'lt', 'gt', 'le', 'ge']

MATH_COMP_OPS_CPP = {
    ('add', 'float'): _cspbaselibimpl.add_f,
    ('add', 'int'): _cspbaselibimpl.add_i,
    ('sub', 'float'): _cspbaselibimpl.sub_f,
    ('sub', 'int'): _cspbaselibimpl.sub_i,
    ('multiply', 'float'): _cspbaselibimpl.mul_f,
    ('multiply', 'int'): _cspbaselibimpl.mul_i,
    ('divide', 'float'): _cspbaselibimpl.div_f,
    ('divide', 'int'): _cspbaselibimpl.div_i,
    ('pow', 'float'): _cspbaselibimpl.pow_f,
    ('pow', 'int'): _cspbaselibimpl.pow_i,
    ('max', 'float'): _cspbaselibimpl.max_f,
    ('max', 'int'): _cspbaselibimpl.max_i,
    ('min', 'float'): _cspbaselibimpl.min_f,
    ('min', 'int'): _cspbaselibimpl.min_i,
    ('eq', 'float'): _cspbaselibimpl.eq_f,
    ('eq', 'int'): _cspbaselibimpl.eq_i,
    ('ne', 'float'): _cspbaselibimpl.ne_f,
    ('ne', 'int'): _cspbaselibimpl.ne_i,
    ('lt', 'float'): _cspbaselibimpl.lt_f,
    ('lt', 'int'): _cspbaselibimpl.lt_i,
    ('gt', 'float'): _cspbaselibimpl.gt_f,
    ('gt', 'int'): _cspbaselibimpl.gt_i,
    ('le', 'float'): _cspbaselibimpl.le_f,
    ('le', 'int'): _cspbaselibimpl.le_i,
    ('ge', 'float'): _cspbaselibimpl.ge_f,
    ('ge', 'int'): _cspbaselibimpl.ge_i
}

@lru_cache(maxsize=512)
def define_op( name, op_lambda ):

    float_out_type, int_out_type, generic_out_type = [None]*3
    if name in COMP_OPS:
        float_out_type = bool
        int_out_type = bool
        generic_out_type = bool
    elif name in MATH_OPS:
        float_out_type = float
        if name != 'divide':
            int_out_type = int
            generic_out_type = 'T'
        else:
            int_out_type = float
            generic_out_type = float

    from csp.impl.wiring.node import _node_internal_use
    @_node_internal_use(cppimpl=MATH_COMP_OPS_CPP[(name,'float')], name=name)
    def float_type( x: ts[float], y: ts[float] ) -> ts[float_out_type]:
        if csp.valid( x, y ):
            return op_lambda( x, y )

    @_node_internal_use(cppimpl=MATH_COMP_OPS_CPP[(name,'int')], name=name)
    def int_type( x: ts[int], y: ts[int] ) -> ts[int_out_type]:
        if csp.valid( x, y ):
            return op_lambda( x, y )

    @_node_internal_use(name=name)
    def numpy_type( x: ts['T'], y: ts['U'] ) -> ts[np.ndarray]:
        if csp.valid( x, y ):
            return op_lambda( x, y )

    @_node_internal_use(name=name)
    def generic_type( x: ts['T'], y: ts['T'] ) -> ts[generic_out_type]:
         if csp.valid( x, y ):
            return op_lambda( x, y )

    def comp( x: ts['T'], y: ts['U'] ):

        if x.tstype.typ in [Numpy1DArray[float], NumpyNDArray[float]] or y.tstype.typ in [Numpy1DArray[float], NumpyNDArray[float]]:
            return numpy_type(x, y)
        elif x.tstype.typ is float and y.tstype.typ is float:
            return float_type(x, y)
        elif x.tstype.typ is int and y.tstype.typ is int:
            return int_type(x, y)

        return generic_type( x, y )

    comp.__name__ = name
    return comp

# Math operators

add = define_op( 'add', lambda x,y: x+y )
sub = define_op( 'sub', lambda x,y: x-y )
multiply = define_op( 'multiply', lambda x,y: x*y )
pow = define_op( 'pow', lambda x,y: x**y )
divide = define_op( 'divide', lambda x,y: x/y )
min = define_op( 'min', lambda x,y: x if x < y else y)
max = define_op( 'max', lambda x,y: x if x > y else y)

# Comparison operators

eq = define_op( 'eq', lambda x,y: x==y )
ne = define_op( 'ne', lambda x,y: x!=y )
gt = define_op( 'gt', lambda x,y: x>y)
lt = define_op( 'lt', lambda x,y: x<y)
ge = define_op( 'ge', lambda x,y: x>=y)
le = define_op( 'le', lambda x,y: x<=y)

# Other math ops

@csp.node
def floordiv(x: ts['T'], y: ts['T']) -> ts['T']:
    if csp.ticked(x, y) and csp.valid(x, y):
        return x // y


@csp.node
def accum(x: ts['T'], start: '~T' = 0) -> ts['T']:
    with csp.state():
        s_accum=start

    if csp.ticked(x):
        s_accum += x
        return s_accum


@csp.node(cppimpl=_cspbaselibimpl.ln)
def ln(x: ts[float]) -> ts[float]:
    if csp.ticked(x):
        return math.log(x)

@csp.node(cppimpl=_cspbaselibimpl.exp)
def exp(x: ts[float]) -> ts[float]:
    if csp.ticked(x):
        return math.exp(x)

@csp.node(cppimpl=_cspbaselibimpl.abs)
def abs(x: ts[float]) -> ts[float]:
    if csp.ticked(x):
        return abs(x)

@csp.node(cppimpl=_cspbaselibimpl.exprtk_impl)
def _csp_exprtk_impl(expression_str: str, inputs: {str: csp.ts[object]}, state_vars: dict, constants: dict,
                     functions: dict, trigger: csp.ts[object], use_trigger: bool, out_type: 'T') -> ts['T']:
    raise NotImplementedError("No python implementation of exprtk_impl")
    return None


@csp.graph
def exprtk(expression_str: str, inputs: {str: csp.ts[object]}, state_vars: dict = {}, trigger: csp.ts[object] = None, functions: dict = {}, constants: dict = {}, output_ndarray: bool = False ) -> ts[typing.Union[float, np.ndarray]]:
    ''' given a mathematical expression,
        and a set of timeseries corresponding to variables in that expression, tick out the result (a float) of that expression,
        either every time an input ticks, or on the trigger if provided.
    :param expression_str: a mathematical expression as per the C++ Mathematical Expression Toolkit Library, ExprTk: http://www.partow.net/programming/exprtk/
    :param inputs: a dict basket of timeseries.  The keys will correspond to the variables in the expression.  The timeseries can be of float or string
    :param state_vars: an optional dictionary of variables to be held in state between executions, and assignable within the expression.
        Keys are the variable names and values are the starting values
    :param trigger: an optional trigger for when to calculate.  If not provided, will calculate any time an input ticks
    :param functions: an optional dictionary whose keys are function names that can be used in the expression, and whose values are of the form
        (("arg1", ..), "function body"), for example {"foo": (("x","y"), "x*y")}
    :param constants: an optional dictionary of constants.  Keys are names and values are the constant values
    :param output_ndarray: flag, defaults to False, if True, will output ndarray instead of float.  Must have expression use return like "return [a, b, c]"
    '''
    use_trigger = trigger is not None
    return _csp_exprtk_impl(expression_str, inputs, state_vars, constants, functions, trigger if use_trigger else null_ts(bool), use_trigger, np.ndarray if output_ndarray else float)


@csp.node(cppimpl=_cspbaselibimpl.struct_field)
def struct_field(x: ts['T'], field: str, fieldType: 'Y') -> ts['Y']:
    if csp.ticked(x):
        value = getattr(x, field, UNSET)
        if value is not UNSET:
            return value


@csp.node(cppimpl=_cspbaselibimpl.struct_fromts)
def _struct_fromts(cls: 'T', inputs: {str: ts[object]}, trigger: ts[object], use_trigger:bool) -> ts['T']:
    ''' construct a ticking Struct from the given timeseries.
        Note structs will be created from all valid items '''
    with csp.start():
        if use_trigger:
            csp.make_passive(inputs)

    return cls(**dict(inputs.validitems()))

@csp.graph
def struct_fromts(cls: 'T', inputs: {str: ts[object]}, trigger: ts[object] = None) -> ts['T']:
    ''' construct a ticking Struct from the given timeseries basket.
        Note structs will be created from all valid items.
         trigger - Optional timeseries to control when struct gets created ( defaults to any time a basket input ticks )'''
    use_trigger = trigger is not None
    return _struct_fromts(cls, inputs, trigger if use_trigger else null_ts(bool), use_trigger)

@csp.node(cppimpl=_cspbaselibimpl.struct_collectts)
def struct_collectts(cls: 'T', inputs: {str: ts[object]}) -> ts['T']:
    ''' construct a ticking Struct from the given timeseries.
        Note structs will be created from all ticked items '''
    if csp.ticked(inputs):
        return cls(**dict(inputs.tickeditems()))

@csp.graph
def wrap_feedback(i: ts['T']) -> ts['T']:
    """
    A convenience function to wrap the given time series as a feedback.
    Useful when using in conjunction with some type of delayed nodes. Usually the example would be something like this:

    # The assumption here is that fill generates somehow fills for all orders.
    # So there exists a loop of strategy -> orders -> fills -> strategy, fills must be implemented as some kind of delayed node
    # that is instantiated at late stage after all "ordering" nodes have been created. This must create a feeback loop and as convinience
    # we mark the fills as a feedback by using a call to wrap_feedback.
    my_fills = get_fills_ts(...)
    my_orders = my_strategy(wrap_feedback(my_fills))
    publish_orders(my_orders)

    :param i: The input node to be wrapped as feedback
    :return:
    """
    feedback = csp.feedback(i.tstype.typ)
    feedback.bind(i)
    return feedback.out()


@csp.node
def schedule_on_engine_stop(f: object):
    """Schedules a function to be called on engine stop, useful for cleanup.
    :param f: The function to be called, the function is expected to take no parameters
    """
    with csp.stop():
        f()
    pass

@csp.node(cppimpl=_cspbaselibimpl.times)
def times(x: ts[object]) -> ts[datetime]:
    """
    Returns a time-series of datetimes at which x ticks
    """
    return csp.now()

@csp.node(cppimpl=_cspbaselibimpl.times_ns)
def times_ns(x: ts[object]) -> ts[int]:
    """
    Returns a time-series of ints representing the epoch time (in nanoseconds) at which x ticks
    """
    return csp.now().timestamp()

@csp.graph
def static_cast(x:ts['T'], outType: 'U') -> ts['U']:
    """ "static" cast of the given timeseries type to timeseries of type "U"
    This should only be used when the caller knows with 100% certainty that the type conversion is always valid
    as there will be no runtime type checking.
    """
    # Special case bool / int which are native types, but bool evaluates as a subclass of int
    if not issubclass(outType, x.tstype.typ) or ( outType is bool and x.tstype.typ is int ) :
        raise TypeError( f'Unable to csp.static_cast edge of type {x.tstype.typ.__name__} to {outType.__name__}')
    return Edge(ts[outType], nodedef=x.nodedef, output_idx=x.output_idx, basket_idx=x.basket_idx)

@csp.node
def dynamic_cast(x:ts['T'], outType: 'U') -> ts['U']:
    """ safer version of static_Cast, dynamic_cast will run the data through a node and
        ensure runtime type checking """
    if csp.ticked(x):
        return x

class DelayedDemultiplex(DelayedNodeWrapperDef):
    """ special "advanced use" object for delayed demultiplex.  Useful for writing APIs that susbcribe
        to a fat pipe but want to demux the pipe based on graph-time requests for keys ( subscribe calls )
    """

    def __init__(self, x: ts['T'], key: ts['K'], raise_on_bad_key: bool = False):
        super().__init__()
        self._stream_ts = x
        self._key_ts = key
        self._raise_on_bad_key = raise_on_bad_key
        self._demuxed_by_key = {}

    def copy(self):
        res = DelayedDemultiplex(self._stream_ts, self._key_ts, self._raise_on_bad_key)
        res._demuxed_by_key.update(self._demuxed_by_key)
        return res

    def demultiplex(self, key):
        return self._demultiplex(key, self._key_ts.tstype.typ, self._stream_ts.tstype.typ)

    @csp.graph
    def _demultiplex(self, key: '~K', key_type: 'K', out_type: 'T') -> ts['T']:
        res = self._demuxed_by_key.get(key)
        if res is None:
            res = self._demuxed_by_key[key] = DelayedEdge(self._stream_ts.tstype)
        return res

    def _instantiate(self):
        demuxed = demultiplex(self._stream_ts, self._key_ts, list(self._demuxed_by_key.keys()), self._raise_on_bad_key)
        for k, v in demuxed.items():
            # Note some edges may have been bound from previous run if this was in global context ( see copy() )
            if not self._demuxed_by_key[k].is_bound():
                self._demuxed_by_key[k].bind(v)


class DelayedCollect(DelayedNodeWrapperDef):
    """ special "advanced use" object for delayed collect.  Useful for writing APIs that have publish calls
        which can be called from multiple places, but need to feed into a single sink
    """

    def __init__(self, ts_type, default_to_null:bool = False):
        """
        :param: ts_type - type of input timeseries
        """
        super().__init__()
        self._inputs = []
        self._output = DelayedEdge(ts[[ts_type]], default_to_null)

    def copy(self):
        res = DelayedCollect()
        res._inputs = self._inputs.copy()
        res._output = self._output
        return res

    @csp.graph
    def add_input(self, x : ts['T']):
        if self._output.is_bound():
            raise RuntimeError( 'Attempting to add_input to DelayedCollect which is already bound from a different context( likely global context )')

        self._inputs.append(x)

    def output(self):
        """ returns collected inputs as ts[ typing.List[ input_ts_type] ] """
        return self._output

    def _instantiate(self):
        if not self._output.is_bound() and len(self._inputs):
            self._output.bind(collect(self._inputs))
