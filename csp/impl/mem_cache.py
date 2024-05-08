import copy
import inspect
import threading
from collections import namedtuple
from functools import wraps
from warnings import warn

from csp.impl.constants import UNSET

GraphFunctionObjectKey = namedtuple("GraphFunctionObjectKey", ["func", "args"])


class MemoizeControl(object):
    INST = threading.local()

    def __init__(self, memoize):
        self._memoize = memoize
        self._prev = None

    @classmethod
    def is_memoize_on(cls):
        inst = getattr(cls.INST, "instance", None)
        return not inst or inst._memoize

    def __enter__(self):
        self._prev = getattr(self.INST, "instance", None)
        self.INST.instance = self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._prev:
            self.INST.instance = self._prev
        else:
            del self.INST.instance


def memoize(value=True):
    """Context manager to turn on/off the memoization
    :param value:
    """
    return MemoizeControl(value)


class CspGraphObjectsMemCache(object):
    """An object cache that simplifies graph building

    For simple applications, single factory (singleton) can be used, for more complicated graph
    building applications multiple non default factories can be used
    """

    _THREAD_LOCAL_INSTANCE = threading.local()

    def __init__(self):
        self._instantiated_objects = {}
        self._prev_instance = None
        self._user_objects = {}

    def clear(self, clear_user_objects=True):
        self._instantiated_objects.clear()
        if clear_user_objects:
            self._user_objects.clear()

    def __enter__(self):
        self._prev_instance = self.__class__.instance()
        self.__class__._THREAD_LOCAL_INSTANCE.instance = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__class__._THREAD_LOCAL_INSTANCE.instance = self._prev_instance

    @classmethod
    def new_context(cls):
        new_instance = CspGraphObjectsMemCache()
        current_instance = cls.instance()
        if current_instance:
            new_instance._instantiated_objects.update(current_instance._instantiated_objects)
            new_instance._user_objects.update(current_instance._user_objects)
        return new_instance

    def get_object_stats(self, sort_by="count"):
        assert sort_by is None or sort_by in ["name", "count"]
        res = {}
        for key in self._instantiated_objects.keys():
            assert isinstance(key, GraphFunctionObjectKey)
            name = key.func.__name__
            res[name] = res.get(name, 0) + 1
        if sort_by == "name":
            res = dict(sorted(res.items(), key=lambda t: t[0]))
        elif sort_by == "count":
            res = dict(sorted(res.items(), key=lambda t: (-t[1], t[0])))
        elif sort_by is not None:
            raise RuntimeError(f"Unsupported sort_by value {sort_by}")
        return res

    def __getitem__(self, key):
        assert isinstance(key, GraphFunctionObjectKey)
        return self._instantiated_objects.get(key, UNSET)

    def __setitem__(self, key, value):
        assert isinstance(key, GraphFunctionObjectKey)
        self._instantiated_objects[key] = value

    def get_user_object(self, key):
        assert isinstance(key, GraphFunctionObjectKey)
        return self._user_objects.get(key, UNSET)

    def set_user_object(self, key, value):
        self[key] = value
        self._user_objects[key] = value

    @classmethod
    def instance(cls):
        return getattr(cls._THREAD_LOCAL_INSTANCE, "instance", None)


def _resolve_func_args(func):
    f_spec = inspect.getfullargspec(func)

    if f_spec.varargs:
        raise RuntimeError("varargs are not allowed for graph object provided functions")

    if f_spec.varkw:
        raise RuntimeError("kwargs are not allowed for graph object provided functions")

    all_args = {k: UNSET for k in f_spec.args + f_spec.kwonlyargs}
    if f_spec.kwonlydefaults:
        all_args.update(f_spec.kwonlydefaults)
    if f_spec.defaults:
        n_defaults = len(f_spec.defaults)
        defaults_index = len(f_spec.args) - n_defaults

        all_args.update({f_spec.args[defaults_index + i]: f_spec.defaults[i] for i in range(n_defaults)})
    return all_args


def normalize_arg(arg):
    arg_type = type(arg)
    if arg_type in {list, tuple}:
        return (type(arg),) + tuple(normalize_arg(v) for v in arg)
    elif isinstance(arg, set):
        return (type(arg),) + tuple(normalize_arg(v) for v in sorted(arg))
    elif isinstance(arg, dict):
        return (arg_type,) + tuple((normalize_arg(k), normalize_arg(arg[k])) for k in sorted(arg))
    else:
        return arg_type, arg


def _preprocess_args(args):
    for arg_name, arg_value in args:
        yield (arg_name, normalize_arg(arg_value))


def function_full_name(f):
    """A utility function that can be used for implementation of function_name for csp_memoized_graph_object
    :param f:
    :return:
    """
    module = f.__module__
    if module is None:
        return f.__name__
    else:
        return module + "." + f.__name__


def csp_memoized(func=None, *, force_memoize=False, function_name=None, is_user_data=True):
    """A decorator to register a function as graph object provider.

    The provided object can be any object.

    :param func:
    :param force_memoize
    :param function_name: The name of the function that should be used for error/warning logging.
    :param is_user_data: A flag that specifies whether the memoized object is user object or graph object
    :return:
    """

    def _impl(func):
        func_args = _resolve_func_args(func)
        func_arg_names = list(func_args.keys())

        @wraps(func)
        def __call__(*args, **kwargs):
            if MemoizeControl.is_memoize_on() or force_memoize:
                cur_args = copy.copy(func_args)
                cur_args.update(dict(zip(func_arg_names, args)))
                cur_args.update(kwargs)
                cache_instance = CspGraphObjectsMemCache.instance()
                if cache_instance is None:
                    if force_memoize:
                        logging_context = function_name if function_name else str(func)
                        raise RuntimeError(f"Can't memoize {logging_context} - no memcache instance is set")
                    return func(*args, **kwargs)
                try:
                    key = GraphFunctionObjectKey(func, tuple(_preprocess_args(cur_args.items())))
                    if is_user_data:
                        cur_item = cache_instance.get_user_object(key)
                    else:
                        cur_item = cache_instance[key]
                except TypeError as e:
                    if force_memoize:
                        raise
                    logging_context = function_name if function_name else str(func)
                    warn(f"Not memoizing output of {str(logging_context)}: {str(e)}", Warning)
                    cur_item = func(*args, **kwargs)
                else:
                    if cur_item is UNSET:
                        cur_item = func(*args, **kwargs)
                        if is_user_data:
                            cache_instance.set_user_object(key, cur_item)
                        else:
                            cache_instance[key] = cur_item
            else:
                cur_item = func(*args, **kwargs)

            return cur_item

        return __call__

    if func is None:
        return _impl
    else:
        return _impl(func)


def csp_memoized_graph_object(func=None, *, force_memoize=False, function_name=None):
    return csp_memoized(func=func, force_memoize=force_memoize, function_name=function_name, is_user_data=False)
