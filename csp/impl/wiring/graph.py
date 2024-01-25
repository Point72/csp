import datetime
import inspect
import threading
import types
from contextlib import contextmanager

import csp.impl.wiring.edge
from csp.impl.constants import UNSET
from csp.impl.error_handling import ExceptionContext
from csp.impl.managed_dataset.dateset_name_constants import DatasetNameConstants
from csp.impl.mem_cache import csp_memoized_graph_object, function_full_name
from csp.impl.types.common_definitions import InputDef
from csp.impl.types.instantiation_type_resolver import GraphOutputTypeResolver
from csp.impl.wiring import Signature
from csp.impl.wiring.cache_support.dataset_partition_cached_data import DataSetCachedData, DatasetPartitionCachedData
from csp.impl.wiring.cache_support.graph_building import (
    GraphBuildPartitionCacheManager,
    GraphCacheOptions,
    resolve_graph_cache_options,
)
from csp.impl.wiring.context import Context
from csp.impl.wiring.graph_parser import GraphParser
from csp.impl.wiring.outputs import CacheWriteOnlyOutputsContainer, OutputsContainer
from csp.impl.wiring.special_output_names import ALL_SPECIAL_OUTPUT_NAMES, UNNAMED_OUTPUT_NAME


class NoCachedDataException(RuntimeError):
    pass


class _GraphDefMetaUsingAux:
    def __init__(self, graph_meta, _forced_tvars=None):
        self._graph_meta = graph_meta
        self._forced_tvars = _forced_tvars if _forced_tvars else {}

    def using(self, **_forced_tvars):
        new_forced_tvars = {}
        new_forced_tvars.update(self._forced_tvars)
        new_forced_tvars.update(_forced_tvars)
        return _GraphDefMetaUsingAux(self._graph_meta, new_forced_tvars)

    def __call__(self, *args, **kwargs):
        return self._graph_meta._instantiate(self._forced_tvars, *args, **kwargs)

    def cache_periods(self, start_time, end_time):
        return self._graph_meta.cached_data(start_time, end_time)

    def cached_data(self, data_folder=None, _forced_tvars=None) -> DatasetPartitionCachedData:
        """Get the proxy object for accessing the graph cached data.
        This is the basic interface for inspecting cache files and loading cached data as dataframes
        :param data_folder: The root folder of the cache or an instance of CacheDataPathResolver
        :return: An instance of DatasetPartitionCachedData to access the graph cached data
        """
        if data_folder is None:
            data_folder = Context.instance().config
        return self._graph_meta.cached_data(data_folder, _forced_tvars)

    def cached(self, *args, **kwargs):
        """A utility function to ensure that a graph is read from cache
        For example if there is a cached graph g.
        Calling g(a1, a2, ...) can either read it from cache or write the results to cache if no cached data is found.
        Calling g.cached(a1, a2, ...) forces reading from cache, if no cache data is found then exception will be raised.
        :param args: Positional arguments to the graph
        :param kwargs: Keyword arguments to the graph
        """
        return self._graph_meta.cached(*args, _forced_tvars=self._forced_tvars, **kwargs)


class _ForceCached:
    """This class is an ugly workaround to avoid instantiating cached graphs.
    The problem:
    my_graph.cached(...) - is implemented by calling the regular code path of the graph instantiation and checking whether the graph is actually read from cache. This is a
    problem since the user doesn't expect the graph to be instantiated if they use "cached" property. We can't also provide an argument "force_cached" to the instantiation
    function since it's memcached and extra argument will cause calls to graph.cached(...) and graph(...) to result in different instances which is wrong.

    This class is a workaround to pass this "require_cached" flag not via arguments
    """

    _INSTANCE = threading.local()

    @classmethod
    def is_force_cached(cls):
        if not hasattr(cls._INSTANCE, "force_cached"):
            return False
        return cls._INSTANCE.force_cached

    @classmethod
    @contextmanager
    def force_cached(cls):
        prev_value = cls.is_force_cached()
        try:
            cls._INSTANCE.force_cached = True
            yield
        finally:
            cls._INSTANCE.force_cached = prev_value


class _CacheProxy:
    """A helper class that allows to access cached data in a given time range, that can be smaller than the engine run time

    Usage:
    my_graph.cached[start:end]
    The cached property of the graph will return an instance of _CacheProxy which can then be called with the appropriate parameters.
    """

    def __init__(self, owner, csp_cache_start=None, csp_cache_end=None, csp_timestamp_shift=None):
        self._owner = owner
        self._csp_cache_start = csp_cache_start
        self._csp_cache_end = csp_cache_end
        self._csp_timestamp_shift = csp_timestamp_shift

    def __getitem__(self, item):
        assert isinstance(item, slice), "cached item range must be a slice"
        assert item.step is None, "Providing step for cache range is not supported"
        res = _CacheProxy(self._owner, csp_timestamp_shift=self._csp_timestamp_shift)
        res._csp_cache_start = item.start
        # The range values are exclusive but for caching purposes we need inclusive end time
        res._csp_cache_end = item.stop - datetime.timedelta(microseconds=1)
        return res

    def shifted(self, csp_timestamp_shift: datetime.timedelta):
        return _CacheProxy(
            self._owner,
            csp_cache_start=self._csp_cache_start,
            csp_cache_end=self._csp_cache_end,
            csp_timestamp_shift=csp_timestamp_shift,
        )

    def __call__(self, *args, _forced_tvars=None, **kwargs):
        with _ForceCached.force_cached():
            return self._owner._cached_impl(
                _forced_tvars,
                self._csp_cache_start,
                self._csp_cache_end,
                args,
                kwargs,
                csp_timestamp_shift=self._csp_timestamp_shift,
            )


class GraphDefMeta(type):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._instantiate_func = self._instantiate_impl
        ignored_inputs = getattr(self._cache_options, "ignored_inputs", None)
        if self._cache_options and ignored_inputs is not None:
            non_ignored_inputs = [input for input in self._signature.inputs if input.name not in ignored_inputs]
            self._cache_signature = Signature(
                name=self._signature._name,
                inputs=[
                    InputDef(name=s.name, typ=s.typ, kind=s.kind, basket_kind=s.basket_kind, ts_idx=None, arg_idx=i)
                    for i, s in enumerate(non_ignored_inputs)
                ],
                outputs=self._signature._outputs,
                defaults={k: v for k, v in self._signature._defaults.items() if k not in ignored_inputs},
            )
        else:
            self._cache_signature = self._signature
        if self.memoize or self.force_memoize:
            if self.wrapped_node:
                full_name = function_full_name(self.wrapped_node._impl)
            else:
                full_name = function_full_name(self._func)

            self._instantiate_func = csp_memoized_graph_object(
                self._instantiate_impl, force_memoize=self.force_memoize, function_name=full_name
            )

    def _extract_forced_tvars(cls, d):
        if "_forced_tvars" in d:
            return d.pop("_forced_tvars")
        return {}

    @property
    def _cached_function(self):
        return self.wrapped_node._impl if self.wrapped_node else self._func

    def cache_periods(self, start_time, end_time):
        from csp.impl.managed_dataset.aggregation_period_utils import AggregationPeriodUtils

        agg_period_utils = AggregationPeriodUtils(self._cache_options.time_aggregation)
        return list(agg_period_utils.iterate_periods_in_date_range(start_time, end_time))

    def cached_data(self, data_folder=None, _forced_tvars=None) -> DatasetPartitionCachedData:
        """Get the proxy object for accessing the graph cached data.
        This is the basic interface for inspecting cache files and loading cached data as dataframes
        :param data_folder: An instance of string (data folder), csp.Config or csp.cache_support.CacheConfig with the appropriate cache config. Note, that
        only if one of he configs passed in then the category resolution and custom data types serialization handled properly. Pass in string only if None of the above
        features is used.
        :return: An instance of DatasetPartitionCachedData to access the graph cached data
        """
        if not self._cache_options:
            raise RuntimeError("Trying to get cached data from graph that doesn't cache")
        if data_folder is None:
            data_folder = Context.instance().config

        if isinstance(data_folder, csp.Config):
            cache_config = data_folder.cache_config
        elif isinstance(data_folder, csp.cache_support.CacheConfig):
            cache_config = data_folder
        else:
            cache_config = None

        if cache_config:
            cache_serializers = cache_config.cache_serializers
        else:
            cache_serializers = {}

        cache_signature = self._cache_signature

        dataset = GraphBuildPartitionCacheManager.get_dataset_for_func(
            graph=self, func=self._cached_function, cache_options=self._cache_options, data_folder=data_folder
        )
        if dataset is None:
            return None

        def _get_dataset_partition(*args, **kwargs):
            inputs, scalars, tvars = cache_signature.parse_inputs(_forced_tvars, *args, **kwargs)
            partition_values = dict(zip((i.name for i in cache_signature._inputs), scalars))
            return dataset.get_partition(partition_values)

        return DataSetCachedData(dataset, cache_serializers, _get_dataset_partition)

    @property
    def cached(self) -> _CacheProxy:
        """
        Usage:
            my_graph.cached[start:end]
            Will return an instance of _CacheProxy which can then be called with the appropriate parameters.
        :return: A cache proxy that can be used to limit the time of the returned graph.
        """
        return _CacheProxy(self)

    def _cached_impl(self, _forced_tvars, csp_cache_start, csp_cache_end, args, kwargs, csp_timestamp_shift=None):
        """A utility function to ensure that a graph is read from cache
        For example if there is a cached graph g.
        Calling g(a1, a2, ...) can either read it from cache or write the results to cache if no cached data is found.
        Calling g.cached(a1, a2, ...) forces reading from cache, if no cache data is found then exception will be raised.
        :param args: Positional arguments to the graph
        :param csp_cache_start: The start time of the cached data before which we don't want to load any date
        :param csp_cache_end: The end time of the cached data after which we don't want to load any date
        :param kwargs: Keyword arguments to the graph
        """
        if Context.TLS.instance.config and hasattr(Context.TLS.instance.config, "cache_config") and self._cache_options:
            read_from_cache, res, _ = self._instantiate_func(
                _forced_tvars, self._signature, args, kwargs, csp_cache_start, csp_cache_end, csp_timestamp_shift
            )
            assert read_from_cache
        else:
            raise NoCachedDataException(
                f"No data found in cache for {self._signature._name} for the given run period, seems like cache_config is unset"
            )
        return res

    def _raise_if_forced_cache_read(self, missing_period=None):
        if _ForceCached.is_force_cached():
            if missing_period:
                missing_period_str = f" {str(missing_period[0])} to {str(missing_period[1])}"
                raise NoCachedDataException(
                    f"No data found in cache for {self._signature._name} for period{missing_period_str}"
                )
            else:
                raise NoCachedDataException(
                    f"No data found in cache for {self._signature._name} for the given run period"
                )

    def _instantiate_impl(
        self, _forced_tvars, signature, args, kwargs, csp_cache_start=None, csp_cache_end=None, csp_timestamp_shift=None
    ):
        read_from_cache = False
        inputs, scalars, tvars = signature.parse_inputs(_forced_tvars, *args, allow_none_ts=True, **kwargs)

        basket_shape_eval_inputs = list(scalars)
        for input in inputs:
            if isinstance(input, list) or isinstance(input, dict):
                basket_shape_eval_inputs.append(input)

        expected_outputs = signature.resolve_output_definitions(
            tvars=tvars, basket_shape_eval_inputs=basket_shape_eval_inputs
        )
        if signature.special_outputs:
            expected_regular_outputs = tuple(v for v in expected_outputs if v.name not in ALL_SPECIAL_OUTPUT_NAMES)
        else:
            expected_regular_outputs = expected_outputs

        cache_manager = None
        if (
            hasattr(Context.TLS, "instance")
            and Context.TLS.instance.config
            and hasattr(Context.TLS.instance.config, "cache_config")
            and self._cache_options
        ):
            ignored_inputs = getattr(self._cache_options, "ignored_inputs", set())
            cache_scalars = tuple(s for s, s_def in zip(scalars, signature.scalars) if s_def.name not in ignored_inputs)

            cache_manager = GraphBuildPartitionCacheManager.create_cache_manager(
                graph=self,
                func=self._cached_function,
                signature=self._cache_signature,
                non_ignored_scalars=cache_scalars,
                all_scalars=scalars,
                cache_options=self._cache_options,
                expected_outputs=expected_regular_outputs,
                tvars=tvars,
                csp_cache_start=csp_cache_start,
                csp_cache_end=csp_cache_end,
                csp_timestamp_shift=csp_timestamp_shift,
            )
        allow_non_cached_read = True
        if cache_manager:
            if cache_manager.outputs is not None:
                res = cache_manager.outputs
                read_from_cache = True
            elif cache_manager.written_outputs is not None:
                res = cache_manager.written_outputs
            else:
                self._raise_if_forced_cache_read(cache_manager.first_missing_period)
                res = self._func(*args, **kwargs)
                cache_manager.cache_outputs(res)
            allow_non_cached_read = not cache_manager.is_force_cache_read
        else:
            self._raise_if_forced_cache_read()
            res = self._func(*args, **kwargs)

        if read_from_cache:
            expected_outputs = expected_regular_outputs

        # Validate graph return values
        if isinstance(res, OutputsContainer):
            outputs_raw = []
            for e_o in expected_outputs:
                output_name = (
                    e_o.name
                    if e_o.name
                    else (DatasetNameConstants.UNNAMED_OUTPUT_NAME if read_from_cache else UNNAMED_OUTPUT_NAME)
                )
                cur_o = res._get(output_name, UNSET)
                if cur_o is UNSET:
                    raise KeyError(f"Output {output_name} is not returned from the graph")
                outputs_raw.append(cur_o)
            if len(outputs_raw) != len(res):
                # We have some output for some wrong key
                all_output_names = {e_o.name for e_o in expected_outputs}
                for k in res:
                    if k not in all_output_names:
                        raise KeyError(f"Unexpected returned output name {k}")
                raise RuntimeError("Internal error, unexpected code location")
        elif len(expected_outputs) == 1:
            outputs_raw = (res,)
        else:
            if expected_outputs:
                raise ValueError(
                    f'Unexpected return value "{res}" from {self._signature._name}, expected: {expected_outputs}'
                )
            assert res is None

        if res is not None:
            _ = GraphOutputTypeResolver(
                function_name=self._signature._name,
                output_definitions=expected_outputs,
                values=outputs_raw,
                forced_tvars=tvars,
                allow_subtypes=self._cache_options is None,
            )
        if signature.special_outputs:
            if not read_from_cache:
                if expected_outputs[0].name is None:
                    res = next(iter(res._values()))
                else:
                    res = OutputsContainer(**{k: v for k, v in res._items() if k not in ALL_SPECIAL_OUTPUT_NAMES})

        return read_from_cache, res, allow_non_cached_read

    def _instantiate(self, _forced_tvars, *args, **kwargs):
        _, res, allow_non_cached_read = self._instantiate_func(_forced_tvars, self._signature, args=args, kwargs=kwargs)
        if not allow_non_cached_read:
            if isinstance(res, csp.impl.wiring.edge.Edge):
                return CacheWriteOnlyOutputsContainer(iter([res]))
            else:
                return CacheWriteOnlyOutputsContainer(iter(res))
        else:
            return res

    def __call__(cls, *args, **kwargs):
        return cls._instantiate(None, *args, **kwargs)

    def using(cls, **_forced_tvars):
        return _GraphDefMetaUsingAux(cls, _forced_tvars)

    def __get__(self, instance, owner):
        if instance is not None:
            return types.MethodType(self, instance)
        return self


def _create_graph(
    func_name,
    func_doc,
    impl,
    signature,
    memoize,
    force_memoize,
    cache,
    cache_options,
    wrapped_function=None,
    wrapped_node=None,
):
    resolved_cache_options = resolve_graph_cache_options(
        signature=signature, cache_enabled=cache, cache_options=cache_options
    )
    return GraphDefMeta(
        func_name,
        (object,),
        {
            "_signature": signature,
            "_func": impl,
            "memoize": memoize,
            "force_memoize": force_memoize,
            "_cache_options": resolved_cache_options,
            "__wrapped__": wrapped_function,
            "__module__": wrapped_function.__module__,
            "wrapped_node": wrapped_node,
            "__doc__": func_doc,
        },
    )


def graph(
    func=None,
    *,
    memoize=True,
    force_memoize=False,
    cache: bool = False,
    cache_options: GraphCacheOptions = None,
    name=None,
    debug_print=False,
):
    """
    :param func:
    :param memoize: Specify whether the node should be memoized (default True)
    :param force_memoize: If True, the node will be memoized even if csp.memoize(False) was called. Usually it should not be set, set
    this to True ONLY if memoization required to guarantee correctness of the function (i.e the function must be called at most once with the for each
    set of parameters).
    :param cache_options: The options for graph caching
    :param name: Provide a custom name for the constructed graph type
    :param debug_print: A boolean that specifies that processed function should be printed
    :return:
    """
    func_frame = inspect.currentframe().f_back

    def _impl(func):
        with ExceptionContext():
            add_cache_control_output = cache_options is not None and getattr(cache_options, "controlled_cache", False)
            parser = GraphParser(
                name or func.__name__,
                func,
                func_frame,
                add_cache_control_output=add_cache_control_output,
                debug_print=debug_print,
            )
            parser.parse()

            signature = parser._signature
            return _create_graph(
                name or func.__name__,
                func.__doc__,
                parser._impl,
                signature,
                memoize,
                force_memoize,
                cache,
                cache_options,
                wrapped_function=func,
            )

    if func is None:
        return _impl
    else:
        with ExceptionContext():
            return _impl(func)
