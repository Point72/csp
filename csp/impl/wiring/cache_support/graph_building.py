import copy
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, TypeVar, Union

import csp
from csp.adapters.parquet import ParquetOutputConfig
from csp.impl.config import CacheCategoryConfig, CacheConfig, Config
from csp.impl.managed_dataset.cache_user_custom_object_serializer import CacheObjectSerializer
from csp.impl.managed_dataset.dataset_metadata import TimeAggregation
from csp.impl.managed_dataset.dateset_name_constants import DatasetNameConstants
from csp.impl.managed_dataset.managed_dataset import ManagedDataset
from csp.impl.managed_dataset.managed_dataset_lock_file_util import ManagedDatasetLockUtil
from csp.impl.mem_cache import normalize_arg
from csp.impl.struct import Struct
from csp.impl.types import tstype
from csp.impl.types.common_definitions import ArgKind, OutputBasketContainer, OutputDef
from csp.impl.types.tstype import ts
from csp.impl.types.typing_utils import CspTypingUtils
from csp.utils.qualified_name_utils import QualifiedNameUtils

# relative to avoid cycles
from ..context import Context
from ..edge import Edge
from ..node import node
from ..outputs import OutputsContainer
from ..signature import Signature
from ..special_output_names import UNNAMED_OUTPUT_NAME
from .cache_config_resolver import CacheConfigResolver

T = TypeVar("T")


class _UnhashableObjectWrapper:
    def __init__(self, obj):
        self._obj = obj

    def __hash__(self):
        return hash(id(self._obj))

    def __eq__(self, other):
        return id(self._obj) == id(other._obj)


class _CacheManagerKey:
    def __init__(self, scalars, *extra_args):
        self._normalized_scalars = tuple(self._normalize_scalars(scalars)) + tuple(extra_args)
        self._hash = hash(self._normalized_scalars)

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        return self._hash == other._hash and self._normalized_scalars == other._normalized_scalars

    @classmethod
    def _normalize_scalars(cls, scalars):
        for scalar in scalars:
            normalized_scalar = normalize_arg(scalar)
            try:
                hash(normalized_scalar)
            except TypeError:
                yield _UnhashableObjectWrapper(scalar)
            else:
                yield normalized_scalar


class WrappedStructEdge(Edge):
    def __init__(self, wrapped_edge, parquet_reader, field_map):
        super().__init__(
            tstype=wrapped_edge.tstype,
            nodedef=wrapped_edge.nodedef,
            output_idx=wrapped_edge.output_idx,
            basket_idx=wrapped_edge.basket_idx,
        )
        self._parquet_reader = parquet_reader
        self._field_map = field_map
        self._single_field_edges = {}

    def __getattr__(self, key):
        res = self._single_field_edges.get(key, None)
        if res is not None:
            return res
        elemtype = self.tstype.typ.metadata().get(key)
        if elemtype is None:
            raise AttributeError("'%s' object has no attribute '%s'" % (self.tstype.typ.__name__, key))
        res = self._parquet_reader.subscribe_all(elemtype, field_map=self._field_map[key])
        self._single_field_edges[key] = res
        return res


class WrappedCachedStructBasket(dict):
    def __init__(self, typ, name, wrapped_edges, parquet_reader):
        super().__init__(**wrapped_edges)
        self._typ = typ
        self._name = name
        self._parquet_reader = parquet_reader
        self._shape = None
        self._field_dicts = {}

    def get_basket_field(self, field_name):
        res = self._field_dicts.get(field_name)
        if res is None:
            if self._shape is None:
                self._shape = list(self.keys())
            # res = self._parquet_reader.subscribe_dict_basket_struct_column(self._typ, self._name, self._shape, field_name)
            res = {k: getattr(v, field_name) for k, v in self.items()}
            self._field_dicts[field_name] = res
        return res


class CacheCategoryOverridesTree:
    """A utility class that is used to resolved category overrides for a given category like ['level_1', 'level_2', ...]
    The basic implementation is a tree of levels
    """

    def __init__(self, cur_level_key: str = None, cur_level_value: CacheCategoryConfig = None, parent=None):
        self.cur_level_key = cur_level_key
        self.cur_level_value = cur_level_value
        self.parent = parent
        self.children = {}

    def _get_path_str(self):
        path = []
        cur = self
        while cur is not None:
            if cur.cur_level_value is not None:
                path.append(cur.cur_level_value)
            cur = cur.parent
        return str(reversed(path))

    def __str__(self):
        return f"CacheCategoryOverridesTree({self._get_path_str()}:{self.cur_level_value})"

    def __repr__(self):
        return self.__str__()

    def _get_child(self, key: str):
        res = self.children.get(key)
        if res is None:
            res = CacheCategoryOverridesTree(cur_level_key=key, parent=self)
            self.children[key] = res
        return res

    def _add_override(self, override: CacheCategoryConfig, cur_level_index=0):
        if cur_level_index < len(override.category):
            self._get_child(override.category[cur_level_index])._add_override(override, cur_level_index + 1)
        else:
            if self.cur_level_value is not None:
                raise RuntimeError(f"Trying to override cache directory for {self._get_path_str()} more than once")
            self.cur_level_value = override

    def resolve_root_folder(self, category: List[str], cur_level: int = 0):
        """
        :param category: The category of the dataset
        :return: A config override or the default config for the given category
        """
        if cur_level == len(category):
            return self.cur_level_value

        # We want the longest match possible, so first attempt resolving using children
        child = self.children.get(category[cur_level])
        if child is not None:
            child_res = child.resolve_root_folder(category, cur_level + 1)
            if child_res is not None:
                return child_res

        return self.cur_level_value

    @classmethod
    def construct_from_cache_config(cls, cache_config: Optional[CacheConfig] = None):
        res = CacheCategoryOverridesTree()

        if cache_config is None:
            raise RuntimeError("data_folder must be set in global cache_config to use caching")

        res.cur_level_value = cache_config

        if hasattr(cache_config, "category_overrides"):
            for override in cache_config.category_overrides:
                res._add_override(override)
        return res

    @classmethod
    def construct_from_config(cls, config: Optional[Config] = None):
        if config is not None and hasattr(config, "cache_config"):
            return cls.construct_from_cache_config(config.cache_config)
        return CacheCategoryOverridesTree()


class ContextCacheInfo(Struct):
    """Graph building context storage class

    Should be stored inside a Context object, contains all the data collected during the graph building time to enable support
    of caching
    """

    # A dictionary from tuple (function_id, scalar_arguments) to a corresponding GraphCacheManager
    cache_managers: dict
    # Dictionary of a graph function id to MangedDataset and field mapping that corresponds to it
    managed_datasets_by_graph_object: Dict[object, Tuple[ManagedDataset, Dict[str, str]]]
    # The object that is used to resolve the underlying sub folders of the given graph
    cache_data_paths_resolver: CacheConfigResolver


class _EngineLockRelease:
    """A utility that will release the given lock on engine stop"""

    def __init__(self):
        self.cur_lock = None

    @node
    def release_node(self):
        with csp.stop():
            if self.cur_lock:
                self.cur_lock.unlock()
                self.cur_lock = None


class _MissingPeriodCallback:
    def __init__(self):
        self.first_missing_period = None

    def __call__(self, start, end):
        if self.first_missing_period is None:
            self.first_missing_period = (start, end)
        return True


@node
def _deserialize_value(value: ts[bytes], type_serializer: CacheObjectSerializer, typ: "T") -> ts["T"]:
    if csp.ticked(value):
        return type_serializer.deserialize_from_bytes(value)


class GraphBuildPartitionCacheManager(object):
    """A utility class that "manages" cache at graph building time

    One instance is created per (dataset, partition_values)
    """

    def __init__(
        self,
        function_name,
        dataset: ManagedDataset,
        partition_values,
        expected_outputs,
        cache_options,
        csp_cache_start=None,
        csp_cache_end=None,
        csp_timestamp_shift=None,
    ):
        self._function_name = function_name
        self.dataset_partition = dataset.get_partition(partition_values)
        self._outputs = None
        self._written_outputs = None
        self._cache_options = cache_options
        self._context = Context.TLS.instance
        self._csp_cache_start = csp_cache_start if csp_cache_start else Context.TLS.instance.start_time
        self._csp_cache_end = csp_cache_end if csp_cache_end else Context.TLS.instance.end_time
        self._csp_timestamp_shift = csp_timestamp_shift if csp_timestamp_shift else timedelta()
        missing_period_callback = _MissingPeriodCallback()
        data_for_period, is_full_period_covered = self.get_data_for_period(
            self._csp_cache_start, self._csp_cache_end, missing_range_handler=missing_period_callback
        )
        cache_config = Context.TLS.instance.config.cache_config
        self._first_missing_period = missing_period_callback.first_missing_period
        if is_full_period_covered:
            from csp.adapters.parquet import ParquetReader

            cache_serializers = cache_config.cache_serializers
            # We need to release lock at the end of the run, generator is not guaranteed to do so if it's not called
            engine_lock_releaser = _EngineLockRelease()
            reader = ParquetReader(
                filename_or_list=self._read_files_provider(
                    dataset, data_for_period, cache_config.lock_file_permissions, engine_lock_releaser
                ),
                time_column=self.dataset_partition.dataset.metadata.timestamp_column_name,
                split_columns_to_files=self.dataset_partition.dataset.metadata.split_columns_to_files,
                start_time=csp_cache_start,
                end_time=csp_cache_end,
                allow_overlapping_periods=True,
                time_shift=self._csp_timestamp_shift,
            )
            # We need to instantiate the node ot have it run
            engine_lock_releaser.release_node()
            self._outputs = OutputsContainer()
            is_unnamed_output = False
            for output in expected_outputs:
                output_name = output.name
                if output_name is None:
                    output_name = DatasetNameConstants.UNNAMED_OUTPUT_NAME
                    is_unnamed_output = True
                if output.kind == ArgKind.BASKET_TS:
                    output_dict = reader.subscribe_dict_basket(typ=output.typ.typ, name=output_name, shape=output.shape)
                    output_value = WrappedCachedStructBasket(output.typ.typ, output_name, output_dict, reader)
                else:
                    assert output.kind == ArgKind.TS
                    if isinstance(output.typ.typ, type) and issubclass(output.typ.typ, Struct):
                        # Reverse field mapping
                        write_field_map = cache_options.field_mapping.get(output_name)
                        field_map = {v: k for k, v in write_field_map.items()}
                        # Wrap the edge to allow single column reading
                        output_value = WrappedStructEdge(
                            reader.subscribe_all(typ=output.typ.typ, field_map=field_map), reader, write_field_map
                        )
                    else:
                        type_serializer = cache_serializers.get(output.typ.typ)
                        if type_serializer:
                            output_value = _deserialize_value(
                                reader.subscribe_all(
                                    typ=bytes, field_map=cache_options.field_mapping.get(output_name, output_name)
                                ),
                                type_serializer,
                                output.typ.typ,
                            )
                        else:
                            output_value = reader.subscribe_all(
                                typ=output.typ.typ, field_map=cache_options.field_mapping.get(output_name, output_name)
                            )
                if is_unnamed_output:
                    assert len(expected_outputs) == 1
                    self._outputs = output_value
                else:
                    self._outputs[output_name] = output_value

    def _read_files_provider(self, dataset, data_files, lock_file_permissions, engine_lock_releaser):
        assert data_files
        items_iter = iter(data_files.items())
        finished = False
        num_failures = 0
        next_filename = None
        lock_util = ManagedDatasetLockUtil(lock_file_permissions)
        while not finished:
            if num_failures > 10:
                raise RuntimeError(
                    f"Failed to read cached files too many times, last attempted file is {next_filename}"
                )
            try:
                (next_start_time, next_end_time), next_filename = next(items_iter)
            except StopIteration:
                finished = True
                continue
            is_file = os.path.isfile(next_filename)
            is_dir = os.path.isdir(next_filename)
            if not is_file and not is_dir:
                data_files, _ = self.get_data_for_period(next_start_time, self._csp_cache_end)
                assert data_files
                items_iter = iter(data_files.items())
                num_failures += 1
            with dataset.use_lock_context():
                lock = lock_util.read_lock(next_filename, is_file)
                lock.lock()
            engine_lock_releaser.cur_lock = lock
            try:
                if os.path.exists(next_filename):
                    num_failures = 0
                    yield next_filename
                else:
                    data_files, _ = self.get_data_for_period(next_start_time, self._csp_cache_end)
                    assert data_files
                    items_iter = iter(data_files.items())
                    num_failures += 1
            finally:
                lock.unlock()
                engine_lock_releaser.cur_lock = None

    @property
    def first_missing_period(self):
        return self._first_missing_period

    @property
    def is_force_cache_read(self):
        return hasattr(self._cache_options, "data_timestamp_column_name")

    @property
    def outputs(self):
        return self._outputs

    @property
    def written_outputs(self):
        return self._written_outputs

    @classmethod
    def _resolve_anonymous_dataset_category(cls, cache_options):
        category = getattr(cache_options, "category", None)
        if category is None:
            category = ["csp_unnamed_cache"]
        return category

    @classmethod
    def get_dataset_for_func(cls, graph, func, cache_options, data_folder):
        category = cls._resolve_anonymous_dataset_category(cache_options)
        cache_config_resolver = None
        if isinstance(data_folder, Config):
            cache_config_resolver = CacheConfigResolver(data_folder.cache_config)
        elif isinstance(data_folder, CacheConfig):
            cache_config_resolver = CacheConfigResolver(data_folder)
        if isinstance(data_folder, CacheConfigResolver):
            cache_config_resolver = data_folder

        if cache_config_resolver is None:
            cache_config_resolver = CacheConfigResolver(CacheConfig(data_folder=data_folder))

        cache_config = cache_config_resolver.resolve_cache_config(graph, category)

        dataset_name = getattr(cache_options, "dataset_name", None) if cache_options else None
        if dataset_name is None:
            dataset_name = f"{QualifiedNameUtils.get_qualified_object_name(func)}"
        return ManagedDataset.load_from_disk(cache_config=cache_config, name=dataset_name, data_category=category)

    @classmethod
    def _resolve_dataset(cls, graph, func, signature, cache_options, expected_outputs, tvars):
        context_cache_data = Context.TLS.instance.cache_data
        # We might have the dataset already

        func_id = id(func)
        existing_dataset_and_field_mapping = context_cache_data.managed_datasets_by_graph_object.get(func_id)
        if existing_dataset_and_field_mapping is not None:
            cache_options.field_mapping = existing_dataset_and_field_mapping[1]
            return existing_dataset_and_field_mapping[0]

        dataset_name = getattr(cache_options, "dataset_name", None)
        partition_columns = {input.name: input.typ for input in signature.scalars}
        column_types = {}
        dict_basket_column_types = {}

        if len(expected_outputs) == 1 and expected_outputs[0].name is None:
            timestamp_column_auto_prefix = f"{DatasetNameConstants.UNNAMED_OUTPUT_NAME}."
            cur_def = expected_outputs[0]
            expected_outputs = (
                OutputDef(
                    name=DatasetNameConstants.UNNAMED_OUTPUT_NAME,
                    typ=cur_def.typ,
                    kind=cur_def.kind,
                    ts_idx=cur_def.ts_idx,
                    shape=cur_def.shape,
                ),
            )
        else:
            timestamp_column_auto_prefix = ""

        field_mapping = cache_options.field_mapping
        for i, out in enumerate(expected_outputs):
            if out.kind == ArgKind.BASKET_TS:
                # Let's make sure that we're handling dict basket
                if isinstance(out.shape, list) and tstype.isTsType(out.typ):
                    dict_basket_column_types[out.name] = signature.resolve_basket_key_type(i, tvars), out.typ.typ
                else:
                    raise NotImplementedError(f"Caching of basket output {out.name} of type {out.typ} is unsupported")
            elif isinstance(out.typ.typ, type) and issubclass(out.typ.typ, Struct):
                struct_field_mapping = field_mapping.get(out.name)
                if struct_field_mapping is None:
                    if cache_options.prefix_struct_names:
                        struct_col_types = {f"{out.name}.{k}": v for k, v in out.typ.typ.metadata().items()}
                    else:
                        struct_col_types = out.typ.typ.metadata()
                    column_types.update(struct_col_types)
                    struct_field_mapping = {n1: n2 for n1, n2 in zip(out.typ.typ.metadata(), struct_col_types)}
                    field_mapping[out.name] = struct_field_mapping
                else:
                    for k, v in out.typ.typ.metadata().items():
                        cache_col_name = struct_field_mapping.get(k, k)
                        column_types[cache_col_name] = v
            else:
                name = field_mapping.get(out.name, out.name)
                column_types[name] = out.typ.typ

        if hasattr(cache_options, "data_timestamp_column_name"):
            timestamp_column_name = timestamp_column_auto_prefix + cache_options.data_timestamp_column_name
        else:
            timestamp_column_name = "csp_timestamp"

        category = cls._resolve_anonymous_dataset_category(cache_options)
        resolved_cache_config = context_cache_data.cache_data_paths_resolver.resolve_cache_config(graph, category)
        dataset = cls._create_dataset(
            func,
            resolved_cache_config,
            category,
            dataset_name=dataset_name,
            timestamp_column_name=timestamp_column_name,
            columns_types=column_types,
            partition_columns=partition_columns,
            split_columns_to_files=cache_options.split_columns_to_files,
            time_aggregation=cache_options.time_aggregation,
            dict_basket_column_types=dict_basket_column_types,
        )
        context_cache_data.managed_datasets_by_graph_object[func_id] = dataset, field_mapping
        return dataset

    @classmethod
    def _get_qualified_function_name(cls, func):
        return f"{func.__module__}.{func.__name__}"

    @classmethod
    def _create_dataset(
        cls,
        func,
        cache_config,
        category,
        dataset_name,
        timestamp_column_name: str = None,
        columns_types: Dict[str, object] = None,
        partition_columns: Dict[str, type] = None,
        *,
        split_columns_to_files: Optional[bool],
        time_aggregation: TimeAggregation,
        dict_basket_column_types: Dict[type, Union[type, Tuple[type, type]]],
    ):
        name = dataset_name if dataset_name else f"{QualifiedNameUtils.get_qualified_object_name(func)}"
        dataset = ManagedDataset(
            name=name,
            category=category,
            cache_config=cache_config,
            timestamp_column_name=timestamp_column_name,
            columns_types=columns_types,
            partition_columns=partition_columns,
            split_columns_to_files=split_columns_to_files,
            time_aggregation=time_aggregation,
            dict_basket_column_types=dict_basket_column_types,
        )
        return dataset

    def get_data_for_period(self, starttime: datetime, endtime: datetime, missing_range_handler):
        res, full_period_covered = self.dataset_partition.get_data_for_period(
            starttime=starttime - self._csp_timestamp_shift,
            endtime=endtime - self._csp_timestamp_shift,
            missing_range_handler=missing_range_handler,
        )
        if self._csp_timestamp_shift:
            res = {
                (start + self._csp_timestamp_shift, end + self._csp_timestamp_shift): path
                for (start, end), path in res.items()
            }

        return res, full_period_covered

    def _fix_outputs_for_caching(self, outputs):
        if isinstance(outputs, OutputsContainer) and UNNAMED_OUTPUT_NAME in outputs:
            outputs_dict = dict(outputs._items())
            outputs_dict[DatasetNameConstants.UNNAMED_OUTPUT_NAME] = outputs_dict.pop(UNNAMED_OUTPUT_NAME)
            return OutputsContainer(**outputs_dict)
        else:
            return outputs

    def cache_outputs(self, outputs):
        from csp.impl.managed_dataset.managed_parquet_writer import create_managed_parquet_writer_node

        outputs = self._fix_outputs_for_caching(outputs)
        assert self._written_outputs is None
        self._written_outputs = outputs
        create_managed_parquet_writer_node(
            function_name=self._function_name,
            dataset_partition=self.dataset_partition,
            values=outputs,
            field_mapping=self._cache_options.field_mapping,
            config=getattr(self._cache_options, "parquet_output_config", None),
            data_timestamp_column_name=getattr(self.dataset_partition.dataset.metadata, "timestamp_column_name", None),
            controlled_cache=self._cache_options.controlled_cache,
            default_cache_enabled=self._cache_options.default_cache_enabled,
        )

    @classmethod
    def create_cache_manager(
        cls,
        graph,
        func,
        signature,
        non_ignored_scalars,
        all_scalars,
        cache_options,
        expected_outputs,
        tvars,
        csp_cache_start=None,
        csp_cache_end=None,
        csp_timestamp_shift=None,
    ):
        if not hasattr(Context.TLS, "instance"):
            raise RuntimeError("Graph must be instantiated under a wiring context")

        assert Context.TLS.instance.start_time is not None
        assert Context.TLS.instance.end_time is not None
        key = _CacheManagerKey(
            all_scalars, id(func), tuple(tvars.items()), csp_cache_start, csp_cache_end, csp_timestamp_shift
        )
        existing_cache_manager = Context.TLS.instance.cache_data.cache_managers.get(key)
        if existing_cache_manager is not None:
            return existing_cache_manager

        # We're going to modify field mapping, so we need to make a copy
        cache_options = cache_options.copy()
        cache_options.field_mapping = dict(cache_options.field_mapping)

        for output in expected_outputs:
            if output.kind == ArgKind.TS and isinstance(output.typ.typ, type) and issubclass(output.typ.typ, Struct):
                struct_field_map = cache_options.field_mapping.get(output.name)
                if struct_field_map:
                    # We don't want to omit any fields from the cache otherwise read data will be different from what's written
                    # so whatever the user doesn't map, we will map to the original field.
                    full_field_map = copy.copy(struct_field_map)
                    for k in output.typ.typ.metadata():
                        if k not in full_field_map:
                            full_field_map[k] = k
                    cache_options.field_mapping[output.name] = full_field_map

        dataset = cls._resolve_dataset(graph, func, signature, cache_options, expected_outputs, tvars)

        partition_values = dict(zip((i.name for i in signature.scalars), non_ignored_scalars))
        res = GraphBuildPartitionCacheManager(
            function_name=QualifiedNameUtils.get_qualified_object_name(func),
            dataset=dataset,
            partition_values=partition_values,
            expected_outputs=expected_outputs,
            cache_options=cache_options,
            csp_cache_start=csp_cache_start,
            csp_cache_end=csp_cache_end,
            csp_timestamp_shift=csp_timestamp_shift,
        )
        Context.TLS.instance.cache_data.cache_managers[key] = res
        return res


class GraphCacheOptions(Struct):
    # The name of the dataset to which the data will be written - optional
    dataset_name: str
    # An optional column mapping for scalar time series, the mapping should be string (the name of the column in parquet file),
    # for struct time series it should be a map of {struct_field_name:column_name}.
    field_mapping: Dict[str, Union[str, Dict[str, str]]]
    # A boolean that specifies whether struct fields should be prefixed with the output name. For example for a graph output
    # named "o" and a field named "f", if prefix_struct_names is True then the column will be "o.f" else the column will be "f"
    prefix_struct_names: bool = True
    # This is an advanced usage of graph caching, in some instances we want to override timestamp and write data with custom timestamp,
    # if this is specified, the values from the given column will be used as the timestamp column
    data_timestamp_column_name: str
    # Optional category specification for the dataset, can only be specified if using the default dataset. An example of category
    # can be ['daily_statistics', 'market_prices']. This category will be part of the files path. Additionally, cache paths can be
    # overridden for different categories.
    category: List[str]
    # Inputs to ignore for caching purposes
    ignored_inputs: Set[str]
    # A boolean that specifies whether each column should be written to a separate file
    split_columns_to_files: bool
    # The configuration of the written files
    parquet_output_config: ParquetOutputConfig
    # Data aggregation period
    time_aggregation: TimeAggregation = TimeAggregation.DAY
    # A boolean flag that specifies whether the node/graph provides a ts that specifies that the output should be cached
    controlled_cache: bool = False
    # The default value of whether at start the outputs should be cached, ignored if controlled_cache is False
    default_cache_enabled: bool = False


class ResolvedGraphCacheOptions(Struct):
    """A struct with all resolved graph cache options"""

    dataset_name: str
    field_mapping: dict
    prefix_struct_names: bool
    data_timestamp_column_name: str
    category: List[str]
    ignored_inputs: Set[str]
    split_columns_to_files: bool
    parquet_output_config: ParquetOutputConfig
    time_aggregation: TimeAggregation
    controlled_cache: bool
    default_cache_enabled: bool


def resolve_graph_cache_options(signature: Signature, cache_enabled, cache_options: GraphCacheOptions):
    """Called at graph building time to validate that the given caching options are valid for given signature

    :param signature: The signature of the cached graph
    :param cache_enabled: A boolean or that specifies whether cache enabled
    :param cache_options: Graph cache read/write options
    :return:
    """
    if cache_enabled:
        if cache_options is None:
            cache_options = GraphCacheOptions()

        field_mapping = getattr(cache_options, "field_mapping", None)
        split_columns_to_files = getattr(cache_options, "split_columns_to_files", None)
        has_basket_outputs = False

        if signature._ts_inputs:
            all_ts_ignored = False
            ignored_inputs = getattr(cache_options, "ignored_inputs", None)
            if ignored_inputs:
                all_ts_ignored = True
                for input in signature._ts_inputs:
                    if input.name not in ignored_inputs:
                        all_ts_ignored = False
            if not all_ts_ignored:
                raise NotImplementedError("Caching of graph with ts arguments is unsupported")
        if not signature._outputs:
            raise NotImplementedError("Caching of graph without outputs is unsupported")
        for output in signature._outputs:
            if isinstance(output.typ, OutputBasketContainer):
                if CspTypingUtils.get_origin(output.typ.typ) is List:
                    raise NotImplementedError("Caching of list basket outputs is unsupported")
                has_basket_outputs = True
            elif not tstype.isTsType(output.typ):
                if tstype.isTsStaticBasket(output.typ):
                    if CspTypingUtils.get_origin(output.typ) is List:
                        raise NotImplementedError("Caching of list basket outputs is unsupported")
                    else:
                        raise TypeError(
                            f"Cached output basket {output.name} must have shape provided using with_shape or with_shape_of"
                        )
            assert tstype.isTsType(output.typ) or isinstance(output.typ, OutputBasketContainer)
        ignored_inputs = getattr(cache_options, "ignored_inputs", set())
        for input in signature.scalars:
            if input.name not in ignored_inputs and not ManagedDataset.is_supported_partition_type(input.typ):
                raise NotImplementedError(
                    f"Caching is unsupported for argument type {input.typ} (argument {input.name})"
                )
        resolved_cache_options = ResolvedGraphCacheOptions(prefix_struct_names=cache_options.prefix_struct_names)

        for attr in (
            "dataset_name",
            "data_timestamp_column_name",
            "category",
            "ignored_inputs",
            "parquet_output_config",
            "time_aggregation",
            "controlled_cache",
            "default_cache_enabled",
        ):
            if hasattr(cache_options, attr):
                setattr(resolved_cache_options, attr, getattr(cache_options, attr))
        if has_basket_outputs:
            if split_columns_to_files is False:
                raise RuntimeError("Cached graph with output basket must set split_columns_to_files to True")
            split_columns_to_files = True
        elif split_columns_to_files is None:
            split_columns_to_files = False

        resolved_cache_options.split_columns_to_files = split_columns_to_files

        resolved_cache_options.field_mapping = {} if field_mapping is None else field_mapping
    else:
        if cache_options:
            raise RuntimeError("cache_options must be None if caching is disabled")
        resolved_cache_options = None
    return resolved_cache_options
