import datetime
import os
from typing import Dict, Optional, TypeVar, Union

import csp
from csp.adapters.parquet import ParquetOutputConfig, ParquetWriter
from csp.impl.managed_dataset.cache_user_custom_object_serializer import CacheObjectSerializer
from csp.impl.managed_dataset.dateset_name_constants import DatasetNameConstants
from csp.impl.managed_dataset.managed_dataset import ManagedDatasetPartition
from csp.impl.managed_dataset.managed_dataset_merge_utils import _create_wip_file
from csp.impl.wiring import Context
from csp.impl.wiring.cache_support.partition_files_container import PartitionFileContainer
from csp.impl.wiring.outputs import OutputsContainer
from csp.impl.wiring.special_output_names import ALL_SPECIAL_OUTPUT_NAMES, CSP_CACHE_ENABLED_OUTPUT

T = TypeVar("T")


def _pa():
    """
    Lazy import pyarrow
    """
    import pyarrow

    return pyarrow


def _create_output_file_or_folder(data_paths, cur_file_start_time, split_columns_to_files):
    output_folder = data_paths.get_output_folder_name(start_time=cur_file_start_time)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
    s_cur_file_path = _create_wip_file(output_folder, cur_file_start_time, is_folder=split_columns_to_files)
    return s_cur_file_path


def _generate_empty_parquet_files(dataset_partition, existing_file, files_to_generate, parquet_output_config):
    if not files_to_generate:
        return

    if os.path.isdir(existing_file):
        file_schemas = {
            f: _pa().parquet.ParquetFile(os.path.join(existing_file, f)).schema.to_arrow_schema()
            for f in os.listdir(existing_file)
            if f.endswith(".parquet")
        }
        for (s, e), dir_name in files_to_generate.items():
            for f_name, schema in file_schemas.items():
                with _pa().parquet.ParquetWriter(
                    os.path.join(dir_name, f_name),
                    schema=schema,
                    compression=parquet_output_config.compression,
                    version=ParquetWriter.PARQUET_VERSION,
                ):
                    pass
            PartitionFileContainer.get_instance().add_generated_file(
                dataset_partition, s, e, dir_name, parquet_output_config
            )
    else:
        file_info = _pa().parquet.ParquetFile(existing_file)
        schema = file_info.schema.to_arrow_schema()

        for (s, e), f_name in files_to_generate.items():
            with _pa().parquet.ParquetWriter(
                f_name,
                schema=schema,
                compression=parquet_output_config.compression,
                version=ParquetWriter.PARQUET_VERSION,
            ):
                PartitionFileContainer.get_instance().add_generated_file(
                    dataset_partition, s, e, f_name, parquet_output_config
                )


@csp.node
def _cache_filename_provider_custom_time(
    dataset_partition: ManagedDatasetPartition,
    config: Optional[ParquetOutputConfig],
    split_columns_to_files: Optional[bool],
    timestamp_ts: csp.ts[datetime.datetime],
) -> csp.ts[str]:
    with csp.state():
        s_data_paths = dataset_partition.data_paths
        s_last_start_time = None
        s_cur_file_path = None
        s_cur_file_cutoff_time = None
        s_empty_files_to_generate = {}
        s_last_closed_file = None

    with csp.start():
        config = config.copy() if config is not None else ParquetOutputConfig()
        config.resolve_compression()

    with csp.stop():
        # We need to chack that s_cur_file_path since if the engine had startup error, s_cur_file_path is undefined
        if "s_cur_file_path" in locals() and s_cur_file_path:
            PartitionFileContainer.get_instance().add_generated_file(
                dataset_partition, s_last_start_time, timestamp_ts, s_cur_file_path, config
            )
            _generate_empty_parquet_files(dataset_partition, s_last_closed_file, s_empty_files_to_generate, config)

    if csp.ticked(timestamp_ts):
        if s_cur_file_cutoff_time is None:
            s_last_start_time = timestamp_ts
            s_cur_file_cutoff_time = s_data_paths.get_file_cutoff_time(s_last_start_time)
            s_cur_file_path = _create_output_file_or_folder(s_data_paths, s_last_start_time, split_columns_to_files)
            return s_cur_file_path
        elif timestamp_ts >= s_cur_file_cutoff_time:
            PartitionFileContainer.get_instance().add_generated_file(
                dataset_partition,
                s_last_start_time,
                s_cur_file_cutoff_time - datetime.timedelta(microseconds=1),
                s_cur_file_path,
                config,
            )
            s_last_closed_file = s_cur_file_path
            s_last_start_time = s_cur_file_cutoff_time
            s_cur_file_cutoff_time = s_data_paths.get_file_cutoff_time(s_last_start_time)
            # There might be some empty files in the middle, we need to take care of this by creating a bunch of empty files on the way
            while s_cur_file_cutoff_time <= timestamp_ts:
                s_cur_file_path = _create_output_file_or_folder(s_data_paths, s_last_start_time, split_columns_to_files)
                s_empty_files_to_generate[
                    (s_last_start_time, s_cur_file_cutoff_time - datetime.timedelta(microseconds=1))
                ] = s_cur_file_path
                s_last_start_time = s_cur_file_cutoff_time
                s_cur_file_cutoff_time = s_data_paths.get_file_cutoff_time(s_last_start_time)

            s_cur_file_path = _create_output_file_or_folder(s_data_paths, s_last_start_time, split_columns_to_files)
            return s_cur_file_path


def _finalize_current_output_file(
    data_paths, config, dataset_partition, now, cur_file_path, split_columns_to_files, last_start_time, cache_enabled
):
    if cur_file_path:
        PartitionFileContainer.get_instance().add_generated_file(
            dataset_partition, last_start_time, now - datetime.timedelta(microseconds=1), cur_file_path, config
        )

    if cache_enabled:
        output_folder = data_paths.get_output_folder_name(start_time=now)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder, exist_ok=True)
        return _create_wip_file(output_folder, now, is_folder=split_columns_to_files)
    else:
        return ""


@csp.node
def _cache_filename_provider(
    dataset_partition: ManagedDatasetPartition,
    config: Optional[ParquetOutputConfig],
    split_columns_to_files: Optional[bool],
    cache_control_ts: csp.ts[bool],
    default_cache_enabled: bool,
) -> csp.ts[str]:
    with csp.alarms():
        a_update_file_alarm = csp.alarm(bool)

    with csp.state():
        s_data_paths = dataset_partition.data_paths
        s_last_start_time = None
        s_cur_file_path = None
        s_cache_enabled = default_cache_enabled

    with csp.start():
        config = config if config is not None else ParquetOutputConfig()
        csp.schedule_alarm(a_update_file_alarm, datetime.timedelta(), False)

    with csp.stop():
        # We need to chack that s_cur_file_path since if the engine had startup error, s_cur_file_path is undefined
        if "s_cur_file_path" in locals() and s_cur_file_path:
            if s_cache_enabled:
                PartitionFileContainer.get_instance().add_generated_file(
                    dataset_partition, s_last_start_time, csp.now(), s_cur_file_path, config
                )

    if csp.ticked(cache_control_ts):
        if cache_control_ts:
            # We didn't write and need to start writing
            if not s_cache_enabled:
                s_cache_enabled = True
                s_cur_file_path = _finalize_current_output_file(
                    s_data_paths,
                    config,
                    dataset_partition,
                    csp.now(),
                    s_cur_file_path,
                    split_columns_to_files,
                    s_last_start_time,
                    s_cache_enabled,
                )
                s_last_start_time = csp.now()
                cutoff_time = s_data_paths.get_file_cutoff_time(s_last_start_time)
                csp.schedule_alarm(a_update_file_alarm, cutoff_time, False)
                return s_cur_file_path
        else:
            # It's a bit ugly for now, we will keep writing even when cache is disabled but then we will throw away the written data.
            # we need a better way to address this in the future
            if s_cache_enabled:
                s_cache_enabled = False
                s_cur_file_path = _finalize_current_output_file(
                    s_data_paths,
                    config,
                    dataset_partition,
                    csp.now(),
                    s_cur_file_path,
                    split_columns_to_files,
                    s_last_start_time,
                    s_cache_enabled,
                )
                s_last_start_time = csp.now()
                cutoff_time = s_data_paths.get_file_cutoff_time(s_last_start_time)
                csp.schedule_alarm(a_update_file_alarm, cutoff_time, False)
                return s_cur_file_path

    if csp.ticked(a_update_file_alarm) and s_last_start_time != csp.now():
        s_cur_file_path = _finalize_current_output_file(
            s_data_paths,
            config,
            dataset_partition,
            csp.now(),
            s_cur_file_path,
            split_columns_to_files,
            s_last_start_time,
            s_cache_enabled,
        )
        s_last_start_time = csp.now()
        cutoff_time = s_data_paths.get_file_cutoff_time(s_last_start_time)
        csp.schedule_alarm(a_update_file_alarm, cutoff_time, False)
        return s_cur_file_path


@csp.node
def _serialize_value(value: csp.ts["T"], type_serializer: CacheObjectSerializer) -> csp.ts[bytes]:
    if csp.ticked(value):
        csp.output(type_serializer.serialize_to_bytes(value))


def create_managed_parquet_writer_node(
    function_name: str,
    dataset_partition: ManagedDatasetPartition,
    values: OutputsContainer,
    field_mapping: Dict[str, Union[str, Dict[str, str]]],
    config: Optional[ParquetOutputConfig] = None,
    data_timestamp_column_name=None,
    controlled_cache: bool = False,
    default_cache_enabled: bool = True,
):
    metadata = dataset_partition.dataset.metadata
    if data_timestamp_column_name is None:
        timestamp_column_name = getattr(metadata, "timestamp_column_name", None)
    else:
        timestamp_column_name = data_timestamp_column_name
    config = config.copy() if config else ParquetOutputConfig()
    config.allow_overwrite = True
    cache_serializers = Context.instance().config.cache_config.cache_serializers

    split_columns_to_files = metadata.split_columns_to_files

    if controlled_cache:
        cache_control_ts = values[CSP_CACHE_ENABLED_OUTPUT]
    else:
        cache_control_ts = csp.const(True)
        default_cache_enabled = True

    if not isinstance(values, OutputsContainer):
        values = OutputsContainer(**{DatasetNameConstants.UNNAMED_OUTPUT_NAME: values})

    if data_timestamp_column_name and data_timestamp_column_name != DatasetNameConstants.CSP_TIMESTAMP:
        timestamp_ts = values
        for k in data_timestamp_column_name.split("."):
            timestamp_ts = getattr(timestamp_ts, k)
        writer = ParquetWriter(
            file_name=None,
            timestamp_column_name=None,
            config=config,
            filename_provider=_cache_filename_provider_custom_time(
                dataset_partition=dataset_partition,
                config=config,
                split_columns_to_files=split_columns_to_files,
                timestamp_ts=timestamp_ts,
            ),
            split_columns_to_files=split_columns_to_files,
        )
    else:
        writer = ParquetWriter(
            file_name=None,
            timestamp_column_name=timestamp_column_name,
            config=config,
            filename_provider=_cache_filename_provider(
                dataset_partition=dataset_partition,
                config=config,
                split_columns_to_files=split_columns_to_files,
                cache_control_ts=cache_control_ts,
                default_cache_enabled=default_cache_enabled,
            ),
            split_columns_to_files=split_columns_to_files,
        )

    all_columns = set()
    for key, value in values._items():
        if key in ALL_SPECIAL_OUTPUT_NAMES:
            continue
        if isinstance(value, dict):
            basket_metadata = metadata.dict_basket_columns[key]
            writer.publish_dict_basket(
                key, value, key_type=basket_metadata.key_type, value_type=basket_metadata.value_type
            )
        elif isinstance(value.tstype.typ, type) and issubclass(value.tstype.typ, csp.Struct):
            s_field_map = field_mapping.get(key)
            for k, v in s_field_map.items():
                try:
                    if v in all_columns:
                        raise RuntimeError(f"Found multiple writers of column {v}")
                except TypeError:
                    raise RuntimeError(f"Invalid cache field name mapping: {v}")
                all_columns.add(v)

            writer.publish_struct(value, field_map=field_mapping.get(key))
        else:
            col_name = field_mapping.get(key, key)
            try:
                if col_name in all_columns:
                    raise RuntimeError(f"Found multiple writers of column {col_name}")
            except TypeError:
                raise RuntimeError(f"Invalid cache field name mapping: {col_name}")
            all_columns.add(col_name)
            type_serializer = cache_serializers.get(value.tstype.typ)
            if type_serializer:
                writer.publish(col_name, _serialize_value(value, type_serializer))
            else:
                writer.publish(col_name, value)
    if (
        data_timestamp_column_name
        and data_timestamp_column_name not in all_columns
        and data_timestamp_column_name != DatasetNameConstants.CSP_TIMESTAMP
    ):
        raise RuntimeError(
            f"{data_timestamp_column_name} specified as timestamp column but no writers for this column found"
        )
