import os
from importlib.metadata import PackageNotFoundError, version as get_package_version
from typing import Callable, Dict, Optional, TypeVar

import numpy
from packaging import version

import csp
from csp.impl.struct import Struct
from csp.impl.types.container_type_normalizer import ContainerTypeNormalizer
from csp.impl.types.tstype import ts
from csp.impl.types.typing_utils import CspTypingUtils
from csp.impl.wiring.adapters import output_adapter_def
from csp.lib import _parquetadapterimpl

__all__ = ["ParquetOutputConfig", "ParquetWriter"]

_ARRAY_DIMENSIONS_SUFFIX = "_csp_dimensions"
K = TypeVar("K")
V = TypeVar("V")


def resolve_array_shape_column_name(column_name, user_provided_shape_column=None):
    assert column_name
    return user_provided_shape_column or f"{column_name}{_ARRAY_DIMENSIONS_SUFFIX}"


class ParquetOutputConfig(Struct):
    allow_overwrite: bool = False
    batch_size: int = 2**15
    compression: str
    write_arrow_binary: bool = False  # If true will write output as binary arrow data rather than parquet

    def resolve_compression(self):
        if not hasattr(self, "compression"):
            self.compression = "" if self.write_arrow_binary else "snappy"
        return self


def _get_default_parquet_version():
    try:
        if version.parse(get_package_version("pyarrow")) >= version.parse("6.0.1"):
            return "2.6"
    except PackageNotFoundError:
        # Don't need to do anything in particular
        ...
    return "2.0"


class ParquetWriter:
    PARQUET_VERSION = _get_default_parquet_version()

    def __init__(
        self,
        file_name: Optional[str],
        timestamp_column_name,
        config: Optional[ParquetOutputConfig] = None,
        filename_provider: Optional[ts[str]] = None,
        split_columns_to_files: bool = False,
        file_metadata: Optional[Dict[str, str]] = None,
        column_metadata: Optional[Dict[str, Dict[str, str]]] = None,
        file_visitor: Optional[Callable[[str], None]] = None,
    ):
        """
        :param file_name: The path of the output parquet file name. Must be provided if no filename_provider specified. If both file_name and filename_provider
        are specified then file_name will be used as the initial output file name until filename_provider provides a new file name.
        :param timestamp_column_name: Required field, if None is provided then no timestamp will be written.
        :param config: Optional configuration of how the file should be written (such as compression, block size,... ).
        :param filename_provider: An optional time series that provides a times series of file paths. When a filename_provider time series
        provides a new file path, the previous open file name will be closed and all subsequent data will be written to the new file provided by the path.
        This enable partitioning and splitting the data based on time.
        :param split_columns_to_files: A boolean flag that specifies that each column should be written to a separate file. If true, then
        file_name must be a folder into which the data will be written.
        :param file_metadata: optional str:str dict that will get written as file-level metadata
        :param column_metadata: optional dict of column : { str:str} that will get written as column-level metadata
        :param file_visitor: optional callable that will be called, after a file is written, with the file name.
        """
        super().__init__()
        config = ParquetOutputConfig() if config is None else config.copy()
        config.resolve_compression()
        assert file_name or filename_provider is not None, "file_name of filename_provider must be specified"
        self._filename_provider = filename_provider
        self._split_columns_to_files = split_columns_to_files
        self._parquet_output_adapter_manager = None
        self._parquet_dict_basket_writer_node_def = None

        if file_name and os.path.exists(file_name):
            if split_columns_to_files:
                assert os.path.isdir(file_name), (
                    f"split_columns_to_files is True, but existing past {file_name} is file, not folder"
                )
            else:
                assert os.path.isfile(file_name), (
                    f"split_columns_to_files is False, but existing past {file_name} is folder, not file"
                )

        self._all_column_names = set()
        self._properties = {
            "file_name": file_name if file_name else "",
            "timestamp_column_name": timestamp_column_name if timestamp_column_name else "",
            "allow_overwrite": config.allow_overwrite,
            "batch_size": config.batch_size,
            "compression": config.compression,
            "write_arrow_binary": config.write_arrow_binary,
            "split_columns_to_files": split_columns_to_files,
            "file_metadata": file_metadata,
            "column_metadata": column_metadata,
        }

        if file_visitor is not None:
            self._properties["file_visitor"] = file_visitor

        if timestamp_column_name:
            self._add_published_columns(timestamp_column_name)
        if self._filename_provider is not None:
            _parquet_output_filename_adapter_def(self, self._filename_provider)

    def _add_published_columns(self, *column_names):
        for column_name in column_names:
            if column_name in self._all_column_names:
                raise KeyError(f"Publishing duplicate column names in parquet/arrow file: {column_name}")
            self._all_column_names.add(column_name)

    def publish_struct(self, value: ts[Struct], field_map: Dict[str, str] = None):
        """Publish a time series of Struct objects to file

        :param value: The time series of Struct objects that should be published.
        :param field_map: An optional dict str->str of the form {struct_field_name:column_name} that maps the names of the
        structure fields to the column names to which the values should be written. If the field_map is non None, then only
        the fields that are specified in the field_map will be written to file. If field_map is not provided then all fields
        of a structure will be written to columns that match exactly the field_name.
        """
        if not field_map:
            field_map = {field: field for field in value.tstype.typ.metadata()}

        array_fields = set()

        for k, v in value.tstype.typ.metadata(typed=True).items():
            if CspTypingUtils.is_numpy_array_type(v) and k in field_map:
                self.publish(field_map[k], getattr(value, k))
                array_fields.add(k)

        for k in array_fields:
            del field_map[k]

        self._add_published_columns(*field_map.values())
        properties = {"field_map": field_map}
        ts_type = value.tstype.typ
        return _parquet_output_adapter_def(self, value, ts_type, properties)

    def publish(self, column_name, value: ts[object], array_dimensions_column_name=None):
        """Publish a time series of primitive type to file
        :param column_name: The name of the parquet file column to which the data should be written to
        :param value: The time series that should be published
        :param array_dimensions_column_name: When publishing array with column name "abc" we also need to publish the dimensions of the array (if it is a multidimensional
        array). The column of the dimensions is specified by this argument, if None provided then default column name will be created by appending '_csp_dimensions' suffix
        """
        from csp.adapters.output_adapters.parquet_utility_nodes import flatten_numpy_array

        properties = {"column_name": column_name}
        ts_type = value.tstype.typ

        if CspTypingUtils.is_numpy_array_type(ts_type):
            if CspTypingUtils.get_origin(ts_type) is csp.typing.NumpyNDArray:
                flat_value, shape = flatten_numpy_array(value)._values()
                array_dimensions_column_name = resolve_array_shape_column_name(
                    column_name, array_dimensions_column_name
                )
                self.publish(array_dimensions_column_name, shape)
                return self.publish(column_name, flat_value)
            value_type = ts_type.__args__[0]
            properties["array_value_type"] = value_type
            properties["is_array"] = True
        self._add_published_columns(column_name)

        return _parquet_output_adapter_def(
            self, value, ContainerTypeNormalizer.normalized_type_to_actual_python_type(ts_type), properties
        )

    @property
    def _parquet_dict_basket_writer(self):
        if self._parquet_dict_basket_writer_node_def:
            return self._parquet_dict_basket_writer_node_def

        def _pre_create_hook(engine, memo):
            if self._parquet_output_adapter_manager is None:
                self._create(engine, memo)
            return (engine, memo)

        from csp.impl.wiring.node import _node_internal_use

        @_node_internal_use(cppimpl=_parquetadapterimpl.parquet_dict_basket_writer, pre_create_hook=_pre_create_hook)
        def _parquet_dict_basket_writer(
            column_name: str, writer: object, input: {"K": ts["V"]}, filename_provider: ts[str]
        ):
            raise NotImplementedError()

        self._parquet_dict_basket_writer_node_def = _parquet_dict_basket_writer
        return self._parquet_dict_basket_writer_node_def

    def publish_dict_basket(self, column_name, value, key_type, value_type):
        if key_type is not str:
            raise NotImplementedError("Writing of baskets with non str key type is not supported")
        if self._filename_provider is not None:
            filename_provider = self._filename_provider
        else:
            filename_provider = csp.null_ts(str)

        if (
            CspTypingUtils.is_generic_container(value_type)
            and CspTypingUtils.get_orig_base(value_type) is numpy.ndarray
        ):
            raise NotImplementedError("Writing of baskets with array values is not supported")

        return self._parquet_dict_basket_writer(
            column_name=column_name, writer=self, input=value, filename_provider=filename_provider
        )

    def _get_output_adapter_manager(self):
        return self._parquet_output_adapter_manager

    def _create(self, engine, memo):
        """method needs to return the wrapped c++ adapter manager"""
        if self._parquet_output_adapter_manager:
            return self._parquet_output_adapter_manager

        self._parquet_output_adapter_manager = _parquetadapterimpl._parquet_output_adapter_manager(
            engine, self._properties
        )
        return self._parquet_output_adapter_manager


_parquet_output_adapter_def = output_adapter_def(
    "parquet_output_adapter",
    _parquetadapterimpl._parquet_output_adapter,
    ParquetWriter,
    input=ts["T"],
    typ="T",
    properties=dict,
)
_parquet_output_filename_adapter_def = output_adapter_def(
    "parquet_output_filename_adapter",
    _parquetadapterimpl._parquet_output_filename_adapter,
    ParquetWriter,
    input=ts[str],
)
