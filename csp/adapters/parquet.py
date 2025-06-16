import datetime
from importlib.metadata import PackageNotFoundError, version as get_package_version
from typing import TypeVar

import numpy
import pyarrow
import pyarrow.parquet
from packaging import version

import csp
from csp.adapters.output_adapters.parquet import ParquetOutputConfig, ParquetWriter, resolve_array_shape_column_name
from csp.adapters.status import Status
from csp.impl.types.common_definitions import PushMode
from csp.impl.types.tstype import ts
from csp.impl.types.typing_utils import CspTypingUtils
from csp.impl.wiring import input_adapter_def, status_adapter_def
from csp.impl.wiring.node import node
from csp.lib import _parquetadapterimpl

__all__ = [
    "ParquetOutputConfig",
    "ParquetReader",
    "ParquetWriter",
]

T = TypeVar("T")

try:
    _CAN_READ_ARROW_BINARY = False
    if version.parse(get_package_version("pyarrow")) >= version.parse("4.0.1"):
        _CAN_READ_ARROW_BINARY = True
except (PackageNotFoundError, ValueError, TypeError):
    # Cannot read binary arrow
    ...


class ParquetReader:
    def __init__(
        self,
        filename_or_list,
        symbol_column=None,
        time_column=None,
        binary_arrow=False,
        tz=None,
        start_time=None,
        end_time=None,
        split_columns_to_files=False,
        read_from_memory_tables=False,
        allow_overlapping_periods=False,
        time_shift: datetime.timedelta = None,
        allow_missing_columns=False,
        allow_missing_files=False,
    ):
        """
        :param filename_or_list: The specifier of the file/files to be read. Can be either:
         - Instance of str, in which case it's interpreted os a path of single file to be read
         - A callable, in which case it's interpreted as a generator function that will be called like f(starttime, endtime) where starttime and endtime
           are the start and end times of the current engine run. It's expected to generate a sequence of filenames to read.
         - Iterable container, for example a list of files to read
        :param symbol_column: An optional parameter that specifies the name of the symbol column if the file if there is any
        :param time_column: A mandatory specification of the time column name in the parquet files. This column will be used to inject the row values
        from parquet at the given timestamps.
        :param tz: The pytz timezone of the timestamp column, should only be provided if the time_column in parquet file doesn't have tz info.
        :param start_time: An optional start time of the date, any data prior to this time will be ignored
        :param end_time: An optional end time of the data, any data after this time will be ignored
        :param split_columns_to_files: A boolean flag that specifies that each column should be written to a separate file. If true, then
        each filename in filename_or_list must be a folder where each corresponding column has a file
        :param read_from_memory_tables: A boolean that specifies that filename_or_list provides in-memory arrow tables instead of filenames. The data
        will be read from those tables instead.
        :param allow_overlapping_periods: A boolean that specifies whether the input files are allowed to have overlaps (the default is no, and will cause an error if the file
        contents have overlapping periods). If set to True, each subsequent file, will be read only from the first timestamp that is AFTER the last timestamp that is read from the last
        file.
        :param time_shift: An optional shift of timestamps in the file (if specified, will be added to each timestamp in the file). NOTE: it will be added only for callback time,
        if subscribing to the timestamp columns as data, it will still return the original non shifted values.
        :param allow_missing_columns: A boolean that specify whether it's allowed for some of the read files to have some missing columns.
        :param allow_missing_files: A boolean that specifies that if some of the input files are missing, it's ok, just skip without raising exception.
        """
        if callable(filename_or_list):
            self._filenames_gen = filename_or_list
        elif isinstance(filename_or_list, str):
            self._filenames_gen = lambda starttime, endtime: (f for f in [filename_or_list])
        elif read_from_memory_tables and isinstance(filename_or_list, pyarrow.Table):
            self._filenames_gen = lambda starttime, endtime: (f for f in [filename_or_list])
        else:
            self._filenames_gen = lambda starttime, endtime: (f for f in filename_or_list)
        if read_from_memory_tables:
            if not _CAN_READ_ARROW_BINARY:
                raise TypeError("CSP Cannot load binary arrows derived from pyarrow versions less than 4.0.1")
            wrapped = self._filenames_gen
            self._filenames_gen = lambda starttime, endtime: self._arrow_c_data_interface(wrapped, starttime, endtime)
            binary_arrow = True
        self._properties = {"split_columns_to_files": split_columns_to_files}
        if symbol_column:
            self._properties["symbol_column"] = symbol_column
        if time_column:
            self._properties["time_column"] = time_column
        if tz:
            self._properties["tz"] = tz.zone
        if binary_arrow:
            self._properties["is_arrow_ipc"] = True
        if start_time:
            self._properties["start_time"] = start_time
        if end_time:
            self._properties["end_time"] = end_time

        self._properties["read_from_memory_tables"] = read_from_memory_tables
        if allow_overlapping_periods:
            self._properties["allow_overlapping_periods"] = allow_overlapping_periods

        if time_shift:
            self._properties["time_shift"] = time_shift
        self._properties["allow_missing_columns"] = allow_missing_columns
        self._properties["allow_missing_files"] = allow_missing_files

    @classmethod
    def _arrow_c_data_interface(cls, gen, startime, endtime):
        for v in gen(startime, endtime):
            if not isinstance(v, pyarrow.Table):
                raise TypeError(f"Expected PyTable from generator, got {type(v).__name__}")
            # Use the PyCapsule C data interface to pass data zero copy
            yield v.__arrow_c_stream__()

    @node
    def _reconstruct_struct_array_fields(self, s: ts["T"], fields: {str: ts[object]}, struct_typ: "T") -> ts["T"]:
        if csp.ticked(s):
            res = s
        else:
            res = struct_typ()
        for k, v in fields.tickeditems():
            setattr(res, k, v)
        return res

    def _subscribe_impl(self, symbol, typ, field_map, push_mode, basket_name=None, array_dimensions_column_name=None):
        properties = self._properties.copy()
        if field_map is None:
            if basket_name is None:
                assert isinstance(typ, csp.impl.struct.StructMeta), "field_map must be provided for non struct types"
                field_map = dict(zip(typ.metadata().keys(), typ.metadata().keys()))
            else:
                if issubclass(typ, csp.Struct):
                    field_map = {f"{basket_name}.{k}": k for k in typ.metadata()}
                else:
                    field_map = ""
        if symbol != "":
            properties["symbol"] = symbol
        elif basket_name and symbol is None:
            raise RuntimeError("No symbol is provided for basket subscription")

        array_fields = {}

        if field_map is not None:
            bad_field_map = False
            if isinstance(field_map, dict):
                bad_field_map = not issubclass(typ, csp.Struct)
                if not bad_field_map:
                    meta_typed = typ.metadata(typed=True)
                    new_field_map = {}
                    for k, v in field_map.items():
                        field_typ = meta_typed[v]
                        if CspTypingUtils.is_numpy_array_type(field_typ):
                            array_fields[v] = self.subscribe(symbol, field_typ, k)
                        else:
                            new_field_map[k] = v
                    field_map = new_field_map
            elif not isinstance(field_map, str):
                bad_field_map = True
            if bad_field_map:
                raise ValueError(f"Invalid field_map type {type(field_map)} for type {typ}")

            properties["field_map"] = field_map
        if basket_name:
            properties["basket_name"] = basket_name

        if CspTypingUtils.is_generic_container(typ) and CspTypingUtils.get_orig_base(typ) is numpy.ndarray:
            value_type = typ.__args__[0]
            if CspTypingUtils.get_origin(typ) is csp.typing.NumpyNDArray:
                array_dimensions_column_name = resolve_array_shape_column_name(field_map, array_dimensions_column_name)
                flat_values = self._subscribe_impl(
                    symbol=symbol,
                    typ=csp.typing.Numpy1DArray[typ.__args__[0]],
                    field_map=field_map,
                    push_mode=push_mode,
                )
                shape = self._subscribe_impl(
                    symbol=symbol,
                    typ=csp.typing.Numpy1DArray[int],
                    field_map=array_dimensions_column_name,
                    push_mode=push_mode,
                )

                from csp.adapters.output_adapters.parquet_utility_nodes import reshape_numpy_array

                return reshape_numpy_array(flat_values, shape)
            else:
                properties["array_value_type"] = value_type
                properties["is_array"] = True

        res = _parquet_input_adapter_def(self, typ, properties, push_mode=push_mode)
        if array_fields:
            return self._reconstruct_struct_array_fields(res, array_fields, typ)
        else:
            return res

    def subscribe(
        self,
        symbol,
        typ,
        field_map=None,
        push_mode: PushMode = PushMode.NON_COLLAPSING,
        array_dimensions_column_name=None,
    ):
        """Subscribe to the rows corresponding to a given symbol

        This form of subscription can be used only if non empty symbol_column was supplied during ParquetReader construction.
        :param symbol: The symbol to subscribe to, for example 'AAPL'
        :param typ: The type of the CSP time series subscription. Can either be a primitive type like int or alternatively a type
        that inherits from csp.Struct, in which case each instance of the struct will be constructed from the matching file columns.
        :param field_map: A map of the fields from parquet columns for the CSP time series. If typ is a primitive, then field_map should be
        a string specifying the column name, if typ is a csp Struct then field_map should be a str->str dictionary of the form
        {column_name:struct_field_name}. For stucts field_map can be omitted in which case we expect a one to one match between the given Struct
        fields and the parquet files columns.
        :param push_mode: A push mode for the output adapter
        :param array_dimensions_column_name: When reading array with column name "abc" we can read the dimension array from a column specified by this name.
        If the read array is NumpyNDArray and the column name is not specified, then the default column name will be used by appending the '_csp_dimensions' to the
        column name.
        """
        return self._subscribe_impl(
            symbol, typ, field_map, push_mode, array_dimensions_column_name=array_dimensions_column_name
        )

    def subscribe_all(
        self,
        typ,
        field_map=None,
        push_mode: PushMode = PushMode.NON_COLLAPSING,
        array_dimensions_column_name=None,
    ):
        """Subscribe to all rows of the input files.

        :param typ: The type of the CSP time series subscription. Can either be a primitive type like int or alternatively a type
        that inherits from csp.Struct, in which case each instance of the struct will be constructed from the matching file columns.
        :param field_map: A map of the fields from parquet columns for the CSP time series. If typ is a primitive, then field_map should be
        a string specifying the column name, if typ is a csp Struct then field_map should be a str->str dictionary of the form
        {column_name:struct_field_name}. For stucts field_map can be omitted in which case we expect a one to one match between the given Struct
        fields and the parquet files columns.
        :param push_mode: A push mode for the output adapter
        :param array_dimensions_column_name: When reading array with column name "abc" we can read the dimension array from a column specified by this name.
        If the read array is NumpyNDArray and the column name is not specified, then the default column name will be used by appending the '_csp_dimensions' to the
        column name.
        """
        if field_map is None:
            assert isinstance(typ, csp.impl.struct.StructMeta), "field_map must be provided for non struct types"
            field_map = dict(zip(typ.metadata().keys(), typ.metadata().keys()))

        return self.subscribe(
            "", typ, field_map, push_mode=push_mode, array_dimensions_column_name=array_dimensions_column_name
        )

    def subscribe_dict_basket(self, typ, name, shape, push_mode: PushMode = PushMode.NON_COLLAPSING):
        return {v: self._subscribe_impl(v, typ, None, push_mode, name) for v in shape}

    def subscribe_dict_basket_struct_column(
        self, typ, name, shape, field_name, push_mode: PushMode = PushMode.NON_COLLAPSING
    ):
        # field_type = typ.metadata()[field_name]
        # return {v: self._subscribe_impl(v, field_type, field_name, push_mode, name) for v in shape}
        raise NotImplementedError

    def status(self, push_mode=PushMode.NON_COLLAPSING):
        ts_type = Status
        return status_adapter_def(self, ts_type, push_mode)

    def _create(self, engine, memo):
        """method needs to return the wrapped c++ adapter manager"""
        return _parquetadapterimpl._parquet_input_adapter_manager(engine, self._properties, self._filenames_gen)


_parquet_input_adapter_def = input_adapter_def(
    "parquet_adapter", _parquetadapterimpl._parquet_input_adapter, ts["T"], ParquetReader, typ="T", properties=dict
)
