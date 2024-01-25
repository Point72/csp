from enum import Enum, auto
from typing import Dict

from csp.impl.struct import Struct
from csp.impl.wiring.cache_support.cache_type_mapper import CacheTypeMapper


class OutputType(Enum):
    PARQUET = auto()


class TimeAggregation(Enum):
    DAY = auto()
    MONTH = auto()
    QUARTER = auto()
    YEAR = auto()


class DictBasketInfo(Struct):
    key_type: object
    value_type: object

    @classmethod
    def _postprocess_dict_to_python(cls, d):
        d["key_type"] = CacheTypeMapper.type_to_json(d["key_type"])
        d["value_type"] = CacheTypeMapper.type_to_json(d["value_type"])
        return d

    @classmethod
    def _preprocess_dict_from_python(cls, d):
        d["key_type"] = CacheTypeMapper.json_to_type(d["key_type"])
        d["value_type"] = CacheTypeMapper.json_to_type(d["value_type"])
        return d


class DatasetMetadata(Struct):
    version: str = "1.0.0"
    name: str
    output_type: OutputType = OutputType.PARQUET
    time_aggregation: TimeAggregation = TimeAggregation.DAY
    columns: Dict[str, object]
    dict_basket_columns: Dict[str, DictBasketInfo]
    partition_columns: Dict[str, type]
    timestamp_column_name: str
    split_columns_to_files: bool = False

    @classmethod
    def _postprocess_dict_to_python(cls, d):
        output_type = d.get("output_type")
        if output_type is not None:
            d["output_type"] = output_type.name
        time_aggregation = d.get("time_aggregation")
        if time_aggregation is not None:
            d["time_aggregation"] = time_aggregation.name
        columns = d["columns"]
        if columns:
            d["columns"] = {k: CacheTypeMapper.type_to_json(v) for k, v in columns.items()}

        partition_columns = d.get("partition_columns")
        if partition_columns:
            d["partition_columns"] = {k: CacheTypeMapper.type_to_json(v) for k, v in partition_columns.items()}

        return d

    @classmethod
    def _preprocess_dict_from_python(cls, d):
        output_type = d.get("output_type")
        if output_type is not None:
            d["output_type"] = OutputType[output_type]
        time_aggregation = d.get("time_aggregation")
        if time_aggregation is not None:
            d["time_aggregation"] = TimeAggregation[time_aggregation]
        columns = d["columns"]
        if columns:
            d["columns"] = {k: CacheTypeMapper.json_to_type(v) for k, v in columns.items()}
        partition_columns = d.get("partition_columns")
        if partition_columns:
            d["partition_columns"] = {k: CacheTypeMapper.json_to_type(v) for k, v in partition_columns.items()}

        return d

    @classmethod
    def load_metadata(cls, file_path: str):
        with open(file_path, "r") as f:
            return DatasetMetadata.from_yaml(f.read())
