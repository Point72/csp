from csp import PushMode, ts
from csp.impl.wiring import input_adapter_def
from csp.lib import _csvadapterimpl


class CsvAdapterManager:
    def __init__(
        self,
        filename,
        time_column,
        delimiter=",",
        symbol_column="",
        has_header=True,
        time_format=None,  # Use YYYY, MM, DD, hh, mm, ss to represent date/time components
    ):
        self._properties = {
            "filename": filename,
            "time_column": time_column,
            "symbol_column": symbol_column,
            "delimiter": delimiter,
            "hasHeader": has_header,
        }

        if time_format is not None:
            self._properties["time_format"] = time_format

    def subscribe(
        self,
        ts_type,
        field_map=None,
        symbol=None,
        push_mode=PushMode.LAST_VALUE,
    ):
        properties = self._properties.copy()

        if field_map is not None:
            properties["field_map"] = field_map
        properties["symbol"] = symbol or ""

        return _csv_input_adapter_def(
            self,
            ts_type,
            properties,
            push_mode=push_mode,
        )

    def subscribe_all(
        self,
        ts_type,
        field_map=None,
        push_mode=PushMode.LAST_VALUE,
    ):
        return self.subscribe(ts_type, field_map, "", push_mode)  # Legacy API

    def _create(self, engine, memo):
        return _csvadapterimpl._csv_adapter_manager(engine, self._properties)


_csv_input_adapter_def = input_adapter_def(
    "csv_input_adapter",
    _csvadapterimpl._csv_input_adapter,
    ts["T"],
    CsvAdapterManager,
    typ="T",
    properties=dict,
)
