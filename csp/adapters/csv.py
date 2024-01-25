import csv as pycsv
from datetime import datetime

from csp import PushMode, ts
from csp.impl.adaptermanager import AdapterManagerImpl, ManagedSimInputAdapter
from csp.impl.wiring import py_managed_adapter_def


def time_converter(column, format_string, tz=None):
    def convert(row):
        v = row[column]
        dt = datetime.strptime(v, format_string)
        if tz is not None:
            dt = tz.localize(dt)
        return dt

    return convert


def YYYYMMDD_TIME_formatter(column, include_fraction=False, tz=None):
    format_string = "%Y%m%d %X"
    if include_fraction:
        format_string += ".%f"

    return time_converter(column, format_string, tz)


# GRAPH TIME
class CSVReader:
    ## TODO we might want to support initial snapshot value
    def __init__(self, filename, time_converter, delimiter=",", symbol_column=None):
        self._filename = filename
        self._symbol_column = symbol_column
        self._delimiter = delimiter
        self._time_converter = time_converter

    def subscribe(self, symbol, typ, field_map=None, push_mode=PushMode.NON_COLLAPSING):
        return self._subscribe(symbol, typ, field_map, push_mode)

    def subscribe_all(self, typ, field_map=None, push_mode=PushMode.NON_COLLAPSING):
        return self._subscribe("", typ, field_map, push_mode)

    def _subscribe(self, symbol, typ, field_map, push_mode):
        return CSVReadAdapter(self, symbol, typ, field_map, push_mode=push_mode)

    def _create(self, engine, memo):
        return CSVReaderImpl(engine, self)


# RUN TIME
class CSVReaderImpl(AdapterManagerImpl):
    def __init__(self, engine, adapterRep):
        super().__init__(engine)

        self._rep = adapterRep
        self._inputs = {}
        self._csv_reader = None
        self._next_row = None

    def start(self, starttime, endtime):
        self._csv_reader = pycsv.DictReader(open(self._rep._filename, "r"), delimiter=self._rep._delimiter)
        self._next_row = None

        for row in self._csv_reader:
            time = self._rep._time_converter(row)
            self._next_row = row
            if time >= starttime:
                break

    def stop(self):
        self._csv_reader = None

    def register_input_adapter(self, symbol, adapter):
        if symbol not in self._inputs:
            self._inputs[symbol] = []
        self._inputs[symbol].append(adapter)

    def process_next_sim_timeslice(self, now):
        if not self._next_row:
            return None

        while True:
            time = self._rep._time_converter(self._next_row)
            if time > now:
                return time
            self.process_row(self._next_row)
            try:
                self._next_row = next(self._csv_reader)
            except StopIteration:
                return None

    def process_row(self, row):
        if self._rep._symbol_column is not None:
            symbol = row[self._rep._symbol_column]

            if symbol in self._inputs:
                for input in self._inputs.get(symbol, []):
                    input.process_dict(row)

        # subscribeAll
        for input in self._inputs.get("", []):
            input.process_dict(row)


class CSVReadAdapterImpl(ManagedSimInputAdapter):
    def __init__(self, managerImpl, symbol, typ, field_map):
        managerImpl.register_input_adapter(symbol, self)
        super().__init__(typ, field_map)


CSVReadAdapter = py_managed_adapter_def(
    "csvadapter", CSVReadAdapterImpl, ts["T"], CSVReader, symbol=str, typ="T", fieldMap=(object, None)
)
