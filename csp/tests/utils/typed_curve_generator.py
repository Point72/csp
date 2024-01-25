import datetime

import csp


class _SimpleStruct(csp.Struct):
    value1: int
    value2: float


class _SimpleSubStruct(_SimpleStruct):
    value3: str


class TypedCurveGenerator(object):
    class SimpleEnum(csp.Enum):
        VALUE1 = csp.Enum.auto()
        VALUE2 = csp.Enum.auto()
        VALUE3 = csp.Enum.auto()

    SimpleStruct = _SimpleStruct
    SimpleSubStruct = _SimpleSubStruct

    class NestedStruct(csp.Struct):
        value1: str
        value2: _SimpleStruct

    def __init__(self, period=datetime.timedelta(seconds=1)):
        self._period = period

    def _generate_int_values(
        self, start_value, num_cycles, increment, duplicate_timestamp_indices, output_on_initial_cycle, skip_indices
    ):
        skip_indices = skip_indices if skip_indices else set()
        duplicate_timestamp_indices = duplicate_timestamp_indices if duplicate_timestamp_indices else set()
        start_dtime = datetime.timedelta() if output_on_initial_cycle else self._period
        if num_cycles > 0:
            values = [(start_dtime, start_value)]
            for i in range(1, num_cycles + 1):
                if i in skip_indices:
                    continue
                if i in duplicate_timestamp_indices:
                    dtime = values[-1][0]
                else:
                    dtime = i * self._period
                value = start_value + increment * i
                values.append((dtime, value))
        else:
            values = []
        return values

    @csp.node
    def _transform_int_ts_node(self, input: csp.ts[int], typ: "T", cast_function: object) -> csp.ts["T"]:
        return cast_function(input)

    def _transform_int_to_datetime(self, value):
        hashed_value = hash(str(value))
        return datetime.datetime(
            year=1950 + hashed_value % 100,
            month=hashed_value % 12 + 1,
            day=hashed_value % 30 + 1,
            hour=hashed_value % 24,
            minute=hashed_value % 60,
            second=hashed_value % 60,
            microsecond=hashed_value % 1000000,
        )

    def _transform_int_to_date(self, value):
        hashed_value = hash(str(value))
        return datetime.date(year=1950 + hashed_value % 100, month=hashed_value % 12 + 1, day=hashed_value % 30 + 1)

    def _transform_int_to_timedelta(self, value):
        hashed_value = hash(str(value))
        return datetime.timedelta(
            days=hashed_value % 23,
            seconds=hashed_value % 90,
            microseconds=hashed_value % 10000001,
            milliseconds=hashed_value % 1000001,
            minutes=hashed_value % 101,
            hours=hashed_value % 557,
            weeks=-100 + (hashed_value % 201),
        )

    def _transform_int_to_simple_struct(self, value):
        hashed_value = hash(str(value))
        res = self.SimpleStruct()
        if hashed_value % 3 != 0:
            res.value1 = value
        if hashed_value % 5 != 0:
            res.value2 = value + value / (value + 1)

        return res

    def _transform_int_to_simple_sub_struct(self, value):
        hashed_value = hash(str(value))
        res = self.SimpleSubStruct()
        if hashed_value % 3 != 0:
            res.value1 = value
        if hashed_value % 5 != 0:
            res.value2 = value + value / (value + 1)
        if hashed_value % 2 != 0:
            res.value3 = f"s_{value}"

        return res

    def _transform_int_to_nested_struct(self, value):
        hashed_value = hash(str(value))
        res = self.NestedStruct()
        if hashed_value % 3 != 0:
            res.value1 = str(value)
        if hashed_value % 5 != 0:
            res.value2 = self._transform_int_to_simple_struct(hashed_value % 53)

        return res

    def _transform_int_curve_to_type(self, typ, input: csp.ts[int]):
        if typ is bool:
            cast_func = bool
        elif typ is int:
            cast_func = int
        elif typ is float:
            cast_func = lambda v: v + v / (v + 1)
        elif typ is datetime.datetime:
            cast_func = self._transform_int_to_datetime
        elif typ is datetime.timedelta:
            cast_func = self._transform_int_to_timedelta
        elif typ is datetime.date:
            cast_func = self._transform_int_to_date
        elif typ is str:
            cast_func = str
        elif typ is self.SimpleEnum:
            cast_func = lambda v: self.SimpleEnum(v % 3)
        elif typ is self.SimpleStruct:
            cast_func = self._transform_int_to_simple_struct
        elif typ is self.SimpleSubStruct:
            cast_func = self._transform_int_to_simple_sub_struct
        elif typ is self.NestedStruct:
            cast_func = self._transform_int_to_nested_struct
        else:
            raise NotImplementedError(f"Transformation to {typ} is not supported")
        return self._transform_int_ts_node(input, typ, cast_func)

    def gen_int_curve(
        self,
        start_value,
        num_cycles,
        increment,
        skip_indices=None,
        output_on_initial_cycle: bool = True,
        duplicate_timestamp_indices=None,
    ):
        values = self._generate_int_values(
            start_value, num_cycles, increment, duplicate_timestamp_indices, output_on_initial_cycle, skip_indices
        )

        return csp.curve(int, values)

    def gen_transformed_curve(self, typ, *args, **kwargs):
        return self._transform_int_curve_to_type(typ, self.gen_int_curve(*args, **kwargs))
