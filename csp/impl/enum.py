import sys
import typing
from enum import EnumMeta, IntEnum, auto

_ = EnumMeta


class Enum(IntEnum):
    auto = staticmethod(auto)

    @classmethod
    def _missing_(cls, value):
        # Allow construction by name, ie MyEnum("FOO") as well as MyEnum(1)
        if isinstance(value, str):
            try:
                return cls._member_map_[value]
            except KeyError:
                pass
        return None

    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        """keep old csp.Enum behavior of auto starting at 0"""
        return last_values[-1] + 1 if last_values else 0

    def __str__(self):
        return f"{type(self).__name__}.{self.name}"

    __hash__ = int.__hash__

    def __eq__(self, other):
        return type(self) is type(other) and self.value == other.value

    def __ne__(self, other):
        return not self.__eq__(other)

    @classmethod
    def _validate(cls, v) -> "Enum":
        if isinstance(v, cls):
            return v
        elif isinstance(v, (str, int)):
            return cls(v)
        raise ValueError(f"Cannot convert value to enum: {v}")

    @staticmethod
    def _serialize(value: typing.Union[str, "Enum"]) -> str:
        return value.name

    @classmethod
    def __get_pydantic_json_schema__(cls, _core_schema, handler):
        from pydantic_core import core_schema

        field_schema = handler(core_schema.str_schema())
        field_schema.update(
            type="string",
            title=cls.__name__,
            description="An enumeration of {}".format(cls.__name__),
            enum=list(cls.__members__.keys()),
        )
        return field_schema

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type,
        handler,
    ):
        from pydantic_core import core_schema

        return core_schema.no_info_before_validator_function(
            cls._validate,
            core_schema.any_schema(),
            serialization=core_schema.plain_serializer_function_ser_schema(
                cls._serialize, info_arg=False, return_schema=core_schema.str_schema(), when_used="json"
            ),
        )


def DynamicEnum(name: str, values: typing.Union[dict, list], start=0, module_name=None):
    """create a csp.Enum type dynamically
    :param name: name of the class type
    :param values: either a dictionary of key : values or a list of enum names
    :param start: when providing a list of values, will start enumerating from start
    """
    if isinstance(values, list):
        values = {k: v + start for v, k in enumerate(values)}

    if module_name is None:
        module_name = sys._getframe(1).f_globals["__name__"]

    return Enum(name, values, module=module_name)
