import builtins
import enum
import inspect
import types
import typing

from csp.impl.__csptypesimpl import _csptypesimpl


class EnumMeta(_csptypesimpl.PyCspEnumMeta):
    def __new__(cls, name, bases, dct):
        metadata = {}
        last_value = -1

        # Disallow subclassing an enum
        EnumMeta._check_for_existing_members(bases)

        for k, v in dct.items():
            # Allow methods and properties and hidden methods
            if (
                (k[0] == "_" and k[-1] == "_")
                or v is enum.auto
                or isinstance(v, (types.FunctionType, builtins.property, builtins.classmethod, builtins.staticmethod))
            ):
                continue

            if isinstance(v, enum.auto):
                v = last_value + 1

            if not isinstance(v, int):
                raise TypeError(f"csp.Enum expected int enum value, got {type(v).__name__} for field {k}")

            metadata[k] = v
            last_value = v

        dct["__metadata__"] = metadata
        return super().__new__(cls, name, bases, dct)

    def __iter__(self):
        for k, v in self.__metadata__.items():
            yield self(k)

    @property
    def __members__(self):
        # For compatibility with python Enum
        return types.MappingProxyType(self.__metadata__)

    @staticmethod
    def _check_for_existing_members(bases):
        for base in bases:
            if isinstance(base, _csptypesimpl.PyCspEnumMeta) and base.__members__:
                raise TypeError("Cannot extend csp.Enum %r: inheriting from an Enum is prohibited" % base.__name__)


class Enum(_csptypesimpl.PyCspEnum, metaclass=EnumMeta):
    auto = enum.auto

    def __reduce__(self):
        return type(self), (self.value,)

    def __repr__(self):
        return f"<{type(self).__name__}.{self.name}: {self.value}>"

    def __str__(self):
        return f"{type(self).__name__}.{self.name}"

    @classmethod
    def _validate(cls, v) -> "Enum":
        if isinstance(v, cls):
            return v
        elif isinstance(v, str):
            return cls[v]
        elif isinstance(v, int):
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
            description=cls.__doc__ or "An enumeration of {}".format(cls.__name__),
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
    else:
        values = values.copy()

    if module_name is None:
        module_name = inspect.currentframe().f_back.f_globals["__name__"]
    values["__module__"] = module_name

    return EnumMeta(name, (Enum,), values)
