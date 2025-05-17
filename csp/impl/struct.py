import io
import typing
from copy import deepcopy

import ruamel.yaml
from deprecated import deprecated

import csp
from csp.impl.__csptypesimpl import _csptypesimpl
from csp.impl.types.container_type_normalizer import ContainerTypeNormalizer
from csp.impl.types.typing_utils import CspTypingUtils, FastList

# Avoid recreating this object every call its expensive!
g_YAML = ruamel.yaml.YAML()


class StructMeta(_csptypesimpl.PyStructMeta):
    def __new__(cls, name, bases, dct):
        full_metadata = {}
        full_metadata_typed = {}
        metadata = {}
        metadata_typed = {}
        defaults = {}

        for base in bases:
            if isinstance(base, StructMeta):
                full_metadata.update(base.__full_metadata__)
                full_metadata_typed.update(base.__full_metadata_typed__)
                defaults.update(base.__defaults__)

        annotations = dct.get("__annotations__", None)
        if annotations:
            for k, v in annotations.items():
                actual_type = v
                # Lists need to be normalized too as potentially we need to add a boolean flag to use FastList
                if v == FastList:
                    raise TypeError(f"{v} annotation is not supported without args")
                if (
                    CspTypingUtils.is_generic_container(v)
                    or CspTypingUtils.is_union_type(v)
                    or CspTypingUtils.is_literal_type(v)
                ):
                    actual_type = ContainerTypeNormalizer.normalized_type_to_actual_python_type(v)
                    if CspTypingUtils.is_generic_container(actual_type):
                        raise TypeError(f"{v} annotation is not supported as a struct field [{actual_type}]")

                if not isinstance(actual_type, type) and not isinstance(actual_type, list):
                    raise TypeError(
                        "struct field '%s' expected field annotation as a type got '%s'" % (k, type(v).__name__)
                    )

                if isinstance(actual_type, list) and (
                    len(actual_type) not in (1, 2)
                    or not isinstance(actual_type[0], type)
                    or (len(actual_type) == 2 and (not isinstance(actual_type[1], bool) or not actual_type[1]))
                    or (isinstance(v, list) and len(actual_type) != 1)
                ):
                    raise TypeError(
                        "struct field '%s' expected list field annotation to be a single-element list of type got '%s'"
                        % (k, (actual_type))
                    )

                metadata_typed[k] = v
                metadata[k] = actual_type

                if k in dct:
                    defaults[k] = dct.pop(k)

        full_metadata.update(metadata)
        full_metadata_typed.update(metadata_typed)
        dct["__full_metadata__"] = full_metadata
        dct["__full_metadata_typed__"] = full_metadata_typed
        dct["__metadata__"] = metadata
        dct["__defaults__"] = defaults

        res = super().__new__(cls, name, bases, dct)
        # This is how we make sure we construct the pydantic schema from the new class
        res.__get_pydantic_core_schema__ = classmethod(res._get_pydantic_core_schema)
        return res

    def layout(self, num_cols=8):
        layout = super()._layout()
        out = ""
        idx = 0
        while idx < len(layout):
            out += layout[idx : idx + num_cols] + "\n"
            idx += num_cols

        out += layout[idx : idx + num_cols]
        return out

    @staticmethod
    def _get_pydantic_core_schema(cls, _source_type, handler):
        """Tell Pydantic how to validate and serialize this Struct class."""
        from pydantic import PydanticSchemaGenerationError
        from pydantic_core import core_schema

        fields = {}
        for field_name, field_type in cls.__full_metadata_typed__.items():
            if field_name.startswith("_"):
                continue  # we skip fields with underscore, like pydantic does
            try:
                field_schema = handler.generate_schema(field_type)
            except PydanticSchemaGenerationError:
                # This logic allows for handling generic types with types we cant get a schema for, only 1 layer deep, same as csp
                item_tp = typing.Any if typing.get_origin(field_type) is None else typing.get_args(field_type)[0]
                try:
                    field_schema = handler.generate_schema(item_tp)
                except PydanticSchemaGenerationError:
                    field_schema = core_schema.any_schema()  # give up finally

            if field_name in cls.__defaults__:
                field_schema = core_schema.with_default_schema(
                    schema=field_schema, default=cls.__defaults__[field_name]
                )
            fields[field_name] = core_schema.typed_dict_field(
                schema=field_schema,
                required=False,  # Make all fields optional
            )
        # Schema for dictionary inputs
        fields_schema = core_schema.typed_dict_schema(
            fields=fields,
            total=False,  # Allow missing fields
            extra_behavior="allow",  # let csp catch extra attributes, allows underscore fields to pass through
        )

        def create_instance(raw_data, validator):
            # We choose to not revalidate, this is the default behavior in pydantic
            if isinstance(raw_data, cls):
                return raw_data
            try:
                return cls(**validator(raw_data))
            except AttributeError as e:
                #  Pydantic can't use AttributeError to check other classes, like in Union annotations
                raise ValueError(str(e)) from None

        def serializer(val, handler):
            # We don't use 'to_dict' since that works recursively, we ignore underscore leading fields
            new_val = {
                k: getattr(val, k) for k in val.__full_metadata_typed__ if not k.startswith("_") and hasattr(val, k)
            }
            return handler(new_val)

        return core_schema.no_info_wrap_validator_function(
            function=create_instance,
            schema=fields_schema,
            serialization=core_schema.wrap_serializer_function_ser_schema(
                function=serializer, schema=fields_schema, when_used="always"
            ),
        )


class Struct(_csptypesimpl.PyStruct, metaclass=StructMeta):
    @classmethod
    def type_adapter(cls):
        # We provide a unique name to make sure that child Structs
        # will get their own type adapters.
        attr_name = f"_{cls.__name__}__pydantic_type_adapter"
        internal_type_adapter = getattr(cls, attr_name, None)
        if internal_type_adapter:
            return internal_type_adapter

        # Late import to avoid autogen issues
        from pydantic import TypeAdapter

        type_adapter = TypeAdapter(cls)
        setattr(cls, attr_name, type_adapter)
        return type_adapter

    @classmethod
    def metadata(cls, typed=False):
        if typed:
            return cls.__full_metadata_typed__
        else:
            return cls.__full_metadata__

    @classmethod
    def fromts(cls, trigger=None, /, **kwargs):
        """convert valid inputs into ts[ struct ]
        trigger - optional position-only argument to control when to sample the inputs and create a struct
                       ( defaults to creating a new struct on any input tick )"""
        import csp

        return csp.struct_fromts(cls, kwargs, trigger)

    @classmethod
    def collectts(cls, **kwargs):
        """convert ticking inputs into ts[ struct ]"""
        import csp

        return csp.struct_collectts(cls, kwargs)

    @classmethod
    def _obj_to_python(cls, obj):
        if isinstance(obj, Struct):
            return {k: cls._obj_to_python(getattr(obj, k)) for k in obj.__full_metadata_typed__ if hasattr(obj, k)}
        elif isinstance(obj, dict):
            return type(obj)({k: cls._obj_to_python(v) for k, v in obj.items()})  # type() for derived dict types
        elif (
            isinstance(obj, (list, tuple, set)) or type(obj).__name__ == "FastList"
        ):  # hack for FastList that is not a list
            return type(obj)(cls._obj_to_python(v) for v in obj)
        elif isinstance(obj, csp.Enum):
            return obj.name  # handled in _obj_from_python
        else:
            return obj

    @classmethod
    def _obj_from_python(cls, json, obj_type):
        obj_type = ContainerTypeNormalizer.normalize_type(obj_type)
        if CspTypingUtils.is_generic_container(obj_type):
            if CspTypingUtils.get_origin(obj_type) in (typing.List, typing.Set, typing.Tuple, FastList):
                return_type = ContainerTypeNormalizer.normalized_type_to_actual_python_type(obj_type)
                # We only take the first item, so like for a Tuple, we would ignore arguments after
                expected_item_type = obj_type.__args__[0]
                return_type = list if isinstance(return_type, list) else return_type
                return return_type(cls._obj_from_python(v, expected_item_type) for v in json)
            elif CspTypingUtils.get_origin(obj_type) is typing.Dict:
                expected_key_type, expected_value_type = obj_type.__args__
                if not isinstance(json, dict):
                    raise TypeError(f"Expected dict, got {type(json)}: {json}")
                return {
                    cls._obj_from_python(k, expected_key_type): cls._obj_from_python(v, expected_value_type)
                    for k, v in json.items()
                }
            elif CspTypingUtils.get_origin(obj_type) in (csp.typing.NumpyNDArray, csp.typing.Numpy1DArray):
                return json
            else:
                raise NotImplementedError(f"Can not deserialize {obj_type} from json")
        elif CspTypingUtils.is_union_type(obj_type):
            return json  ## no checks, just let it through
        elif CspTypingUtils.is_literal_type(obj_type):
            return_type = ContainerTypeNormalizer.normalized_type_to_actual_python_type(obj_type)
            if isinstance(json, return_type):
                return json
            raise ValueError(f"Expected type {return_type} received {json.__class__}")
        elif issubclass(obj_type, Struct):
            if not isinstance(json, dict):
                raise TypeError("Representation of struct as json is expected to be of dict type")
            res = obj_type()
            for k, v in json.items():
                expected_type = obj_type.__full_metadata_typed__.get(k, None)
                if expected_type is None:
                    raise KeyError(f"Unexpected key {k} for type {obj_type}")
                setattr(res, k, cls._obj_from_python(v, expected_type))
            return res
        else:
            if isinstance(json, obj_type):
                return json
            else:
                return obj_type(json)

    @classmethod
    def from_dict(cls, json: dict, use_pydantic: bool = False):
        if use_pydantic:
            return cls.type_adapter().validate_python(json)
        return cls._obj_from_python(json, cls)

    def to_dict_depr(self):
        res = self._obj_to_python(self)
        return res

    #  NOTE: Users can implement this method to customize the output of to_dict
    #  def postprocess_to_dict(self, obj):
    #      """Postprocess hook for to_dict method
    #
    #      This method is invoked by to_dict after converting a struct to a dict
    #      as an additional hook for users to modify the dict before it is returned
    #      by the to_dict method
    #      """
    #      return obj

    def to_dict(self, callback=None, preserve_enums=False):
        """Create a dictionary representation of the struct

        Args:
            callback: Optional function to parse types that are not supported by default in csp and convert them to
                      dicts csp by default can parse Structs, lists, sets, tuples, dicts, datetimes, and primitive types
            preserve_enums: Optional flag to not convert enums to strings when converting structs into dicts
        """
        res = super().to_dict(callback, preserve_enums)
        return res

    def to_json(self, callback=lambda x: x):
        return super().to_json(callback)

    def to_yaml(self):
        string_io = io.StringIO()
        g_YAML.dump(self.to_dict(), string_io)
        return string_io.getvalue()

    @classmethod
    def default_field_map(cls):
        """Used byt adapters to generate default field maps for reading / writing structs"""
        field_map = {}
        for k, v in cls.metadata().items():
            if isinstance(v, type) and issubclass(v, Struct):
                field_map[k] = {k: v.default_field_map()}
            else:
                field_map[k] = k
        return field_map

    @classmethod
    def from_yaml(cls, yaml):
        return cls._obj_from_python(g_YAML.load(yaml), cls)

    def __getstate__(self):
        kwargs = {}
        for k in self.__full_metadata_typed__:
            v = getattr(self, k, csp.UNSET)
            if v is not csp.UNSET:
                kwargs[k] = v
        return kwargs

    def __setstate__(self, state):
        self.update(**state)

    def __deepcopy__(self, memodict={}):
        return self.deepcopy()

    def __dir__(self):
        return sorted(super().__dir__() + list(self.__full_metadata_typed__.keys()))


def define_struct(name, metadata: dict, defaults: dict = {}, base=Struct):
    """Helper method to dynamically create struct types"""

    dct = deepcopy(defaults)
    dct["__annotations__"] = metadata
    clazz = StructMeta(name, (base,), dct)
    return clazz


def define_nested_struct(name, metadata: dict, defaults: dict = {}, base=Struct):
    """Helper method to dynamically create nested struct types.
    metadata and defaults can be a nested dictionaries"""
    metadata = deepcopy(metadata)
    defaults = deepcopy(defaults)
    child_structs = {
        field: define_nested_struct(f"{name}_{field}", submeta, defaults.get(field, {}))
        for field, submeta in metadata.items()
        if isinstance(submeta, dict)
    }
    for fld, struct in child_structs.items():
        if fld in defaults:
            defaults[fld] = struct()
    metadata.update(child_structs)
    return define_struct(name, metadata, defaults, base)


@deprecated(version="0.0.6", reason="Replaced by define_struct")
def defineStruct(name, metadata: dict, defaults: dict = {}, base=Struct):
    return define_struct(name, metadata, defaults, base)


@deprecated(version="0.0.6", reason="Replaced by define_nested_struct")
def defineNestedStruct(name, metadata: dict, defaults: dict = {}, base=Struct):
    return define_nested_struct(name, metadata, defaults, base)
