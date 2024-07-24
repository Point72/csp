import io
import ruamel.yaml
import typing
from copy import deepcopy
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
                if CspTypingUtils.is_generic_container(v):
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

        return super().__new__(cls, name, bases, dct)

    def layout(self, num_cols=8):
        layout = super()._layout()
        out = ""
        idx = 0
        while idx < len(layout):
            out += layout[idx : idx + num_cols] + "\n"
            idx += num_cols

        out += layout[idx : idx + num_cols]
        return out


class Struct(_csptypesimpl.PyStruct, metaclass=StructMeta):
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
                (expected_item_type,) = obj_type.__args__
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
    def from_dict(cls, json: dict):
        return cls._obj_from_python(json, cls)

    def to_dict_depr(self):
        res = self._obj_to_python(self)
        return res

    @classmethod
    def postprocess_to_dict(self, obj):
        """Postprocess hook for to_dict method

        This method is invoked by to_dict after converting a struct to a dict
        as an additional hook for users to modify the dict before it is returned
        by the to_dict method
        """
        return obj

    def to_dict(self, callback=None):
        """Create a dictionary representation of the struct

        Args:
            callback: Optional function to parse types that are not supported by default in csp and convert them to
                      dicts csp by default can parse Structs, lists, sets, tuples, dicts, datetimes, and primitive types
        """
        res = super().to_dict(callback)
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
