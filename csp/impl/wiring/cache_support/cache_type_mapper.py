import datetime
from typing import Union

import csp.typing
from csp.impl.types.typing_utils import CspTypingUtils
from csp.utils.qualified_name_utils import QualifiedNameUtils


class CacheTypeMapper:
    STRING_TO_TYPE_MAPPING = {
        "datetime": datetime.datetime,
        "date": datetime.date,
        "timedelta": datetime.timedelta,
        "int": int,
        "float": float,
        "str": str,
        "bool": bool,
    }
    TYPE_TO_STRING_MAPPING = {v: k for k, v in STRING_TO_TYPE_MAPPING.items()}
    ARRAY_TYPE_NAME_TO_TYPE = {
        "ARRAY": csp.typing.Numpy1DArray,
        "MULTI_DIM_ARRAY": csp.typing.NumpyNDArray,
    }
    ARRAY_TYPE_TO_TYPE_NAME = {v: k for k, v in ARRAY_TYPE_NAME_TO_TYPE.items()}

    @classmethod
    def json_to_type(cls, typ: Union[str, dict]):
        if isinstance(typ, str):
            python_type = cls.STRING_TO_TYPE_MAPPING.get(typ)
            if python_type is None:
                python_type = QualifiedNameUtils.get_object_from_qualified_name(typ)
                if python_type is None:
                    raise TypeError(f"Unsupported arrow serialization type {typ}")
            return python_type
        else:
            array_type = None
            if isinstance(typ, dict) and len(typ) == 1:
                typ_key, typ_value = next(iter(typ.items()))
                array_type = cls.ARRAY_TYPE_NAME_TO_TYPE.get(typ_key)
            if array_type is None:
                raise TypeError(f"Trying to deserialize invalid type: {typ}")
            return array_type[cls.json_to_type(typ_value)]

    @classmethod
    def type_to_json(cls, typ):
        str_type = cls.TYPE_TO_STRING_MAPPING.get(typ)
        if str_type is None:
            if CspTypingUtils.is_generic_container(typ):
                origin = CspTypingUtils.get_origin(typ)
                type_name = cls.ARRAY_TYPE_TO_TYPE_NAME.get(origin)
                if type_name is not None:
                    return {type_name: cls.type_to_json(typ.__args__[0])}

            return QualifiedNameUtils.get_qualified_object_name(typ)
        return str_type
