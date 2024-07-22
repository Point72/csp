import datetime
import typing

import numba

from csp.impl.types.typing_utils import CspTypingUtils
from csp.impl.wiring.numba_utils import datetime_extension


class NumbaTypeResolver(object):
    _EPOCH = datetime.datetime.utcfromtimestamp(0)

    _PRIMITIVE_TYPE_MAPPING = {
        bool: numba.types.boolean,
        int: numba.types.int64,
        float: numba.types.double,
        str: numba.types.string,
        datetime.datetime: datetime_extension.csp_numba_datetime_type,
        datetime.timedelta: datetime_extension.csp_numba_timedelta_type,
    }

    @classmethod
    def _resolve_container_type(cls, python_type):
        if CspTypingUtils.get_origin(python_type) is typing.Dict:
            k, v = python_type.__args__
            numba_k = cls.resolve_numba_type(k)
            numba_v = cls.resolve_numba_type(v)
            return numba.types.DictType(numba_k, numba_v)
        elif CspTypingUtils.get_origin(python_type) is typing.List:
            (t,) = python_type.__args__
            numba_t = cls.resolve_numba_type(t)
            return numba.types.ListType(numba_t)
        elif CspTypingUtils.get_origin(python_type) is typing.Set:
            raise NotImplementedError("Set scalars are not currently supported in numba nodes")
        else:
            raise RuntimeError(f"Unable to resolve numba type for {python_type}")

    @classmethod
    def resolve_numba_type(cls, python_type):
        res = cls._PRIMITIVE_TYPE_MAPPING.get(python_type)
        if res is None:
            if CspTypingUtils.is_generic_container(python_type):
                return cls._resolve_container_type(python_type)
            else:
                raise RuntimeError(f"Unable to resolve numba type for {python_type}")
        return res

    @classmethod
    def _instantiate_container(cls, numba_type):
        if isinstance(numba_type, numba.types.DictType):
            return
        # elif
        raise TypeError(f"Don't know how to isntantiate {numba_type}")

    @classmethod
    def transform_scalar(cls, scalar, scalar_type):
        if isinstance(scalar_type, numba.types.DictType):
            res = numba.typed.Dict.empty(
                key_type=scalar_type.key_type,
                value_type=scalar_type.value_type,
            )

            for k, v in scalar.items():
                res[cls.transform_scalar(k, scalar_type.key_type)] = cls.transform_scalar(v, scalar_type.value_type)
            return res
        elif isinstance(scalar_type, numba.types.ListType):
            res = numba.typed.List.empty_list(scalar_type.item_type)
            for v in scalar:
                res.append(v)
            return res

        elif isinstance(scalar, datetime.timedelta):
            return datetime_extension.csp_numba_timedelta_type(int(scalar.total_seconds() * 1e9))
        elif isinstance(scalar, datetime.datetime):
            datetime_extension.csp_numba_datetime_type(int((scalar - cls._EPOCH).total_seconds() * 1e9))
        else:
            return scalar
