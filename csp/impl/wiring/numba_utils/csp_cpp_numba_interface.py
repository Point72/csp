from cffi import FFI

from csp.impl.__cspimpl import _cspimpl

__all__ = []

ffi = FFI()
ffi.cdef(
    """
    bool __csp_numba_node_ticked__(int64_t node_ptr, int32_t input_index);
    bool __csp_numba_node_valid__(int64_t node_ptr, int32_t input_index);
    bool __csp_numba_node_make_passive__(int64_t node_ptr, int32_t input_index);
    bool __csp_numba_node_make_active__(int64_t node_ptr, int32_t input_index);
    bool __csp_get_node_bool_value__(int64_t node_ptr, int32_t input_index);
    int64_t __csp_get_node_int64_value__(int64_t node_ptr, int32_t input_index);
    double  __csp_get_node_double_value__(int64_t node_ptr, int32_t input_index);
    void __csp_return_bool_value__(int64_t node_ptr, int32_t input_index, bool value);
    void __csp_return_int64_value__(int64_t node_ptr, int32_t input_index, int64_t value);
    void __csp_return_double_value__(int64_t node_ptr, int32_t output_index, double value);    
    int64_t __csp_create_datetime_nanoseconds__(int32_t year, int32_t month, int32_t day,
                                            int32_t hour, int32_t minute, int32_t second,
                                            int32_t nanosecond);
"""
)

C = ffi.dlopen(_cspimpl.__file__)

CSP_NUMBA_CPP_FUNCTIONS = {}

for s in dir(C):
    if s.startswith("__csp_"):
        CSP_NUMBA_CPP_FUNCTIONS[s] = getattr(C, s)


class NumbaTSTypedFunctionResolver:
    _SUPPORTED_TYPE_C_NAMES = {bool: "bool", int: "int64", float: "double"}

    __VALUE_GETTERS__ = {}
    __VALUE_RETURNERS__ = {}

    def __init__(self):
        pass

    @classmethod
    def init_getters_and_setters_maps(cls):
        for typ, typ_name in cls._SUPPORTED_TYPE_C_NAMES.items():
            cls.__VALUE_GETTERS__[typ] = CSP_NUMBA_CPP_FUNCTIONS[f"__csp_get_node_{typ_name}_value__"]
            cls.__VALUE_RETURNERS__[typ] = CSP_NUMBA_CPP_FUNCTIONS[f"__csp_return_{typ_name}_value__"]

    @classmethod
    def get_value_getter_function(cls, typ):
        res = cls.__VALUE_GETTERS__.get(typ)
        if res is None:
            raise RuntimeError(f"Unable to resolve getter function for type {typ}")
        return res

    @classmethod
    def get_value_returner_function(cls, typ):
        res = cls.__VALUE_RETURNERS__.get(typ)
        if res is None:
            raise RuntimeError(f"Unable to resolve return function for type {typ}")
        return res


NumbaTSTypedFunctionResolver.init_getters_and_setters_maps()


def ffi_ptr_to_int(ffi_ptr_obj):
    return int(ffi.cast("int64_t", ffi_ptr_obj))
