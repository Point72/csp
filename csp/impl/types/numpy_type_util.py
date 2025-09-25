import numpy

_TYPE_MAPPING = {numpy.dtype("float64"): float, numpy.dtype("int64"): int, numpy.dtype("bool"): bool}


def map_numpy_dtype_to_python_type(numpy_dtype):
    if numpy.issubdtype(numpy_dtype, numpy.str_):
        return str
    return _TYPE_MAPPING.get(numpy_dtype, object)
