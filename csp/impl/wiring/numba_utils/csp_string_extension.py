"""
An implementation of utility functions to work with numba strings
"""

import numba
from numba import cgutils

_BYTES_TYPE = numba.types.Bytes(numba.types.uint8, 1, "C", readonly=True)


def _csp_make_constant_bytes(context, builder, nbytes):
    bstr_ctor = cgutils.create_struct_proxy(_BYTES_TYPE)
    bstr = bstr_ctor(context, builder)

    if isinstance(nbytes, int):
        nbytes = numba.llvmlite.ir.Constant(bstr.nitems.type, nbytes)

    bstr.meminfo = context.nrt.meminfo_alloc(builder, nbytes)
    bstr.nitems = nbytes
    bstr.itemsize = numba.llvmlite.ir.Constant(bstr.itemsize.type, 1)
    bstr.data = context.nrt.meminfo_data(builder, bstr.meminfo)
    bstr.parent = cgutils.get_null_value(bstr.parent.type)
    # bstr.shape and bstr.strides are not used
    bstr.shape = cgutils.get_null_value(bstr.shape.type)
    bstr.strides = cgutils.get_null_value(bstr.strides.type)
    return bstr


@numba.extending.lower_cast(numba.types.UnicodeType, numba.types.Bytes)
def csp_unicode_to_bytes_cast(context, builder, fromty, toty, val):
    uni_str = cgutils.create_struct_proxy(fromty)(context, builder, value=val)
    src1 = builder.bitcast(uni_str.data, numba.llvmlite.ir.IntType(8).as_pointer())
    notkind1 = builder.icmp_unsigned("!=", uni_str.kind, numba.llvmlite.ir.Constant(uni_str.kind.type, 1))
    src_length = uni_str.length

    with builder.if_then(notkind1):
        context.call_conv.return_user_exc(builder, ValueError, ("cannot cast higher than 8-bit unicode_type to bytes",))

    bstr = _csp_make_constant_bytes(context, builder, src_length)
    cgutils.memcpy(builder, bstr.data, src1, bstr.nitems)
    return bstr


@numba.extending.intrinsic
def csp_unicode_to_bytes(typingctx, s):
    # used in _to_bytes method
    assert s == numba.types.unicode_type
    sig = _BYTES_TYPE(s)

    def codegen(context, builder, signature, args):
        return csp_unicode_to_bytes_cast(context, builder, s, _BYTES_TYPE, args[0])._getvalue()

    return sig, codegen
