from csp.impl.struct import Struct


CSP_AUTOGEN_HINTS = {
    'cpp_header' : 'mystruct.h'
}

class MyStruct(Struct):
    a: int
    b: str
