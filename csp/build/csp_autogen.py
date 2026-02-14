import argparse
import importlib
import importlib.machinery
import importlib.util
import os.path
import sys
import types

# We need to patch mock modules into sys.modules to avoid pulling in csp/__init__.py and all of its baggage, which include imports of
# _cspimpl, which would be a circular dep
spec = importlib.util.find_spec("csp")
loader = importlib.machinery.SourceFileLoader(spec.name, spec.origin)
spec = importlib.util.spec_from_loader("csp", loader)
csp_mod = importlib.util.module_from_spec(spec)
sys.modules["csp"] = csp_mod

from csp.impl.enum import Enum  # noqa: E402
from csp.impl.struct import Struct  # noqa: E402


def struct_type(type_info):
    # If struct pytype is None then we are dealing with a generic csp.Struct field type
    if type_info["pytype"] is None:
        return "csp::StructPtr"
    return f"""csp::autogen::{type_info["pytype"].__name__}::Ptr"""


def enum_type(type_info):
    return f"""csp::autogen::{type_info["pytype"].__name__}"""


def array_type(type_info):
    elem_typeinfo = type_info["elemtype"]
    elemctype = CSP_CPP_TYPE_MAP[elem_typeinfo["type"]]
    if isinstance(elemctype, types.FunctionType):
        elemctype = elemctype(elem_typeinfo)

    return f"""std::vector<{elemctype}>"""


CSP_CPP_TYPE_MAP = {
    "BOOL": "bool",
    "INT8": "int8_t",
    "UINT8": "uint8_t",
    "INT16": "int16_t",
    "UINT16": "uint16_t",
    "INT32": "int32_t",
    "UINT32": "uint32_t",
    "INT64": "int64_t",
    "UINT64": "uint64_t",
    "DOUBLE": "double",
    "DATETIME": "csp::DateTime",
    "TIMEDELTA": "csp::TimeDelta",
    "DATE": "csp::Date",
    "STRING": "std::string",
    "DIALECT_GENERIC": "csp::DialectGenericType",
    "STRUCT": struct_type,
    "ENUM": enum_type,
    "ARRAY": array_type,
}


class CodeGenerator:
    def __init__(self, module_name: str, output_filename: str, namespace: str, generate_imported_types: bool):
        self._module_name = module_name
        self._module = importlib.import_module(module_name)
        self._namespace = namespace

        self._header_filename = f"{output_filename}.h"
        self._cpp_filename = f"{output_filename}.cpp"

        self._struct_types = []
        self._enum_types = []

        self._external_types = {}

        # extract struct and enum types types
        for k, v in self._module.__dict__.items():
            if not isinstance(v, type):
                continue

            if v.__module__ != module_name and not generate_imported_types:
                self._external_types[v] = importlib.import_module(v.__module__)
                continue

            if issubclass(v, Struct) and v is not Struct:
                self._struct_types.append(v)
            elif issubclass(v, Enum) and v is not Enum:
                self._enum_types.append(v)

    def _get_dependent_headers(self):
        """see if there are any dependent headers we need to include"""
        headers = set()
        for struct_type in self._struct_types:
            metainfo = struct_type._metadata_info()
            fields = metainfo["fields"]
            for field in fields:
                type_info = field["type"]
                if type_info["type"] not in ("STRUCT", "ENUM"):
                    continue

                pytype = type_info["pytype"]
                extmod = self._external_types.get(pytype)
                if extmod is not None:
                    headers.add(extmod.CSP_AUTOGEN_HINTS["cpp_header"])

            # Include dependent bases as well
            base_struct = struct_type.__bases__[0] if struct_type.__bases__[0] != Struct else None
            extmod = self._external_types.get(base_struct)
            if extmod is not None:
                headers.add(extmod.CSP_AUTOGEN_HINTS["cpp_header"])

        return headers

    def header_filename(self):
        return self._header_filename

    def cpp_filename(self):
        return self._cpp_filename

    def generate_header_code(self):
        include_guard = "_IN_CSP_AUTOGEN_" + self._module_name.replace(".", "_").upper()
        out = f"""
#ifndef {include_guard}
#define {include_guard}

"""
        out += self._generate_headers()

        out += f"""

namespace {self._namespace}
{{
"""

        for enum_type in self._enum_types:
            out += self._generate_enum_class(enum_type)

        for struct_type in self._struct_types:
            out += self._generate_struct_class(struct_type)

        out += "\n}\n#endif"
        return out

    def _generate_headers(self):
        common_headers = ["csp/core/Exception.h", "csp/engine/Struct.h", "cstddef"]

        common_headers.extend(self._get_dependent_headers())
        return "\n".join(f"#include <{h}>" for h in common_headers)

    def _generate_enum_class(self, enum_type):
        enum_name = enum_type.__name__

        value_decls = ",\n".join(f"        {x.name} = {x.value}" for x in enum_type)
        cspenum_decls = "\n".join(f"    static {enum_name} {x.name};" for x in enum_type)

        out = f"""
class {enum_name} : public csp::CspEnum
{{
public:
    // Raw value quick access
    enum class enum_
    {{
{value_decls}
    }};

    // CspEnum types
{cspenum_decls}

    const char * asCString() const                {{ return name().c_str(); }}
    const std::string & asString() const          {{ return name(); }}

    static {enum_name} create( enum_ v )          {{ return s_meta -> create( ( int64_t ) v ); }}
    static {enum_name} create( const char * name) {{ return s_meta -> fromString( name ); }}
    static {enum_name} create( const std::string & s ) {{ return create( s.c_str() ); }}

    enum_ enum_value() const {{ return ( enum_ ) value(); }}

    static constexpr uint32_t num_types() {{ return {len([x for x in enum_type])}; }}

    static bool static_init();

    {enum_name}( const csp::CspEnum & v ) : csp::CspEnum( v ) {{ CSP_TRUE_OR_THROW( v.meta() == s_meta.get(), AssertionError, "Mismatched enum meta" ); }}

private:

    static std::shared_ptr<csp::CspEnumMeta> s_meta;
}};
"""

        return out

    def _generate_struct_class(self, struct_type):
        struct_name = struct_type.__name__
        metainfo = struct_type._metadata_info()

        base_struct = struct_type.__bases__[0] if struct_type.__bases__[0] != Struct else None

        mask_loc = metainfo["mask_loc"]
        mask_size = metainfo["mask_size"]

        field_getsets = []
        field_decls = []

        fields = metainfo["fields"]
        if base_struct:
            base_metainfo = base_struct._metadata_info()
            base_fields = set(f["fieldname"] for f in base_metainfo["fields"])

            # remove fields from the base, we will derive from it
            fields = [f for f in fields if f["fieldname"] not in base_fields]

        mask_fieldname = f"m_{struct_name}_mask"

        for field in fields:
            fieldname = field["fieldname"]
            type_info = field["type"]
            field_offset = field["offset"]
            field_size = field["size"]
            field_alignment = field["alignment"]
            csp_type = type_info["type"]
            mask_offset = field["mask_offset"]
            mask_bitmask = field["mask_bitmask"]

            ctype = CSP_CPP_TYPE_MAP[csp_type]
            if isinstance(ctype, types.FunctionType):
                ctype = ctype(type_info)

            decl = f"    {ctype} m_{fieldname};\n"
            field_decls.append(decl)

            # index into mask char array
            mask_byte = mask_offset - mask_loc

            field_offset_assert = f"static_assert( offsetof( {struct_name},m_{fieldname} ) == {field_offset} );"
            mask_offset_assert = (
                f"static_assert(( offsetof( {struct_name},{mask_fieldname}) + {mask_byte} ) == {mask_offset} );"
            )

            # Unfortunately our version of GCC doesnt allow offset of on derived members!
            if base_struct is not None:
                field_offset_assert = (
                    "//Moved to runtime check due to offsetof issue on derived types\n        //" + field_offset_assert
                )
                mask_offset_assert = (
                    "//Moved to runtime check due to offsetof issue on derived types\n        //" + mask_offset_assert
                )

            common_setter = f"""
        {field_offset_assert}
        static_assert( alignof( {ctype} ) == {field_alignment} );
        static_assert( sizeof( {ctype} ) == {field_size} );

        {mask_fieldname}[{mask_byte}] |= {mask_bitmask};
"""
            extra_setter = ""
            if ctype == "std::string":
                extra_setter = f"""
    void set_{fieldname}( const char * value )
    {{
        {common_setter}
        m_{fieldname} = value;
    }}

    void set_{fieldname}( std::string_view value )
    {{
        {common_setter}
        m_{fieldname} = value;
    }}
"""
            getset = f"""
    const {ctype} & {fieldname}() const
    {{
        {field_offset_assert}
        static_assert( alignof( {ctype} ) == {field_alignment} );
        static_assert( sizeof( {ctype} ) == {field_size} );

        if( !{fieldname}_isSet() )
            CSP_THROW( csp::ValueError, "field {fieldname} on struct {struct_name} is not set" );

        return m_{fieldname};
    }}

    void set_{fieldname}( const {ctype} & value )
    {{
        {common_setter}

        //TODO employ move semantics where it makes sense
        m_{fieldname} = value;
    }}

    {extra_setter}

    bool {fieldname}_isSet() const
    {{
        {mask_offset_assert}
        return {mask_fieldname}[{mask_byte}] & {mask_bitmask};
    }}

    void clear_{fieldname}()
    {{
        {mask_offset_assert}
        {mask_fieldname}[{mask_byte}] &= ~{mask_bitmask};
    }}
"""
            field_getsets.append(getset)

        field_decls.append(f"    char {mask_fieldname}[{mask_size}];\n")
        getset = "".join(field_getsets)
        decls = "".join(field_decls)

        base_class = "csp::Struct" if base_struct is None else base_struct.__name__

        maskloc_offset_assert = f"static_assert( offsetof( {struct_name}, {mask_fieldname} ) == {mask_loc} );"
        if base_struct is not None:
            maskloc_offset_assert = (
                "//Moved to runtime check due to offsetof issue on derived types\n        //" + maskloc_offset_assert
            )

        out = f"""
class {struct_name} : public {base_class}
{{
public:

    using Ptr = csp::TypedStructPtr<{struct_name}>;

    {struct_name}()  = delete;
    ~{struct_name}() = delete;
    {struct_name}( const {struct_name} & ) = delete;
    {struct_name}( {struct_name} && ) = delete;

    Ptr copy() const {{ return csp::structptr_cast<{struct_name}>( Struct::copy() ); }}

    static {struct_name}::Ptr create()
    {{
        return Ptr( static_cast<{struct_name} *>( s_meta -> createRaw() ) );
    }}

    static const csp::StructMetaPtr & meta() {{ return s_meta; }}

{getset}

    static bool static_init();

private:

{decls}

    static csp::StructMetaPtr s_meta;

    static void assert_mask()
    {{
        {maskloc_offset_assert}
    }}
}};
"""
        return out

    def generate_cpp_code(self):
        struct_inits = []
        for struct_type in self._struct_types:
            struct_name = struct_type.__name__

            # Get derived fields to account for runtime asserts that were lacking!
            assertions = []
            base_struct = struct_type.__bases__[0] if struct_type.__bases__[0] != Struct else None
            if base_struct:
                metainfo = struct_type._metadata_info()
                base_metainfo = base_struct._metadata_info()
                base_fields = set(f["fieldname"] for f in base_metainfo["fields"])
                fields = [f for f in metainfo["fields"] if f["fieldname"] not in base_fields]
                mask_fieldname = f"m_{struct_name}_mask"
                mask_loc = metainfo["mask_loc"]

                for field in fields:
                    fieldname = field["fieldname"]
                    field_offset = field["offset"]
                    mask_offset = field["mask_offset"]
                    mask_byte = mask_offset - mask_loc
                    assertions.append(f"_offsetof( {struct_name},m_{fieldname} ) == {field_offset}")
                    assertions.append(f"( _offsetof( {struct_name},{mask_fieldname}) + {mask_byte} ) == {mask_offset}")

                assertions.append(f"_offsetof( {struct_name}, {mask_fieldname} ) == {mask_loc}")

            assertions = "\n".join(f'    assert_or_die( {assertion}, "{assertion}" );' for assertion in assertions)

            struct_init = f"""
bool {struct_name}::static_init()
{{
{assertions}
    if( Py_IsInitialized() )
    {{
        //Note that windows requires we grab the GIL since the windows DLL loading code releases GIL
        csp::python::AcquireGIL gil;

        // initialize StructMeta from python type if we're in python
        PyObject * pymodule = PyImport_ImportModule( "{self._module_name}" );
        assert_or_die( pymodule != nullptr, "failed to import struct module {self._module_name}" );

        PyObject * structType = PyObject_GetAttrString(pymodule, "{struct_name}" );
        assert_or_die( structType != nullptr, "failed to find struct type {struct_name} in module {self._module_name}" );

        // should add some assertion here..
        csp::python::PyStructMeta * pymeta = ( csp::python::PyStructMeta * ) structType;
        s_meta = pymeta -> structMeta;
    }}

    return true;
}}

bool static_init_{struct_name} = {struct_name}::static_init();
csp::StructMetaPtr {struct_name}::s_meta;
"""

            struct_inits.append(struct_init)

        struct_inits = "\n".join(struct_inits)

        enum_inits = []
        for enum_type in self._enum_types:
            enum_name = enum_type.__name__

            static_decls = "\n".join(
                f'{enum_name} {enum_name}::{x.name} = {enum_name}::create("{x.name}");' for x in enum_type
            )

            enum_init = f"""
bool {enum_name}::static_init()
{{
    if( Py_IsInitialized() )
    {{
        csp::python::AcquireGIL gil;

        // initialize EnumMeta from python type if we're in python
        PyObject * pymodule = PyImport_ImportModule( "{self._module_name}" );
        assert_or_die( pymodule != nullptr, "failed to import struct module {self._module_name}" );

        PyObject * enumType = PyObject_GetAttrString(pymodule, "{enum_name}" );
        assert_or_die( enumType != nullptr, "failed to find num type {enum_name} in module {self._module_name}" );

        // should add some assertion here..
        csp::python::PyCspEnumMeta * pymeta = ( csp::python::PyCspEnumMeta * ) enumType;
        s_meta = pymeta -> enumMeta;
    }}

    return true;
}}

bool static_init_{enum_name} = {enum_name}::static_init();
std::shared_ptr<csp::CspEnumMeta> {enum_name}::s_meta;
{static_decls}
"""

            enum_inits.append(enum_init)

        enum_inits = "\n".join(enum_inits)

        out = f"""
#include "{self._header_filename}"
#include <csp/python/Common.h>
#include <csp/python/PyStruct.h>
#include <csp/python/PyCspEnum.h>
#include <iostream>
#include <stdlib.h>
#include <Python.h>

namespace {self._namespace}
{{

#define _offsetof( C, M ) ( ( char * ) &( ( C * ) nullptr ) -> M - ( char * ) 0 )

static void assert_or_die( bool assertion, const char * error )
{{
    if( !assertion )
    {{
        std::cerr << "Fatal error on import of " << __FILE__ << ": " << error << std::endl;
        if( PyErr_Occurred() )
            PyErr_Print();
        abort();
    }}
}}

{enum_inits}
{struct_inits}
}}
"""

        return out


class Test(Struct):
    bl: bool
    a: int
    b: float
    s: str
    # o: object


class Derived(Test):
    string: str
    flt: float


# Test2 = csp.impl.struct.define_struct( 'Test2', { 'A' + str(i) : bool for i in range(25 )})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", dest="module_name", action="store", required=True, help="module to parse and autogen")
    parser.add_argument("-o", dest="outname", action="store", required=True, help="name of output .h/.cpp files")
    parser.add_argument(
        "-n",
        dest="namespace",
        action="store",
        required=False,
        default="csp::autogen",
        help="c++ namespace to generate in",
    )
    parser.add_argument(
        "-d", dest="output_directory", action="store", required=True, help="directory to write files to"
    )
    parser.add_argument(
        "--generate_imports",
        dest="generate_imports",
        action="store_true",
        required=False,
        help="pass flag to generate types that are imported into the specified module as well",
    )
    parser.add_argument(
        "--dryRun", dest="dry_run", action="store_true", required=False, help="if true write output to stdout"
    )

    args = parser.parse_args()

    struct_gen = CodeGenerator(args.module_name, args.outname, args.namespace, bool(args.generate_imports))

    header_file = os.path.join(args.output_directory, struct_gen.header_filename())
    cpp_file = os.path.join(args.output_directory, struct_gen.cpp_filename())

    header_code = struct_gen.generate_header_code()
    cpp_code = struct_gen.generate_cpp_code()

    if args.dry_run:
        print(header_file, ":")
        print(header_code)
        print(cpp_file)
        print(cpp_code)
    else:
        with open(header_file, "w") as f:
            f.write(header_code)

        with open(cpp_file, "w") as f:
            f.write(cpp_code)
