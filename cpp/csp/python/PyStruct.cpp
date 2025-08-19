#include <csp/python/Conversions.h>
#include <csp/python/CspTypeFactory.h>
#include <csp/python/InitHelper.h>
#include <csp/python/PyObjectPtr.h>
#include <csp/python/PyStruct.h>
#include <csp/python/PyStructFastList_impl.h>
#include <csp/python/PyStructList_impl.h>
#include <csp/python/PyStructToJson.h>
#include <csp/python/PyStructToDict.h>
#include <unordered_set>
#include <type_traits>

namespace csp::python { class PyObjectStructField; }

namespace csp::python
{

class PyObjectStructField final : public DialectGenericStructField
{
public:
    using BASE = DialectGenericStructField;
    PyObjectStructField( const std::string & name,
                         PyTypeObjectPtr pytype ) : DialectGenericStructField( name, sizeof( PyObjectPtr ), alignof( PyObjectPtr ) ),
                                                    m_pytype( pytype )
    {}


    PyTypeObject * pytype() const { return m_pytype.ptr(); }

    void setValue( Struct * s, const DialectGenericType & obj ) const override
    {
        auto& pyobj = reinterpret_cast<const PyObjectPtr &>(obj);
        if( !PyObject_IsInstance( pyobj.ptr(), ( PyObject * ) m_pytype.ptr() ) )
            CSP_THROW( TypeError, "Invalid " << m_pytype -> tp_name << " type, expected " << m_pytype -> tp_name << " got " <<
                       Py_TYPE( pyobj.ptr() ) -> tp_name << " for field '" << fieldname() << "'" );

        BASE::setValue(s, obj);
    }

private:
    PyTypeObjectPtr m_pytype;
};

DialectStructMeta::DialectStructMeta( PyTypeObject * pyType, const std::string & name,
                                      const Fields & flds, std::shared_ptr<StructMeta> base ) :
    StructMeta( name, flds, base ),
    m_pyType( PyTypeObjectPtr::incref( pyType ) )
{
}

/*
MetaClass Madness NOTES!!!

It took quite a while to get this straight.  PyStructMeta::PyType is the StructMeta type for StructMeta type instances.
Instances of PyStructMeta::PyType are PyStrutMeta types, which are Struct class definitions.
PyStructMeta types ( aka Struct class types ) are forced to inherit from PyStruct::PyType.
This is done in python code where csp.Struct is defined as class Struct( _cspimpl.PyStruct, metaclass = StructMeta ).
StructMeta is defined in python and is a derived type of _cspimpl.PyStructMeta so that we can defined some helper code in python.

This is a lot to take in... so again
In python we have:
class StructMeta( _cspimpl.PyStructMeta ):
    def __new__( ... ):
        #extract meta info
        super().__new__( ..., metainfo )  <-- this calls into PyStructMeta_new in c++, the base type, which will return a new PyStructMeta instance ( a new struct type )

class Struct( _cspimpl.PyStruct, metaclass = StructMeta ):
    # python side methods...

class MyStruct( Struct ):
   a = int
   ...

when MyStruct is declared, it will call StructMeta::__new__ which will call PyStructMeta_new and create the MyStruct type, which is PyStructMeta instance
when MyStruct() instance is created, it'll defer required calls to _cspimpl.PyStruct, its baseclass

Hope this all makes sense when I read this in 3 months
*/

static PyObject * PyStructMeta_new( PyTypeObject *subtype, PyObject *args, PyObject *kwds )
{
    CSP_BEGIN_METHOD;

    PyObject * pyname;
    PyObject * bases;
    PyObject * dict;
    if( !PyArg_ParseTuple( args, "UO!O!",
                           &pyname,
                           &PyTuple_Type, &bases,
                           &PyDict_Type, &dict ) )
        CSP_THROW( PythonPassthrough, "" );

    //subtype is python defined StructMeta class
    PyStructMeta * pymeta = ( PyStructMeta * ) PyType_Type.tp_new( subtype, args, kwds );

    //Note that we call ctor without parents so as not to 0-init the base POD PyTypeObject class after its been initialized
    new ( pymeta ) PyStructMeta;

    //this would be the Struct class on python side, it doesnt create any metastruct for itself
    if( pymeta -> ht_type.tp_base == &PyStruct::PyType )
        return ( PyObject * ) pymeta;

    std::string name = PyUnicode_AsUTF8( pyname );

    PyObject * metadata = PyDict_GetItemString( dict, "__metadata__" );
    if( !metadata )
        CSP_THROW( KeyError, "StructMeta missing __metadata__" );

    StructMeta::Fields fields;
    {
        PyObject *key, *type;
        Py_ssize_t pos = 0;
        while( PyDict_Next( metadata, &pos, &key, &type ) )
        {
            const char * keystr = PyUnicode_AsUTF8( key );
            if( !keystr )
                CSP_THROW( PythonPassthrough, "" );

            if( !PyType_Check( type ) && !PyList_Check( type ) )
                CSP_THROW( TypeError, "Struct metadata for key " << keystr << " expected a type, got " << PyObjectPtr::incref( type ) );

            std::shared_ptr<StructField> field;
            CspTypePtr csptype = pyTypeAsCspType( type );

            switch( csptype -> type() )
            {
                case csp::CspType::Type::BOOL:      field = std::make_shared<BoolStructField>( keystr ); break;
                case csp::CspType::Type::INT64:     field = std::make_shared<Int64StructField>( keystr ); break;
                case csp::CspType::Type::DOUBLE:    field = std::make_shared<DoubleStructField>( keystr ); break;
                case csp::CspType::Type::DATETIME:  field = std::make_shared<DateTimeStructField>( keystr ); break;
                case csp::CspType::Type::TIMEDELTA: field = std::make_shared<TimeDeltaStructField>( keystr ); break;
                case csp::CspType::Type::DATE:      field = std::make_shared<DateStructField>( keystr ); break;
                case csp::CspType::Type::TIME:      field = std::make_shared<TimeStructField>( keystr ); break;
                case csp::CspType::Type::STRING:    field = std::make_shared<StringStructField>( csptype, keystr ); break;
                case csp::CspType::Type::ENUM:      field = std::make_shared<CspEnumStructField>( csptype, keystr ); break;
                case csp::CspType::Type::STRUCT:    field = std::make_shared<StructStructField>( csptype, keystr ); break;
                case csp::CspType::Type::ARRAY:
                {
                    const CspArrayType & arrayType = static_cast<const CspArrayType&>( *csptype );
                    field = ArraySubTypeSwitch::invoke( arrayType.elemType(), [csptype,keystr]( auto tag ) -> std::shared_ptr<StructField>
                    {
                        using CElemType = typename decltype(tag)::type;
                        using CType = typename CspType::Type::toCArrayType<CElemType>::type;
                        return std::make_shared<ArrayStructField<CType>>( csptype, keystr );
                    } );

                    break;
                }

                case csp::CspType::Type::DIALECT_GENERIC: field = std::make_shared<PyObjectStructField>( keystr, PyTypeObjectPtr::incref( ( PyTypeObject * ) type ) ); break;
                default:
                    CSP_THROW( ValueError, "Unexpected csp type " << csptype -> type() << " on struct " << name );
            }

            fields.emplace_back( field );
        }
    }
    std::shared_ptr<StructMeta> metabase = nullptr;
    Py_ssize_t numbases = PyTuple_GET_SIZE( bases );

    for( Py_ssize_t idx = 0; idx < numbases; ++idx )
    {
        PyTypeObject * base = ( PyTypeObject * ) PyTuple_GET_ITEM( bases, idx );

        //Check if base is a PyStruct type.  Note that this might be the python side csp.Struct class, but we dont create structMeta on that
        //( see above ) so it will be null here, no need to check explicitly
        if( PyType_IsSubtype( base, &PyStruct::PyType ) )
        {
            if( metabase )
                CSP_THROW( TypeError, "Struct " << name << " defined with multiple struct bases.  Only single-struct hierarchy is supported" );
            metabase = ( ( PyStructMeta * ) base ) -> structMeta;
        }
    }

    /*back reference to the struct type that will be accessible on the csp struct -> meta()
      DialectStructMeta needs a strong reference to the type.  This creates a known strong circular dep
      whiech effectively will keep the struct type instances around beyond their need

      This is the layout of references between all these types
                              StructMeta (shared_ptr) <-------- strong ref
                                  |                              |
                           DialectStructMeta --> strong ref to PyStructMeta ( the PyType )
                                 /\                              /\
                                  |                              |
                                  | (strong ref )                |
                              Struct (shared_ptr)            python ref to its PyType
                                 /\                              |
                                  |                              |
                                  |                              |
                              PyStruct  --------------------------
    */
    auto structMeta = std::make_shared<DialectStructMeta>( ( PyTypeObject * ) pymeta, name, fields, metabase );

    //Setup fast attr dict lookup
    pymeta -> attrDict = PyObjectPtr::own( PyDict_New() );
    for( auto & field : structMeta -> fields() )
    {
        if( PyDict_SetItem( pymeta -> attrDict.get(),
                            PyObjectPtr::own( PyUnicode_InternFromString( field -> fieldname().c_str() ) ).get(),
                            PyObjectPtr::own( PyCapsule_New( field.get(), nullptr, nullptr ) ).get() ) < 0 )
            CSP_THROW( PythonPassthrough, "" );
    }

    //Setup default image
    PyObject * defaults = PyDict_GetItemString( dict, "__defaults__" );
    if( !defaults )
        CSP_THROW( KeyError, "StructMeta missing __defaults__" );   \

    //only setup default struct if defaults are defined, otherwise we will do the fast 0-init only
    if( PyDict_Size( defaults ) > 0 )
    {
        StructPtr defaultStruct = structMeta -> create();

        PyObject *key, *value;
        Py_ssize_t pos = 0;
        while( PyDict_Next( defaults, &pos, &key, &value ) )
        {
            //Ensure key is properly interned
            Py_INCREF( key );
            PyUnicode_InternInPlace( &key );
            PyStruct::setattr( defaultStruct.get(), key, value );
            Py_DECREF( key );
        }

        structMeta -> setDefault( defaultStruct );
    }

    pymeta -> structMeta = structMeta;

    /*printf( "Struct %s size %lu ( partial %lu )\n", name.c_str(), structMeta -> size(), structMeta -> partialSize() );
    for( auto & field : structMeta -> fields() )
    {
        printf( "\t%s loc %lu size %lu maskloc %lu maskbit %d type %s\n", field -> fieldname().c_str(), field -> offset(), field -> size(), field -> maskOffset(), field -> maskBit(),
                field -> type().asCString() );
                }*/

    return ( PyObject * ) pymeta;
    CSP_RETURN_NULL;
}

void PyStructMeta_dealloc( PyStructMeta * m )
{
    CspTypeFactory::instance().removeCachedType( reinterpret_cast<PyTypeObject*>( m ) );
    m -> ~PyStructMeta();
    PyStructMeta::PyType.tp_free( m );
}

PyObject * PyStructMeta_layout( PyStructMeta * m )
{
    CSP_BEGIN_METHOD;

    return PyUnicode_FromString( m -> structMeta -> layout().c_str() );
    CSP_RETURN_NULL;
}

static PyObjectPtr PyStructMeta_typeinfo( const CspType * type )
{
    PyObjectPtr out = PyObjectPtr::own( PyDict_New() );
    if( PyDict_SetItemString( out.get(), "type", PyObjectPtr::own( toPython( type -> type().asString() ) ).get() ) < 0 )
        CSP_THROW( PythonPassthrough, "" );

    if( PyDict_SetItemString( out.get(), "type_id", PyObjectPtr::own( toPython( type -> type().value() ) ).get() ) < 0 )
        CSP_THROW( PythonPassthrough, "" );

    if( type -> type() == CspType::Type::ENUM )
    {
        auto const * enumType = static_cast<const CspEnumType*>( type );
        auto const * enumMeta = static_cast<const DialectCspEnumMeta*>( enumType -> meta().get() );
        if( PyDict_SetItemString( out.get(), "pytype", ( PyObject * ) enumMeta -> pyType().get() ) < 0 )
            CSP_THROW( PythonPassthrough, "" );
    }
    else if( type -> type() == CspType::Type::STRUCT )
    {
        auto const * structType = static_cast<const CspStructType*>( type );
        auto const * structMeta = static_cast<const DialectStructMeta*>( structType -> meta().get() );
        PyObject * pyType = structMeta ? ( PyObject * ) structMeta -> pyType() : Py_None;
        if( PyDict_SetItemString( out.get(), "pytype", pyType ) < 0 )
            CSP_THROW( PythonPassthrough, "" );
    }
    else if( type -> type() == CspType::Type::ARRAY )
    {
        auto const * arrayType = static_cast<const CspArrayType*>( type );
        if( PyDict_SetItemString( out.get(), "elemtype", PyStructMeta_typeinfo( arrayType -> elemType().get() ).get() ) < 0 )
            CSP_THROW( PythonPassthrough, "" );
    }

    return out;
}

static PyObjectPtr PyStructMeta_fieldinfo( StructField * field )
{
    PyObjectPtr out = PyObjectPtr::own( PyDict_New() );
    if( PyDict_SetItemString( out.get(), "fieldname", PyObjectPtr::own( toPython( field -> fieldname() ) ).get() ) < 0 )
        CSP_THROW( PythonPassthrough, "" );

    if( PyDict_SetItemString( out.get(), "type", PyStructMeta_typeinfo( field -> type().get() ).get() ) < 0 )
        CSP_THROW( PythonPassthrough, "" );

    if( PyDict_SetItemString( out.get(), "offset", PyObjectPtr::own( toPython( static_cast<std::uint64_t>( field -> offset() ) ) ).get() ) < 0 )
        CSP_THROW( PythonPassthrough, "" );

    if( PyDict_SetItemString( out.get(), "size", PyObjectPtr::own( toPython( static_cast<std::uint64_t>( field -> size() ) ) ).get() ) < 0 )
        CSP_THROW( PythonPassthrough, "" );

    if( PyDict_SetItemString( out.get(), "alignment", PyObjectPtr::own( toPython( static_cast<std::uint64_t>( field -> alignment() ) ) ).get() ) < 0 )
        CSP_THROW( PythonPassthrough, "" );

    if( PyDict_SetItemString( out.get(), "mask_offset", PyObjectPtr::own( toPython( static_cast<std::uint64_t>( field -> maskOffset() ) ) ).get() ) < 0 )
        CSP_THROW( PythonPassthrough, "" );

    if( PyDict_SetItemString( out.get(), "mask_bit", PyObjectPtr::own( toPython( field -> maskBit() ) ).get() ) < 0 )
        CSP_THROW( PythonPassthrough, "" );

    if( PyDict_SetItemString( out.get(), "mask_bitmask", PyObjectPtr::own( toPython( field -> maskBitMask() ) ).get() ) < 0 )
        CSP_THROW( PythonPassthrough, "" );

    return out;
}

static PyObject * PyStructMeta_metadata_info( PyStructMeta * m )
{
    PyObjectPtr out = PyObjectPtr::own( PyDict_New() );

    auto * meta = m -> structMeta.get();

    PyObjectPtr fields = PyObjectPtr::own( PyList_New( meta -> fields().size() ) );
    for( size_t i = 0; i < meta -> fields().size(); ++i )
        PyList_SET_ITEM( fields.get(), i, PyStructMeta_fieldinfo( meta -> fields()[i].get() ).release() );

    if( PyDict_SetItemString( out.get(), "fields", fields.get() ) < 0 )
        CSP_THROW( PythonPassthrough, "" );

    if( PyDict_SetItemString( out.get(), "size", PyObjectPtr::own( toPython( static_cast<std::uint64_t>( meta -> size() ) ) ).get() )< 0 )
        CSP_THROW( PythonPassthrough, "" );

    if( PyDict_SetItemString( out.get(), "partial_size", PyObjectPtr::own( toPython( static_cast<std::uint64_t>( meta -> partialSize() ) ) ).get() ) < 0 )
        CSP_THROW( PythonPassthrough, "" );

    if( PyDict_SetItemString( out.get(), "is_native", PyObjectPtr::own( toPython( meta -> isNative() ) ).get() ) < 0 )
        CSP_THROW( PythonPassthrough, "" );

    if( PyDict_SetItemString( out.get(), "mask_loc", PyObjectPtr::own( toPython( static_cast<std::uint64_t>( meta -> maskLoc() ) ) ).get() ) < 0 )
        CSP_THROW( PythonPassthrough, "" );

    if( PyDict_SetItemString( out.get(), "mask_size", PyObjectPtr::own( toPython( static_cast<std::uint64_t>( meta -> maskSize() ) ) ).get() ) < 0 )
        CSP_THROW( PythonPassthrough, "" );

    return out.release();
}

static PyMethodDef PyStructMeta_methods[] = {
    {"_layout",        (PyCFunction) PyStructMeta_layout,        METH_NOARGS,  "debug view of structs internal mem layout"},
    {"_metadata_info", (PyCFunction) PyStructMeta_metadata_info, METH_NOARGS,  "provide detailed information about struct layout"},
    {NULL}
};

PyTypeObject PyStructMeta::PyType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "_cspimpl.PyStructMeta",   /* tp_name */
    sizeof(PyStructMeta),      /* tp_basicsize */
    0,                         /* tp_itemsize */
    (destructor) PyStructMeta_dealloc, /* tp_dealloc */
    0,                         /* tp_print */
    0,                         /* tp_getattr */
    0,                         /* tp_setattr */
    0,                         /* tp_reserved */
    0,                         /* tp_repr */
    0,                         /* tp_as_number */
    0,                         /* tp_as_sequence */
    0,                         /* tp_as_mapping */
    0,                         /* tp_hash  */
    0,                         /* tp_call */
    0,                         /* tp_str */
    0,                         /* tp_getattro */
    0,                         /* tp_setattro */
    0,                         /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT |
    Py_TPFLAGS_BASETYPE | Py_TPFLAGS_TYPE_SUBCLASS, /* tp_flags */
    "csp struct metaclass",    /* tp_doc */
    0,                         /* tp_traverse */
    0,                         /* tp_clear */
    0,                         /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    0,                         /* tp_iter */
    0,                         /* tp_iternext */
    PyStructMeta_methods,      /* tp_methods */
    0,                         /* tp_members */
    0,                         /* tp_getset */
    &PyType_Type,              /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    0,                         /*tp_init*/
    0,//PyType_GenericAlloc,       /* tp_alloc */
    (newfunc) PyStructMeta_new,/* tp_new */
    PyObject_GC_Del,           /* tp_free */
};


//PyStruct
PyObject * getattr_( const StructField * field, const Struct * struct_ )
{
    assert( field -> type() -> type() != CspType::Type::ARRAY );

    PyObject *v = switchCspType( field -> type(), [ field, struct_ ]( auto tag )
    {
        using CType = typename decltype(tag)::type;
        auto *typedField = static_cast<const typename StructField::upcast<CType>::type *>( field );
        return toPython( typedField -> value( struct_ ), *field -> type() );
    } );

    return v;
}

PyObject * getarrayattr_( const StructField * field, const PyStruct * pystruct )
{
    assert( field -> type() -> type() == CspType::Type::ARRAY );
    
    const CspArrayType * arrayType = static_cast<const CspArrayType *>( field -> type().get() );
    PyObject *v = ArraySubTypeSwitch::invoke( arrayType -> elemType(), [ field, pystruct ]( auto tag )
    {
        using StorageT  = typename CspType::Type::toCArrayStorageType<typename decltype(tag)::type>::type;
        using ArrayT    = typename StructField::upcast<std::vector<StorageT>>::type;
        auto * typedField = static_cast<const ArrayT *>( field );
        return toPython( typedField -> value( pystruct -> struct_.get()  ), *field -> type(), pystruct );
    } );
    return v;
}

PyObject * PyStruct::getattr( PyObject * attr )
{
    auto * field = structMeta() -> field( attr );

    if( !field )
        return PyObject_GenericGetAttr( this, attr );

    if( !field -> isSet( struct_.get() ) )
    {
        //For efficiency reasons we set err here rather than rely on exceptions, since this
        //can get called pretty regularly, ie getattr( s, "f", default ) or hasattr checks
        //we also pass the attribute as the exception for efficiency, expensive to format a nice error here
        //that wont get used for hasattr calls
        PyErr_SetObject( PyExc_AttributeError, attr );
        return nullptr;
    }

    if( field -> type() -> type() == CspType::Type::ARRAY )
        return getarrayattr_( field, this );
    return getattr_( field, ( const Struct * ) struct_.get() );
}

void PyStruct::setattr( Struct * s, PyObject * attr, PyObject * value )
{
    auto * field = static_cast<const DialectStructMeta *>( s -> meta() ) -> field( attr );

    if( !field )
        CSP_THROW( AttributeError, "'" << s -> meta() -> name() << "' object has no attribute '" << PyUnicode_AsUTF8( attr ) << "'" );

    try
    {
        switchCspType( field -> type(), [field,&struct_=s,value]( auto tag )
        {
            //workaround for MS compiler bug, separate into two using lines... :/
            using TagType = decltype( tag );
            using CType  = typename TagType::type;
            using fieldType = typename StructField::upcast<CType>::type;
            auto *typedField = static_cast<const fieldType *>( field );

            if( value )
                typedField -> setValue( struct_, fromPython<CType>( value, *field -> type() ) );
            else
                typedField -> clearValue( struct_ );
        } );
    }
    catch( const TypeError & err )
    {
        CSP_THROW( TypeError, "on field '" << PyUnicode_AsUTF8( attr ) << "' : " << err.description() );
    }
}

// Struct printing code

// forward declaration
void repr_struct( const Struct * struct_, std::string & tl_repr, bool show_unset );

// helper functions for formatting to Python standard
void format_bool( const bool val, std::string & tl_repr ) {  tl_repr += ( ( val ? "True" : "False" ) ); }
void format_double( const double val, std::string & tl_repr )
{
    char * data = PyOS_double_to_string( val, 'r', 0, Py_DTSF_ADD_DOT_0, NULL);
    tl_repr += std::string( data );
    PyMem_Free( data );
}
void format_pyobject( const PyObjectPtr & pyptr, std::string & tl_repr )
{
    auto repr = PyObjectPtr::check( PyObject_Repr( pyptr.get() ) );
    tl_repr += ( char * ) PyUnicode_DATA(repr.get() );
}

// type checkers to remove switches and do everything at compile-time
template<class T>
struct is_vector { static bool const value = false; };

template<class T>
struct is_vector<std::vector<T> > { static bool const value = true; };

void repr_field( const Struct * struct_, const StructFieldPtr & field, std::string & tl_repr, bool show_unset )
{
    // Helper function to convert the values contained in a struct to strings
    auto type = field -> type() -> type();
    switch( type )
    {
        case CspType::Type::BOOL:
            format_bool( field -> value<bool>( struct_ ), tl_repr );
            break;
        case CspType::Type::INT8:
            tl_repr += std::to_string( field -> value<int8_t>( struct_ ) );
            break;
        case CspType::Type::UINT8:
            tl_repr += std::to_string( field -> value<uint8_t>( struct_ ) );
            break;
        case CspType::Type::INT16:
            tl_repr += std::to_string( field -> value<int16_t>( struct_ ) );
            break;
        case CspType::Type::UINT16:
            tl_repr += std::to_string( field -> value<uint16_t>( struct_ ) );
            break;
        case CspType::Type::INT32:
            tl_repr += std::to_string( field -> value<int32_t>( struct_ ) );
            break;
        case CspType::Type::UINT32:
            tl_repr += std::to_string( field -> value<uint32_t>( struct_ ) );
            break;
        case CspType::Type::INT64:
            tl_repr += std::to_string( field -> value<int64_t>( struct_ ) );
            break;
        case CspType::Type::UINT64:
            tl_repr += std::to_string( field -> value<uint64_t>( struct_ ) );
            break;
        case CspType::Type::DOUBLE:
            format_double( field -> value<double>( struct_ ), tl_repr );
            break;
        case CspType::Type::STRING:
        {
            auto const * stringType = static_cast<const CspStringType*>( field -> type().get() );
            const auto & val = field -> value<std::string>( struct_ );
            //Use Python's bytes repr for bytes if this is a bytes string
            if( stringType -> isBytes() )
                format_pyobject( PyObjectPtr::own( toPython( val, *stringType ) ), tl_repr );
            else
                tl_repr += val;
            break;
        }
        case CspType::Type::STRUCT:
        {
            StructStructField* as_sf = static_cast<StructStructField*>( field.get() );
            const Struct * val = as_sf -> value( struct_ ).get();
            repr_struct( val, tl_repr, show_unset ); // recursive
            break;
        }
        case CspType::Type::ARRAY:
        {
            auto const * arrayType = static_cast<const CspArrayType*>( field -> type().get() );
            const CspType * elemType = arrayType -> elemType().get();

            switchCspType( elemType, [ field, struct_, &elemType, &tl_repr, show_unset ]( auto tag )
            {
                //workaround for MS compiler bug, separate into two using lines... :/
                using TagType = decltype( tag );
                using CElemType = typename TagType::type;
                using ArrayType = typename CspType::Type::toCArrayType<CElemType>::type;
                const ArrayType & val = field -> value<ArrayType>( struct_ );
                repr_array( val, *elemType, tl_repr, show_unset );
            } );

            break;
        }
        case CspType::Type::DATETIME:
        case CspType::Type::TIMEDELTA:
        case CspType::Type::TIME:
        case CspType::Type::DATE:
        case CspType::Type::ENUM:
        case CspType::Type::DIALECT_GENERIC:
        {
            PyObjectPtr attr = PyObjectPtr::own( getattr_( field.get(), struct_ ) );
            format_pyobject( attr, tl_repr ); // switch to generic PyObject
            break;
        }
        case CspType::Type::UNKNOWN:
        case CspType::Type::NUM_TYPES:
            break;
    }
}

template<typename StorageT>
void repr_array( const std::vector<StorageT> & val, const CspType & elemType, std::string & tl_repr, bool show_unset )
{
    using ElemT = typename CspType::Type::toCArrayElemType<StorageT>::type;
    tl_repr += "[";

    bool first = true;
    for( auto it = val.begin(); it != val.end(); it++ )
    {
        if( unlikely( first ) ) first = false;
        else tl_repr += ", ";

        // print array types
        if constexpr( std::is_same<ElemT, bool>::value )
            format_bool( *it, tl_repr );
        else if constexpr( std::is_same<ElemT, double>::value )
            format_double( *it, tl_repr );
        else if constexpr( std::is_same<ElemT, std::string>::value )
            tl_repr += *it;
        else if constexpr( std::is_same<ElemT, StructPtr>::value )
        {
            const Struct * s = ( *it ).get();
            repr_struct( s, tl_repr, show_unset );
        }
        else if constexpr( std::is_integral<ElemT>::value )
            tl_repr += std::to_string( *it );
        else if constexpr( is_vector<ElemT>::value )
            repr_array( *it, elemType, tl_repr, show_unset ); // recursive, allows for nested arrays!
        else
        {
            // if the element is an enum, generic or datetime type, convert to python
            PyObjectPtr attr = PyObjectPtr::own( toPython( *it, elemType ) );
            format_pyobject( attr, tl_repr );
        }
    }

    tl_repr += "]";
}

void repr_struct( const Struct * struct_, std::string & tl_repr, bool show_unset )
{
    static thread_local std::unordered_set<const Struct *> tl_structReprsVisited;

    auto [_, inserted] = tl_structReprsVisited.insert( struct_ );
    if ( !inserted )
    {
        tl_repr += "( ... )";
        return;
    }

    auto * meta = struct_ -> meta();
    // Helper function to convert a struct to a string representation
    tl_repr += meta -> name();
    tl_repr += "( ";

    // Extract fields and attributes
    bool first = true;
    for( auto & fieldname : meta -> fieldNames() )
    {
        auto & field = meta -> field( fieldname );
        if( !( field -> isSet( struct_ ) ) && !show_unset )
            continue;

        if( unlikely( first ) ) first = false;
        else tl_repr += ", ";

        tl_repr += fieldname;
        tl_repr += "=";
        if( !( field -> isSet( struct_ ) ) )
            tl_repr += "<unset>";
        else
            repr_field( struct_, field, tl_repr, show_unset );
    }
    tl_repr += " )";

    tl_structReprsVisited.erase( struct_ );
}

PyObject * PyStruct::repr( bool show_unset ) const
{
    static thread_local std::string tl_repr;

    // Each struct is responsible for clearing the TLS string after
    size_t offset = tl_repr.size();
    const Struct * val = this -> struct_.get();
    repr_struct( val, tl_repr, show_unset );

    PyObject * rv = PyUnicode_FromString( tl_repr.c_str() + offset );
    tl_repr.erase( offset );
    return rv;
}


/***************************
A note on GC madness...

We ensure to maintain a single PyStruct instance per c++ StructPtr instance.  This instance is cached in the Struct's hidden() data and will
be created on the fly if it doesnt exist when needed ( see Conversions.h )
The instance on the StructPtr does NOT own a python ref.  When the PyStruct refcount drops to 0, or when its garbage collected and destroyed,
we clear the dialectPtr from the c++ struct.
Now in tp_traverse and tp_clear, we ensure not to do traversal or clearing unless we are the only holders of the c++ StructPtr, otherwise we
would end up clearing / freeing data that is still used by the struct ( instance can still exist in c++ / in the timeseries buffers )
The reason we ensure a single PyStruct instance exists is to avoid leaks where we can have a self referencing container with the same struct
twice, ie [ S1, S2 ] where S1 and S2 point to the same StructPtr and hold a ref to the list as well.  If we allowed multiple PyStruct instances, the
refcount would stay at 2 and this would never clean up
****************************/

PyObject * PyStruct_new( PyTypeObject * type, PyObject *args, PyObject *kwds )
{
    CSP_BEGIN_METHOD;

    //base struct class
    if( ! ( (PyStructMeta * ) type ) -> structMeta )
        CSP_THROW( TypeError, "csp.Struct cannot be instantiated" );

    StructPtr struct_ =  ( (PyStructMeta * ) type ) -> structMeta -> create();

    PyStruct * pystruct = ( PyStruct * ) type -> tp_alloc( type, 0 );

    //assign dialectptr, but we DO NOT incref the instance on the struct
    struct_ -> setDialectPtr( pystruct );

    new ( pystruct ) PyStruct( std::move( struct_ ) );

    return pystruct;
    CSP_RETURN_NULL;
}

int PyStruct_tp_clear( PyStruct * self )
{
    //important to only invoke cleanup if we are the actual last refcount holder
    if( self -> struct_ -> refcount() != 1 )
        return 0;

    for( auto & field : self -> structMeta() -> fields() )
    {
        if( field -> type() -> type() == CspType::Type::DIALECT_GENERIC && field -> isSet( self -> struct_.get() ) )
        {
            auto * pyfield = static_cast<PyObjectStructField*>( field.get() );
            PyObject * o = reinterpret_cast<PyObjectPtr &>(pyfield -> value( self -> struct_.get() ) ).ptr();
            Py_XINCREF( o );
            pyfield -> clearValue( self -> struct_.get() );
            Py_CLEAR( o );
        }
    }

    return 0;
}

int PyStruct_traverse( PyStruct * self, visitproc visit, void * arg )
{
    //important to only traverse the actual last refcount holder
    if( self -> struct_ -> refcount() != 1 )
        return 0;

    for( auto & field : self -> structMeta() -> fields() )
    {
        if( field -> type() -> type() == CspType::Type::DIALECT_GENERIC )
        {
            auto * pyfield = static_cast<PyObjectStructField*>( field.get() );
            Py_VISIT( reinterpret_cast<PyObjectPtr &>( pyfield -> value( self -> struct_.get() ) ).ptr() );
        }
    }

    return 0;
}

void PyStruct_dealloc( PyStruct * self )
{
    PyObject_GC_UnTrack( self );
    PyStruct_tp_clear( self );

    //clear the dialectPtr at this point
    self -> struct_ -> setDialectPtr( nullptr );

    self -> ~PyStruct();
    PyStruct::PyType.tp_free( self );
}

void PyStruct_setattrs( PyStruct * self, PyObject * args, PyObject * kwargs, const char * methodName )
{
    if( PyTuple_GET_SIZE( args ) > 0 )
        CSP_THROW( TypeError, "'" << self -> ob_type -> tp_name << '.' << methodName << "' takes 0 positional arguments but " << PyTuple_GET_SIZE( args ) << " were given" );

    if( !kwargs )
        return;

    PyObject *key, *value;
    Py_ssize_t pos = 0;
    while( PyDict_Next( kwargs, &pos, &key, &value ) )
    {
        Py_INCREF( key );
        PyUnicode_InternInPlace( &key );
        self -> setattr( key, value );
        Py_DECREF( key );
    }
}

int PyStruct_init( PyStruct * self, PyObject * args, PyObject * kwargs )
{
    CSP_BEGIN_METHOD;

    PyStruct_setattrs( self, args, kwargs, "__init__" );

    CSP_RETURN_INT;
}

PyObject * PyStruct_update( PyStruct * self, PyObject * args, PyObject * kwargs )
{
    CSP_BEGIN_METHOD;

    PyStruct_setattrs( self, args, kwargs, "update" );

    CSP_RETURN_NONE;
}

PyObject * PyStruct_richcompare( PyStruct * self, PyObject * other, int op )
{
    CSP_BEGIN_METHOD;

    if( !PyType_IsSubtype( other -> ob_type, &PyStruct::PyType ) ||
        ( op != Py_EQ && op != Py_NE ) )
    {
        Py_INCREF( Py_NotImplemented );
        return Py_NotImplemented;
    }

    PyStruct * rhs = ( PyStruct * ) other;

    bool rv = ( *self -> struct_.get() ) == ( *rhs -> struct_.get() );
    rv = ( rv == ( op == Py_EQ ) );
    return toPython( rv );

    CSP_RETURN_NONE;
}

Py_hash_t PyStruct_hash( PyStruct * self )
{
    CSP_BEGIN_METHOD;

    Py_hash_t hash = self -> struct_ -> hash();
    //wheres the "so incredibly unlikely" macro
    if( unlikely( hash == -1 ) )
        hash = 2;

    return hash;

    CSP_RETURN_INT;
}

PyObject * PyStruct_str( PyStruct * self )
{
    CSP_BEGIN_METHOD;

    return self -> repr( true );

    CSP_RETURN_NULL;
}

PyObject * PyStruct_repr( PyStruct * self )
{
    CSP_BEGIN_METHOD;

    return self -> repr( false );

    CSP_RETURN_NULL;
}


PyObject * PyStruct_getattro( PyStruct * self, PyObject * attr )
{
    CSP_BEGIN_METHOD;

    return self -> getattr( attr );

    CSP_RETURN_NULL;
}

int PyStruct_setattro( PyStruct * self, PyObject * attr, PyObject * value )
{
    CSP_BEGIN_METHOD;

    self -> setattr( attr, value );

    CSP_RETURN_INT;
}

PyObject * PyStruct_copy( PyStruct * self )
{
    CSP_BEGIN_METHOD;
    PyObject * copy = self -> ob_type -> tp_alloc( self -> ob_type, 0 );
    new ( copy ) PyStruct( self -> struct_ -> copy() );
    return copy;
    CSP_RETURN_NULL;
}

PyObject * PyStruct_deepcopy( PyStruct * self )
{
    CSP_BEGIN_METHOD;
    //Note that once tp_alloc is called, the object will get added to GC
    //deepcopy traversal may kick in a GC collect, so we have to call that first before the PyStruct is created
    //of it may traverse a partially consturcted object and crash
    auto deepcopy = self -> struct_ -> deepcopy();
    PyObject * pyDeepcopy = self -> ob_type -> tp_alloc( self -> ob_type, 0 );
    new ( pyDeepcopy ) PyStruct( deepcopy );
    return pyDeepcopy;
    CSP_RETURN_NULL;
}

PyObject * PyStruct_clear( PyStruct * self )
{
    CSP_BEGIN_METHOD;
    self -> struct_ -> clear();
    CSP_RETURN_NONE;
}

PyObject * PyStruct_copy_from( PyStruct * self, PyObject * o )
{
    CSP_BEGIN_METHOD;

    if( !PyType_IsSubtype( Py_TYPE( o ), &PyStruct::PyType ) )
        CSP_THROW( TypeError, "Attempting to copy from non-struct type " << Py_TYPE( o ) -> tp_name );

    self -> struct_ -> copyFrom( ( ( PyStruct * ) o ) -> struct_.get() );
    CSP_RETURN_NONE;
}

PyObject * PyStruct_deepcopy_from( PyStruct * self, PyObject * o )
{
    CSP_BEGIN_METHOD;

    if( !PyType_IsSubtype( Py_TYPE( o ), &PyStruct::PyType ) )
        CSP_THROW( TypeError, "Attempting to deepcopy from non-struct type " << Py_TYPE( o ) -> tp_name );

    self -> struct_ -> deepcopyFrom( ( ( PyStruct * ) o ) -> struct_.get() );
    CSP_RETURN_NONE;
}

PyObject * PyStruct_update_from( PyStruct * self, PyObject * o )
{
    CSP_BEGIN_METHOD;

    if( !PyType_IsSubtype( Py_TYPE( o ), &PyStruct::PyType ) )
        CSP_THROW( TypeError, "Attempting to update_from from non-struct type " << Py_TYPE( o ) -> tp_name );

    self -> struct_ -> updateFrom( ( ( PyStruct * ) o ) -> struct_.get() );
    CSP_RETURN_NONE;
}

PyObject * PyStruct_all_fields_set( PyStruct * self )
{
    return toPython( self -> struct_ -> allFieldsSet() );
}

PyObject * PyStruct_to_dict( PyStruct * self, PyObject * args, PyObject * kwargs )
{
    CSP_BEGIN_METHOD;

    // NOTE: Consider grouping customization properties into a dictionary
    PyObject * callable = nullptr;
    int preserve_enums = 0;
    if( PyArg_ParseTuple( args, "Op:to_dict", &callable, &preserve_enums ) )
    {
        if( ( callable != Py_None ) && !PyCallable_Check( callable ) )
        {
            CSP_THROW( TypeError, "Parameter must be callable or None got " + std::string( Py_TYPE( callable ) -> tp_name ) );
        }
    }
    if( callable == Py_None )
        callable = nullptr;
    auto struct_ptr = self -> struct_;
    auto pyobj_ptr = structToDict( struct_ptr, callable, ( preserve_enums != 0 ) );
    return pyobj_ptr.release();

    CSP_RETURN_NULL;
}

PyObject * PyStruct_to_json( PyStruct * self, PyObject * args, PyObject * kwargs )
{
    CSP_BEGIN_METHOD;

    // NOTE: Consider grouping customization properties into a dictionary
    PyObject * callable = nullptr;

    if( PyArg_ParseTuple( args, "O:to_json", &callable ) )
    {
        if( !PyCallable_Check( callable ) )
        {
            CSP_THROW( TypeError, "Parameter must be callable" );
        }
    }
    else
    {
        CSP_THROW( TypeError, "Expected a callable as the argument" );
    }
    auto struct_ptr = self -> struct_;
    auto buffer = structToJson( struct_ptr, callable );
    return toPython( buffer );
    CSP_RETURN_NULL;
}

static PyMethodDef PyStruct_methods[] = {
    { "copy",           (PyCFunction) PyStruct_copy,           METH_NOARGS, "make a shallow copy of the struct" },
    { "deepcopy",       (PyCFunction) PyStruct_deepcopy,       METH_NOARGS, "make a deep copy of the struct" },
    { "clear",          (PyCFunction) PyStruct_clear,          METH_NOARGS, "clear all fields" },
    { "copy_from",      (PyCFunction) PyStruct_copy_from,      METH_O,      "copy from struct. struct must be same type or a derived type. unset fields will copy over" },
    { "deepcopy_from",  (PyCFunction) PyStruct_deepcopy_from,  METH_O,      "deepcopy from struct. struct must be same type or a derived type. unset fields will copy over" },
    { "update_from",    (PyCFunction) PyStruct_update_from,    METH_O,      "update from struct. struct must be same type or a derived type. unset fields will be not be copied" },
    { "update",         (PyCFunction) PyStruct_update,         METH_VARARGS | METH_KEYWORDS, "update from key=val.  given fields will be set on struct.  other fields will remain as is in struct" },
    { "all_fields_set", (PyCFunction) PyStruct_all_fields_set, METH_NOARGS, "return true if all fields on the struct are set" },
    { "to_dict",        (PyCFunction) PyStruct_to_dict,        METH_VARARGS | METH_KEYWORDS, "return a python dict of the struct by recursively converting struct members into python dicts" },
    { "to_json",        (PyCFunction) PyStruct_to_json,        METH_VARARGS | METH_KEYWORDS, "return a json string of the struct by recursively converting struct members into json format" },
    { NULL}
};

PyTypeObject PyStruct::PyType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "_cspimpl.PyStruct",       /* tp_name */
    sizeof(PyStruct),          /* tp_basicsize */
    0,                         /* tp_itemsize */
    ( destructor ) PyStruct_dealloc, /* tp_dealloc */
    0,                         /* tp_print */
    0,                         /* tp_getattr */
    0,                         /* tp_setattr */
    0,                         /* tp_reserved */
    ( reprfunc ) PyStruct_repr,  /* tp_repr */
    0,                         /* tp_as_number */
    0,                         /* tp_as_sequence */
    0,                         /* tp_as_mapping */
    ( hashfunc ) PyStruct_hash,  /* tp_hash  */
    0,                         /* tp_call */
    ( reprfunc ) PyStruct_str,   /* tp_str */
    ( getattrofunc ) PyStruct_getattro, /* tp_getattro */
    ( setattrofunc ) PyStruct_setattro, /* tp_setattro */
    0,                         /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC |
       Py_TPFLAGS_BASETYPE, /* tp_flags */
    "csp struct",              /* tp_doc */
    ( traverseproc ) PyStruct_traverse,         /* tp_traverse */
    ( inquiry ) PyStruct_tp_clear,             /* tp_clear */
    ( richcmpfunc ) PyStruct_richcompare,      /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    0,                         /* tp_iter */
    0,                         /* tp_iternext */
    PyStruct_methods,          /* tp_methods */
    0,                         /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    ( initproc ) PyStruct_init,  /* tp_init */
    PyType_GenericAlloc,       /* tp_alloc */
    ( newfunc ) PyStruct_new,    /* tp_new */
    PyObject_GC_Del,           /* tp_free */
};

REGISTER_TYPE_INIT( &PyStructMeta::PyType, "PyStructMeta" )
REGISTER_TYPE_INIT( &PyStruct::PyType,     "PyStruct" )

// Instantiate all templates for PyStructList class
template struct PyStructList<int8_t>;
template struct PyStructList<uint8_t>;
template struct PyStructList<int16_t>;
template struct PyStructList<uint16_t>;
template struct PyStructList<int32_t>;
template struct PyStructList<uint32_t>;
template struct PyStructList<int64_t>;
template struct PyStructList<uint64_t>;
template struct PyStructList<double>;
template struct PyStructList<DateTime>;
template struct PyStructList<TimeDelta>;
template struct PyStructList<Date>;
template struct PyStructList<Time>;
template struct PyStructList<std::string>;
template struct PyStructList<DialectGenericType>;
template struct PyStructList<StructPtr>;
template struct PyStructList<CspEnum>;

// Instantiate all templates for PyStructFastList class
template struct PyStructFastList<int8_t>;
template struct PyStructFastList<uint8_t>;
template struct PyStructFastList<int16_t>;
template struct PyStructFastList<uint16_t>;
template struct PyStructFastList<int32_t>;
template struct PyStructFastList<uint32_t>;
template struct PyStructFastList<int64_t>;
template struct PyStructFastList<uint64_t>;
template struct PyStructFastList<double>;
template struct PyStructFastList<DateTime>;
template struct PyStructFastList<TimeDelta>;
template struct PyStructFastList<Date>;
template struct PyStructFastList<Time>;
template struct PyStructFastList<std::string>;
template struct PyStructFastList<DialectGenericType>;
template struct PyStructFastList<StructPtr>;
template struct PyStructFastList<CspEnum>;

}
