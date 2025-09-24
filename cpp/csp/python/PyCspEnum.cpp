#include <csp/python/Conversions.h>
#include <csp/python/CspTypeFactory.h>
#include <csp/python/InitHelper.h>
#include <csp/python/PyCspEnum.h>
#include <csp/python/PyObjectPtr.h>

namespace csp::python
{

DialectCspEnumMeta::DialectCspEnumMeta( PyTypeObjectPtr pyType, const std::string & name,
                                        const CspEnumMeta::ValueDef & def ) :
    CspEnumMeta( name, def ),
    m_pyType( pyType )
{
}

/*
MetaClass Madness NOTES!!! -- see PyStruct.cpp for note, same idea
*/

static PyObject * PyCspEnumMeta_new( PyTypeObject *subtype, PyObject *args, PyObject *kwds )
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

    //subtype is python defined CspEnumMeta class
    PyCspEnumMeta * pymeta = ( PyCspEnumMeta * ) PyType_Type.tp_new( subtype, args, kwds );

    //Note that we call ctor without parents so as not to 0-init the base POD PyTypeObject class after its been initialized
    new ( pymeta ) PyCspEnumMeta;

    //this would be the CspEnum class on python side, it doesnt create any metadata for itself
    if( pymeta -> ht_type.tp_base == &PyCspEnum::PyType )
        return ( PyObject * ) pymeta;

    std::string name = PyUnicode_AsUTF8( pyname );

    PyObject * metadata = PyDict_GetItemString( dict, "__metadata__" );
    if( !metadata )
        CSP_THROW( KeyError, "CspEnumMeta missing __metadata__" );

    CspEnumMeta::ValueDef def;

    {
        PyObject *key, *value;
        Py_ssize_t pos = 0;
        while( PyDict_Next( metadata, &pos, &key, &value ) )
        {
            const char * keystr = PyUnicode_AsUTF8( key );
            if( !keystr )
                CSP_THROW( PythonPassthrough, "" );

            if( !PyLong_Check( value ) )
                CSP_THROW( TypeError, "csp.Enum key " << keystr << " expected an integer got " << PyObjectPtr::incref( value ) );

            def[ keystr ] = fromPython<int64_t>( value );
        }
    }

    //back reference to the csp enum type that will be accessible on the csp enum -> meta()
    //intentionally dont incref here to break the circular dep of type -> shared_ptr on CspEnumMeta
    PyTypeObjectPtr typePtr = PyTypeObjectPtr::own( ( PyTypeObject * ) pymeta );
    auto enumMeta = std::make_shared<DialectCspEnumMeta>( typePtr, name, def );

    pymeta -> enumMeta = enumMeta;

    //pre-create instances
    pymeta -> enumsByName  = PyObjectPtr::own( PyDict_New() );
    pymeta -> enumsByValue = PyObjectPtr::own( PyDict_New() );

    for( auto & [ key, value ] : def )
    {
        PyCspEnum * enum_ = ( PyCspEnum * ) ( (PyTypeObject * ) pymeta ) -> tp_alloc( (PyTypeObject * ) pymeta, 0 );

        new( enum_ ) PyCspEnum( enumMeta -> create( value ) );
        enum_ -> enumName  = PyObjectPtr::own( toPython( key ) );
        enum_ -> enumValue = PyObjectPtr::own( toPython<int64_t>( value ) );

        pymeta -> enumsByCValue[ value ] = PyObjectPtr::incref( enum_ );

        if( PyDict_SetItem( pymeta -> enumsByName.get(), enum_ -> enumName.get(), enum_ ) < 0 )
            CSP_THROW( PythonPassthrough, "" );

        if( PyDict_SetItem( pymeta -> enumsByValue.get(), enum_ -> enumValue.get(), enum_ ) < 0 )
            CSP_THROW( PythonPassthrough, "" );

        //We also have to update the items in the actual type's dict so FooEnum.A is a PyCspEnum!
        if( PyDict_SetItem( ( ( PyTypeObject * ) pymeta ) -> tp_dict, enum_ -> enumName.get(), enum_ ) < 0 )
            CSP_THROW( PythonPassthrough, "" );
    }

    return ( PyObject * ) pymeta;
    CSP_RETURN_NULL;
}

PyObject * PyCspEnumMeta::toPyEnum( CspEnum e ) const
{
    auto it = enumsByCValue.find( e.value() );
    if( it == enumsByCValue.end() )
        return nullptr;

    PyObject * rv = it -> second.get();
    Py_INCREF( rv );
    return rv;
}

void PyCspEnumMeta_dealloc( PyCspEnumMeta * m )
{
    CspTypeFactory::instance().removeCachedType( reinterpret_cast<PyTypeObject*>( m ) );
    m -> ~PyCspEnumMeta();
    PyCspEnumMeta::PyType.tp_free( m );
}

PyObject * PyCspEnumMeta_subscript( PyCspEnumMeta * self, PyObject * key )
{
    CSP_BEGIN_METHOD;
   
    PyObject * obj = PyDict_GetItem( self -> enumsByName.get(), key );

    if( !obj )
        CSP_THROW( ValueError, PyObjectPtr::incref( key ) << " is not a valid name on csp.enum type " << ( ( PyTypeObject * ) self ) -> tp_name );

    Py_INCREF( obj );
    return obj;
    CSP_RETURN_NULL;
}


static PyMappingMethods PyCspEnumMeta_MappingMethods = {
    0,                               /*mp_length */
    (binaryfunc) PyCspEnumMeta_subscript, /*mp_subscript */
};

PyTypeObject PyCspEnumMeta::PyType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "_cspimpl.PyCspEnumMeta",  /* tp_name */
    sizeof(PyCspEnumMeta),     /* tp_basicsize */
    0,                         /* tp_itemsize */
    (destructor) PyCspEnumMeta_dealloc, /* tp_dealloc */
    0,                         /* tp_print */
    0,                         /* tp_getattr */
    0,                         /* tp_setattr */
    0,                         /* tp_reserved */
    0,                         /* tp_repr */
    0,                         /* tp_as_number */
    0,                         /* tp_as_sequence */
    &PyCspEnumMeta_MappingMethods, /* tp_as_mapping */
    0,                         /* tp_hash  */
    0,                         /* tp_call */
    0,                         /* tp_str */
    0,                         /* tp_getattro */
    0,                         /* tp_setattro */
    0,                         /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT |
    Py_TPFLAGS_BASETYPE | Py_TPFLAGS_TYPE_SUBCLASS, /* tp_flags */
    "csp enum metaclass",      /* tp_doc */
    0,                         /* tp_traverse */
    0,                         /* tp_clear */
    0,                         /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    0,                         /* tp_iter */
    0,                         /* tp_iternext */
    0,                         /* tp_methods */
    0,                         /* tp_members */
    0,                         /* tp_getset */
    &PyType_Type,              /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    0,                         /*tp_init*/
    0,                         /* tp_alloc */
    (newfunc) PyCspEnumMeta_new,/* tp_new */
    PyObject_GC_Del,           /* tp_free */
};


//PyCspEnum
void PyCspEnum_dealloc( PyCspEnum * self )
{
    self -> ~PyCspEnum();
    PyCspEnum::PyType.tp_free( self );
}

PyObject * PyCspEnum_new( PyTypeObject * type, PyObject *args, PyObject *kwds )
{
    CSP_BEGIN_METHOD;

    PyObject * pyvalue;
    if( !PyArg_ParseTuple( args, "O", &pyvalue ) )
        CSP_THROW( PythonPassthrough, "" );

    auto pymeta = (PyCspEnumMeta * ) type;
    PyObject * obj = nullptr;
    if( PyLong_Check( pyvalue ) )
        obj = PyDict_GetItem( pymeta -> enumsByValue.get(), pyvalue );
    else if( PyUnicode_Check( pyvalue ) )
        obj = PyDict_GetItem( pymeta -> enumsByName.get(), pyvalue );

    if( !obj )
        CSP_THROW( ValueError, PyObjectPtr::incref( pyvalue ) << " is not a valid value on csp.enum type " << type -> tp_name );

    Py_INCREF( obj );
    return obj;
    CSP_RETURN_NULL;
}

PyObject * PyCspEnum_name( PyCspEnum * self, void * )
{
    Py_INCREF( self -> enumName.get() );
    return self -> enumName.get();
}

PyObject * PyCspEnum_value( PyCspEnum * self, void * )
{
    Py_INCREF( self -> enumValue.get() );
    return self -> enumValue.get();
}

static PyGetSetDef PyCspEnum_getset[] = {
    { ( char * ) "name",  (getter) PyCspEnum_name,  0, ( char * ) "string name of the enum instance", 0 },
    { ( char * ) "value", (getter) PyCspEnum_value, 0, ( char * ) "long value of the enum instance", 0 },
    { NULL }
};

PyTypeObject PyCspEnum::PyType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "_cspimpl.PyCspEnum",      /* tp_name */
    sizeof(PyCspEnum),         /* tp_basicsize */
    0,                         /* tp_itemsize */
    (destructor) PyCspEnum_dealloc, /* tp_dealloc */
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
       Py_TPFLAGS_BASETYPE,    /* tp_flags */
    "csp enum",                /* tp_doc */
    0,                         /* tp_traverse */
    0,                         /* tp_clear */
    0,                         /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    0,                         /* tp_iter */
    0,                         /* tp_iternext */
    0,                         /* tp_methods */
    0,                         /* tp_members */
    PyCspEnum_getset,          /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    0,                         /* tp_init */
    0,                         /* tp_alloc */
    (newfunc) PyCspEnum_new,   /* tp_new */
    0,                         /* tp_free */
};

REGISTER_TYPE_INIT( &PyCspEnumMeta::PyType, "PyCspEnumMeta" )
REGISTER_TYPE_INIT( &PyCspEnum::PyType,     "PyCspEnum" )

}
