#include <csp/python/PyStructToDict.h>
#include <csp/python/PyIterator.h>

namespace csp::python
{
static thread_local std::unordered_set<const void *> g_tl_ptrsVisited;

class CircularRefCheck {
public:
    CircularRefCheck( const void * ptr ): m_ptr( ptr )
    {
        auto [_, inserted] = g_tl_ptrsVisited.insert( m_ptr );
        if( !inserted )
        {
            CSP_THROW( RecursionError, "Cannot handle objects with circular reference" );
        }
    }

    ~CircularRefCheck()
    {
        g_tl_ptrsVisited.erase( m_ptr );
    }

private:
    const void * m_ptr;
};

// Helper function to convert csp Structs into python objects recursively
PyObjectPtr parseStructToDictRecursive( const StructPtr& self, PyObject * callable );

// Helper function to parse some python objects in cpp, this should not be used extensively.
// instead add support for those python types to csp so that they can be handled more generically
// and in a language agnostic way
PyObjectPtr parsePyObject( PyObject * value, PyObject * callable, bool is_recursing );

// Helper fallback function to convert any type into python object recursively
template<typename T>
inline PyObjectPtr parseCspToPython( const T& val, const CspType& typ, PyObject * callable )
{
    // Default handler for any unknown T
    return PyObjectPtr::own( toPython( val ) );
}

// Helper function to convert Enums into python object recursively
template<>
inline PyObjectPtr parseCspToPython( const CspEnum& val, const CspType& typ, PyObject * callable )
{
    // NOTE: Customization parameter to return the enum instead of string to be added
    return PyObjectPtr::own( toPython( val.name() ) );
}

// Helper function to convert csp Structs into python object recursively
template<>
inline PyObjectPtr parseCspToPython( const StructPtr& val, const CspType& typ, PyObject * callable )
{
    return parseStructToDictRecursive( val, callable );
}

// Helper function to convert python objects in csp Structs into python object recursively
template<>
inline PyObjectPtr parseCspToPython( const DialectGenericType& val, const CspType& typ, PyObject * callable )
{
    auto py_obj = PyObjectPtr::own( toPython<DialectGenericType>( val ) );
    return parsePyObject( py_obj.get(), callable, false );
}

// Helper function to convert arrays in csp Structs into python lists recursively
template<typename StorageT>
inline PyObjectPtr parseCspToPython( const std::vector<StorageT>& val, const CspType& typ, PyObject * callable )
{
    using ElemT = typename CspType::Type::toCArrayElemType<StorageT>::type;

    auto const * arrayType = static_cast<const CspArrayType*>( &typ );
    const CspType * elemType = arrayType -> elemType().get();
    auto new_list = PyObjectPtr::own( PyList_New( val.size() ) );

    for( size_t idx = 0; idx < val.size(); ++idx )
    {
        auto py_obj = parseCspToPython<ElemT>( val[idx], *elemType, callable );
        // PyList_SET_ITEM steals a reference, so we need to release here to avoid
        // having the py_obj deleted when the PyObjectPtr is destroyed
        PyList_SET_ITEM( new_list.get(), idx, py_obj.release() );
    }
    return new_list;
}

PyObjectPtr parseStructToDictRecursive( const StructPtr& self, PyObject * callable )
{
    auto * struct_ptr = self.get();
    CircularRefCheck checker( struct_ptr );

    auto * meta = static_cast<const DialectStructMeta *>( self -> meta() );
    auto new_dict = PyObjectPtr::own( PyDict_New() );
    auto& fields = meta -> fields();

    for( const auto& field: fields )
    {
        // NOTE: Add customization parameter to skip fields starting with underscore("_")
        if( !field -> isSet( struct_ptr ) )
            continue;

        auto& key = field -> fieldname();
        auto py_obj = switchCspType( field -> type(), [field, struct_ptr, callable]( auto tag )
            {
                using CType = typename decltype( tag )::type;
                auto * typedField = static_cast<const typename StructField::upcast<CType>::type *>( field.get() );
                return parseCspToPython( typedField -> value( struct_ptr ), *field -> type(), callable );
            } );
        PyDict_SetItemString( new_dict.get(), key.c_str(), py_obj.get() );
    }

    // Optional postprocess hook in python to allow caller to customize to_dict behavior for struct
    PyObject * py_type = ( PyObject * ) meta -> pyType();
    if( PyObject_HasAttrString( py_type, "postprocess_to_dict" ) )
    {
        auto postprocess_dict_callable = PyObjectPtr::own( PyObject_GetAttrString( py_type, "postprocess_to_dict" ) );
        new_dict = PyObjectPtr::check( PyObject_CallFunction( postprocess_dict_callable.get(), "(O)", new_dict.get() ) );
    }
    return new_dict;
}

class PySequenceIterator
{
public:
    PySequenceIterator( PyObject * iter, PyObject * callable ):
         m_iter( iter ), m_callable( callable )
    {
    }
    PyObject * iternext()
    {
        auto py_obj = PyObjectPtr::own( PyIter_Next( m_iter ) );
        if( py_obj.get() == NULL )
        {
            return NULL;
        }
        else
        {
            auto parsed_obj = parsePyObject( py_obj.get(), m_callable, false );
            return parsed_obj.release();
        }
    }
private:
    PyObject * m_iter;
    PyObject * m_callable;
};

// Helper function to parse python lists/tuples/sets recursively
PyObjectPtr parsePySequence( PyObject * py_seq, PyObject * callable )
{
    CircularRefCheck checker( py_seq );
    auto raw_iter = PyObjectPtr::own( PyObject_GetIter( py_seq ) );
    if( raw_iter.get() == NULL )
        CSP_THROW( ValueError, "Cannot extract iterator from python sequence" );
    PySequenceIterator py_seq_iter( raw_iter.get(), callable );
    auto iter = PyObjectPtr::own( PyIterator<PySequenceIterator>::create( py_seq_iter ) );
    PyTypeObject * orig_type = py_seq -> ob_type;
    return PyObjectPtr::own( PyObject_CallFunction( ( PyObject * ) orig_type, "(O)", iter.get() ) );
}

// Helper function to parse python dicts recursively
PyObjectPtr parsePyDict( PyObject * py_dict, PyObject * callable )
{
    CircularRefCheck checker( py_dict );
    PyObject * py_key = NULL;
    PyObject * py_value = NULL;
    Py_ssize_t pos = 0;
    PyTypeObject * orig_type = py_dict -> ob_type;
    auto parsed_dict = PyObjectPtr::own( PyObject_CallFunction( ( PyObject * ) orig_type, "" ) );
    while( PyDict_Next( py_dict, &pos, &py_key, &py_value ) )
    {
        auto py_obj = parsePyObject( py_value, callable, false );
        PyDict_SetItem( parsed_dict.get(), py_key, py_obj.get() );
    }
    return parsed_dict;
}

PyObjectPtr parsePyObject( PyObject * value, PyObject * callable, bool is_recursing )
{
    INIT_PYDATETIME;

    if( ( value == Py_None ) ||                                                             // None check
        ( PyBool_Check( value ) || PyLong_Check( value ) || PyFloat_Check( value ) ) ||     // Primitives check
        ( PyUnicode_Check( value ) || PyBytes_Check( value ) ) ||                           // Unicode/bytes check
        ( PyTime_CheckExact( value ) || PyDate_CheckExact( value ) ||
          PyDateTime_CheckExact( value ) || PyDelta_CheckExact( value ) ) )                 // Datetime check
        return PyObjectPtr::incref( value );
    else if( PyTuple_Check( value ) || PyList_Check( value ) || PySet_Check( value ) )
        return parsePySequence( value, callable );
    else if( PyDict_Check( value ) )
        return parsePyDict( value, callable );
    else if( PyType_IsSubtype( Py_TYPE( value ), &PyStruct::PyType ) )
    {
        auto struct_ptr = static_cast<PyStruct *>( value ) -> struct_;
        return parseStructToDictRecursive( struct_ptr, callable );
    }
    else if( PyType_IsSubtype( Py_TYPE( value ), &PyCspEnum::PyType ) )
    {
        auto enum_ptr = static_cast<PyCspEnum *>( value ) -> enum_;
        return parseCspToPython( enum_ptr, CspType( CspType::Type::ENUM ), callable );
    }
    else
    {
        if( ( callable == nullptr ) || is_recursing )
        {
            // We are recursing now, just return the object as is
            // NOTE: Any modifications to the returned object will also reflect in the Struct
            return PyObjectPtr::incref( value );
        }
        else
        {
            // Not a known type, try invoking callable
            auto res = PyObjectPtr::check( PyObject_CallFunction( callable, "(O)", value ) );
            return parsePyObject( res.get(), callable, true );
        }
    }
}

PyObjectPtr structToDict( const StructPtr& struct_ptr, PyObject * callable )
{
    // Reset circular reference checker state
    g_tl_ptrsVisited.clear();
    return parseStructToDictRecursive( struct_ptr, callable );
}

}
