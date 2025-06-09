#include <csp/python/PyStructToDict.h>
#include <csp/python/PyIterator.h>

namespace csp::python
{
class StructToDictHelper
{
public:
    StructToDictHelper( bool preserve_enums, PyObject * callable ):
        m_preserveEnums( preserve_enums ),
        m_callable( callable )
    {
        m_ptrsVisited.clear();
    }

    PyObjectPtr toDict( const StructPtr& struct_ptr )
    {
        return parseStructToDictRecursive( struct_ptr );
    }

private:
    // Helper fallback function to convert any type into python object recursively
    template<typename T>
    inline PyObjectPtr parseCspToPython( const T& val, const CspType& typ );

    // Helper fallback function to convert any type into python object recursively
    template<typename StorageT>
    inline PyObjectPtr parseCspToPython( const std::vector<StorageT>& val, const CspType& typ );

    PyObjectPtr parseStructToDictRecursive( const StructPtr& self );

    // Helper function to parse python lists/tuples/sets recursively
    PyObjectPtr parsePySequence( PyObject * py_seq );

    // Helper function to parse python dicts recursively
    PyObjectPtr parsePyDict( PyObject * py_dict );

    // Helper function to parse some python objects in cpp, this should not be used extensively.
    // instead add support for those python types to csp so that they can be handled more generically
    // and in a language agnostic way
    PyObjectPtr parsePyObject( PyObject * value, bool is_recursing );

    class CircularRefCheck
    {
    public:
        CircularRefCheck( std::unordered_set<const void *>& ptrs_visited, const void * ptr ): m_ptrsVisited( ptrs_visited ), m_ptr( ptr )
        {
            auto [_, inserted] = m_ptrsVisited.insert( m_ptr );
            if( !inserted )
            {
                CSP_THROW( RecursionError, "Cannot handle objects with circular reference" );
            }
        }

        ~CircularRefCheck() { m_ptrsVisited.erase( m_ptr ); }

    private:
        std::unordered_set<const void *>& m_ptrsVisited;
        const void * m_ptr;
    };

    class PySequenceIterator
    {
    public:
        PySequenceIterator( PyObject * iter, StructToDictHelper * helper ):
            m_iter( iter ), m_helper( helper )
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
                auto parsed_obj = m_helper -> parsePyObject( py_obj.get(), false );
                return parsed_obj.release();
            }
        }
    private:
        PyObject * m_iter;
        StructToDictHelper * m_helper;
    };

    std::unordered_set<const void *> m_ptrsVisited;
    bool m_preserveEnums;
    PyObject * m_callable;

    friend class PySequenceIterator;
};

// Helper fallback function to convert any type into python object recursively
template<typename T>
inline PyObjectPtr StructToDictHelper::parseCspToPython( const T& val, const CspType& typ )
{
    // Default handler for any unknown T
    return PyObjectPtr::own( toPython( val ) );
}

// Helper function to convert Enums into python object recursively
template<>
inline PyObjectPtr StructToDictHelper::parseCspToPython<CspEnum>( const CspEnum& val, const CspType& typ )
{
    if( m_preserveEnums )
        return PyObjectPtr::own( toPython( val, typ ) );
    else
        return PyObjectPtr::own( toPython( val.name() ) );
}

// Helper function to convert csp Structs into python object recursively
template<>
inline PyObjectPtr StructToDictHelper::parseCspToPython<StructPtr>( const StructPtr& val, const CspType& typ )
{
    return parseStructToDictRecursive( val );
}

// Helper function to convert python objects in csp Structs into python object recursively
template<>
inline PyObjectPtr StructToDictHelper::parseCspToPython<DialectGenericType>( const DialectGenericType& val, const CspType& typ )
{
    auto py_obj = PyObjectPtr::own( toPython<DialectGenericType>( val ) );
    return parsePyObject( py_obj.get(), false );
}

// Helper function to convert arrays in csp Structs into python lists recursively
template<typename StorageT>
inline PyObjectPtr StructToDictHelper::parseCspToPython( const std::vector<StorageT>& val, const CspType& typ )
{
    using ElemT = typename CspType::Type::toCArrayElemType<StorageT>::type;

    auto const * arrayType = static_cast<const CspArrayType*>( &typ );
    const CspType * elemType = arrayType -> elemType().get();
    auto new_list = PyObjectPtr::own( PyList_New( val.size() ) );

    for( size_t idx = 0; idx < val.size(); ++idx )
    {
        auto py_obj = parseCspToPython<ElemT>( val[idx], *elemType );
        // PyList_SET_ITEM steals a reference, so we need to release here to avoid
        // having the py_obj deleted when the PyObjectPtr is destroyed
        PyList_SET_ITEM( new_list.get(), idx, py_obj.release() );
    }
    return new_list;
}

PyObjectPtr StructToDictHelper::parseStructToDictRecursive( const StructPtr& self )
{
    auto * struct_ptr = self.get();
    CircularRefCheck checker( m_ptrsVisited, struct_ptr );

    auto * meta = static_cast<const DialectStructMeta *>( self -> meta() );
    auto new_dict = PyObjectPtr::own( PyDict_New() );
    auto& fields = meta -> fields();

    for( const auto& field: fields )
    {
        // NOTE: Add customization parameter to skip fields starting with underscore("_")
        if( !field -> isSet( struct_ptr ) )
            continue;

        auto& key = field -> fieldname();
        auto py_obj = switchCspType( field -> type(), [field, struct_ptr, this]( auto tag )
            {
                using CType = typename decltype( tag )::type;
                auto * typedField = static_cast<const typename StructField::upcast<CType>::type *>( field.get() );
                return this -> parseCspToPython( typedField -> value( struct_ptr ), *field -> type() );
            } );
        PyDict_SetItemString( new_dict.get(), key.c_str(), py_obj.get() );
    }

    // Optional postprocess hook in python to allow caller to customize to_dict behavior for struct
    auto py_struct = PyObjectPtr::own( toPython( self ) );
    if( PyObject_HasAttrString( py_struct.get(), "postprocess_to_dict" ) )
    {
        auto postprocess_dict_callable = PyObjectPtr::own( PyObject_GetAttrString( py_struct.get(), "postprocess_to_dict" ) );
        new_dict = PyObjectPtr::check( PyObject_CallFunction( postprocess_dict_callable.get(), "(O)", new_dict.get() ) );
    }
    return new_dict;
}

// Helper function to parse python lists/tuples/sets recursively
PyObjectPtr StructToDictHelper::parsePySequence( PyObject * py_seq )
{
    CircularRefCheck checker( m_ptrsVisited, py_seq );
    auto raw_iter = PyObjectPtr::own( PyObject_GetIter( py_seq ) );
    if( raw_iter.get() == NULL )
        CSP_THROW( ValueError, "Cannot extract iterator from python sequence" );
    PySequenceIterator py_seq_iter( raw_iter.get(), this );
    auto iter = PyObjectPtr::own( PyIterator<PySequenceIterator>::create( py_seq_iter ) );
    PyTypeObject * orig_type = py_seq -> ob_type;
    return PyObjectPtr::own( PyObject_CallFunction( ( PyObject * ) orig_type, "(O)", iter.get() ) );
}

// Helper function to parse python dicts recursively
PyObjectPtr StructToDictHelper::parsePyDict( PyObject * py_dict )
{
    CircularRefCheck checker( m_ptrsVisited, py_dict );
    PyObject * py_key = NULL;
    PyObject * py_value = NULL;
    Py_ssize_t pos = 0;
    PyTypeObject * orig_type = py_dict -> ob_type;
    auto parsed_dict = PyObjectPtr::own( PyObject_CallFunction( ( PyObject * ) orig_type, "" ) );
    while( PyDict_Next( py_dict, &pos, &py_key, &py_value ) )
    {
        auto py_obj = parsePyObject( py_value, false );
        PyDict_SetItem( parsed_dict.get(), py_key, py_obj.get() );
    }
    return parsed_dict;
}

// Helper function to parse some python objects in cpp, this should not be used extensively.
// instead add support for those python types to csp so that they can be handled more generically
// and in a language agnostic way
PyObjectPtr StructToDictHelper::parsePyObject( PyObject * value, bool is_recursing )
{
    INIT_PYDATETIME;

    if( ( value == Py_None ) ||                                                             // None check
        ( PyBool_Check( value ) || PyLong_Check( value ) || PyFloat_Check( value ) ) ||     // Primitives check
        ( PyUnicode_Check( value ) || PyBytes_Check( value ) ) ||                           // Unicode/bytes check
        ( PyTime_CheckExact( value ) || PyDate_CheckExact( value ) ||
        PyDateTime_CheckExact( value ) || PyDelta_CheckExact( value ) ) )                 // Datetime check
        return PyObjectPtr::incref( value );
    else if( PyTuple_Check( value ) || PyList_Check( value ) || PySet_Check( value ) )
        return parsePySequence( value );
    else if( PyDict_Check( value ) )
        return parsePyDict( value );
    else if( PyType_IsSubtype( Py_TYPE( value ), &PyStruct::PyType ) )
    {
        auto struct_ptr = static_cast<PyStruct *>( value ) -> struct_;
        return parseStructToDictRecursive( struct_ptr );
    }
    else if( PyType_IsSubtype( Py_TYPE( value ), &PyCspEnum::PyType ) )
    {
        auto enum_ptr = static_cast<PyCspEnum *>( value ) -> enum_;
        PyObject* py_type = ( PyObject* ) ( Py_TYPE( value ) );
        return parseCspToPython( enum_ptr, *pyTypeAsCspType( py_type ) );
    }
    else
    {
        if( ( m_callable == nullptr ) || is_recursing )
        {
            // We are recursing now, just return the object as is
            // NOTE: Any modifications to the returned object will also reflect in the Struct
            return PyObjectPtr::incref( value );
        }
        else
        {
            // Not a known type, try invoking callable
            auto res = PyObjectPtr::check( PyObject_CallFunction( m_callable, "(O)", value ) );
            return parsePyObject( res.get(), true );
        }
    }
}

PyObjectPtr structToDict( const StructPtr& struct_ptr, PyObject * callable, bool preserve_enums )
{
    StructToDictHelper helper( preserve_enums, callable );
    return helper.toDict( struct_ptr );
}

}
