#include <csp/python/PyStructToJson.h>
#include <rapidjson/document.h>
#include <rapidjson/writer.h>

namespace csp::python
{

// Helper function to convert csp Structs into json objects recursively
rapidjson::Value toJsonRecursive( const StructPtr& self, rapidjson::Document& doc, PyObject * callable );

// Helper function to parse some python objects in cpp, this should not be used extensively.
// instead add support for those python types to csp so that they can be handled more generically
// and in a language agnostic way
rapidjson::Value pyObjectToJson( PyObject * value, rapidjson::Document& doc, PyObject * callable, bool is_recursing );

// Helper fallback function to convert any type into json format recursively
template<typename T>
inline rapidjson::Value toJson( const T& val, const CspType& typ, rapidjson::Document& doc, PyObject * callable )
{
    // Default handler for any unknown T
    return rapidjson::Value( val );
}

// Helper function for parsing doubles
inline rapidjson::Value doubleToJson( const double& val, rapidjson::Document& doc )
{
    // NOTE: Rapidjson adds support for this in a future release. Remove this when we upgrade rapidjson to a version
    // after 07/16/2023 and use kWriteNanAndInfNullFlag in the writer.
    //
    // To be compatible with other JSON libraries, we cannot use the default approach that rapidjson has to
    // serializing NaN, and (+/-)Infs. We need to manually convert them to NULLs. Rapidjson adds support for this
    // in a future release.
    if ( std::isnan( val ) || std::isinf( val ) )
    {
        return rapidjson::Value();
    }
    else
    {
        return rapidjson::Value( val );
    }
}

// Helper function to convert doubles into json format recursively, by properly handlings NaNs, and Infs
template<>
inline rapidjson::Value toJson( const double& val, const CspType& typ, rapidjson::Document& doc, PyObject * callable )
{
    return doubleToJson( val, doc );
}

// Helper function to convert Enums into json format recursively
template<>
inline rapidjson::Value toJson( const CspEnum& val, const CspType& typ, rapidjson::Document& doc, PyObject * callable )
{
    // NOTE: Assume that passed enum has greater lifetime than the json object
    return rapidjson::Value( rapidjson::StringRef( val.name() ) );
}

// Helper function to convert strings into json format recursively
template<>
inline rapidjson::Value toJson( const std::string& val, const CspType& typ, rapidjson::Document& doc, PyObject * callable )
{
    // NOTE: Assume that passed string has greater lifetime than the json object
    return rapidjson::Value( rapidjson::StringRef( val ) );
}

// Helper function to convert TimeDelta into json format recursively
template<>
inline rapidjson::Value toJson( const TimeDelta& val, const CspType& typ, rapidjson::Document& doc, PyObject * callable )
{
    if( val.isNone() ) return rapidjson::Value();

    // Convert TimeDelta to <sign><seconds>.<microseconds>
    // sign( 1 ) + seconds ( 18 ) + '.'( 1 ) + microseconds( 9 ) + '\0'( 1 )
    char buf[32] = {};
    long seconds = val.abs().asSeconds();
    auto microseconds = static_cast<unsigned>( val.abs().nanoseconds() / NANOS_PER_MICROSECOND );
    auto len = sprintf( buf, "%c%ld.%06u", ( val.sign() >= 0 ) ? '+' : '-', seconds, microseconds );
    rapidjson::Value res;
    res.SetString( buf, len, doc.GetAllocator() );
    return res;
}

// Helper function to convert Date into json format recursively
template<>
inline rapidjson::Value toJson( const Date& val, const CspType& typ, rapidjson::Document& doc, PyObject * callable )
{
    if( val.isNone() ) return rapidjson::Value();

    // Convert Date to <year>-<month>-<day>
    // year( 4 ) + '-'( 1 ) + month( 2 ) + '-'( 1 ) + day( 2 ) + '\0'( 1 )
    char buf[32] = {};
    auto len = sprintf( buf, "%04u-%02u-%02u", val.year(), val.month(), val.day() );
    rapidjson::Value res;
    res.SetString( buf, len, doc.GetAllocator() );
    return res;
}

// Helper function to convert Time into json format recursively
template<>
inline rapidjson::Value toJson( const Time& val, const CspType& typ, rapidjson::Document& doc, PyObject * callable )
{
    if( val.isNone() ) return rapidjson::Value();

    // Convert the Time to <hours>:<minutes>:<seconds>.<microseconds>
    // hours( 2 ) + ':'( 1 ) + minutes( 2 ) + ':'( 1 ) + seconds( 2 ) + '.'( 1 ) + micros( 6 ) + '\0'( 1 )
    char buf[48] = {};
    auto micros = static_cast<unsigned>( val.nanosecond() / NANOS_PER_MICROSECOND );
    auto len = sprintf( buf, "%02u:%02u:%02u.%06u", val.hour(), val.minute(), val.second(), micros );
    rapidjson::Value res;
    res.SetString( buf, len, doc.GetAllocator() );
    return res;
}

// Helper function to convert DateTime into json format recursively
template<>
inline rapidjson::Value toJson( const DateTime& val, const CspType& typ, rapidjson::Document& doc, PyObject * callable )
{
    if( val.isNone() ) return rapidjson::Value();

    // Convert the datetime value into an ISO 8601 formatted string
    DateTimeEx dtx( val );

    char iso_str[80] = {};
    static const std::string utc_offset = "+00:00";
    auto micros = static_cast<unsigned>( dtx.nanoseconds() / NANOS_PER_MICROSECOND );
    // Hardcode UTC since all times in csp are UTC
    // NOTE: Python's datetime.fromisoformat() API does not support nanoseconds in the string
    // Hence we truncate from nanos to micros to allow easy conversion to python datetime objects
    auto len = sprintf( iso_str, "%04u-%02u-%02uT%02u:%02u:%02u.%06u%s",
            dtx.year(), dtx.month(), dtx.day(),
            dtx.hour(), dtx.minute(), dtx.second(), micros, utc_offset.c_str() );
    rapidjson::Value res;
    res.SetString( iso_str, len, doc.GetAllocator() );
    return res;
}

// Helper function to convert csp Structs into json format recursively
template<>
inline rapidjson::Value toJson( const StructPtr& val, const CspType& typ, rapidjson::Document& doc, PyObject * callable )
{
    return toJsonRecursive( val, doc, callable );
}

// Helper function to convert python objects in csp Structs into json format recursively
template<>
inline rapidjson::Value toJson( const DialectGenericType& val, const CspType& typ, rapidjson::Document& doc, PyObject * callable )
{
    auto py_obj = PyObjectPtr::own( toPython<DialectGenericType>( val ) );
    return pyObjectToJson( py_obj.get(), doc, callable, false );
}

// Helper function to convert arrays in csp Structs into json lists recursively
template<typename StorageT>
inline rapidjson::Value toJson( const std::vector<StorageT>& val, const CspType& typ, rapidjson::Document& doc, PyObject * callable )
{
    using ElemT = typename CspType::Type::toCArrayElemType<StorageT>::type;

    auto const * arrayType = static_cast<const CspArrayType*>( &typ );
    const CspType * elemType = arrayType -> elemType().get();
    rapidjson::Value new_list;
    new_list.SetArray();

    for( size_t idx = 0; idx < val.size(); ++idx )
    {
        auto sub_json = toJson<ElemT>( val[idx], *elemType, doc, callable );
        new_list.PushBack( sub_json, doc.GetAllocator() );
    }
    return new_list;
}

rapidjson::Value toJsonRecursive( const StructPtr& self, rapidjson::Document& doc, PyObject * callable )
{
    auto * struct_ptr = self.get();
    if( struct_ptr == nullptr )
    {
        CSP_THROW( ValueError, "Cannot call to_json on NULL struct object" );
    }
    auto * meta = static_cast<const DialectStructMeta *>( self -> meta() );
    rapidjson::Value new_dict;
    new_dict.SetObject();

    auto& fields = meta -> fields();
    for( const auto& field: fields )
    {
        if( !field -> isSet( struct_ptr ) )
        {
            continue;
        }
        auto& key = field -> fieldname();
        auto sub_json = switchCspType( field -> type(), [field, struct_ptr, callable, &doc]( auto tag )
            {
            using CType = typename decltype( tag )::type;
            auto *typedField = static_cast<const typename StructField::upcast<CType>::type *>( field.get() );
            return toJson( typedField -> value( struct_ptr ), *field -> type(), doc, callable );
            } );
        new_dict.AddMember( rapidjson::StringRef( key ), sub_json, doc.GetAllocator() );
    }
    return new_dict;
}

rapidjson::Value pyDictKeyToName( PyObject * py_key, rapidjson::Document& doc, PyObject * callable )
{
    // NOTE: Only support None, bool, strings, ints, and floats, date, time, datetime, enums, csp.Enums as keys
    // JSON encoding requires all names to be strings so convert them to strings

    static thread_local PyTypeObjectPtr s_tl_enum_type;
    // Get the enum type on the first call and save it for future use
    if( s_tl_enum_type.get() == nullptr ) [[unlikely]]
    {
        // Import enum module to extract the Enum type
        auto py_enum_module = PyObjectPtr::own( PyImport_ImportModule( "enum" ) );
        if( py_enum_module.get() )
        {
            s_tl_enum_type = PyTypeObjectPtr::own( reinterpret_cast<PyTypeObject*>( PyObject_GetAttrString( py_enum_module.get(), "Enum" ) ) );
        }
        else
        {
            CSP_THROW( RuntimeException, "Unable to import enum module from the python standard library" );
        }
    }

    rapidjson::Value val;
    if( py_key == Py_None )
    {
        val.SetString( "null" );
    }
    else if( PyBool_Check( py_key ) )
    {
        auto str_obj = PyObjectPtr::own( PyObject_Str( py_key ) );
        Py_ssize_t len = 0;
        const char * str = PyUnicode_AsUTF8AndSize( str_obj.get(), &len );
        val.SetString( str, len, doc.GetAllocator() );
    }
    else if( PyUnicode_Check( py_key ) )
    {
        Py_ssize_t len;
        auto str = PyUnicode_AsUTF8AndSize( py_key, &len );
        val.SetString( str, len, doc.GetAllocator() );
    }
    else if( PyLong_Check( py_key ) )
    {
        auto key = PyLong_AsLong( py_key );
        val.SetString( std::to_string( key ), doc.GetAllocator() );
    }
    else if( PyFloat_Check( py_key ) )
    {
        auto key = PyFloat_AsDouble( py_key );
        auto json_obj = doubleToJson( key, doc );
        if ( json_obj.IsNull() )
        {
            auto str_obj = PyObjectPtr::own( PyObject_Str( py_key ) );
            const char * str = PyUnicode_AsUTF8( str_obj.get() );
            CSP_THROW( ValueError, "Cannot serialize " + std::string( str ) + " to key in JSON" );
        }
        else
        {
            // Convert to string
            std::stringstream s;
            s << key;
            val.SetString( s.str(), doc.GetAllocator() );
        }
    }
    else if( PyTime_CheckExact( py_key ) )
    {
        auto v = fromPython<Time>( py_key );
        val = toJson( v, CspType( CspType::Type::TIME ), doc, callable );
    }
    else if( PyDate_CheckExact( py_key ) )
    {
        auto v = fromPython<Date>( py_key );
        val = toJson( v, CspType( CspType::Type::DATE ), doc, callable );
    }
    else if( PyDateTime_CheckExact( py_key ) )
    {
        auto v = fromPython<DateTime>( py_key );
        val = toJson( v, CspType( CspType::Type::DATETIME ), doc, callable );
    }
    else if( PyType_IsSubtype( Py_TYPE( py_key ), &PyCspEnum::PyType ) )
    {
        auto enum_ptr = static_cast<PyCspEnum *>( py_key ) -> enum_;
        val = toJson( enum_ptr, CspType( CspType::Type::ENUM ), doc, callable );
    }
    else if( PyType_IsSubtype( Py_TYPE( py_key ), s_tl_enum_type.get() ) )
    {
        // Use the `name` attribute of the enum for the string representation
        auto py_enum_name = PyObjectPtr::own( PyObject_GetAttrString( py_key, "name" ) );
        auto str_obj = PyObjectPtr::own( PyObject_Str( py_enum_name.get() ) );
        Py_ssize_t len = 0;
        const char * str = PyUnicode_AsUTF8AndSize( str_obj.get(), &len );
        val.SetString( str, len, doc.GetAllocator() );
    }
    else
    {
        CSP_THROW( ValueError, "Cannot serialize key of type: " + std::string( Py_TYPE( py_key ) -> tp_name ) );
        // Never reaches here
    }
    return val;
}

// Helper function to parse python lists into json arrays recursively
rapidjson::Value pyListToJson( PyObject * py_list, rapidjson::Document& doc, PyObject * callable )
{
    size_t size = PyList_GET_SIZE( py_list );
    rapidjson::Value new_list;
    new_list.SetArray();

    for( size_t idx = 0; idx < size; ++idx )
    {
        auto * item = PyList_GET_ITEM( py_list, idx );
        auto res = pyObjectToJson( item, doc, callable, false );
        new_list.PushBack( res, doc.GetAllocator() );
    }
    return new_list;
}

// Helper function to parse python tuples into json arrays recursively
rapidjson::Value pyTupleToJson( PyObject * py_tuple, rapidjson::Document& doc, PyObject * callable )
{
    size_t size = PyTuple_GET_SIZE( py_tuple );
    rapidjson::Value new_list;
    new_list.SetArray();

    for( size_t idx = 0; idx < size; ++idx )
    {
        auto * item = PyTuple_GetItem( py_tuple, idx );
        auto res = pyObjectToJson( item, doc, callable, false );
        new_list.PushBack( res, doc.GetAllocator() );
    }
    return new_list;
}

// Helper function to parse python dicts into json objects recursively
rapidjson::Value pyDictToJson( PyObject * py_dict, rapidjson::Document& doc, PyObject * callable )
{
    PyObject * py_key = NULL;
    PyObject * py_value = NULL;
    Py_ssize_t pos = 0;
    rapidjson::Value new_dict;
    new_dict.SetObject();

    while( PyDict_Next( py_dict, &pos, &py_key, &py_value ) )
    {
        auto key = pyDictKeyToName( py_key, doc, callable );
        auto res = pyObjectToJson( py_value, doc, callable, false );
        new_dict.AddMember( key, res, doc.GetAllocator() );
    }
    return new_dict;
}

rapidjson::Value pyObjectToJson( PyObject * value, rapidjson::Document& doc, PyObject * callable, bool is_recursing )
{
    INIT_PYDATETIME;
    if( value == Py_None )
    {
        return rapidjson::Value();
    }
    else if( PyBool_Check( value ) )
    {
        return rapidjson::Value( fromPython<bool>( value ) );
    }
    else if( PyLong_Check( value ) )
    {
        return rapidjson::Value( fromPython<int64_t>( value ) );
    }
    else if( PyFloat_Check( value ) )
    {
        return doubleToJson( fromPython<double>( value ), doc );
    }
    else if( PyUnicode_Check( value ) )
    {
        Py_ssize_t len;
        auto str = PyUnicode_AsUTF8AndSize( value, &len );
        rapidjson::Value str_val;
        str_val.SetString( str, len, doc.GetAllocator() );
        return str_val;
    }
    else if( PyBytes_Check( value ) )
    {
        Py_ssize_t len = PyBytes_Size( value );
        auto str = PyBytes_AsString( value );
        rapidjson::Value str_val;
        str_val.SetString( str, len, doc.GetAllocator() );
        return str_val;
    }
    else if( PyTime_CheckExact( value ) )
    {
        auto v = fromPython<Time>( value );
        return toJson( v, CspType( CspType::Type::TIME ), doc, callable );
    }
    else if( PyDate_CheckExact( value ) )
    {
        auto v = fromPython<Date>( value );
        return toJson( v, CspType( CspType::Type::DATE ), doc, callable );
    }
    else if( PyDateTime_CheckExact( value ) )
    {
        auto v = fromPython<DateTime>( value );
        return toJson( v, CspType( CspType::Type::DATETIME ), doc, callable );
    }
    else if( PyDelta_CheckExact( value ) )
    {
        auto v = fromPython<TimeDelta>( value );
        return toJson( v, CspType( CspType::Type::TIMEDELTA ), doc, callable );
    }
    else if( PyTuple_CheckExact( value ) )
    {
        // NOTE: Remove this logic when generic python tuples are supported as an internal type in csp
        return pyTupleToJson( value, doc, callable );
    }
    else if( PyList_CheckExact( value ) )
    {
        // NOTE: Remove this logic when generic python lists are supported as an internal type in csp
        return pyListToJson( value, doc, callable );
    }
    else if( PyDict_CheckExact( value ) )
    {
        // NOTE: Remove this logic when generic python dicts are supported as an internal type in csp
        return pyDictToJson( value, doc, callable );
    }
    else if( PyType_IsSubtype( Py_TYPE( value ), &PyStruct::PyType ) )
    {
        auto struct_ptr = static_cast<PyStruct *>( value ) -> struct_;
        return toJsonRecursive( struct_ptr, doc, callable );
    }
    else if( PyType_IsSubtype( Py_TYPE( value ), &PyCspEnum::PyType ) )
    {
        auto enum_ptr = static_cast<PyCspEnum *>( value ) -> enum_;
        return toJson( enum_ptr, CspType( CspType::Type::ENUM ), doc, callable );
    }
    else
    {
        if( is_recursing )
        {
            // Looks like we are recursively calling pyObjectToJson for a generic unsupported type
            // just return the object since we don't know how to jsonify it
            CSP_THROW( ValueError, "Cannot serialize value of type: " + std::string( Py_TYPE( value ) -> tp_name ) );
            // Never reaches here
        }
        else
        {
            // Not a known type
            // We expect callback to return a py object consisting of either csp types or dicts and lists
            auto res_py_obj = PyObjectPtr::own( PyObject_CallFunction( callable, "(O)", value ) );
            if( res_py_obj )
            {
                // NOTE: We could add checks to verify the returned py object is jsonified,
                // but that would have performance implications
                return pyObjectToJson( res_py_obj.get(), doc, callable, true );
            }
            else
            {
                // The callback failed, just pass the error back up to the caller
                CSP_THROW( PythonPassthrough, "" );
            }
        }
    }
}

// Note: This class provides a way to write the rapidjson to a string stored in the holder
// This can help avoid creating an extra copy of the string that StringBuffer would cause since
// StringBuffer returns a char* instead of a string object
class StringHolder
{
public:
    typedef char Ch;
    StringHolder( std::string& s ) : s_( s )
    {
        s_.reserve( DEFAULT_BUFFER_SIZE );
    }
    void Put( char c )
    {
        s_.push_back( c );
    }
    void Clear()
    {
        s_.clear();
    }
    void Flush()
    {
        return;
    }
    size_t Size() const
    {
        return s_.length();
    }
    const size_t DEFAULT_BUFFER_SIZE = 4096;
private:
    std::string& s_;
};

std::string structToJson( const StructPtr& struct_ptr, PyObject * callable )
{
    // Need this just for the Allocator
    rapidjson::Document placeholder_doc;
    auto json = toJsonRecursive( struct_ptr, placeholder_doc, callable );
    std::string output;
    StringHolder holder( output );
    rapidjson::Writer<StringHolder> writer( holder );
    json.Accept( writer );
    placeholder_doc.GetAllocator().Clear();
    return output;
}

}
