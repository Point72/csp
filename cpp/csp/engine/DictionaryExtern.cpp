/*
 * Implementation of the C API for Dictionary access.
 */
#include <csp/engine/c/CspDictionary.h>
#include <csp/engine/c/CspError.h>
#include <csp/engine/Dictionary.h>
#include <csp/core/Exception.h>
#include <cstring>
#include <variant>

// ============================================================================
// Iterator State
// ============================================================================

struct CCspDictIteratorImpl
{
    const csp::Dictionary * dict;
    csp::Dictionary::const_iterator current;
    csp::Dictionary::const_iterator end;
    bool started;  // Have we called next() at least once?
    
    CCspDictIteratorImpl( const csp::Dictionary * d )
        : dict( d )
        , current( d -> begin() )
        , end( d -> end() )
        , started( false )
    {}
};

// ============================================================================
// Helper to map std::variant index to CCspDictValueType
// ============================================================================

static CCspDictValueType variantIndexToType( size_t index )
{
    // Dictionary::Value = std::variant<std::monostate,bool,int32_t,uint32_t,int64_t,uint64_t,double,
    //                                  std::string,DateTime,TimeDelta,StructMetaPtr,DialectGenericType,
    //                                  DictionaryPtr,Vector,std::shared_ptr<Data>>
    switch( index )
    {
        case 0:  return CCSP_DICT_TYPE_NONE;       // monostate
        case 1:  return CCSP_DICT_TYPE_BOOL;
        case 2:  return CCSP_DICT_TYPE_INT32;
        case 3:  return CCSP_DICT_TYPE_UINT32;
        case 4:  return CCSP_DICT_TYPE_INT64;
        case 5:  return CCSP_DICT_TYPE_UINT64;
        case 6:  return CCSP_DICT_TYPE_DOUBLE;
        case 7:  return CCSP_DICT_TYPE_STRING;
        case 8:  return CCSP_DICT_TYPE_DATETIME;
        case 9:  return CCSP_DICT_TYPE_TIMEDELTA;
        case 10: return CCSP_DICT_TYPE_STRUCT_META;
        case 11: return CCSP_DICT_TYPE_DIALECT;
        case 12: return CCSP_DICT_TYPE_DICTIONARY;
        case 13: return CCSP_DICT_TYPE_VECTOR;
        case 14: return CCSP_DICT_TYPE_DATA;
        default: return CCSP_DICT_TYPE_NONE;
    }
}

// ============================================================================
// C API Implementation
// ============================================================================

extern "C" {

// ----------------------------------------------------------------------------
// Basic Operations
// ----------------------------------------------------------------------------

int ccsp_dictionary_exists( CCspDictionaryHandle dict, const char * key )
{
    if( !dict || !key ) return 0;
    auto * d = reinterpret_cast<const csp::Dictionary *>( dict );
    return d -> exists( key ) ? 1 : 0;
}

size_t ccsp_dictionary_size( CCspDictionaryHandle dict )
{
    if( !dict ) return 0;
    auto * d = reinterpret_cast<const csp::Dictionary *>( dict );
    return d -> size();
}

int ccsp_dictionary_is_empty( CCspDictionaryHandle dict )
{
    if( !dict ) return 1;
    auto * d = reinterpret_cast<const csp::Dictionary *>( dict );
    return d -> empty() ? 1 : 0;
}

CCspDictValueType ccsp_dictionary_get_type( CCspDictionaryHandle dict, const char * key )
{
    if( !dict || !key ) return CCSP_DICT_TYPE_NONE;
    auto * d = reinterpret_cast<const csp::Dictionary *>( dict );
    
    if( !d -> exists( key ) )
        return CCSP_DICT_TYPE_NONE;
    
    try
    {
        const csp::Dictionary::Value & value = d -> getUntypedValue( key );
        return variantIndexToType( value.index() );
    }
    catch( ... )
    {
        return CCSP_DICT_TYPE_NONE;
    }
}

// ----------------------------------------------------------------------------
// Type-Safe Getters
// ----------------------------------------------------------------------------

CCspErrorCode ccsp_dictionary_get_bool( CCspDictionaryHandle dict, const char * key, int8_t * out_value )
{
    if( !dict || !key || !out_value )
    {
        ccsp_set_error( CCSP_ERROR_NULL_POINTER, "null argument" );
        return CCSP_ERROR_NULL_POINTER;
    }
    
    auto * d = reinterpret_cast<const csp::Dictionary *>( dict );
    try
    {
        *out_value = d -> get<bool>( key ) ? 1 : 0;
        return CCSP_OK;
    }
    catch( const csp::KeyError & )
    {
        ccsp_set_error( CCSP_ERROR_KEY_NOT_FOUND, "key not found" );
        return CCSP_ERROR_KEY_NOT_FOUND;
    }
    catch( const csp::TypeError & )
    {
        ccsp_set_error( CCSP_ERROR_TYPE_MISMATCH, "type mismatch" );
        return CCSP_ERROR_TYPE_MISMATCH;
    }
    catch( const std::exception & e )
    {
        ccsp_set_error( CCSP_ERROR_RUNTIME, e.what() );
        return CCSP_ERROR_RUNTIME;
    }
}

CCspErrorCode ccsp_dictionary_get_int32( CCspDictionaryHandle dict, const char * key, int32_t * out_value )
{
    if( !dict || !key || !out_value )
    {
        ccsp_set_error( CCSP_ERROR_NULL_POINTER, "null argument" );
        return CCSP_ERROR_NULL_POINTER;
    }
    
    auto * d = reinterpret_cast<const csp::Dictionary *>( dict );
    try
    {
        *out_value = d -> get<int32_t>( key );
        return CCSP_OK;
    }
    catch( const csp::KeyError & )
    {
        ccsp_set_error( CCSP_ERROR_KEY_NOT_FOUND, "key not found" );
        return CCSP_ERROR_KEY_NOT_FOUND;
    }
    catch( const csp::TypeError & )
    {
        ccsp_set_error( CCSP_ERROR_TYPE_MISMATCH, "type mismatch" );
        return CCSP_ERROR_TYPE_MISMATCH;
    }
    catch( const std::exception & e )
    {
        ccsp_set_error( CCSP_ERROR_RUNTIME, e.what() );
        return CCSP_ERROR_RUNTIME;
    }
}

CCspErrorCode ccsp_dictionary_get_uint32( CCspDictionaryHandle dict, const char * key, uint32_t * out_value )
{
    if( !dict || !key || !out_value )
    {
        ccsp_set_error( CCSP_ERROR_NULL_POINTER, "null argument" );
        return CCSP_ERROR_NULL_POINTER;
    }
    
    auto * d = reinterpret_cast<const csp::Dictionary *>( dict );
    try
    {
        *out_value = d -> get<uint32_t>( key );
        return CCSP_OK;
    }
    catch( const csp::KeyError & )
    {
        ccsp_set_error( CCSP_ERROR_KEY_NOT_FOUND, "key not found" );
        return CCSP_ERROR_KEY_NOT_FOUND;
    }
    catch( const csp::TypeError & )
    {
        ccsp_set_error( CCSP_ERROR_TYPE_MISMATCH, "type mismatch" );
        return CCSP_ERROR_TYPE_MISMATCH;
    }
    catch( const std::exception & e )
    {
        ccsp_set_error( CCSP_ERROR_RUNTIME, e.what() );
        return CCSP_ERROR_RUNTIME;
    }
}

CCspErrorCode ccsp_dictionary_get_int64( CCspDictionaryHandle dict, const char * key, int64_t * out_value )
{
    if( !dict || !key || !out_value )
    {
        ccsp_set_error( CCSP_ERROR_NULL_POINTER, "null argument" );
        return CCSP_ERROR_NULL_POINTER;
    }
    
    auto * d = reinterpret_cast<const csp::Dictionary *>( dict );
    try
    {
        *out_value = d -> get<int64_t>( key );
        return CCSP_OK;
    }
    catch( const csp::KeyError & )
    {
        ccsp_set_error( CCSP_ERROR_KEY_NOT_FOUND, "key not found" );
        return CCSP_ERROR_KEY_NOT_FOUND;
    }
    catch( const csp::TypeError & )
    {
        ccsp_set_error( CCSP_ERROR_TYPE_MISMATCH, "type mismatch" );
        return CCSP_ERROR_TYPE_MISMATCH;
    }
    catch( const std::exception & e )
    {
        ccsp_set_error( CCSP_ERROR_RUNTIME, e.what() );
        return CCSP_ERROR_RUNTIME;
    }
}

CCspErrorCode ccsp_dictionary_get_uint64( CCspDictionaryHandle dict, const char * key, uint64_t * out_value )
{
    if( !dict || !key || !out_value )
    {
        ccsp_set_error( CCSP_ERROR_NULL_POINTER, "null argument" );
        return CCSP_ERROR_NULL_POINTER;
    }
    
    auto * d = reinterpret_cast<const csp::Dictionary *>( dict );
    try
    {
        *out_value = d -> get<uint64_t>( key );
        return CCSP_OK;
    }
    catch( const csp::KeyError & )
    {
        ccsp_set_error( CCSP_ERROR_KEY_NOT_FOUND, "key not found" );
        return CCSP_ERROR_KEY_NOT_FOUND;
    }
    catch( const csp::TypeError & )
    {
        ccsp_set_error( CCSP_ERROR_TYPE_MISMATCH, "type mismatch" );
        return CCSP_ERROR_TYPE_MISMATCH;
    }
    catch( const std::exception & e )
    {
        ccsp_set_error( CCSP_ERROR_RUNTIME, e.what() );
        return CCSP_ERROR_RUNTIME;
    }
}

CCspErrorCode ccsp_dictionary_get_double( CCspDictionaryHandle dict, const char * key, double * out_value )
{
    if( !dict || !key || !out_value )
    {
        ccsp_set_error( CCSP_ERROR_NULL_POINTER, "null argument" );
        return CCSP_ERROR_NULL_POINTER;
    }
    
    auto * d = reinterpret_cast<const csp::Dictionary *>( dict );
    try
    {
        *out_value = d -> get<double>( key );
        return CCSP_OK;
    }
    catch( const csp::KeyError & )
    {
        ccsp_set_error( CCSP_ERROR_KEY_NOT_FOUND, "key not found" );
        return CCSP_ERROR_KEY_NOT_FOUND;
    }
    catch( const csp::TypeError & )
    {
        ccsp_set_error( CCSP_ERROR_TYPE_MISMATCH, "type mismatch" );
        return CCSP_ERROR_TYPE_MISMATCH;
    }
    catch( const std::exception & e )
    {
        ccsp_set_error( CCSP_ERROR_RUNTIME, e.what() );
        return CCSP_ERROR_RUNTIME;
    }
}

CCspErrorCode ccsp_dictionary_get_datetime( CCspDictionaryHandle dict, const char * key, CCspDateTime * out_value )
{
    if( !dict || !key || !out_value )
    {
        ccsp_set_error( CCSP_ERROR_NULL_POINTER, "null argument" );
        return CCSP_ERROR_NULL_POINTER;
    }
    
    auto * d = reinterpret_cast<const csp::Dictionary *>( dict );
    try
    {
        csp::DateTime dt = d -> get<csp::DateTime>( key );
        *out_value = dt.asNanoseconds();
        return CCSP_OK;
    }
    catch( const csp::KeyError & )
    {
        ccsp_set_error( CCSP_ERROR_KEY_NOT_FOUND, "key not found" );
        return CCSP_ERROR_KEY_NOT_FOUND;
    }
    catch( const csp::TypeError & )
    {
        ccsp_set_error( CCSP_ERROR_TYPE_MISMATCH, "type mismatch" );
        return CCSP_ERROR_TYPE_MISMATCH;
    }
    catch( const std::exception & e )
    {
        ccsp_set_error( CCSP_ERROR_RUNTIME, e.what() );
        return CCSP_ERROR_RUNTIME;
    }
}

CCspErrorCode ccsp_dictionary_get_timedelta( CCspDictionaryHandle dict, const char * key, CCspTimeDelta * out_value )
{
    if( !dict || !key || !out_value )
    {
        ccsp_set_error( CCSP_ERROR_NULL_POINTER, "null argument" );
        return CCSP_ERROR_NULL_POINTER;
    }
    
    auto * d = reinterpret_cast<const csp::Dictionary *>( dict );
    try
    {
        csp::TimeDelta td = d -> get<csp::TimeDelta>( key );
        *out_value = td.asNanoseconds();
        return CCSP_OK;
    }
    catch( const csp::KeyError & )
    {
        ccsp_set_error( CCSP_ERROR_KEY_NOT_FOUND, "key not found" );
        return CCSP_ERROR_KEY_NOT_FOUND;
    }
    catch( const csp::TypeError & )
    {
        ccsp_set_error( CCSP_ERROR_TYPE_MISMATCH, "type mismatch" );
        return CCSP_ERROR_TYPE_MISMATCH;
    }
    catch( const std::exception & e )
    {
        ccsp_set_error( CCSP_ERROR_RUNTIME, e.what() );
        return CCSP_ERROR_RUNTIME;
    }
}

CCspErrorCode ccsp_dictionary_get_string( CCspDictionaryHandle dict, const char * key,
                                          const char ** out_data, size_t * out_length )
{
    if( !dict || !key || !out_data || !out_length )
    {
        ccsp_set_error( CCSP_ERROR_NULL_POINTER, "null argument" );
        return CCSP_ERROR_NULL_POINTER;
    }
    
    auto * d = reinterpret_cast<const csp::Dictionary *>( dict );
    try
    {
        const std::string & str = d -> get<std::string>( key );
        *out_data = str.data();
        *out_length = str.size();
        return CCSP_OK;
    }
    catch( const csp::KeyError & )
    {
        ccsp_set_error( CCSP_ERROR_KEY_NOT_FOUND, "key not found" );
        return CCSP_ERROR_KEY_NOT_FOUND;
    }
    catch( const csp::TypeError & )
    {
        ccsp_set_error( CCSP_ERROR_TYPE_MISMATCH, "type mismatch" );
        return CCSP_ERROR_TYPE_MISMATCH;
    }
    catch( const std::exception & e )
    {
        ccsp_set_error( CCSP_ERROR_RUNTIME, e.what() );
        return CCSP_ERROR_RUNTIME;
    }
}

CCspErrorCode ccsp_dictionary_get_dict( CCspDictionaryHandle dict, const char * key,
                                        CCspDictionaryHandle * out_dict )
{
    if( !dict || !key || !out_dict )
    {
        ccsp_set_error( CCSP_ERROR_NULL_POINTER, "null argument" );
        return CCSP_ERROR_NULL_POINTER;
    }
    
    auto * d = reinterpret_cast<const csp::Dictionary *>( dict );
    try
    {
        csp::DictionaryPtr nested = d -> get<csp::DictionaryPtr>( key );
        *out_dict = reinterpret_cast<CCspDictionaryHandle>( nested.get() );
        return CCSP_OK;
    }
    catch( const csp::KeyError & )
    {
        ccsp_set_error( CCSP_ERROR_KEY_NOT_FOUND, "key not found" );
        return CCSP_ERROR_KEY_NOT_FOUND;
    }
    catch( const csp::TypeError & )
    {
        ccsp_set_error( CCSP_ERROR_TYPE_MISMATCH, "type mismatch" );
        return CCSP_ERROR_TYPE_MISMATCH;
    }
    catch( const std::exception & e )
    {
        ccsp_set_error( CCSP_ERROR_RUNTIME, e.what() );
        return CCSP_ERROR_RUNTIME;
    }
}

// ----------------------------------------------------------------------------
// Getters with Defaults
// ----------------------------------------------------------------------------

int8_t ccsp_dictionary_get_bool_or( CCspDictionaryHandle dict, const char * key, int8_t default_value )
{
    if( !dict || !key ) return default_value;
    auto * d = reinterpret_cast<const csp::Dictionary *>( dict );
    try
    {
        return d -> get<bool>( key, default_value != 0 ) ? 1 : 0;
    }
    catch( ... )
    {
        return default_value;
    }
}

int32_t ccsp_dictionary_get_int32_or( CCspDictionaryHandle dict, const char * key, int32_t default_value )
{
    if( !dict || !key ) return default_value;
    auto * d = reinterpret_cast<const csp::Dictionary *>( dict );
    try
    {
        return d -> get<int32_t>( key, default_value );
    }
    catch( ... )
    {
        return default_value;
    }
}

uint32_t ccsp_dictionary_get_uint32_or( CCspDictionaryHandle dict, const char * key, uint32_t default_value )
{
    if( !dict || !key ) return default_value;
    auto * d = reinterpret_cast<const csp::Dictionary *>( dict );
    try
    {
        return d -> get<uint32_t>( key, default_value );
    }
    catch( ... )
    {
        return default_value;
    }
}

int64_t ccsp_dictionary_get_int64_or( CCspDictionaryHandle dict, const char * key, int64_t default_value )
{
    if( !dict || !key ) return default_value;
    auto * d = reinterpret_cast<const csp::Dictionary *>( dict );
    try
    {
        return d -> get<int64_t>( key, default_value );
    }
    catch( ... )
    {
        return default_value;
    }
}

uint64_t ccsp_dictionary_get_uint64_or( CCspDictionaryHandle dict, const char * key, uint64_t default_value )
{
    if( !dict || !key ) return default_value;
    auto * d = reinterpret_cast<const csp::Dictionary *>( dict );
    try
    {
        return d -> get<uint64_t>( key, default_value );
    }
    catch( ... )
    {
        return default_value;
    }
}

double ccsp_dictionary_get_double_or( CCspDictionaryHandle dict, const char * key, double default_value )
{
    if( !dict || !key ) return default_value;
    auto * d = reinterpret_cast<const csp::Dictionary *>( dict );
    try
    {
        return d -> get<double>( key, default_value );
    }
    catch( ... )
    {
        return default_value;
    }
}

CCspDateTime ccsp_dictionary_get_datetime_or( CCspDictionaryHandle dict, const char * key, CCspDateTime default_value )
{
    if( !dict || !key ) return default_value;
    auto * d = reinterpret_cast<const csp::Dictionary *>( dict );
    try
    {
        csp::DateTime def = csp::DateTime::fromNanoseconds( default_value );
        csp::DateTime dt = d -> get<csp::DateTime>( key, def );
        return dt.asNanoseconds();
    }
    catch( ... )
    {
        return default_value;
    }
}

CCspTimeDelta ccsp_dictionary_get_timedelta_or( CCspDictionaryHandle dict, const char * key, CCspTimeDelta default_value )
{
    if( !dict || !key ) return default_value;
    auto * d = reinterpret_cast<const csp::Dictionary *>( dict );
    try
    {
        csp::TimeDelta def = csp::TimeDelta::fromNanoseconds( default_value );
        csp::TimeDelta td = d -> get<csp::TimeDelta>( key, def );
        return td.asNanoseconds();
    }
    catch( ... )
    {
        return default_value;
    }
}

const char * ccsp_dictionary_get_string_or( CCspDictionaryHandle dict, const char * key,
                                            const char * default_value, size_t * out_length )
{
    if( !dict || !key )
    {
        if( out_length && default_value )
            *out_length = strlen( default_value );
        else if( out_length )
            *out_length = 0;
        return default_value;
    }
    
    auto * d = reinterpret_cast<const csp::Dictionary *>( dict );
    try
    {
        if( !d -> exists( key ) )
        {
            if( out_length && default_value )
                *out_length = strlen( default_value );
            else if( out_length )
                *out_length = 0;
            return default_value;
        }
        
        const std::string & str = d -> get<std::string>( key );
        if( out_length )
            *out_length = str.size();
        return str.data();
    }
    catch( ... )
    {
        if( out_length && default_value )
            *out_length = strlen( default_value );
        else if( out_length )
            *out_length = 0;
        return default_value;
    }
}

// ----------------------------------------------------------------------------
// Iteration
// ----------------------------------------------------------------------------

CCspDictIteratorHandle ccsp_dictionary_iter_create( CCspDictionaryHandle dict )
{
    if( !dict ) return nullptr;
    
    auto * d = reinterpret_cast<const csp::Dictionary *>( dict );
    auto * iter = new CCspDictIteratorImpl( d );
    return reinterpret_cast<CCspDictIteratorHandle>( iter );
}

void ccsp_dictionary_iter_destroy( CCspDictIteratorHandle iter )
{
    if( !iter ) return;
    auto * impl = reinterpret_cast<CCspDictIteratorImpl *>( iter );
    delete impl;
}

int ccsp_dictionary_iter_next( CCspDictIteratorHandle iter, const char ** out_key )
{
    if( !iter || !out_key ) return 0;
    
    auto * impl = reinterpret_cast<CCspDictIteratorImpl *>( iter );
    
    if( !impl -> started )
    {
        impl -> started = true;
    }
    else
    {
        ++( impl -> current );
    }
    
    if( impl -> current == impl -> end )
        return 0;
    
    *out_key = impl -> current.key().c_str();
    return 1;
}

CCspDictValueType ccsp_dictionary_iter_value_type( CCspDictIteratorHandle iter )
{
    if( !iter ) return CCSP_DICT_TYPE_NONE;
    
    auto * impl = reinterpret_cast<CCspDictIteratorImpl *>( iter );
    if( !impl -> started || impl -> current == impl -> end )
        return CCSP_DICT_TYPE_NONE;
    
    const csp::Dictionary::Value & value = impl -> current.getUntypedValue();
    return variantIndexToType( value.index() );
}

CCspErrorCode ccsp_dictionary_iter_get_bool( CCspDictIteratorHandle iter, int8_t * out_value )
{
    if( !iter || !out_value )
    {
        ccsp_set_error( CCSP_ERROR_NULL_POINTER, "null argument" );
        return CCSP_ERROR_NULL_POINTER;
    }
    
    auto * impl = reinterpret_cast<CCspDictIteratorImpl *>( iter );
    if( !impl -> started || impl -> current == impl -> end )
    {
        ccsp_set_error( CCSP_ERROR_INVALID_ARGUMENT, "iterator not positioned" );
        return CCSP_ERROR_INVALID_ARGUMENT;
    }
    
    try
    {
        *out_value = impl -> current.value<bool>() ? 1 : 0;
        return CCSP_OK;
    }
    catch( ... )
    {
        ccsp_set_error( CCSP_ERROR_TYPE_MISMATCH, "type mismatch" );
        return CCSP_ERROR_TYPE_MISMATCH;
    }
}

CCspErrorCode ccsp_dictionary_iter_get_int32( CCspDictIteratorHandle iter, int32_t * out_value )
{
    if( !iter || !out_value )
    {
        ccsp_set_error( CCSP_ERROR_NULL_POINTER, "null argument" );
        return CCSP_ERROR_NULL_POINTER;
    }
    
    auto * impl = reinterpret_cast<CCspDictIteratorImpl *>( iter );
    if( !impl -> started || impl -> current == impl -> end )
    {
        ccsp_set_error( CCSP_ERROR_INVALID_ARGUMENT, "iterator not positioned" );
        return CCSP_ERROR_INVALID_ARGUMENT;
    }
    
    try
    {
        *out_value = impl -> current.value<int32_t>();
        return CCSP_OK;
    }
    catch( ... )
    {
        ccsp_set_error( CCSP_ERROR_TYPE_MISMATCH, "type mismatch" );
        return CCSP_ERROR_TYPE_MISMATCH;
    }
}

CCspErrorCode ccsp_dictionary_iter_get_uint32( CCspDictIteratorHandle iter, uint32_t * out_value )
{
    if( !iter || !out_value )
    {
        ccsp_set_error( CCSP_ERROR_NULL_POINTER, "null argument" );
        return CCSP_ERROR_NULL_POINTER;
    }
    
    auto * impl = reinterpret_cast<CCspDictIteratorImpl *>( iter );
    if( !impl -> started || impl -> current == impl -> end )
    {
        ccsp_set_error( CCSP_ERROR_INVALID_ARGUMENT, "iterator not positioned" );
        return CCSP_ERROR_INVALID_ARGUMENT;
    }
    
    try
    {
        *out_value = impl -> current.value<uint32_t>();
        return CCSP_OK;
    }
    catch( ... )
    {
        ccsp_set_error( CCSP_ERROR_TYPE_MISMATCH, "type mismatch" );
        return CCSP_ERROR_TYPE_MISMATCH;
    }
}

CCspErrorCode ccsp_dictionary_iter_get_int64( CCspDictIteratorHandle iter, int64_t * out_value )
{
    if( !iter || !out_value )
    {
        ccsp_set_error( CCSP_ERROR_NULL_POINTER, "null argument" );
        return CCSP_ERROR_NULL_POINTER;
    }
    
    auto * impl = reinterpret_cast<CCspDictIteratorImpl *>( iter );
    if( !impl -> started || impl -> current == impl -> end )
    {
        ccsp_set_error( CCSP_ERROR_INVALID_ARGUMENT, "iterator not positioned" );
        return CCSP_ERROR_INVALID_ARGUMENT;
    }
    
    try
    {
        *out_value = impl -> current.value<int64_t>();
        return CCSP_OK;
    }
    catch( ... )
    {
        ccsp_set_error( CCSP_ERROR_TYPE_MISMATCH, "type mismatch" );
        return CCSP_ERROR_TYPE_MISMATCH;
    }
}

CCspErrorCode ccsp_dictionary_iter_get_uint64( CCspDictIteratorHandle iter, uint64_t * out_value )
{
    if( !iter || !out_value )
    {
        ccsp_set_error( CCSP_ERROR_NULL_POINTER, "null argument" );
        return CCSP_ERROR_NULL_POINTER;
    }
    
    auto * impl = reinterpret_cast<CCspDictIteratorImpl *>( iter );
    if( !impl -> started || impl -> current == impl -> end )
    {
        ccsp_set_error( CCSP_ERROR_INVALID_ARGUMENT, "iterator not positioned" );
        return CCSP_ERROR_INVALID_ARGUMENT;
    }
    
    try
    {
        *out_value = impl -> current.value<uint64_t>();
        return CCSP_OK;
    }
    catch( ... )
    {
        ccsp_set_error( CCSP_ERROR_TYPE_MISMATCH, "type mismatch" );
        return CCSP_ERROR_TYPE_MISMATCH;
    }
}

CCspErrorCode ccsp_dictionary_iter_get_double( CCspDictIteratorHandle iter, double * out_value )
{
    if( !iter || !out_value )
    {
        ccsp_set_error( CCSP_ERROR_NULL_POINTER, "null argument" );
        return CCSP_ERROR_NULL_POINTER;
    }
    
    auto * impl = reinterpret_cast<CCspDictIteratorImpl *>( iter );
    if( !impl -> started || impl -> current == impl -> end )
    {
        ccsp_set_error( CCSP_ERROR_INVALID_ARGUMENT, "iterator not positioned" );
        return CCSP_ERROR_INVALID_ARGUMENT;
    }
    
    try
    {
        *out_value = impl -> current.value<double>();
        return CCSP_OK;
    }
    catch( ... )
    {
        ccsp_set_error( CCSP_ERROR_TYPE_MISMATCH, "type mismatch" );
        return CCSP_ERROR_TYPE_MISMATCH;
    }
}

CCspErrorCode ccsp_dictionary_iter_get_datetime( CCspDictIteratorHandle iter, CCspDateTime * out_value )
{
    if( !iter || !out_value )
    {
        ccsp_set_error( CCSP_ERROR_NULL_POINTER, "null argument" );
        return CCSP_ERROR_NULL_POINTER;
    }
    
    auto * impl = reinterpret_cast<CCspDictIteratorImpl *>( iter );
    if( !impl -> started || impl -> current == impl -> end )
    {
        ccsp_set_error( CCSP_ERROR_INVALID_ARGUMENT, "iterator not positioned" );
        return CCSP_ERROR_INVALID_ARGUMENT;
    }
    
    try
    {
        csp::DateTime dt = impl -> current.value<csp::DateTime>();
        *out_value = dt.asNanoseconds();
        return CCSP_OK;
    }
    catch( ... )
    {
        ccsp_set_error( CCSP_ERROR_TYPE_MISMATCH, "type mismatch" );
        return CCSP_ERROR_TYPE_MISMATCH;
    }
}

CCspErrorCode ccsp_dictionary_iter_get_timedelta( CCspDictIteratorHandle iter, CCspTimeDelta * out_value )
{
    if( !iter || !out_value )
    {
        ccsp_set_error( CCSP_ERROR_NULL_POINTER, "null argument" );
        return CCSP_ERROR_NULL_POINTER;
    }
    
    auto * impl = reinterpret_cast<CCspDictIteratorImpl *>( iter );
    if( !impl -> started || impl -> current == impl -> end )
    {
        ccsp_set_error( CCSP_ERROR_INVALID_ARGUMENT, "iterator not positioned" );
        return CCSP_ERROR_INVALID_ARGUMENT;
    }
    
    try
    {
        csp::TimeDelta td = impl -> current.value<csp::TimeDelta>();
        *out_value = td.asNanoseconds();
        return CCSP_OK;
    }
    catch( ... )
    {
        ccsp_set_error( CCSP_ERROR_TYPE_MISMATCH, "type mismatch" );
        return CCSP_ERROR_TYPE_MISMATCH;
    }
}

CCspErrorCode ccsp_dictionary_iter_get_string( CCspDictIteratorHandle iter, const char ** out_data, size_t * out_length )
{
    if( !iter || !out_data || !out_length )
    {
        ccsp_set_error( CCSP_ERROR_NULL_POINTER, "null argument" );
        return CCSP_ERROR_NULL_POINTER;
    }
    
    auto * impl = reinterpret_cast<CCspDictIteratorImpl *>( iter );
    if( !impl -> started || impl -> current == impl -> end )
    {
        ccsp_set_error( CCSP_ERROR_INVALID_ARGUMENT, "iterator not positioned" );
        return CCSP_ERROR_INVALID_ARGUMENT;
    }
    
    try
    {
        const std::string & str = impl -> current.value<std::string>();
        *out_data = str.data();
        *out_length = str.size();
        return CCSP_OK;
    }
    catch( ... )
    {
        ccsp_set_error( CCSP_ERROR_TYPE_MISMATCH, "type mismatch" );
        return CCSP_ERROR_TYPE_MISMATCH;
    }
}

CCspErrorCode ccsp_dictionary_iter_get_dict( CCspDictIteratorHandle iter, CCspDictionaryHandle * out_dict )
{
    if( !iter || !out_dict )
    {
        ccsp_set_error( CCSP_ERROR_NULL_POINTER, "null argument" );
        return CCSP_ERROR_NULL_POINTER;
    }
    
    auto * impl = reinterpret_cast<CCspDictIteratorImpl *>( iter );
    if( !impl -> started || impl -> current == impl -> end )
    {
        ccsp_set_error( CCSP_ERROR_INVALID_ARGUMENT, "iterator not positioned" );
        return CCSP_ERROR_INVALID_ARGUMENT;
    }
    
    try
    {
        csp::DictionaryPtr nested = impl -> current.value<csp::DictionaryPtr>();
        *out_dict = reinterpret_cast<CCspDictionaryHandle>( nested.get() );
        return CCSP_OK;
    }
    catch( ... )
    {
        ccsp_set_error( CCSP_ERROR_TYPE_MISMATCH, "type mismatch" );
        return CCSP_ERROR_TYPE_MISMATCH;
    }
}

} // extern "C"
