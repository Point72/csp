/*
 * Implementation of the C API tagged value container.
 *
 * Ownership rules:
 *   - Scalars are stored inline.
 *   - Strings are either owned (allocated here, released by ccsp_value_free) or borrowed views.
 *   - Struct handles and enum metadata are opaque and owned by CSP, never released here.
 *   - Arrays carry an is_owned flag; owned arrays of fixed-size element types are deep copied.
 */

#include <csp/engine/c/CspValue.h>
#include <csp/engine/c/CspError.h>

#include <stdlib.h>
#include <string.h>

namespace
{

/* Size of one array element, or 0 for element types with no flat representation here */
size_t arrayElemSize( CCspType type )
{
    switch( type )
    {
        case CCSP_TYPE_BOOL:
        case CCSP_TYPE_INT8:
        case CCSP_TYPE_UINT8:      return 1;
        case CCSP_TYPE_INT16:
        case CCSP_TYPE_UINT16:     return 2;
        case CCSP_TYPE_INT32:
        case CCSP_TYPE_UINT32:
        case CCSP_TYPE_DATE:       return 4;
        case CCSP_TYPE_INT64:
        case CCSP_TYPE_UINT64:
        case CCSP_TYPE_DOUBLE:
        case CCSP_TYPE_DATETIME:
        case CCSP_TYPE_TIMEDELTA:
        case CCSP_TYPE_TIME:       return 8;
        default:                   return 0;
    }
}

void releaseOwned( CCspValue * value )
{
    if( value -> type == CCSP_TYPE_STRING && value -> string_val.is_owned )
    {
        free( ( void * ) value -> string_val.data );
    }
    else if( value -> type == CCSP_TYPE_ARRAY && value -> array_val.is_owned )
    {
        free( value -> array_val.data );
    }
}

/* Every setter overwrites whatever was there, so it must release first */
void resetTo( CCspValue * value, CCspType type )
{
    releaseOwned( value );
    memset( value, 0, sizeof( *value ) );
    value -> type = type;
}

CCspErrorCode checkGet( const CCspValue * value, CCspType expected, const void * out )
{
    if( !value || !out )
    {
        ccsp_set_error( CCSP_ERROR_NULL_POINTER, "null argument" );
        return CCSP_ERROR_NULL_POINTER;
    }
    if( value -> type != expected )
    {
        ccsp_set_error( CCSP_ERROR_TYPE_MISMATCH, "value type mismatch" );
        return CCSP_ERROR_TYPE_MISMATCH;
    }
    return CCSP_OK;
}

}

extern "C" {

// ============================================================================
// Lifecycle
// ============================================================================

void ccsp_value_init( CCspValue * value )
{
    if( !value )
        return;
    memset( value, 0, sizeof( *value ) );
    value -> type = CCSP_TYPE_UNKNOWN;
}

void ccsp_value_free( CCspValue * value )
{
    if( !value )
        return;
    releaseOwned( value );
    memset( value, 0, sizeof( *value ) );
    value -> type = CCSP_TYPE_UNKNOWN;
}

CCspErrorCode ccsp_value_copy( CCspValue * dest, const CCspValue * src )
{
    if( !dest || !src )
    {
        ccsp_set_error( CCSP_ERROR_NULL_POINTER, "null argument" );
        return CCSP_ERROR_NULL_POINTER;
    }

    if( dest == src )
        return CCSP_OK;

    if( src -> type == CCSP_TYPE_STRING && src -> string_val.is_owned )
    {
        if( src -> string_val.length == SIZE_MAX )
        {
            ccsp_set_error( CCSP_ERROR_OUT_OF_RANGE, "string too long to copy" );
            return CCSP_ERROR_OUT_OF_RANGE;
        }

        char * buffer = ( char * ) malloc( src -> string_val.length + 1 );
        if( !buffer )
        {
            ccsp_set_error( CCSP_ERROR_OUT_OF_MEMORY, "out of memory" );
            return CCSP_ERROR_OUT_OF_MEMORY;
        }

        if( src -> string_val.length > 0 )
            memcpy( buffer, src -> string_val.data, src -> string_val.length );
        buffer[ src -> string_val.length ] = '\0';

        ccsp_value_free( dest );
        dest -> type = CCSP_TYPE_STRING;
        dest -> string_val.data = buffer;
        dest -> string_val.length = src -> string_val.length;
        dest -> string_val.is_owned = 1;
        return CCSP_OK;
    }

    if( src -> type == CCSP_TYPE_ARRAY && src -> array_val.is_owned )
    {
        const size_t elemSize = arrayElemSize( src -> array_val.elem_type );
        if( elemSize == 0 )
        {
            ccsp_set_error( CCSP_ERROR_NOT_IMPLEMENTED,
                            "deep copy of an owned array is only supported for fixed-size element types" );
            return CCSP_ERROR_NOT_IMPLEMENTED;
        }
        if( src -> array_val.length != 0 && src -> array_val.length > SIZE_MAX / elemSize )
        {
            ccsp_set_error( CCSP_ERROR_OUT_OF_RANGE, "array too large to copy" );
            return CCSP_ERROR_OUT_OF_RANGE;
        }

        const size_t bytes = src -> array_val.length * elemSize;
        void * buffer = malloc( bytes ? bytes : 1 );
        if( !buffer )
        {
            ccsp_set_error( CCSP_ERROR_OUT_OF_MEMORY, "out of memory" );
            return CCSP_ERROR_OUT_OF_MEMORY;
        }
        if( bytes > 0 )
            memcpy( buffer, src -> array_val.data, bytes );

        ccsp_value_free( dest );
        dest -> type = CCSP_TYPE_ARRAY;
        dest -> array_val.data = buffer;
        dest -> array_val.length = src -> array_val.length;
        dest -> array_val.elem_type = src -> array_val.elem_type;
        dest -> array_val.is_owned = 1;
        return CCSP_OK;
    }

    /* Everything else is either inline or a borrowed reference */
    ccsp_value_free( dest );
    memcpy( dest, src, sizeof( *dest ) );
    return CCSP_OK;
}

void ccsp_value_move( CCspValue * dest, CCspValue * src )
{
    if( !dest || !src || dest == src )
        return;

    ccsp_value_free( dest );
    memcpy( dest, src, sizeof( *dest ) );
    memset( src, 0, sizeof( *src ) );
    src -> type = CCSP_TYPE_UNKNOWN;
}

// ============================================================================
// Setters
// ============================================================================

#define IMPLEMENT_VALUE_SETTER( name, c_type, type_enum, field )     \
    void ccsp_value_set_##name( CCspValue * value, c_type v )        \
    {                                                                \
        if( !value )                                                 \
            return;                                                  \
        resetTo( value, type_enum );                                 \
        value -> field = v;                                          \
    }

IMPLEMENT_VALUE_SETTER( bool,      int8_t,        CCSP_TYPE_BOOL,      bool_val )
IMPLEMENT_VALUE_SETTER( int8,      int8_t,        CCSP_TYPE_INT8,      int8_val )
IMPLEMENT_VALUE_SETTER( uint8,     uint8_t,       CCSP_TYPE_UINT8,     uint8_val )
IMPLEMENT_VALUE_SETTER( int16,     int16_t,       CCSP_TYPE_INT16,     int16_val )
IMPLEMENT_VALUE_SETTER( uint16,    uint16_t,      CCSP_TYPE_UINT16,    uint16_val )
IMPLEMENT_VALUE_SETTER( int32,     int32_t,       CCSP_TYPE_INT32,     int32_val )
IMPLEMENT_VALUE_SETTER( uint32,    uint32_t,      CCSP_TYPE_UINT32,    uint32_val )
IMPLEMENT_VALUE_SETTER( int64,     int64_t,       CCSP_TYPE_INT64,     int64_val )
IMPLEMENT_VALUE_SETTER( uint64,    uint64_t,      CCSP_TYPE_UINT64,    uint64_val )
IMPLEMENT_VALUE_SETTER( double,    double,        CCSP_TYPE_DOUBLE,    double_val )
IMPLEMENT_VALUE_SETTER( datetime,  CCspDateTime,  CCSP_TYPE_DATETIME,  datetime_val )
IMPLEMENT_VALUE_SETTER( timedelta, CCspTimeDelta, CCSP_TYPE_TIMEDELTA, timedelta_val )
IMPLEMENT_VALUE_SETTER( date,      CCspDate,      CCSP_TYPE_DATE,      date_val )
IMPLEMENT_VALUE_SETTER( time,      CCspTime,      CCSP_TYPE_TIME,      time_val )

CCspErrorCode ccsp_value_set_string( CCspValue * value, const char * data, size_t length )
{
    if( !value )
    {
        ccsp_set_error( CCSP_ERROR_NULL_POINTER, "null value" );
        return CCSP_ERROR_NULL_POINTER;
    }
    if( !data && length > 0 )
    {
        ccsp_set_error( CCSP_ERROR_INVALID_ARGUMENT, "null data with non-zero length" );
        return CCSP_ERROR_INVALID_ARGUMENT;
    }
    if( length == SIZE_MAX )
    {
        ccsp_set_error( CCSP_ERROR_OUT_OF_RANGE, "string too long" );
        return CCSP_ERROR_OUT_OF_RANGE;
    }

    char * buffer = ( char * ) malloc( length + 1 );
    if( !buffer )
    {
        ccsp_set_error( CCSP_ERROR_OUT_OF_MEMORY, "out of memory" );
        return CCSP_ERROR_OUT_OF_MEMORY;
    }

    if( length > 0 )
        memcpy( buffer, data, length );
    buffer[ length ] = '\0';

    resetTo( value, CCSP_TYPE_STRING );
    value -> string_val.data = buffer;
    value -> string_val.length = length;
    value -> string_val.is_owned = 1;
    return CCSP_OK;
}

CCspErrorCode ccsp_value_set_string_cstr( CCspValue * value, const char * cstr )
{
    return ccsp_value_set_string( value, cstr, cstr ? strlen( cstr ) : 0 );
}

void ccsp_value_set_string_view( CCspValue * value, const char * data, size_t length )
{
    if( !value )
        return;

    resetTo( value, CCSP_TYPE_STRING );
    value -> string_val.data = data;
    value -> string_val.length = data ? length : 0;
    value -> string_val.is_owned = 0;
}

void ccsp_value_set_struct( CCspValue * value, CCspStructHandle s )
{
    if( !value )
        return;

    resetTo( value, CCSP_TYPE_STRUCT );
    value -> struct_val = s;
}

void ccsp_value_set_enum( CCspValue * value, int32_t ordinal, CCspEnumMetaHandle meta )
{
    if( !value )
        return;

    resetTo( value, CCSP_TYPE_ENUM );
    value -> enum_val.ordinal = ordinal;
    value -> enum_val.meta = meta;
}

// ============================================================================
// Getters
// ============================================================================

#define IMPLEMENT_VALUE_GETTER( name, c_type, type_enum, field )                    \
    CCspErrorCode ccsp_value_get_##name( const CCspValue * value, c_type * out )    \
    {                                                                               \
        CCspErrorCode err = checkGet( value, type_enum, out );                      \
        if( err != CCSP_OK )                                                        \
            return err;                                                             \
        *out = value -> field;                                                      \
        return CCSP_OK;                                                             \
    }

IMPLEMENT_VALUE_GETTER( bool,      int8_t,        CCSP_TYPE_BOOL,      bool_val )
IMPLEMENT_VALUE_GETTER( int8,      int8_t,        CCSP_TYPE_INT8,      int8_val )
IMPLEMENT_VALUE_GETTER( uint8,     uint8_t,       CCSP_TYPE_UINT8,     uint8_val )
IMPLEMENT_VALUE_GETTER( int16,     int16_t,       CCSP_TYPE_INT16,     int16_val )
IMPLEMENT_VALUE_GETTER( uint16,    uint16_t,      CCSP_TYPE_UINT16,    uint16_val )
IMPLEMENT_VALUE_GETTER( int32,     int32_t,       CCSP_TYPE_INT32,     int32_val )
IMPLEMENT_VALUE_GETTER( uint32,    uint32_t,      CCSP_TYPE_UINT32,    uint32_val )
IMPLEMENT_VALUE_GETTER( int64,     int64_t,       CCSP_TYPE_INT64,     int64_val )
IMPLEMENT_VALUE_GETTER( uint64,    uint64_t,      CCSP_TYPE_UINT64,    uint64_val )
IMPLEMENT_VALUE_GETTER( double,    double,        CCSP_TYPE_DOUBLE,    double_val )
IMPLEMENT_VALUE_GETTER( datetime,  CCspDateTime,  CCSP_TYPE_DATETIME,  datetime_val )
IMPLEMENT_VALUE_GETTER( timedelta, CCspTimeDelta, CCSP_TYPE_TIMEDELTA, timedelta_val )
IMPLEMENT_VALUE_GETTER( date,      CCspDate,      CCSP_TYPE_DATE,      date_val )
IMPLEMENT_VALUE_GETTER( time,      CCspTime,      CCSP_TYPE_TIME,      time_val )

CCspErrorCode ccsp_value_get_string( const CCspValue * value, const char ** out_data, size_t * out_length )
{
    CCspErrorCode err = checkGet( value, CCSP_TYPE_STRING, out_data );
    if( err != CCSP_OK )
        return err;
    if( !out_length )
    {
        ccsp_set_error( CCSP_ERROR_NULL_POINTER, "null argument" );
        return CCSP_ERROR_NULL_POINTER;
    }

    *out_data = value -> string_val.data;
    *out_length = value -> string_val.length;
    return CCSP_OK;
}

CCspErrorCode ccsp_value_get_struct( const CCspValue * value, CCspStructHandle * out )
{
    CCspErrorCode err = checkGet( value, CCSP_TYPE_STRUCT, out );
    if( err != CCSP_OK )
        return err;

    *out = value -> struct_val;
    return CCSP_OK;
}

CCspErrorCode ccsp_value_get_enum( const CCspValue * value, int32_t * out_ordinal, CCspEnumMetaHandle * out_meta )
{
    CCspErrorCode err = checkGet( value, CCSP_TYPE_ENUM, out_ordinal );
    if( err != CCSP_OK )
        return err;

    *out_ordinal = value -> enum_val.ordinal;
    if( out_meta )
        *out_meta = value -> enum_val.meta;
    return CCSP_OK;
}

// ============================================================================
// Type checking
// ============================================================================

int ccsp_value_is_numeric( const CCspValue * value )
{
    if( !value )
        return 0;

    switch( value -> type )
    {
        case CCSP_TYPE_INT8:
        case CCSP_TYPE_UINT8:
        case CCSP_TYPE_INT16:
        case CCSP_TYPE_UINT16:
        case CCSP_TYPE_INT32:
        case CCSP_TYPE_UINT32:
        case CCSP_TYPE_INT64:
        case CCSP_TYPE_UINT64:
        case CCSP_TYPE_DOUBLE:
            return 1;
        default:
            return 0;
    }
}

int ccsp_value_is_integer( const CCspValue * value )
{
    if( !value )
        return 0;

    switch( value -> type )
    {
        case CCSP_TYPE_INT8:
        case CCSP_TYPE_UINT8:
        case CCSP_TYPE_INT16:
        case CCSP_TYPE_UINT16:
        case CCSP_TYPE_INT32:
        case CCSP_TYPE_UINT32:
        case CCSP_TYPE_INT64:
        case CCSP_TYPE_UINT64:
            return 1;
        default:
            return 0;
    }
}

}
