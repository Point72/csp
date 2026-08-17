/*
 * Implementation of C API for CSP Struct Access
 */

#include <csp/engine/c/CspStruct.h>
#include <csp/engine/c/CspError.h>
#include <csp/engine/ExternBoundary.h>
#include <csp/engine/Struct.h>
#include <cstring>

// ============================================================================
// Helper Functions
// ============================================================================

static CCspType cspTypeToC( csp::CspType::Type t )
{
    switch( t )
    {
        case csp::CspType::Type::BOOL:            return CCSP_TYPE_BOOL;
        case csp::CspType::Type::INT8:            return CCSP_TYPE_INT8;
        case csp::CspType::Type::UINT8:           return CCSP_TYPE_UINT8;
        case csp::CspType::Type::INT16:           return CCSP_TYPE_INT16;
        case csp::CspType::Type::UINT16:          return CCSP_TYPE_UINT16;
        case csp::CspType::Type::INT32:           return CCSP_TYPE_INT32;
        case csp::CspType::Type::UINT32:          return CCSP_TYPE_UINT32;
        case csp::CspType::Type::INT64:           return CCSP_TYPE_INT64;
        case csp::CspType::Type::UINT64:          return CCSP_TYPE_UINT64;
        case csp::CspType::Type::DOUBLE:          return CCSP_TYPE_DOUBLE;
        case csp::CspType::Type::STRING:          return CCSP_TYPE_STRING;
        case csp::CspType::Type::DATETIME:        return CCSP_TYPE_DATETIME;
        case csp::CspType::Type::TIMEDELTA:       return CCSP_TYPE_TIMEDELTA;
        case csp::CspType::Type::DATE:            return CCSP_TYPE_DATE;
        case csp::CspType::Type::TIME:            return CCSP_TYPE_TIME;
        case csp::CspType::Type::ENUM:            return CCSP_TYPE_ENUM;
        case csp::CspType::Type::STRUCT:          return CCSP_TYPE_STRUCT;
        case csp::CspType::Type::ARRAY:           return CCSP_TYPE_ARRAY;
        case csp::CspType::Type::DIALECT_GENERIC: return CCSP_TYPE_DIALECT_GENERIC;
        default:                                  return CCSP_TYPE_UNKNOWN;
    }
}

// CCspStructHandle is actually a csp::StructPtr* (heap-allocated smart pointer)
// This helper extracts the raw Struct* for read operations
static inline const csp::Struct * getStructConst( CCspStructHandle s )
{
    auto * ptr = reinterpret_cast<const csp::StructPtr *>( s );
    return ptr -> get();
}

// For write operations
static inline csp::Struct * getStruct( CCspStructHandle s )
{
    auto * ptr = reinterpret_cast<csp::StructPtr *>( s );
    return ptr -> get();
}

// A field handle is a raw offset into its own struct type's storage, so applying one to an
// instance of a different type reads or writes outside that instance. A derived struct may
// override an inherited field with a distinct StructField sharing the base's layout, so accept
// any handle whose layout matches and resolve to this instance's own field.
static const csp::StructField * resolveField( const csp::Struct * st, CCspStructFieldHandle field )
{
    auto * f = reinterpret_cast<const csp::StructField *>( field );
    const csp::StructField * owned = st -> meta() -> field( f -> fieldname() ).get();
    if( !owned || ( owned != f && ( owned -> offset()     != f -> offset()     ||
                                    owned -> maskOffset() != f -> maskOffset() ||
                                    owned -> maskBit()    != f -> maskBit()    ) ) )
    {
        ccsp_set_error( CCSP_ERROR_INVALID_ARGUMENT, "field does not belong to this struct type" );
        return nullptr;
    }
    return owned;
}

extern "C" {

// ============================================================================
// StructMeta Functions
// ============================================================================

const char * ccsp_struct_meta_name( CCspStructMetaHandle meta )
{
    if( !meta ) return nullptr;
    
    auto * m = reinterpret_cast<const csp::StructMeta *>( meta );
    return m -> name().c_str();
}

size_t ccsp_struct_meta_field_count( CCspStructMetaHandle meta )
{
    if( !meta ) return 0;
    
    auto * m = reinterpret_cast<const csp::StructMeta *>( meta );
    return m -> fieldNames().size();
}

CCspStructFieldHandle ccsp_struct_meta_field_by_index( CCspStructMetaHandle meta, size_t index )
{
    if( !meta ) return nullptr;
    
    auto * m = reinterpret_cast<const csp::StructMeta *>( meta );
    const auto & fieldNames = m -> fieldNames();
    
    if( index >= fieldNames.size() )
        return nullptr;
    
    const auto & fieldPtr = m -> field( fieldNames[index] );
    if( !fieldPtr )
        return nullptr;
    
    return reinterpret_cast<CCspStructFieldHandle>( fieldPtr.get() );
}

CCspStructFieldHandle ccsp_struct_meta_field_by_name( CCspStructMetaHandle meta, const char * name )
{
    if( !meta || !name ) return nullptr;
    
    auto * m = reinterpret_cast<const csp::StructMeta *>( meta );
    const auto & fieldPtr = m -> field( name );
    
    if( !fieldPtr )
        return nullptr;
    
    return reinterpret_cast<CCspStructFieldHandle>( fieldPtr.get() );
}

const char * ccsp_struct_meta_field_name_by_index( CCspStructMetaHandle meta, size_t index )
{
    if( !meta ) return nullptr;
    
    auto * m = reinterpret_cast<const csp::StructMeta *>( meta );
    const auto & fieldNames = m -> fieldNames();
    
    if( index >= fieldNames.size() )
        return nullptr;
    
    return fieldNames[index].c_str();
}

int ccsp_struct_meta_is_strict( CCspStructMetaHandle meta )
{
    if( !meta ) return 0;
    
    auto * m = reinterpret_cast<const csp::StructMeta *>( meta );
    return m -> isStrict() ? 1 : 0;
}

// ============================================================================
// StructField Functions
// ============================================================================

const char * ccsp_struct_field_name( CCspStructFieldHandle field )
{
    if( !field ) return nullptr;
    
    auto * f = reinterpret_cast<const csp::StructField *>( field );
    return f -> fieldname().c_str();
}

CCspType ccsp_struct_field_type( CCspStructFieldHandle field )
{
    if( !field ) return CCSP_TYPE_UNKNOWN;
    
    auto * f = reinterpret_cast<const csp::StructField *>( field );
    return cspTypeToC( f -> type() -> type() );
}

int ccsp_struct_field_is_optional( CCspStructFieldHandle field )
{
    if( !field ) return 0;
    
    auto * f = reinterpret_cast<const csp::StructField *>( field );
    return f -> isOptional() ? 1 : 0;
}

// ============================================================================
// Struct Instance Functions
// ============================================================================

CCspStructMetaHandle ccsp_struct_meta( CCspStructHandle s )
{
    if( !s ) return nullptr;
    
    const csp::Struct * st = getStructConst( s );
    return reinterpret_cast<CCspStructMetaHandle>( const_cast<csp::StructMeta *>( st -> meta() ) );
}

int ccsp_struct_field_is_set( CCspStructHandle s, CCspStructFieldHandle field )
{
    if( !s || !field ) return 0;
    
    const csp::Struct * st = getStructConst( s );
    auto * f = resolveField( st, field );
    return f && f -> isSet( st ) ? 1 : 0;
}

int ccsp_struct_field_is_none( CCspStructHandle s, CCspStructFieldHandle field )
{
    if( !s || !field ) return 0;
    
    const csp::Struct * st = getStructConst( s );
    auto * f = resolveField( st, field );
    return f && f -> isNone( st ) ? 1 : 0;
}

// ============================================================================
// Field Value Getters
// ============================================================================

#define IMPLEMENT_GETTER( name, out_type, csp_field_type, type_enum, value_expr )           \
    CCspErrorCode ccsp_struct_get_##name( CCspStructHandle s, CCspStructFieldHandle field,  \
                                          out_type * out_value )                           \
    {                                                                                       \
        if( !s || !field || !out_value )                                                    \
        {                                                                                   \
            ccsp_set_error( CCSP_ERROR_NULL_POINTER, "null argument" );                     \
            return CCSP_ERROR_NULL_POINTER;                                                 \
        }                                                                                   \
        try                                                                                 \
        {                                                                                   \
            const csp::Struct * st = getStructConst( s );                                   \
            auto * f = resolveField( st, field );                                           \
            if( !f )                                                                        \
                return CCSP_ERROR_INVALID_ARGUMENT;                                         \
                                                                                            \
            if( !f -> isSet( st ) )                                                         \
            {                                                                               \
                ccsp_set_error( CCSP_ERROR_KEY_NOT_FOUND, "field not set" );                \
                return CCSP_ERROR_KEY_NOT_FOUND;                                            \
            }                                                                               \
                                                                                            \
            if( f -> type() -> type() != csp::CspType::Type::type_enum )                    \
            {                                                                               \
                ccsp_set_error( CCSP_ERROR_TYPE_MISMATCH, "field type mismatch" );          \
                return CCSP_ERROR_TYPE_MISMATCH;                                            \
            }                                                                               \
                                                                                            \
            auto * tf = static_cast<const csp::csp_field_type *>( f );                      \
            *out_value = value_expr;                                                        \
            return CCSP_OK;                                                                 \
        }                                                                                   \
        CCSP_CATCH_ERR                                                                      \
    }

#define IMPLEMENT_SETTER( name, in_type, csp_field_type, type_enum, value_expr )            \
    CCspErrorCode ccsp_struct_set_##name( CCspStructHandle s, CCspStructFieldHandle field,  \
                                          in_type value )                                  \
    {                                                                                       \
        if( !s || !field )                                                                  \
        {                                                                                   \
            ccsp_set_error( CCSP_ERROR_NULL_POINTER, "null argument" );                     \
            return CCSP_ERROR_NULL_POINTER;                                                 \
        }                                                                                   \
        try                                                                                 \
        {                                                                                   \
            csp::Struct * st = getStruct( s );                                              \
            auto * f = resolveField( st, field );                                           \
            if( !f )                                                                        \
                return CCSP_ERROR_INVALID_ARGUMENT;                                         \
                                                                                            \
            if( f -> type() -> type() != csp::CspType::Type::type_enum )                    \
            {                                                                               \
                ccsp_set_error( CCSP_ERROR_TYPE_MISMATCH, "field type mismatch" );          \
                return CCSP_ERROR_TYPE_MISMATCH;                                            \
            }                                                                               \
                                                                                            \
            auto * tf = static_cast<const csp::csp_field_type *>( f );                      \
            tf -> setValue( st, value_expr );                                               \
            return CCSP_OK;                                                                 \
        }                                                                                   \
        CCSP_CATCH_ERR                                                                      \
    }

IMPLEMENT_GETTER( bool,      int8_t,        BoolStructField,      BOOL,      tf -> value( st ) ? 1 : 0 )
IMPLEMENT_GETTER( int8,      int8_t,        Int8StructField,      INT8,      tf -> value( st ) )
IMPLEMENT_GETTER( uint8,     uint8_t,       UInt8StructField,     UINT8,     tf -> value( st ) )
IMPLEMENT_GETTER( int16,     int16_t,       Int16StructField,     INT16,     tf -> value( st ) )
IMPLEMENT_GETTER( uint16,    uint16_t,      UInt16StructField,    UINT16,    tf -> value( st ) )
IMPLEMENT_GETTER( int32,     int32_t,       Int32StructField,     INT32,     tf -> value( st ) )
IMPLEMENT_GETTER( uint32,    uint32_t,      UInt32StructField,    UINT32,    tf -> value( st ) )
IMPLEMENT_GETTER( int64,     int64_t,       Int64StructField,     INT64,     tf -> value( st ) )
IMPLEMENT_GETTER( uint64,    uint64_t,      UInt64StructField,    UINT64,    tf -> value( st ) )
IMPLEMENT_GETTER( double,    double,        DoubleStructField,    DOUBLE,    tf -> value( st ) )
IMPLEMENT_GETTER( datetime,  CCspDateTime,  DateTimeStructField,  DATETIME,  tf -> value( st ).asNanoseconds() )
IMPLEMENT_GETTER( timedelta, CCspTimeDelta, TimeDeltaStructField, TIMEDELTA, tf -> value( st ).asNanoseconds() )
IMPLEMENT_GETTER( enum,      int32_t,       CspEnumStructField,   ENUM,      static_cast<int32_t>( tf -> value( st ).value() ) )

CCspErrorCode ccsp_struct_get_string( CCspStructHandle s, CCspStructFieldHandle field, const char ** out_data, size_t * out_length )
{
    if( !s || !field || !out_data || !out_length )
    {
        ccsp_set_error( CCSP_ERROR_NULL_POINTER, "null argument" );
        return CCSP_ERROR_NULL_POINTER;
    }
    
    const csp::Struct * st = getStructConst( s );
    auto * f = resolveField( st, field );
    if( !f )
        return CCSP_ERROR_INVALID_ARGUMENT;
    
    if( !f -> isSet( st ) )
    {
        ccsp_set_error( CCSP_ERROR_KEY_NOT_FOUND, "field not set" );
        return CCSP_ERROR_KEY_NOT_FOUND;
    }
    
    if( f -> type() -> type() != csp::CspType::Type::STRING )
    {
        ccsp_set_error( CCSP_ERROR_TYPE_MISMATCH, "field type mismatch" );
        return CCSP_ERROR_TYPE_MISMATCH;
    }
    
    auto * tf = static_cast<const csp::StringStructField *>( f );
    const std::string & str = tf -> value( st );
    *out_data = str.data();
    *out_length = str.size();
    return CCSP_OK;
}

CCspErrorCode ccsp_struct_get_struct( CCspStructHandle s, CCspStructFieldHandle field, CCspStructHandle * out_struct )
{
    if( !s || !field || !out_struct )
    {
        ccsp_set_error( CCSP_ERROR_NULL_POINTER, "null argument" );
        return CCSP_ERROR_NULL_POINTER;
    }
    
    const csp::Struct * st = getStructConst( s );
    auto * f = resolveField( st, field );
    if( !f )
        return CCSP_ERROR_INVALID_ARGUMENT;
    
    if( !f -> isSet( st ) )
    {
        ccsp_set_error( CCSP_ERROR_KEY_NOT_FOUND, "field not set" );
        return CCSP_ERROR_KEY_NOT_FOUND;
    }
    
    if( f -> type() -> type() != csp::CspType::Type::STRUCT )
    {
        ccsp_set_error( CCSP_ERROR_TYPE_MISMATCH, "field type mismatch" );
        return CCSP_ERROR_TYPE_MISMATCH;
    }
    
    /* Hand back an independent handle sharing ownership; released with ccsp_struct_destroy */
    auto * tf = static_cast<const csp::StructStructField *>( f );
    const csp::StructPtr & nested = tf -> value( st );
    if( !nested.get() )
    {
        ccsp_set_error( CCSP_ERROR_VALUE, "nested struct is null" );
        return CCSP_ERROR_VALUE;
    }

    *out_struct = reinterpret_cast<CCspStructHandle>( new csp::StructPtr( nested ) );
    return CCSP_OK;
}

// ============================================================================
// Field Value Getters by Name
// ============================================================================

CCspErrorCode ccsp_struct_get_bool_by_name( CCspStructHandle s, const char * name, int8_t * out_value )
{
    if( !s || !name )
    {
        ccsp_set_error( CCSP_ERROR_NULL_POINTER, "null argument" );
        return CCSP_ERROR_NULL_POINTER;
    }
    
    CCspStructMetaHandle meta = ccsp_struct_meta( s );
    CCspStructFieldHandle field = ccsp_struct_meta_field_by_name( meta, name );
    if( !field )
    {
        ccsp_set_error( CCSP_ERROR_KEY_NOT_FOUND, "field not found" );
        return CCSP_ERROR_KEY_NOT_FOUND;
    }
    
    return ccsp_struct_get_bool( s, field, out_value );
}

CCspErrorCode ccsp_struct_get_int32_by_name( CCspStructHandle s, const char * name, int32_t * out_value )
{
    if( !s || !name )
    {
        ccsp_set_error( CCSP_ERROR_NULL_POINTER, "null argument" );
        return CCSP_ERROR_NULL_POINTER;
    }
    
    CCspStructMetaHandle meta = ccsp_struct_meta( s );
    CCspStructFieldHandle field = ccsp_struct_meta_field_by_name( meta, name );
    if( !field )
    {
        ccsp_set_error( CCSP_ERROR_KEY_NOT_FOUND, "field not found" );
        return CCSP_ERROR_KEY_NOT_FOUND;
    }
    
    return ccsp_struct_get_int32( s, field, out_value );
}

CCspErrorCode ccsp_struct_get_int64_by_name( CCspStructHandle s, const char * name, int64_t * out_value )
{
    if( !s || !name )
    {
        ccsp_set_error( CCSP_ERROR_NULL_POINTER, "null argument" );
        return CCSP_ERROR_NULL_POINTER;
    }
    
    CCspStructMetaHandle meta = ccsp_struct_meta( s );
    CCspStructFieldHandle field = ccsp_struct_meta_field_by_name( meta, name );
    if( !field )
    {
        ccsp_set_error( CCSP_ERROR_KEY_NOT_FOUND, "field not found" );
        return CCSP_ERROR_KEY_NOT_FOUND;
    }
    
    return ccsp_struct_get_int64( s, field, out_value );
}

CCspErrorCode ccsp_struct_get_double_by_name( CCspStructHandle s, const char * name, double * out_value )
{
    if( !s || !name )
    {
        ccsp_set_error( CCSP_ERROR_NULL_POINTER, "null argument" );
        return CCSP_ERROR_NULL_POINTER;
    }
    
    CCspStructMetaHandle meta = ccsp_struct_meta( s );
    CCspStructFieldHandle field = ccsp_struct_meta_field_by_name( meta, name );
    if( !field )
    {
        ccsp_set_error( CCSP_ERROR_KEY_NOT_FOUND, "field not found" );
        return CCSP_ERROR_KEY_NOT_FOUND;
    }
    
    return ccsp_struct_get_double( s, field, out_value );
}

CCspErrorCode ccsp_struct_get_datetime_by_name( CCspStructHandle s, const char * name, CCspDateTime * out_value )
{
    if( !s || !name )
    {
        ccsp_set_error( CCSP_ERROR_NULL_POINTER, "null argument" );
        return CCSP_ERROR_NULL_POINTER;
    }
    
    CCspStructMetaHandle meta = ccsp_struct_meta( s );
    CCspStructFieldHandle field = ccsp_struct_meta_field_by_name( meta, name );
    if( !field )
    {
        ccsp_set_error( CCSP_ERROR_KEY_NOT_FOUND, "field not found" );
        return CCSP_ERROR_KEY_NOT_FOUND;
    }
    
    return ccsp_struct_get_datetime( s, field, out_value );
}

CCspErrorCode ccsp_struct_get_string_by_name( CCspStructHandle s, const char * name, const char ** out_data, size_t * out_length )
{
    if( !s || !name )
    {
        ccsp_set_error( CCSP_ERROR_NULL_POINTER, "null argument" );
        return CCSP_ERROR_NULL_POINTER;
    }
    
    CCspStructMetaHandle meta = ccsp_struct_meta( s );
    CCspStructFieldHandle field = ccsp_struct_meta_field_by_name( meta, name );
    if( !field )
    {
        ccsp_set_error( CCSP_ERROR_KEY_NOT_FOUND, "field not found" );
        return CCSP_ERROR_KEY_NOT_FOUND;
    }
    
    return ccsp_struct_get_string( s, field, out_data, out_length );
}

// ============================================================================
// Field Value Setters
// ============================================================================

IMPLEMENT_SETTER( bool,      int8_t,        BoolStructField,      BOOL,      value != 0 )
IMPLEMENT_SETTER( int8,      int8_t,        Int8StructField,      INT8,      value )
IMPLEMENT_SETTER( uint8,     uint8_t,       UInt8StructField,     UINT8,     value )
IMPLEMENT_SETTER( int16,     int16_t,       Int16StructField,     INT16,     value )
IMPLEMENT_SETTER( uint16,    uint16_t,      UInt16StructField,    UINT16,    value )
IMPLEMENT_SETTER( int32,     int32_t,       Int32StructField,     INT32,     value )
IMPLEMENT_SETTER( uint32,    uint32_t,      UInt32StructField,    UINT32,    value )
IMPLEMENT_SETTER( int64,     int64_t,       Int64StructField,     INT64,     value )
IMPLEMENT_SETTER( uint64,    uint64_t,      UInt64StructField,    UINT64,    value )
IMPLEMENT_SETTER( double,    double,        DoubleStructField,    DOUBLE,    value )
IMPLEMENT_SETTER( datetime,  CCspDateTime,  DateTimeStructField,  DATETIME,  csp::DateTime::fromNanoseconds( value ) )
IMPLEMENT_SETTER( timedelta, CCspTimeDelta, TimeDeltaStructField, TIMEDELTA, csp::TimeDelta::fromNanoseconds( value ) )

CCspErrorCode ccsp_struct_set_string( CCspStructHandle s, CCspStructFieldHandle field,
                                      const char * data, size_t length )
{
    if( !s || !field )
    {
        ccsp_set_error( CCSP_ERROR_NULL_POINTER, "null argument" );
        return CCSP_ERROR_NULL_POINTER;
    }
    
    if( !data && length > 0 )
    {
        ccsp_set_error( CCSP_ERROR_INVALID_ARGUMENT, "null data with non-zero length" );
        return CCSP_ERROR_INVALID_ARGUMENT;
    }
    
    try
    {
        csp::Struct * st = getStruct( s );
        auto * f = resolveField( st, field );
        if( !f )
            return CCSP_ERROR_INVALID_ARGUMENT;
        
        if( f -> type() -> type() != csp::CspType::Type::STRING )
        {
            ccsp_set_error( CCSP_ERROR_TYPE_MISMATCH, "field type mismatch" );
            return CCSP_ERROR_TYPE_MISMATCH;
        }
        
        auto * tf = static_cast<const csp::StringStructField *>( f );
        tf -> setValue( st, std::string( data ? data : "", length ) );
        return CCSP_OK;
    }
    CCSP_CATCH_ERR
}

CCspErrorCode ccsp_struct_set_enum( CCspStructHandle s, CCspStructFieldHandle field, int32_t ordinal )
{
    if( !s || !field )
    {
        ccsp_set_error( CCSP_ERROR_NULL_POINTER, "null argument" );
        return CCSP_ERROR_NULL_POINTER;
    }

    try
    {
        csp::Struct * st = getStruct( s );
        auto * f = resolveField( st, field );
        if( !f )
            return CCSP_ERROR_INVALID_ARGUMENT;

        if( f -> type() -> type() != csp::CspType::Type::ENUM )
        {
            ccsp_set_error( CCSP_ERROR_TYPE_MISMATCH, "field type mismatch" );
            return CCSP_ERROR_TYPE_MISMATCH;
        }

        /* The ordinal is only meaningful against the field's own enum meta */
        const auto & meta = static_cast<const csp::CspEnumType *>( f -> type().get() ) -> meta();
        auto * tf = static_cast<const csp::CspEnumStructField *>( f );
        tf -> setValue( st, meta -> create( ordinal ) );
        return CCSP_OK;
    }
    CCSP_CATCH_ERR
}

// ============================================================================
// Struct Creation
// ============================================================================

// Note: CCspStructHandle is actually a csp::StructPtr* (pointer to smart pointer)
// This allows proper reference counting without needing access to private incref/decref

CCspStructHandle ccsp_struct_create( CCspStructMetaHandle meta )
{
    if( !meta )
    {
        ccsp_set_error( CCSP_ERROR_NULL_POINTER, "null meta" );
        return nullptr;
    }
    
    auto * m = reinterpret_cast<csp::StructMeta *>( meta );
    
    try
    {
        csp::StructPtr * ptr = new csp::StructPtr( m -> create() );
        return reinterpret_cast<CCspStructHandle>( ptr );
    }
    catch( ... )
    {
        ccsp_set_error( CCSP_ERROR_OUT_OF_MEMORY, "failed to create struct" );
        return nullptr;
    }
}

void ccsp_struct_destroy( CCspStructHandle s )
{
    if( !s ) return;
    
    auto * ptr = reinterpret_cast<csp::StructPtr *>( s );
    delete ptr;
}

CCspStructHandle ccsp_struct_copy( CCspStructHandle s )
{
    if( !s )
    {
        ccsp_set_error( CCSP_ERROR_NULL_POINTER, "null struct" );
        return nullptr;
    }
    
    auto * srcPtr = reinterpret_cast<csp::StructPtr *>( s );
    const csp::Struct * st = srcPtr -> get();
    
    try
    {
        csp::StructPtr * copy = new csp::StructPtr( st -> meta() -> create() );
        csp::StructMeta::deepcopyFrom( st, copy -> get() );
        
        return reinterpret_cast<CCspStructHandle>( copy );
    }
    catch( ... )
    {
        ccsp_set_error( CCSP_ERROR_OUT_OF_MEMORY, "failed to copy struct" );
        return nullptr;
    }
}

} // extern "C"
