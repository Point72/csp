/*
 * Implementation of C API for CSP Struct Access
 */

#include <csp/engine/c/CspStruct.h>
#include <csp/engine/c/CspError.h>
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
    auto * f = reinterpret_cast<const csp::StructField *>( field );
    return f -> isSet( st ) ? 1 : 0;
}

int ccsp_struct_field_is_none( CCspStructHandle s, CCspStructFieldHandle field )
{
    if( !s || !field ) return 0;
    
    const csp::Struct * st = getStructConst( s );
    auto * f = reinterpret_cast<const csp::StructField *>( field );
    return f -> isNone( st ) ? 1 : 0;
}

// ============================================================================
// Field Value Getters
// ============================================================================

// Helper macro for type-safe getters
#define IMPLEMENT_GETTER( c_type, csp_field_type, type_enum )                              \
    CCspErrorCode ccsp_struct_get_##c_type( CCspStructHandle s, CCspStructFieldHandle field,\
                                            c_type##_t * out_value )                       \
    {                                                                                       \
        if( !s || !field || !out_value )                                                   \
        {                                                                                   \
            ccsp_set_error( CCSP_ERROR_NULL_POINTER, "null argument" );                    \
            return CCSP_ERROR_NULL_POINTER;                                                 \
        }                                                                                   \
                                                                                            \
        const csp::Struct * st = getStructConst( s );                            \
        auto * f = reinterpret_cast<const csp::StructField *>( field );                    \
                                                                                            \
        if( !f -> isSet( st ) )                                                            \
        {                                                                                   \
            ccsp_set_error( CCSP_ERROR_KEY_NOT_FOUND, "field not set" );                   \
            return CCSP_ERROR_KEY_NOT_FOUND;                                                \
        }                                                                                   \
                                                                                            \
        if( f -> type() -> type() != csp::CspType::Type::type_enum )                       \
        {                                                                                   \
            ccsp_set_error( CCSP_ERROR_TYPE_MISMATCH, "field type mismatch" );             \
            return CCSP_ERROR_TYPE_MISMATCH;                                                \
        }                                                                                   \
                                                                                            \
        auto * tf = static_cast<const csp::csp_field_type *>( f );                         \
        *out_value = tf -> value( st );                                                    \
        return CCSP_OK;                                                                     \
    }

CCspErrorCode ccsp_struct_get_bool( CCspStructHandle s, CCspStructFieldHandle field, int8_t * out_value )
{
    if( !s || !field || !out_value )
    {
        ccsp_set_error( CCSP_ERROR_NULL_POINTER, "null argument" );
        return CCSP_ERROR_NULL_POINTER;
    }
    
    const csp::Struct * st = getStructConst( s );
    auto * f = reinterpret_cast<const csp::StructField *>( field );
    
    if( !f -> isSet( st ) )
    {
        ccsp_set_error( CCSP_ERROR_KEY_NOT_FOUND, "field not set" );
        return CCSP_ERROR_KEY_NOT_FOUND;
    }
    
    if( f -> type() -> type() != csp::CspType::Type::BOOL )
    {
        ccsp_set_error( CCSP_ERROR_TYPE_MISMATCH, "field type mismatch" );
        return CCSP_ERROR_TYPE_MISMATCH;
    }
    
    auto * tf = static_cast<const csp::BoolStructField *>( f );
    *out_value = tf -> value( st ) ? 1 : 0;
    return CCSP_OK;
}

CCspErrorCode ccsp_struct_get_int8( CCspStructHandle s, CCspStructFieldHandle field, int8_t * out_value )
{
    if( !s || !field || !out_value )
    {
        ccsp_set_error( CCSP_ERROR_NULL_POINTER, "null argument" );
        return CCSP_ERROR_NULL_POINTER;
    }
    
    const csp::Struct * st = getStructConst( s );
    auto * f = reinterpret_cast<const csp::StructField *>( field );
    
    if( !f -> isSet( st ) )
    {
        ccsp_set_error( CCSP_ERROR_KEY_NOT_FOUND, "field not set" );
        return CCSP_ERROR_KEY_NOT_FOUND;
    }
    
    if( f -> type() -> type() != csp::CspType::Type::INT8 )
    {
        ccsp_set_error( CCSP_ERROR_TYPE_MISMATCH, "field type mismatch" );
        return CCSP_ERROR_TYPE_MISMATCH;
    }
    
    auto * tf = static_cast<const csp::Int8StructField *>( f );
    *out_value = tf -> value( st );
    return CCSP_OK;
}

CCspErrorCode ccsp_struct_get_uint8( CCspStructHandle s, CCspStructFieldHandle field, uint8_t * out_value )
{
    if( !s || !field || !out_value )
    {
        ccsp_set_error( CCSP_ERROR_NULL_POINTER, "null argument" );
        return CCSP_ERROR_NULL_POINTER;
    }
    
    const csp::Struct * st = getStructConst( s );
    auto * f = reinterpret_cast<const csp::StructField *>( field );
    
    if( !f -> isSet( st ) )
    {
        ccsp_set_error( CCSP_ERROR_KEY_NOT_FOUND, "field not set" );
        return CCSP_ERROR_KEY_NOT_FOUND;
    }
    
    if( f -> type() -> type() != csp::CspType::Type::UINT8 )
    {
        ccsp_set_error( CCSP_ERROR_TYPE_MISMATCH, "field type mismatch" );
        return CCSP_ERROR_TYPE_MISMATCH;
    }
    
    auto * tf = static_cast<const csp::UInt8StructField *>( f );
    *out_value = tf -> value( st );
    return CCSP_OK;
}

CCspErrorCode ccsp_struct_get_int16( CCspStructHandle s, CCspStructFieldHandle field, int16_t * out_value )
{
    if( !s || !field || !out_value )
    {
        ccsp_set_error( CCSP_ERROR_NULL_POINTER, "null argument" );
        return CCSP_ERROR_NULL_POINTER;
    }
    
    const csp::Struct * st = getStructConst( s );
    auto * f = reinterpret_cast<const csp::StructField *>( field );
    
    if( !f -> isSet( st ) )
    {
        ccsp_set_error( CCSP_ERROR_KEY_NOT_FOUND, "field not set" );
        return CCSP_ERROR_KEY_NOT_FOUND;
    }
    
    if( f -> type() -> type() != csp::CspType::Type::INT16 )
    {
        ccsp_set_error( CCSP_ERROR_TYPE_MISMATCH, "field type mismatch" );
        return CCSP_ERROR_TYPE_MISMATCH;
    }
    
    auto * tf = static_cast<const csp::Int16StructField *>( f );
    *out_value = tf -> value( st );
    return CCSP_OK;
}

CCspErrorCode ccsp_struct_get_uint16( CCspStructHandle s, CCspStructFieldHandle field, uint16_t * out_value )
{
    if( !s || !field || !out_value )
    {
        ccsp_set_error( CCSP_ERROR_NULL_POINTER, "null argument" );
        return CCSP_ERROR_NULL_POINTER;
    }
    
    const csp::Struct * st = getStructConst( s );
    auto * f = reinterpret_cast<const csp::StructField *>( field );
    
    if( !f -> isSet( st ) )
    {
        ccsp_set_error( CCSP_ERROR_KEY_NOT_FOUND, "field not set" );
        return CCSP_ERROR_KEY_NOT_FOUND;
    }
    
    if( f -> type() -> type() != csp::CspType::Type::UINT16 )
    {
        ccsp_set_error( CCSP_ERROR_TYPE_MISMATCH, "field type mismatch" );
        return CCSP_ERROR_TYPE_MISMATCH;
    }
    
    auto * tf = static_cast<const csp::UInt16StructField *>( f );
    *out_value = tf -> value( st );
    return CCSP_OK;
}

CCspErrorCode ccsp_struct_get_int32( CCspStructHandle s, CCspStructFieldHandle field, int32_t * out_value )
{
    if( !s || !field || !out_value )
    {
        ccsp_set_error( CCSP_ERROR_NULL_POINTER, "null argument" );
        return CCSP_ERROR_NULL_POINTER;
    }
    
    const csp::Struct * st = getStructConst( s );
    auto * f = reinterpret_cast<const csp::StructField *>( field );
    
    if( !f -> isSet( st ) )
    {
        ccsp_set_error( CCSP_ERROR_KEY_NOT_FOUND, "field not set" );
        return CCSP_ERROR_KEY_NOT_FOUND;
    }
    
    if( f -> type() -> type() != csp::CspType::Type::INT32 )
    {
        ccsp_set_error( CCSP_ERROR_TYPE_MISMATCH, "field type mismatch" );
        return CCSP_ERROR_TYPE_MISMATCH;
    }
    
    auto * tf = static_cast<const csp::Int32StructField *>( f );
    *out_value = tf -> value( st );
    return CCSP_OK;
}

CCspErrorCode ccsp_struct_get_uint32( CCspStructHandle s, CCspStructFieldHandle field, uint32_t * out_value )
{
    if( !s || !field || !out_value )
    {
        ccsp_set_error( CCSP_ERROR_NULL_POINTER, "null argument" );
        return CCSP_ERROR_NULL_POINTER;
    }
    
    const csp::Struct * st = getStructConst( s );
    auto * f = reinterpret_cast<const csp::StructField *>( field );
    
    if( !f -> isSet( st ) )
    {
        ccsp_set_error( CCSP_ERROR_KEY_NOT_FOUND, "field not set" );
        return CCSP_ERROR_KEY_NOT_FOUND;
    }
    
    if( f -> type() -> type() != csp::CspType::Type::UINT32 )
    {
        ccsp_set_error( CCSP_ERROR_TYPE_MISMATCH, "field type mismatch" );
        return CCSP_ERROR_TYPE_MISMATCH;
    }
    
    auto * tf = static_cast<const csp::UInt32StructField *>( f );
    *out_value = tf -> value( st );
    return CCSP_OK;
}

CCspErrorCode ccsp_struct_get_int64( CCspStructHandle s, CCspStructFieldHandle field, int64_t * out_value )
{
    if( !s || !field || !out_value )
    {
        ccsp_set_error( CCSP_ERROR_NULL_POINTER, "null argument" );
        return CCSP_ERROR_NULL_POINTER;
    }
    
    const csp::Struct * st = getStructConst( s );
    auto * f = reinterpret_cast<const csp::StructField *>( field );
    
    if( !f -> isSet( st ) )
    {
        ccsp_set_error( CCSP_ERROR_KEY_NOT_FOUND, "field not set" );
        return CCSP_ERROR_KEY_NOT_FOUND;
    }
    
    if( f -> type() -> type() != csp::CspType::Type::INT64 )
    {
        ccsp_set_error( CCSP_ERROR_TYPE_MISMATCH, "field type mismatch" );
        return CCSP_ERROR_TYPE_MISMATCH;
    }
    
    auto * tf = static_cast<const csp::Int64StructField *>( f );
    *out_value = tf -> value( st );
    return CCSP_OK;
}

CCspErrorCode ccsp_struct_get_uint64( CCspStructHandle s, CCspStructFieldHandle field, uint64_t * out_value )
{
    if( !s || !field || !out_value )
    {
        ccsp_set_error( CCSP_ERROR_NULL_POINTER, "null argument" );
        return CCSP_ERROR_NULL_POINTER;
    }
    
    const csp::Struct * st = getStructConst( s );
    auto * f = reinterpret_cast<const csp::StructField *>( field );
    
    if( !f -> isSet( st ) )
    {
        ccsp_set_error( CCSP_ERROR_KEY_NOT_FOUND, "field not set" );
        return CCSP_ERROR_KEY_NOT_FOUND;
    }
    
    if( f -> type() -> type() != csp::CspType::Type::UINT64 )
    {
        ccsp_set_error( CCSP_ERROR_TYPE_MISMATCH, "field type mismatch" );
        return CCSP_ERROR_TYPE_MISMATCH;
    }
    
    auto * tf = static_cast<const csp::UInt64StructField *>( f );
    *out_value = tf -> value( st );
    return CCSP_OK;
}

CCspErrorCode ccsp_struct_get_double( CCspStructHandle s, CCspStructFieldHandle field, double * out_value )
{
    if( !s || !field || !out_value )
    {
        ccsp_set_error( CCSP_ERROR_NULL_POINTER, "null argument" );
        return CCSP_ERROR_NULL_POINTER;
    }
    
    const csp::Struct * st = getStructConst( s );
    auto * f = reinterpret_cast<const csp::StructField *>( field );
    
    if( !f -> isSet( st ) )
    {
        ccsp_set_error( CCSP_ERROR_KEY_NOT_FOUND, "field not set" );
        return CCSP_ERROR_KEY_NOT_FOUND;
    }
    
    if( f -> type() -> type() != csp::CspType::Type::DOUBLE )
    {
        ccsp_set_error( CCSP_ERROR_TYPE_MISMATCH, "field type mismatch" );
        return CCSP_ERROR_TYPE_MISMATCH;
    }
    
    auto * tf = static_cast<const csp::DoubleStructField *>( f );
    *out_value = tf -> value( st );
    return CCSP_OK;
}

CCspErrorCode ccsp_struct_get_datetime( CCspStructHandle s, CCspStructFieldHandle field, CCspDateTime * out_value )
{
    if( !s || !field || !out_value )
    {
        ccsp_set_error( CCSP_ERROR_NULL_POINTER, "null argument" );
        return CCSP_ERROR_NULL_POINTER;
    }
    
    const csp::Struct * st = getStructConst( s );
    auto * f = reinterpret_cast<const csp::StructField *>( field );
    
    if( !f -> isSet( st ) )
    {
        ccsp_set_error( CCSP_ERROR_KEY_NOT_FOUND, "field not set" );
        return CCSP_ERROR_KEY_NOT_FOUND;
    }
    
    if( f -> type() -> type() != csp::CspType::Type::DATETIME )
    {
        ccsp_set_error( CCSP_ERROR_TYPE_MISMATCH, "field type mismatch" );
        return CCSP_ERROR_TYPE_MISMATCH;
    }
    
    auto * tf = static_cast<const csp::DateTimeStructField *>( f );
    *out_value = tf -> value( st ).asNanoseconds();
    return CCSP_OK;
}

CCspErrorCode ccsp_struct_get_timedelta( CCspStructHandle s, CCspStructFieldHandle field, CCspTimeDelta * out_value )
{
    if( !s || !field || !out_value )
    {
        ccsp_set_error( CCSP_ERROR_NULL_POINTER, "null argument" );
        return CCSP_ERROR_NULL_POINTER;
    }
    
    const csp::Struct * st = getStructConst( s );
    auto * f = reinterpret_cast<const csp::StructField *>( field );
    
    if( !f -> isSet( st ) )
    {
        ccsp_set_error( CCSP_ERROR_KEY_NOT_FOUND, "field not set" );
        return CCSP_ERROR_KEY_NOT_FOUND;
    }
    
    if( f -> type() -> type() != csp::CspType::Type::TIMEDELTA )
    {
        ccsp_set_error( CCSP_ERROR_TYPE_MISMATCH, "field type mismatch" );
        return CCSP_ERROR_TYPE_MISMATCH;
    }
    
    auto * tf = static_cast<const csp::TimeDeltaStructField *>( f );
    *out_value = tf -> value( st ).asNanoseconds();
    return CCSP_OK;
}

CCspErrorCode ccsp_struct_get_string( CCspStructHandle s, CCspStructFieldHandle field, const char ** out_data, size_t * out_length )
{
    if( !s || !field || !out_data || !out_length )
    {
        ccsp_set_error( CCSP_ERROR_NULL_POINTER, "null argument" );
        return CCSP_ERROR_NULL_POINTER;
    }
    
    const csp::Struct * st = getStructConst( s );
    auto * f = reinterpret_cast<const csp::StructField *>( field );
    
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

CCspErrorCode ccsp_struct_get_enum( CCspStructHandle s, CCspStructFieldHandle field, int32_t * out_ordinal )
{
    if( !s || !field || !out_ordinal )
    {
        ccsp_set_error( CCSP_ERROR_NULL_POINTER, "null argument" );
        return CCSP_ERROR_NULL_POINTER;
    }
    
    const csp::Struct * st = getStructConst( s );
    auto * f = reinterpret_cast<const csp::StructField *>( field );
    
    if( !f -> isSet( st ) )
    {
        ccsp_set_error( CCSP_ERROR_KEY_NOT_FOUND, "field not set" );
        return CCSP_ERROR_KEY_NOT_FOUND;
    }
    
    if( f -> type() -> type() != csp::CspType::Type::ENUM )
    {
        ccsp_set_error( CCSP_ERROR_TYPE_MISMATCH, "field type mismatch" );
        return CCSP_ERROR_TYPE_MISMATCH;
    }
    
    auto * tf = static_cast<const csp::CspEnumStructField *>( f );
    *out_ordinal = static_cast<int32_t>( tf -> value( st ).value() );
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
    auto * f = reinterpret_cast<const csp::StructField *>( field );
    
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
    
    // StructStructField stores StructPtr, need to get raw pointer
    // This is a bit tricky - we need to access the nested struct
    ccsp_set_error( CCSP_ERROR_NOT_IMPLEMENTED, "nested struct access not implemented yet" );
    return CCSP_ERROR_NOT_IMPLEMENTED;
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

CCspErrorCode ccsp_struct_set_bool( CCspStructHandle s, CCspStructFieldHandle field, int8_t value )
{
    if( !s || !field )
    {
        ccsp_set_error( CCSP_ERROR_NULL_POINTER, "null argument" );
        return CCSP_ERROR_NULL_POINTER;
    }
    
    csp::Struct * st = getStruct( s );
    auto * f = reinterpret_cast<const csp::StructField *>( field );
    
    if( f -> type() -> type() != csp::CspType::Type::BOOL )
    {
        ccsp_set_error( CCSP_ERROR_TYPE_MISMATCH, "field type mismatch" );
        return CCSP_ERROR_TYPE_MISMATCH;
    }
    
    auto * tf = static_cast<const csp::BoolStructField *>( f );
    tf -> setValue( st, value != 0 );
    return CCSP_OK;
}

CCspErrorCode ccsp_struct_set_int8( CCspStructHandle s, CCspStructFieldHandle field, int8_t value )
{
    if( !s || !field )
    {
        ccsp_set_error( CCSP_ERROR_NULL_POINTER, "null argument" );
        return CCSP_ERROR_NULL_POINTER;
    }
    
    csp::Struct * st = getStruct( s );
    auto * f = reinterpret_cast<const csp::StructField *>( field );
    
    if( f -> type() -> type() != csp::CspType::Type::INT8 )
    {
        ccsp_set_error( CCSP_ERROR_TYPE_MISMATCH, "field type mismatch" );
        return CCSP_ERROR_TYPE_MISMATCH;
    }
    
    auto * tf = static_cast<const csp::Int8StructField *>( f );
    tf -> setValue( st, value );
    return CCSP_OK;
}

CCspErrorCode ccsp_struct_set_uint8( CCspStructHandle s, CCspStructFieldHandle field, uint8_t value )
{
    if( !s || !field )
    {
        ccsp_set_error( CCSP_ERROR_NULL_POINTER, "null argument" );
        return CCSP_ERROR_NULL_POINTER;
    }
    
    csp::Struct * st = getStruct( s );
    auto * f = reinterpret_cast<const csp::StructField *>( field );
    
    if( f -> type() -> type() != csp::CspType::Type::UINT8 )
    {
        ccsp_set_error( CCSP_ERROR_TYPE_MISMATCH, "field type mismatch" );
        return CCSP_ERROR_TYPE_MISMATCH;
    }
    
    auto * tf = static_cast<const csp::UInt8StructField *>( f );
    tf -> setValue( st, value );
    return CCSP_OK;
}

CCspErrorCode ccsp_struct_set_int16( CCspStructHandle s, CCspStructFieldHandle field, int16_t value )
{
    if( !s || !field )
    {
        ccsp_set_error( CCSP_ERROR_NULL_POINTER, "null argument" );
        return CCSP_ERROR_NULL_POINTER;
    }
    
    csp::Struct * st = getStruct( s );
    auto * f = reinterpret_cast<const csp::StructField *>( field );
    
    if( f -> type() -> type() != csp::CspType::Type::INT16 )
    {
        ccsp_set_error( CCSP_ERROR_TYPE_MISMATCH, "field type mismatch" );
        return CCSP_ERROR_TYPE_MISMATCH;
    }
    
    auto * tf = static_cast<const csp::Int16StructField *>( f );
    tf -> setValue( st, value );
    return CCSP_OK;
}

CCspErrorCode ccsp_struct_set_uint16( CCspStructHandle s, CCspStructFieldHandle field, uint16_t value )
{
    if( !s || !field )
    {
        ccsp_set_error( CCSP_ERROR_NULL_POINTER, "null argument" );
        return CCSP_ERROR_NULL_POINTER;
    }
    
    csp::Struct * st = getStruct( s );
    auto * f = reinterpret_cast<const csp::StructField *>( field );
    
    if( f -> type() -> type() != csp::CspType::Type::UINT16 )
    {
        ccsp_set_error( CCSP_ERROR_TYPE_MISMATCH, "field type mismatch" );
        return CCSP_ERROR_TYPE_MISMATCH;
    }
    
    auto * tf = static_cast<const csp::UInt16StructField *>( f );
    tf -> setValue( st, value );
    return CCSP_OK;
}

CCspErrorCode ccsp_struct_set_int32( CCspStructHandle s, CCspStructFieldHandle field, int32_t value )
{
    if( !s || !field )
    {
        ccsp_set_error( CCSP_ERROR_NULL_POINTER, "null argument" );
        return CCSP_ERROR_NULL_POINTER;
    }
    
    csp::Struct * st = getStruct( s );
    auto * f = reinterpret_cast<const csp::StructField *>( field );
    
    if( f -> type() -> type() != csp::CspType::Type::INT32 )
    {
        ccsp_set_error( CCSP_ERROR_TYPE_MISMATCH, "field type mismatch" );
        return CCSP_ERROR_TYPE_MISMATCH;
    }
    
    auto * tf = static_cast<const csp::Int32StructField *>( f );
    tf -> setValue( st, value );
    return CCSP_OK;
}

CCspErrorCode ccsp_struct_set_uint32( CCspStructHandle s, CCspStructFieldHandle field, uint32_t value )
{
    if( !s || !field )
    {
        ccsp_set_error( CCSP_ERROR_NULL_POINTER, "null argument" );
        return CCSP_ERROR_NULL_POINTER;
    }
    
    csp::Struct * st = getStruct( s );
    auto * f = reinterpret_cast<const csp::StructField *>( field );
    
    if( f -> type() -> type() != csp::CspType::Type::UINT32 )
    {
        ccsp_set_error( CCSP_ERROR_TYPE_MISMATCH, "field type mismatch" );
        return CCSP_ERROR_TYPE_MISMATCH;
    }
    
    auto * tf = static_cast<const csp::UInt32StructField *>( f );
    tf -> setValue( st, value );
    return CCSP_OK;
}

CCspErrorCode ccsp_struct_set_int64( CCspStructHandle s, CCspStructFieldHandle field, int64_t value )
{
    if( !s || !field )
    {
        ccsp_set_error( CCSP_ERROR_NULL_POINTER, "null argument" );
        return CCSP_ERROR_NULL_POINTER;
    }
    
    csp::Struct * st = getStruct( s );
    auto * f = reinterpret_cast<const csp::StructField *>( field );
    
    if( f -> type() -> type() != csp::CspType::Type::INT64 )
    {
        ccsp_set_error( CCSP_ERROR_TYPE_MISMATCH, "field type mismatch" );
        return CCSP_ERROR_TYPE_MISMATCH;
    }
    
    auto * tf = static_cast<const csp::Int64StructField *>( f );
    tf -> setValue( st, value );
    return CCSP_OK;
}

CCspErrorCode ccsp_struct_set_uint64( CCspStructHandle s, CCspStructFieldHandle field, uint64_t value )
{
    if( !s || !field )
    {
        ccsp_set_error( CCSP_ERROR_NULL_POINTER, "null argument" );
        return CCSP_ERROR_NULL_POINTER;
    }
    
    csp::Struct * st = getStruct( s );
    auto * f = reinterpret_cast<const csp::StructField *>( field );
    
    if( f -> type() -> type() != csp::CspType::Type::UINT64 )
    {
        ccsp_set_error( CCSP_ERROR_TYPE_MISMATCH, "field type mismatch" );
        return CCSP_ERROR_TYPE_MISMATCH;
    }
    
    auto * tf = static_cast<const csp::UInt64StructField *>( f );
    tf -> setValue( st, value );
    return CCSP_OK;
}

CCspErrorCode ccsp_struct_set_double( CCspStructHandle s, CCspStructFieldHandle field, double value )
{
    if( !s || !field )
    {
        ccsp_set_error( CCSP_ERROR_NULL_POINTER, "null argument" );
        return CCSP_ERROR_NULL_POINTER;
    }
    
    csp::Struct * st = getStruct( s );
    auto * f = reinterpret_cast<const csp::StructField *>( field );
    
    if( f -> type() -> type() != csp::CspType::Type::DOUBLE )
    {
        ccsp_set_error( CCSP_ERROR_TYPE_MISMATCH, "field type mismatch" );
        return CCSP_ERROR_TYPE_MISMATCH;
    }
    
    auto * tf = static_cast<const csp::DoubleStructField *>( f );
    tf -> setValue( st, value );
    return CCSP_OK;
}

CCspErrorCode ccsp_struct_set_datetime( CCspStructHandle s, CCspStructFieldHandle field, CCspDateTime value )
{
    if( !s || !field )
    {
        ccsp_set_error( CCSP_ERROR_NULL_POINTER, "null argument" );
        return CCSP_ERROR_NULL_POINTER;
    }
    
    csp::Struct * st = getStruct( s );
    auto * f = reinterpret_cast<const csp::StructField *>( field );
    
    if( f -> type() -> type() != csp::CspType::Type::DATETIME )
    {
        ccsp_set_error( CCSP_ERROR_TYPE_MISMATCH, "field type mismatch" );
        return CCSP_ERROR_TYPE_MISMATCH;
    }
    
    auto * tf = static_cast<const csp::DateTimeStructField *>( f );
    tf -> setValue( st, csp::DateTime::fromNanoseconds( value ) );
    return CCSP_OK;
}

CCspErrorCode ccsp_struct_set_timedelta( CCspStructHandle s, CCspStructFieldHandle field, CCspTimeDelta value )
{
    if( !s || !field )
    {
        ccsp_set_error( CCSP_ERROR_NULL_POINTER, "null argument" );
        return CCSP_ERROR_NULL_POINTER;
    }
    
    csp::Struct * st = getStruct( s );
    auto * f = reinterpret_cast<const csp::StructField *>( field );
    
    if( f -> type() -> type() != csp::CspType::Type::TIMEDELTA )
    {
        ccsp_set_error( CCSP_ERROR_TYPE_MISMATCH, "field type mismatch" );
        return CCSP_ERROR_TYPE_MISMATCH;
    }
    
    auto * tf = static_cast<const csp::TimeDeltaStructField *>( f );
    tf -> setValue( st, csp::TimeDelta::fromNanoseconds( value ) );
    return CCSP_OK;
}

CCspErrorCode ccsp_struct_set_string( CCspStructHandle s, CCspStructFieldHandle field,
                                      const char * data, size_t length )
{
    if( !s || !field )
    {
        ccsp_set_error( CCSP_ERROR_NULL_POINTER, "null argument" );
        return CCSP_ERROR_NULL_POINTER;
    }
    
    csp::Struct * st = getStruct( s );
    auto * f = reinterpret_cast<const csp::StructField *>( field );
    
    if( f -> type() -> type() != csp::CspType::Type::STRING )
    {
        ccsp_set_error( CCSP_ERROR_TYPE_MISMATCH, "field type mismatch" );
        return CCSP_ERROR_TYPE_MISMATCH;
    }
    
    auto * tf = static_cast<const csp::StringStructField *>( f );
    tf -> setValue( st, std::string( data, length ) );
    return CCSP_OK;
}

CCspErrorCode ccsp_struct_set_enum( CCspStructHandle s, CCspStructFieldHandle field, int32_t ordinal )
{
    if( !s || !field )
    {
        ccsp_set_error( CCSP_ERROR_NULL_POINTER, "null argument" );
        return CCSP_ERROR_NULL_POINTER;
    }
    
    auto * f = reinterpret_cast<const csp::StructField *>( field );
    
    if( f -> type() -> type() != csp::CspType::Type::ENUM )
    {
        ccsp_set_error( CCSP_ERROR_TYPE_MISMATCH, "field type mismatch" );
        return CCSP_ERROR_TYPE_MISMATCH;
    }
    
    // Setting enum requires creating a CspEnum with the correct meta
    // This is more complex - for now mark as not implemented
    ccsp_set_error( CCSP_ERROR_NOT_IMPLEMENTED, "enum set not implemented" );
    return CCSP_ERROR_NOT_IMPLEMENTED;
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
