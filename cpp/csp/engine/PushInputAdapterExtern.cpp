/*
 * Implementation of the C++ PushInputAdapterExtern wrapper and C API functions.
 */
#include <csp/engine/PushInputAdapterExtern.h>
#include <csp/engine/CspType.h>
#include <csp/engine/Engine.h>
#include <csp/engine/ExternBoundary.h>
#include <csp/engine/c/InputAdapter.h>
#include <csp/engine/c/CspError.h>
#include <csp/core/Exception.h>
#include <cstring>

namespace csp
{

// ============================================================================
// PushInputAdapterExtern Implementation
// ============================================================================

PushInputAdapterExtern::PushInputAdapterExtern( Engine * engine, CspTypePtr & type,
                                                PushMode pushMode, PushGroup * group,
                                                const CCspPushInputAdapterVTable & vtable )
    : PushInputAdapter( engine, type, pushMode, group )
    , m_vtable()
    , m_startTime( DateTime::NONE() )
    , m_endTime( DateTime::NONE() )
{
    if( !ccsp_vtable_adopt( &m_vtable, sizeof( m_vtable ), &vtable ) )
    {
        CSP_THROW( ValueError, "PushInputAdapterExtern: vtable was not initialized with CCSP_VTABLE_INIT, "
                               "or was built against a newer csp ABI than this build supports" );
    }

    if( !m_vtable.destroy )
    {
        CSP_THROW( ValueError, "PushInputAdapterExtern: destroy callback is required" );
    }
}

PushInputAdapterExtern::~PushInputAdapterExtern()
{
    if( m_vtable.destroy )
    {
        // Destructors are implicitly noexcept, so a throwing callback would terminate the process
        try { m_vtable.destroy( m_vtable.user_data ); } catch( ... ) {}
    }
}

void PushInputAdapterExtern::start( DateTime startTime, DateTime endTime )
{
    m_startTime = startTime;
    m_endTime = endTime;

    if( m_vtable.start )
    {
        CCspEngineHandle engineHandle = reinterpret_cast<CCspEngineHandle>( rootEngine() );
        CCspPushInputAdapterHandle adapterHandle = reinterpret_cast<CCspPushInputAdapterHandle>( this );

        try
        {
            m_vtable.start( m_vtable.user_data, engineHandle, adapterHandle,
                            startTime.asNanoseconds(), endTime.asNanoseconds() );
        }
        CSP_CATCH_EXTERN_CALLBACK( "push input adapter start" )
    }
}

void PushInputAdapterExtern::stop()
{
    if( m_vtable.stop )
    {
        try
        {
            m_vtable.stop( m_vtable.user_data );
        }
        CSP_CATCH_EXTERN_CALLBACK( "push input adapter stop" )
    }
}

} // namespace csp

namespace
{

// PushInputAdapter::consumeEvent re-derives the payload type from the adapter's declared type and
// static_casts the event to it, so pushing a mismatched type reinterprets the value's bytes.
template<typename T>
bool checkPushType( const csp::PushInputAdapter * adapter )
{
    constexpr csp::CspType::TypeTraits::_enum expected = csp::CspType::TypeTraits::fromCType<T>::type;
    if( adapter -> dataType() -> type() != expected )
    {
        ccsp_set_error( CCSP_ERROR_TYPE_MISMATCH, "pushed value type does not match the adapter's declared type" );
        return false;
    }
    return true;
}

} // namespace

// ============================================================================
// C API Implementation
// ============================================================================

extern "C" {

// ============================================================================
// Push Input Adapter Creation
// ============================================================================

CCspPushInputAdapterHandle ccsp_push_input_adapter_extern_create( CCspEngineHandle engine, CCspType type,
                                                                  CCspPushMode push_mode, CCspPushGroupHandle group,
                                                                  const CCspPushInputAdapterVTable* vtable )
{
    if( !engine || !vtable )
    {
        ccsp_set_error( CCSP_ERROR_NULL_POINTER, "null engine or vtable" );
        return nullptr;
    }

    try
    {
        auto * eng = reinterpret_cast<csp::Engine *>( engine );
        auto * grp = reinterpret_cast<csp::PushGroup *>( group );

        // Convert CCspType to CspTypePtr
        csp::CspTypePtr cspType;
        switch( type )
        {
            case CCSP_TYPE_BOOL:      cspType = csp::CspType::BOOL();      break;
            case CCSP_TYPE_INT8:      cspType = csp::CspType::INT8();      break;
            case CCSP_TYPE_UINT8:     cspType = csp::CspType::UINT8();     break;
            case CCSP_TYPE_INT16:     cspType = csp::CspType::INT16();     break;
            case CCSP_TYPE_UINT16:    cspType = csp::CspType::UINT16();    break;
            case CCSP_TYPE_INT32:     cspType = csp::CspType::INT32();     break;
            case CCSP_TYPE_UINT32:    cspType = csp::CspType::UINT32();    break;
            case CCSP_TYPE_INT64:     cspType = csp::CspType::INT64();     break;
            case CCSP_TYPE_UINT64:    cspType = csp::CspType::UINT64();    break;
            case CCSP_TYPE_DOUBLE:    cspType = csp::CspType::DOUBLE();    break;
            case CCSP_TYPE_DATETIME:  cspType = csp::CspType::DATETIME();  break;
            case CCSP_TYPE_TIMEDELTA: cspType = csp::CspType::TIMEDELTA(); break;
            case CCSP_TYPE_STRING:    cspType = csp::CspType::STRING();    break;
            default:
                ccsp_set_error( CCSP_ERROR_INVALID_ARGUMENT, "unsupported type for push input adapter" );
                return nullptr;
        }

        // Convert push mode
        csp::PushMode cspPushMode;
        switch( push_mode )
        {
            case CCSP_PUSH_MODE_LAST_VALUE:      cspPushMode = csp::PushMode::LAST_VALUE;     break;
            case CCSP_PUSH_MODE_NON_COLLAPSING:  cspPushMode = csp::PushMode::NON_COLLAPSING; break;
            case CCSP_PUSH_MODE_BURST:           cspPushMode = csp::PushMode::BURST;          break;
            default:
                ccsp_set_error( CCSP_ERROR_INVALID_ARGUMENT, "invalid push mode" );
                return nullptr;
        }

        auto * adapter = eng -> createOwnedObject<csp::PushInputAdapterExtern>( cspType, cspPushMode, grp, *vtable );

        return reinterpret_cast<CCspPushInputAdapterHandle>( adapter );
    }
    CCSP_CATCH_RET( nullptr )
}

void ccsp_push_input_adapter_extern_destroy( CCspPushInputAdapterHandle adapter )
{
    // The adapter is owned by the engine, destruction is handled there
    ( void ) adapter;
}

// ============================================================================
// Type-specific push functions
// ============================================================================

#define IMPLEMENT_PUSH( name, c_type, cpp_type, value_expr )                                      \
    CCspErrorCode ccsp_push_input_adapter_push_##name( CCspPushInputAdapterHandle adapter,        \
                                                       c_type value, CCspPushBatchHandle batch )  \
    {                                                                                             \
        if( !adapter )                                                                            \
        {                                                                                         \
            ccsp_set_error( CCSP_ERROR_NULL_POINTER, "null adapter" );                            \
            return CCSP_ERROR_NULL_POINTER;                                                       \
        }                                                                                         \
        try                                                                                       \
        {                                                                                         \
            auto * pushAdapter = reinterpret_cast<csp::PushInputAdapter *>( adapter );            \
            if( !checkPushType<cpp_type>( pushAdapter ) )                                         \
                return CCSP_ERROR_TYPE_MISMATCH;                                                  \
            auto * pushBatch = reinterpret_cast<csp::PushBatch *>( batch );                       \
            cpp_type val = value_expr;                                                            \
            pushAdapter -> pushTick( std::move( val ), pushBatch );                               \
            return CCSP_OK;                                                                       \
        }                                                                                         \
        CCSP_CATCH_ERR                                                                            \
    }

IMPLEMENT_PUSH( bool,      int8_t,        bool,           static_cast<bool>( value ) )
IMPLEMENT_PUSH( int8,      int8_t,        int8_t,         value )
IMPLEMENT_PUSH( uint8,     uint8_t,       uint8_t,        value )
IMPLEMENT_PUSH( int16,     int16_t,       int16_t,        value )
IMPLEMENT_PUSH( uint16,    uint16_t,      uint16_t,       value )
IMPLEMENT_PUSH( int32,     int32_t,       int32_t,        value )
IMPLEMENT_PUSH( uint32,    uint32_t,      uint32_t,       value )
IMPLEMENT_PUSH( int64,     int64_t,       int64_t,        value )
IMPLEMENT_PUSH( uint64,    uint64_t,      uint64_t,       value )
IMPLEMENT_PUSH( double,    double,        double,         value )
IMPLEMENT_PUSH( datetime,  CCspDateTime,  csp::DateTime,  csp::DateTime::fromNanoseconds( value ) )
IMPLEMENT_PUSH( timedelta, CCspTimeDelta, csp::TimeDelta, csp::TimeDelta::fromNanoseconds( value ) )

CCspErrorCode ccsp_push_input_adapter_push_string( CCspPushInputAdapterHandle adapter, const char* data, size_t length, CCspPushBatchHandle batch )
{
    if( !adapter )
    {
        ccsp_set_error( CCSP_ERROR_NULL_POINTER, "null adapter" );
        return CCSP_ERROR_NULL_POINTER;
    }

    if( !data && length > 0 )
    {
        ccsp_set_error( CCSP_ERROR_INVALID_ARGUMENT, "null data with non-zero length" );
        return CCSP_ERROR_INVALID_ARGUMENT;
    }

    try
    {
        auto * pushAdapter = reinterpret_cast<csp::PushInputAdapter *>( adapter );
        if( !checkPushType<std::string>( pushAdapter ) )
            return CCSP_ERROR_TYPE_MISMATCH;
        auto * pushBatch = reinterpret_cast<csp::PushBatch *>( batch );
        std::string str( data ? data : "", length );
        pushAdapter -> pushTick( std::move( str ), pushBatch );
        return CCSP_OK;
    }
    CCSP_CATCH_ERR
}

CCspErrorCode ccsp_push_input_adapter_push_struct( CCspPushInputAdapterHandle adapter, CCspStructHandle value, CCspPushBatchHandle batch )
{
    if( !adapter || !value )
    {
        ccsp_set_error( CCSP_ERROR_NULL_POINTER, "null adapter or struct" );
        return CCSP_ERROR_NULL_POINTER;
    }

    try
    {
        auto * pushAdapter = reinterpret_cast<csp::PushInputAdapter *>( adapter );
        if( !checkPushType<csp::StructPtr>( pushAdapter ) )
            return CCSP_ERROR_TYPE_MISMATCH;

        auto * pushBatch = reinterpret_cast<csp::PushBatch *>( batch );

        /* The handle is a StructPtr owned by the caller, so push a copy of the reference */
        csp::StructPtr sp = *reinterpret_cast<csp::StructPtr *>( value );
        if( !sp.get() )
        {
            ccsp_set_error( CCSP_ERROR_INVALID_ARGUMENT, "struct handle holds no struct" );
            return CCSP_ERROR_INVALID_ARGUMENT;
        }

        pushAdapter -> pushTick( std::move( sp ), pushBatch );
        return CCSP_OK;
    }
    CCSP_CATCH_ERR
}

CCspErrorCode ccsp_push_input_adapter_push_value(
    CCspPushInputAdapterHandle adapter,
    const CCspValue* value,
    CCspPushBatchHandle batch )
{
    if( !adapter || !value )
    {
        ccsp_set_error( CCSP_ERROR_NULL_POINTER, "null adapter or value" );
        return CCSP_ERROR_NULL_POINTER;
    }

    switch( value -> type )
    {
        case CCSP_TYPE_BOOL:      return ccsp_push_input_adapter_push_bool( adapter, value -> bool_val, batch );
        case CCSP_TYPE_INT8:      return ccsp_push_input_adapter_push_int8( adapter, value -> int8_val, batch );
        case CCSP_TYPE_UINT8:     return ccsp_push_input_adapter_push_uint8( adapter, value -> uint8_val, batch );
        case CCSP_TYPE_INT16:     return ccsp_push_input_adapter_push_int16( adapter, value -> int16_val, batch );
        case CCSP_TYPE_UINT16:    return ccsp_push_input_adapter_push_uint16( adapter, value -> uint16_val, batch );
        case CCSP_TYPE_INT32:     return ccsp_push_input_adapter_push_int32( adapter, value -> int32_val, batch );
        case CCSP_TYPE_UINT32:    return ccsp_push_input_adapter_push_uint32( adapter, value -> uint32_val, batch );
        case CCSP_TYPE_INT64:     return ccsp_push_input_adapter_push_int64( adapter, value -> int64_val, batch );
        case CCSP_TYPE_UINT64:    return ccsp_push_input_adapter_push_uint64( adapter, value -> uint64_val, batch );
        case CCSP_TYPE_DOUBLE:    return ccsp_push_input_adapter_push_double( adapter, value -> double_val, batch );
        case CCSP_TYPE_DATETIME:  return ccsp_push_input_adapter_push_datetime( adapter, value -> datetime_val, batch );
        case CCSP_TYPE_TIMEDELTA: return ccsp_push_input_adapter_push_timedelta( adapter, value -> timedelta_val, batch );
        case CCSP_TYPE_STRING:    return ccsp_push_input_adapter_push_string( adapter, value -> string_val.data,
                                                                             value -> string_val.length, batch );
        case CCSP_TYPE_STRUCT:    return ccsp_push_input_adapter_push_struct( adapter, value -> struct_val, batch );
        default:
            ccsp_set_error( CCSP_ERROR_NOT_IMPLEMENTED, "value type cannot be pushed" );
            return CCSP_ERROR_NOT_IMPLEMENTED;
    }
}

// ============================================================================
// Push Batch Management
// ============================================================================

CCspPushBatchHandle ccsp_push_batch_create( CCspEngineHandle engine )
{
    if( !engine )
    {
        ccsp_set_error( CCSP_ERROR_NULL_POINTER, "null engine" );
        return nullptr;
    }

    try
    {
        auto * eng = reinterpret_cast<csp::Engine *>( engine );
        auto * batch = new csp::PushBatch( eng -> rootEngine() );
        return reinterpret_cast<CCspPushBatchHandle>( batch );
    }
    CCSP_CATCH_RET( nullptr )
}

void ccsp_push_batch_flush( CCspPushBatchHandle batch )
{
    if( !batch ) return;

    try
    {
        auto * pushBatch = reinterpret_cast<csp::PushBatch *>( batch );
        pushBatch -> flush();
    }
    catch( ... )
    {
        // Ignore errors during flush
    }
}

void ccsp_push_batch_destroy( CCspPushBatchHandle batch )
{
    if( !batch ) return;

    try
    {
        auto * pushBatch = reinterpret_cast<csp::PushBatch *>( batch );
        delete pushBatch;
    }
    catch( ... )
    {
        // Ignore errors during destroy
    }
}

// ============================================================================
// Push Group Management
// ============================================================================

CCspPushGroupHandle ccsp_push_group_create( void )
{
    try
    {
        auto * group = new csp::PushGroup();
        return reinterpret_cast<CCspPushGroupHandle>( group );
    }
    CCSP_CATCH_RET( nullptr )
}

void ccsp_push_group_destroy( CCspPushGroupHandle group )
{
    if( !group ) return;

    try
    {
        auto * pushGroup = reinterpret_cast<csp::PushGroup *>( group );
        delete pushGroup;
    }
    catch( ... )
    {
        // Ignore errors during destroy
    }
}

} // extern "C"
