/*
 * Implementation of the C++ PushInputAdapterExtern wrapper and C API functions.
 */
#include <csp/engine/PushInputAdapterExtern.h>
#include <csp/engine/Engine.h>
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
    , m_vtable( vtable )
    , m_startTime( DateTime::NONE() )
    , m_endTime( DateTime::NONE() )
{
    if( !vtable.destroy )
    {
        CSP_THROW( ValueError, "PushInputAdapterExtern: destroy callback is required" );
    }
}

PushInputAdapterExtern::~PushInputAdapterExtern()
{
    if( m_vtable.destroy )
    {
        m_vtable.destroy( m_vtable.user_data );
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

        m_vtable.start( m_vtable.user_data, engineHandle, adapterHandle,
                        startTime.asNanoseconds(), endTime.asNanoseconds() );
    }
}

void PushInputAdapterExtern::stop()
{
    if( m_vtable.stop )
    {
        m_vtable.stop( m_vtable.user_data );
    }
}

} // namespace csp

// ============================================================================
// C API Implementation
// ============================================================================

extern "C" {

// Forward declaration of error functions from OutputAdapterExtern.cpp
extern void ccsp_set_error( CCspErrorCode code, const char * message );

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
    catch( const std::exception & e )
    {
        ccsp_set_error( CCSP_ERROR_RUNTIME, e.what() );
        return nullptr;
    }
}

void ccsp_push_input_adapter_extern_destroy( CCspPushInputAdapterHandle adapter )
{
    // The adapter is owned by the engine, destruction is handled there
    ( void ) adapter;
}

// ============================================================================
// Type-specific push functions
// ============================================================================

CCspErrorCode ccsp_push_input_adapter_push_bool( CCspPushInputAdapterHandle adapter, int8_t value, CCspPushBatchHandle batch )
{
    if( !adapter )
    {
        ccsp_set_error( CCSP_ERROR_NULL_POINTER, "null adapter" );
        return CCSP_ERROR_NULL_POINTER;
    }

    try
    {
        auto * pushAdapter = reinterpret_cast<csp::PushInputAdapter *>( adapter );
        auto * pushBatch = reinterpret_cast<csp::PushBatch *>( batch );
        bool bvalue = static_cast<bool>( value );
        pushAdapter -> pushTick( std::move( bvalue ), pushBatch );
        return CCSP_OK;
    }
    catch( const std::exception & e )
    {
        ccsp_set_error( CCSP_ERROR_RUNTIME, e.what() );
        return CCSP_ERROR_RUNTIME;
    }
}

CCspErrorCode ccsp_push_input_adapter_push_int8( CCspPushInputAdapterHandle adapter, int8_t value, CCspPushBatchHandle batch )
{
    if( !adapter )
    {
        ccsp_set_error( CCSP_ERROR_NULL_POINTER, "null adapter" );
        return CCSP_ERROR_NULL_POINTER;
    }

    try
    {
        auto * pushAdapter = reinterpret_cast<csp::PushInputAdapter *>( adapter );
        auto * pushBatch = reinterpret_cast<csp::PushBatch *>( batch );
        int8_t val = value;
        pushAdapter -> pushTick( std::move( val ), pushBatch );
        return CCSP_OK;
    }
    catch( const std::exception & e )
    {
        ccsp_set_error( CCSP_ERROR_RUNTIME, e.what() );
        return CCSP_ERROR_RUNTIME;
    }
}

CCspErrorCode ccsp_push_input_adapter_push_uint8( CCspPushInputAdapterHandle adapter, uint8_t value, CCspPushBatchHandle batch )
{
    if( !adapter )
    {
        ccsp_set_error( CCSP_ERROR_NULL_POINTER, "null adapter" );
        return CCSP_ERROR_NULL_POINTER;
    }

    try
    {
        auto * pushAdapter = reinterpret_cast<csp::PushInputAdapter *>( adapter );
        auto * pushBatch = reinterpret_cast<csp::PushBatch *>( batch );
        uint8_t val = value;
        pushAdapter -> pushTick( std::move( val ), pushBatch );
        return CCSP_OK;
    }
    catch( const std::exception & e )
    {
        ccsp_set_error( CCSP_ERROR_RUNTIME, e.what() );
        return CCSP_ERROR_RUNTIME;
    }
}

CCspErrorCode ccsp_push_input_adapter_push_int16( CCspPushInputAdapterHandle adapter, int16_t value, CCspPushBatchHandle batch )
{
    if( !adapter )
    {
        ccsp_set_error( CCSP_ERROR_NULL_POINTER, "null adapter" );
        return CCSP_ERROR_NULL_POINTER;
    }

    try
    {
        auto * pushAdapter = reinterpret_cast<csp::PushInputAdapter *>( adapter );
        auto * pushBatch = reinterpret_cast<csp::PushBatch *>( batch );
        int16_t val = value;
        pushAdapter -> pushTick( std::move( val ), pushBatch );
        return CCSP_OK;
    }
    catch( const std::exception & e )
    {
        ccsp_set_error( CCSP_ERROR_RUNTIME, e.what() );
        return CCSP_ERROR_RUNTIME;
    }
}

CCspErrorCode ccsp_push_input_adapter_push_uint16( CCspPushInputAdapterHandle adapter, uint16_t value, CCspPushBatchHandle batch )
{
    if( !adapter )
    {
        ccsp_set_error( CCSP_ERROR_NULL_POINTER, "null adapter" );
        return CCSP_ERROR_NULL_POINTER;
    }

    try
    {
        auto * pushAdapter = reinterpret_cast<csp::PushInputAdapter *>( adapter );
        auto * pushBatch = reinterpret_cast<csp::PushBatch *>( batch );
        uint16_t val = value;
        pushAdapter -> pushTick( std::move( val ), pushBatch );
        return CCSP_OK;
    }
    catch( const std::exception & e )
    {
        ccsp_set_error( CCSP_ERROR_RUNTIME, e.what() );
        return CCSP_ERROR_RUNTIME;
    }
}

CCspErrorCode ccsp_push_input_adapter_push_int32( CCspPushInputAdapterHandle adapter, int32_t value, CCspPushBatchHandle batch )
{
    if( !adapter )
    {
        ccsp_set_error( CCSP_ERROR_NULL_POINTER, "null adapter" );
        return CCSP_ERROR_NULL_POINTER;
    }

    try
    {
        auto * pushAdapter = reinterpret_cast<csp::PushInputAdapter *>( adapter );
        auto * pushBatch = reinterpret_cast<csp::PushBatch *>( batch );
        int32_t val = value;
        pushAdapter -> pushTick( std::move( val ), pushBatch );
        return CCSP_OK;
    }
    catch( const std::exception & e )
    {
        ccsp_set_error( CCSP_ERROR_RUNTIME, e.what() );
        return CCSP_ERROR_RUNTIME;
    }
}

CCspErrorCode ccsp_push_input_adapter_push_uint32( CCspPushInputAdapterHandle adapter, uint32_t value, CCspPushBatchHandle batch )
{
    if( !adapter )
    {
        ccsp_set_error( CCSP_ERROR_NULL_POINTER, "null adapter" );
        return CCSP_ERROR_NULL_POINTER;
    }

    try
    {
        auto * pushAdapter = reinterpret_cast<csp::PushInputAdapter *>( adapter );
        auto * pushBatch = reinterpret_cast<csp::PushBatch *>( batch );
        uint32_t val = value;
        pushAdapter -> pushTick( std::move( val ), pushBatch );
        return CCSP_OK;
    }
    catch( const std::exception & e )
    {
        ccsp_set_error( CCSP_ERROR_RUNTIME, e.what() );
        return CCSP_ERROR_RUNTIME;
    }
}

CCspErrorCode ccsp_push_input_adapter_push_int64( CCspPushInputAdapterHandle adapter, int64_t value, CCspPushBatchHandle batch )
{
    if( !adapter )
    {
        ccsp_set_error( CCSP_ERROR_NULL_POINTER, "null adapter" );
        return CCSP_ERROR_NULL_POINTER;
    }

    try
    {
        auto * pushAdapter = reinterpret_cast<csp::PushInputAdapter *>( adapter );
        auto * pushBatch = reinterpret_cast<csp::PushBatch *>( batch );
        int64_t val = value;
        pushAdapter -> pushTick( std::move( val ), pushBatch );
        return CCSP_OK;
    }
    catch( const std::exception & e )
    {
        ccsp_set_error( CCSP_ERROR_RUNTIME, e.what() );
        return CCSP_ERROR_RUNTIME;
    }
}

CCspErrorCode ccsp_push_input_adapter_push_uint64( CCspPushInputAdapterHandle adapter, uint64_t value, CCspPushBatchHandle batch )
{
    if( !adapter )
    {
        ccsp_set_error( CCSP_ERROR_NULL_POINTER, "null adapter" );
        return CCSP_ERROR_NULL_POINTER;
    }

    try
    {
        auto * pushAdapter = reinterpret_cast<csp::PushInputAdapter *>( adapter );
        auto * pushBatch = reinterpret_cast<csp::PushBatch *>( batch );
        uint64_t val = value;
        pushAdapter -> pushTick( std::move( val ), pushBatch );
        return CCSP_OK;
    }
    catch( const std::exception & e )
    {
        ccsp_set_error( CCSP_ERROR_RUNTIME, e.what() );
        return CCSP_ERROR_RUNTIME;
    }
}

CCspErrorCode ccsp_push_input_adapter_push_double( CCspPushInputAdapterHandle adapter, double value, CCspPushBatchHandle batch )
{
    if( !adapter )
    {
        ccsp_set_error( CCSP_ERROR_NULL_POINTER, "null adapter" );
        return CCSP_ERROR_NULL_POINTER;
    }

    try
    {
        auto * pushAdapter = reinterpret_cast<csp::PushInputAdapter *>( adapter );
        auto * pushBatch = reinterpret_cast<csp::PushBatch *>( batch );
        double val = value;
        pushAdapter -> pushTick( std::move( val ), pushBatch );
        return CCSP_OK;
    }
    catch( const std::exception & e )
    {
        ccsp_set_error( CCSP_ERROR_RUNTIME, e.what() );
        return CCSP_ERROR_RUNTIME;
    }
}

CCspErrorCode ccsp_push_input_adapter_push_datetime( CCspPushInputAdapterHandle adapter, CCspDateTime value, CCspPushBatchHandle batch )
{
    if( !adapter )
    {
        ccsp_set_error( CCSP_ERROR_NULL_POINTER, "null adapter" );
        return CCSP_ERROR_NULL_POINTER;
    }

    try
    {
        auto * pushAdapter = reinterpret_cast<csp::PushInputAdapter *>( adapter );
        auto * pushBatch = reinterpret_cast<csp::PushBatch *>( batch );
        csp::DateTime dt = csp::DateTime::fromNanoseconds( value );
        pushAdapter -> pushTick( std::move( dt ), pushBatch );
        return CCSP_OK;
    }
    catch( const std::exception & e )
    {
        ccsp_set_error( CCSP_ERROR_RUNTIME, e.what() );
        return CCSP_ERROR_RUNTIME;
    }
}

CCspErrorCode ccsp_push_input_adapter_push_timedelta( CCspPushInputAdapterHandle adapter, CCspTimeDelta value, CCspPushBatchHandle batch )
{
    if( !adapter )
    {
        ccsp_set_error( CCSP_ERROR_NULL_POINTER, "null adapter" );
        return CCSP_ERROR_NULL_POINTER;
    }

    try
    {
        auto * pushAdapter = reinterpret_cast<csp::PushInputAdapter *>( adapter );
        auto * pushBatch = reinterpret_cast<csp::PushBatch *>( batch );
        csp::TimeDelta td = csp::TimeDelta::fromNanoseconds( value );
        pushAdapter -> pushTick( std::move( td ), pushBatch );
        return CCSP_OK;
    }
    catch( const std::exception & e )
    {
        ccsp_set_error( CCSP_ERROR_RUNTIME, e.what() );
        return CCSP_ERROR_RUNTIME;
    }
}

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
        auto * pushBatch = reinterpret_cast<csp::PushBatch *>( batch );
        std::string str( data ? data : "", length );
        pushAdapter -> pushTick( std::move( str ), pushBatch );
        return CCSP_OK;
    }
    catch( const std::exception & e )
    {
        ccsp_set_error( CCSP_ERROR_RUNTIME, e.what() );
        return CCSP_ERROR_RUNTIME;
    }
}

CCspErrorCode ccsp_push_input_adapter_push_struct( CCspPushInputAdapterHandle adapter, CCspStructHandle value, CCspPushBatchHandle batch )
{
    // TODO: Implement struct push when struct support is complete
    (void)adapter;
    (void)value;
    (void)batch;
    ccsp_set_error( CCSP_ERROR_NOT_IMPLEMENTED, "struct push not yet implemented" );
    return CCSP_ERROR_NOT_IMPLEMENTED;
}

CCspErrorCode ccsp_push_input_adapter_push_value(
    CCspPushInputAdapterHandle adapter,
    const CCspValue* value,
    CCspPushBatchHandle batch )
{
    // TODO: Implement generic value push
    (void)adapter;
    (void)value;
    (void)batch;
    ccsp_set_error( CCSP_ERROR_NOT_IMPLEMENTED, "generic value push not yet implemented" );
    return CCSP_ERROR_NOT_IMPLEMENTED;
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
    catch( const std::exception & e )
    {
        ccsp_set_error( CCSP_ERROR_RUNTIME, e.what() );
        return nullptr;
    }
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
    catch( const std::exception & e )
    {
        ccsp_set_error( CCSP_ERROR_RUNTIME, e.what() );
        return nullptr;
    }
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
