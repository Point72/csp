/*
 * Implementation of the C++ AdapterManagerExtern wrapper and C API functions.
 */
#include <csp/engine/AdapterManagerExtern.h>
#include <csp/engine/Engine.h>
#include <csp/engine/OutputAdapterExtern.h>
#include <csp/engine/c/CspError.h>
#include <csp/core/Exception.h>
#include <cstring>

namespace csp
{

// ============================================================================
// AdapterManagerExtern Implementation
// ============================================================================

AdapterManagerExtern::AdapterManagerExtern( Engine * engine, const CCspAdapterManagerVTable & vtable )
    : AdapterManager( engine )
    , m_vtable( vtable )
{
    if( !vtable.name )
    {
        CSP_THROW( ValueError, "AdapterManagerExtern: name callback is required" );
    }
    if( !vtable.process_next_sim_time_slice )
    {
        CSP_THROW( ValueError, "AdapterManagerExtern: process_next_sim_time_slice callback is required" );
    }
    if( !vtable.destroy )
    {
        CSP_THROW( ValueError, "AdapterManagerExtern: destroy callback is required" );
    }
}

AdapterManagerExtern::~AdapterManagerExtern()
{
    if( m_vtable.destroy )
    {
        m_vtable.destroy( m_vtable.user_data );
    }
}

const char * AdapterManagerExtern::name() const
{
    if( m_name.empty() && m_vtable.name )
    {
        const char * n = m_vtable.name( m_vtable.user_data );
        if( n )
        {
            m_name = n;
        }
    }
    return m_name.c_str();
}

void AdapterManagerExtern::start( DateTime startTime, DateTime endTime )
{
    AdapterManager::start( startTime, endTime );

    if( m_vtable.start )
    {
        CCspAdapterManagerHandle handle = reinterpret_cast<CCspAdapterManagerHandle>( this );
        m_vtable.start( m_vtable.user_data, handle,
                        startTime.asNanoseconds(), endTime.asNanoseconds() );
    }
}

void AdapterManagerExtern::stop()
{
    if( m_vtable.stop )
    {
        m_vtable.stop( m_vtable.user_data );
    }
    AdapterManager::stop();
}

DateTime AdapterManagerExtern::processNextSimTimeSlice( DateTime time )
{
    CCspDateTime result = m_vtable.process_next_sim_time_slice( m_vtable.user_data, time.asNanoseconds() );
    if( result == 0 )
    {
        return DateTime::NONE();
    }
    return DateTime::fromNanoseconds( result );
}

} // namespace csp

// ============================================================================
// C API Implementation
// ============================================================================

extern "C" {

CCspAdapterManagerHandle ccsp_adapter_manager_extern_create(
    CCspEngineHandle engine,
    const CCspAdapterManagerVTable * vtable )
{
    if( !engine || !vtable )
    {
        ccsp_set_error( CCSP_ERROR_NULL_POINTER, "null engine or vtable" );
        return nullptr;
    }

    try
    {
        auto * eng = reinterpret_cast<csp::Engine *>( engine );
        auto * manager = eng -> createOwnedObject<csp::AdapterManagerExtern>( *vtable );
        return reinterpret_cast<CCspAdapterManagerHandle>( manager );
    }
    catch( const std::exception & e )
    {
        ccsp_set_error( CCSP_ERROR_RUNTIME, e.what() );
        return nullptr;
    }
}

void ccsp_adapter_manager_extern_destroy( CCspAdapterManagerHandle manager )
{
    // Engine-owned objects are cleaned up by the engine when the graph stops
    // The vtable's destroy callback will be called from the destructor
    (void)manager;
}

CCspEngineHandle ccsp_adapter_manager_engine( CCspAdapterManagerHandle manager )
{
    if( !manager ) return nullptr;
    auto * m = reinterpret_cast<csp::AdapterManagerExtern *>( manager );
    return reinterpret_cast<CCspEngineHandle>( m -> engine() );
}

CCspDateTime ccsp_adapter_manager_start_time( CCspAdapterManagerHandle manager )
{
    if( !manager ) return 0;
    auto * m = reinterpret_cast<csp::AdapterManagerExtern *>( manager );
    return m -> starttime().asNanoseconds();
}

CCspDateTime ccsp_adapter_manager_end_time( CCspAdapterManagerHandle manager )
{
    if( !manager ) return 0;
    auto * m = reinterpret_cast<csp::AdapterManagerExtern *>( manager );
    return m -> endtime().asNanoseconds();
}

CCspOutputAdapterHandle ccsp_adapter_manager_create_output_adapter(
    CCspAdapterManagerHandle manager,
    CCspType input_type,
    const CCspOutputAdapterVTable * vtable )
{
    if( !manager || !vtable )
    {
        ccsp_set_error( CCSP_ERROR_NULL_POINTER, "null manager or vtable" );
        return nullptr;
    }

    try
    {
        auto * m = reinterpret_cast<csp::AdapterManagerExtern *>( manager );

        // Map CCspType to CspTypePtr
        csp::CspTypePtr cspType;
        switch( input_type )
        {
            case CCSP_TYPE_BOOL:      cspType = csp::CspType::BOOL(); break;
            case CCSP_TYPE_INT8:      cspType = csp::CspType::INT8(); break;
            case CCSP_TYPE_UINT8:     cspType = csp::CspType::UINT8(); break;
            case CCSP_TYPE_INT16:     cspType = csp::CspType::INT16(); break;
            case CCSP_TYPE_UINT16:    cspType = csp::CspType::UINT16(); break;
            case CCSP_TYPE_INT32:     cspType = csp::CspType::INT32(); break;
            case CCSP_TYPE_UINT32:    cspType = csp::CspType::UINT32(); break;
            case CCSP_TYPE_INT64:     cspType = csp::CspType::INT64(); break;
            case CCSP_TYPE_UINT64:    cspType = csp::CspType::UINT64(); break;
            case CCSP_TYPE_DOUBLE:    cspType = csp::CspType::DOUBLE(); break;
            case CCSP_TYPE_STRING:    cspType = csp::CspType::STRING(); break;
            case CCSP_TYPE_DATETIME:  cspType = csp::CspType::DATETIME(); break;
            case CCSP_TYPE_TIMEDELTA: cspType = csp::CspType::TIMEDELTA(); break;
            case CCSP_TYPE_DATE:      cspType = csp::CspType::DATE(); break;
            case CCSP_TYPE_TIME:      cspType = csp::CspType::TIME(); break;
            default:
                ccsp_set_error( CCSP_ERROR_INVALID_ARGUMENT, "unsupported input type" );
                return nullptr;
        }

        auto * adapter = m -> engine() -> createOwnedObject<csp::OutputAdapterExtern>( cspType, *vtable );
        return reinterpret_cast<CCspOutputAdapterHandle>( adapter );
    }
    catch( const std::exception & e )
    {
        ccsp_set_error( CCSP_ERROR_RUNTIME, e.what() );
        return nullptr;
    }
}

CCspErrorCode ccsp_adapter_manager_push_status(
    CCspAdapterManagerHandle manager,
    CCspStatusLevel level,
    int64_t err_code,
    const char * message )
{
    if( !manager )
    {
        ccsp_set_error( CCSP_ERROR_NULL_POINTER, "null manager" );
        return CCSP_ERROR_NULL_POINTER;
    }

    try
    {
        auto * m = reinterpret_cast<csp::AdapterManagerExtern *>( manager );
        m -> pushStatus( static_cast<int64_t>( level ), err_code,
                         message ? message : "" );
        return CCSP_OK;
    }
    catch( const std::exception & e )
    {
        ccsp_set_error( CCSP_ERROR_RUNTIME, e.what() );
        return CCSP_ERROR_RUNTIME;
    }
}

// TODO: Implement these when ManagedSimInputAdapter extern support is added
CCspManagedSimInputAdapterHandle ccsp_adapter_manager_create_managed_sim_input_adapter(
    CCspAdapterManagerHandle manager,
    CCspType type,
    CCspPushMode push_mode )
{
    ccsp_set_error( CCSP_ERROR_NOT_IMPLEMENTED, "managed sim input adapter not yet implemented" );
    return nullptr;
}

CCspErrorCode ccsp_managed_sim_input_adapter_push_bool( CCspManagedSimInputAdapterHandle adapter, int8_t value )
{
    ccsp_set_error( CCSP_ERROR_NOT_IMPLEMENTED, "managed sim input adapter not yet implemented" );
    return CCSP_ERROR_NOT_IMPLEMENTED;
}

CCspErrorCode ccsp_managed_sim_input_adapter_push_int64( CCspManagedSimInputAdapterHandle adapter, int64_t value )
{
    ccsp_set_error( CCSP_ERROR_NOT_IMPLEMENTED, "managed sim input adapter not yet implemented" );
    return CCSP_ERROR_NOT_IMPLEMENTED;
}

CCspErrorCode ccsp_managed_sim_input_adapter_push_double( CCspManagedSimInputAdapterHandle adapter, double value )
{
    ccsp_set_error( CCSP_ERROR_NOT_IMPLEMENTED, "managed sim input adapter not yet implemented" );
    return CCSP_ERROR_NOT_IMPLEMENTED;
}

CCspErrorCode ccsp_managed_sim_input_adapter_push_string( CCspManagedSimInputAdapterHandle adapter, const char * data, size_t length )
{
    ccsp_set_error( CCSP_ERROR_NOT_IMPLEMENTED, "managed sim input adapter not yet implemented" );
    return CCSP_ERROR_NOT_IMPLEMENTED;
}

CCspErrorCode ccsp_managed_sim_input_adapter_push_datetime( CCspManagedSimInputAdapterHandle adapter, CCspDateTime value )
{
    ccsp_set_error( CCSP_ERROR_NOT_IMPLEMENTED, "managed sim input adapter not yet implemented" );
    return CCSP_ERROR_NOT_IMPLEMENTED;
}

// Push input adapter creation from manager
CCspPushInputAdapterHandle ccsp_adapter_manager_create_push_input_adapter(
    CCspAdapterManagerHandle manager,
    CCspType type,
    CCspPushMode push_mode,
    const CCspPushInputAdapterVTable * vtable )
{
    // For now, delegate to the standalone creation
    // In a full implementation, the manager would track these adapters
    if( !manager )
    {
        ccsp_set_error( CCSP_ERROR_NULL_POINTER, "null manager" );
        return nullptr;
    }

    auto * m = reinterpret_cast<csp::AdapterManagerExtern *>( manager );
    (void)m;  // Currently unused - will be used when implemented

    // Use the standalone push input adapter creation
    // This is a simplification - full implementation would track adapters
    ccsp_set_error( CCSP_ERROR_NOT_IMPLEMENTED, "push input adapter creation via manager not yet implemented" );
    return nullptr;
}

} // extern "C"
