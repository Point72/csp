/*
 * Implementation of the C++ AdapterManagerExtern wrapper and C API functions.
 */
#include <csp/engine/AdapterManagerExtern.h>
#include <csp/engine/Engine.h>
#include <csp/engine/ExternBoundary.h>
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
    , m_vtable()
{
    if( !ccsp_vtable_adopt( &m_vtable, sizeof( m_vtable ), &vtable ) )
    {
        CSP_THROW( ValueError, "AdapterManagerExtern: vtable was not initialized with CCSP_VTABLE_INIT, "
                               "or was built against a newer csp ABI than this build supports" );
    }

    if( !m_vtable.name )
    {
        CSP_THROW( ValueError, "AdapterManagerExtern: name callback is required" );
    }
    if( !m_vtable.process_next_sim_time_slice )
    {
        CSP_THROW( ValueError, "AdapterManagerExtern: process_next_sim_time_slice callback is required" );
    }
    if( !m_vtable.destroy )
    {
        CSP_THROW( ValueError, "AdapterManagerExtern: destroy callback is required" );
    }
}

AdapterManagerExtern::~AdapterManagerExtern()
{
    if( m_vtable.destroy )
    {
        // Destructors are implicitly noexcept, so a throwing callback would terminate the process
        try { m_vtable.destroy( m_vtable.user_data ); } catch( ... ) {}
    }
}

const char * AdapterManagerExtern::name() const
{
    if( m_name.empty() && m_vtable.name )
    {
        // name() is used to format engine errors, so it must not raise one of its own
        try
        {
            const char * n = m_vtable.name( m_vtable.user_data );
            if( n )
            {
                m_name = n;
            }
        }
        catch( ... ) {}
    }
    return m_name.c_str();
}

void AdapterManagerExtern::start( DateTime startTime, DateTime endTime )
{
    AdapterManager::start( startTime, endTime );

    if( m_vtable.start )
    {
        CCspAdapterManagerHandle handle = reinterpret_cast<CCspAdapterManagerHandle>( this );
        try
        {
            m_vtable.start( m_vtable.user_data, handle,
                            startTime.asNanoseconds(), endTime.asNanoseconds() );
        }
        CSP_CATCH_EXTERN_CALLBACK( "adapter manager start" )
    }
}

void AdapterManagerExtern::stop()
{
    if( m_vtable.stop )
    {
        try
        {
            m_vtable.stop( m_vtable.user_data );
        }
        CSP_CATCH_EXTERN_CALLBACK( "adapter manager stop" )
    }
    AdapterManager::stop();
}

DateTime AdapterManagerExtern::processNextSimTimeSlice( DateTime time )
{
    CCspDateTime result;
    try
    {
        result = m_vtable.process_next_sim_time_slice( m_vtable.user_data, time.asNanoseconds() );
    }
    CSP_CATCH_EXTERN_CALLBACK( "process_next_sim_time_slice" )

    if( result == CCSP_DATETIME_NONE )
    {
        return DateTime::NONE();
    }
    return DateTime::fromNanoseconds( result );
}

} // namespace csp

namespace
{

/* Sim pushes are only legal from inside processNextSimTimeSlice, on the engine thread */
template<typename T>
CCspErrorCode pushSimTick( CCspManagedSimInputAdapterHandle adapter, csp::CspType::Type::_enum expected, T && value )
{
    if( !adapter )
    {
        ccsp_set_error( CCSP_ERROR_NULL_POINTER, "null adapter" );
        return CCSP_ERROR_NULL_POINTER;
    }

    try
    {
        auto * sim = reinterpret_cast<csp::ManagedSimInputAdapter *>( adapter );
        if( sim -> dataType() -> type() != expected )
        {
            ccsp_set_error( CCSP_ERROR_TYPE_MISMATCH, "pushed value type does not match the adapter's declared type" );
            return CCSP_ERROR_TYPE_MISMATCH;
        }

        sim -> pushTick( std::forward<T>( value ) );
        return CCSP_OK;
    }
    CCSP_CATCH_ERR
}

} // namespace

// ============================================================================
// C API Implementation
// ============================================================================

extern "C" {

CCspAdapterManagerHandle ccsp_adapter_manager_extern_create( CCspEngineHandle engine, const CCspAdapterManagerVTable * vtable )
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
    CCSP_CATCH_RET( nullptr )
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

CCspOutputAdapterHandle ccsp_adapter_manager_create_output_adapter( CCspAdapterManagerHandle manager, CCspType input_type, const CCspOutputAdapterVTable * vtable )
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
    CCSP_CATCH_RET( nullptr )
}

CCspErrorCode ccsp_adapter_manager_push_status( CCspAdapterManagerHandle manager, CCspStatusLevel level, int64_t err_code, const char * message )
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
    CCSP_CATCH_ERR
}

/* ManagedSimInputAdapter is the one adapter kind that genuinely registers with its manager: the
 * manager feeds it during processNextSimTimeSlice. */
CCspManagedSimInputAdapterHandle ccsp_adapter_manager_create_managed_sim_input_adapter( CCspAdapterManagerHandle manager, CCspType type, CCspPushMode push_mode )
{
    if( !manager )
    {
        ccsp_set_error( CCSP_ERROR_NULL_POINTER, "null manager" );
        return nullptr;
    }

    try
    {
        auto * m = reinterpret_cast<csp::AdapterManagerExtern *>( manager );

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
            case CCSP_TYPE_STRING:    cspType = csp::CspType::STRING();    break;
            case CCSP_TYPE_DATETIME:  cspType = csp::CspType::DATETIME();  break;
            case CCSP_TYPE_TIMEDELTA: cspType = csp::CspType::TIMEDELTA(); break;
            default:
                ccsp_set_error( CCSP_ERROR_INVALID_ARGUMENT, "unsupported type for managed sim input adapter" );
                return nullptr;
        }

        csp::PushMode cspPushMode;
        switch( push_mode )
        {
            case CCSP_PUSH_MODE_LAST_VALUE:     cspPushMode = csp::PushMode::LAST_VALUE;     break;
            case CCSP_PUSH_MODE_NON_COLLAPSING: cspPushMode = csp::PushMode::NON_COLLAPSING; break;
            case CCSP_PUSH_MODE_BURST:          cspPushMode = csp::PushMode::BURST;          break;
            default:
                ccsp_set_error( CCSP_ERROR_INVALID_ARGUMENT, "invalid push mode" );
                return nullptr;
        }

        auto * adapter = m -> engine() -> createOwnedObject<csp::ManagedSimInputAdapter>( cspType, m, cspPushMode );
        return reinterpret_cast<CCspManagedSimInputAdapterHandle>( adapter );
    }
    CCSP_CATCH_RET( nullptr )
}

CCspErrorCode ccsp_managed_sim_input_adapter_push_bool( CCspManagedSimInputAdapterHandle adapter, int8_t value )
{
    return pushSimTick( adapter, csp::CspType::Type::BOOL, value != 0 );
}

CCspErrorCode ccsp_managed_sim_input_adapter_push_int64( CCspManagedSimInputAdapterHandle adapter, int64_t value )
{
    return pushSimTick( adapter, csp::CspType::Type::INT64, value );
}

CCspErrorCode ccsp_managed_sim_input_adapter_push_double( CCspManagedSimInputAdapterHandle adapter, double value )
{
    return pushSimTick( adapter, csp::CspType::Type::DOUBLE, value );
}

CCspErrorCode ccsp_managed_sim_input_adapter_push_string( CCspManagedSimInputAdapterHandle adapter, const char * data, size_t length )
{
    if( !data && length > 0 )
    {
        ccsp_set_error( CCSP_ERROR_INVALID_ARGUMENT, "null data with non-zero length" );
        return CCSP_ERROR_INVALID_ARGUMENT;
    }
    return pushSimTick( adapter, csp::CspType::Type::STRING, std::string( data ? data : "", length ) );
}

CCspErrorCode ccsp_managed_sim_input_adapter_push_datetime( CCspManagedSimInputAdapterHandle adapter, CCspDateTime value )
{
    return pushSimTick( adapter, csp::CspType::Type::DATETIME, csp::DateTime::fromNanoseconds( value ) );
}

// Push input adapter creation from manager
CCspPushInputAdapterHandle ccsp_adapter_manager_create_push_input_adapter( CCspAdapterManagerHandle manager, CCspType type, CCspPushMode push_mode, const CCspPushInputAdapterVTable * vtable )
{
    // For now, delegate to the standalone creation
    // In a full implementation, the manager would track these adapters
    if( !manager )
    {
        ccsp_set_error( CCSP_ERROR_NULL_POINTER, "null manager" );
        return nullptr;
    }

    auto * m = reinterpret_cast<csp::AdapterManagerExtern *>( manager );
    ( void ) m;  // Currently unused - will be used when implemented

    // Use the standalone push input adapter creation
    // This is a simplification - full implementation would track adapters
    ccsp_set_error( CCSP_ERROR_NOT_IMPLEMENTED, "push input adapter creation via manager not yet implemented" );
    return nullptr;
}

} // extern "C"
