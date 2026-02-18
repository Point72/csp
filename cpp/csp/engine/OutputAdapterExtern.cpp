/*
 * Implementation of the C++ OutputAdapterExtern wrapper and C API functions.
 */
#include <csp/engine/OutputAdapterExtern.h>
#include <csp/engine/Engine.h>
#include <csp/engine/TimeSeriesProvider.h>
#include <csp/engine/c/OutputAdapter.h>
#include <csp/engine/c/CspError.h>
#include <csp/core/Exception.h>
#include <cstring>

namespace csp
{

// ============================================================================
// OutputAdapterExtern Implementation
// ============================================================================

OutputAdapterExtern::OutputAdapterExtern( Engine * engine, const CspTypePtr & type,
                                          const CCspOutputAdapterVTable & vtable )
    : OutputAdapter( engine )
    , m_vtable( vtable )
    , m_startTime( DateTime::NONE() )
    , m_endTime( DateTime::NONE() )
{
    if( !vtable.execute )
    {
        CSP_THROW( ValueError, "OutputAdapterExtern: execute callback is required" );
    }
    if( !vtable.destroy )
    {
        CSP_THROW( ValueError, "OutputAdapterExtern: destroy callback is required" );
    }
}

OutputAdapterExtern::~OutputAdapterExtern()
{
    if( m_vtable.destroy )
    {
        m_vtable.destroy( m_vtable.user_data );
    }
}

void OutputAdapterExtern::start()
{
    OutputAdapter::start();
    m_startTime = rootEngine() -> startTime();
    m_endTime = rootEngine() -> endTime();

    if( m_vtable.start )
    {
        CCspEngineHandle engineHandle = reinterpret_cast<CCspEngineHandle>( engine() );
        m_vtable.start( m_vtable.user_data, engineHandle,
                        m_startTime.asNanoseconds(), m_endTime.asNanoseconds() );
    }
}

void OutputAdapterExtern::stop()
{
    if( m_vtable.stop )
    {
        m_vtable.stop( m_vtable.user_data );
    }
    OutputAdapter::stop();
}

void OutputAdapterExtern::executeImpl()
{
    CCspEngineHandle engineHandle = reinterpret_cast<CCspEngineHandle>( engine() );
    CCspInputHandle inputHandle = reinterpret_cast<CCspInputHandle>( input() );

    m_vtable.execute( m_vtable.user_data, engineHandle, inputHandle );
}

} // namespace csp

// ============================================================================
// C API Implementation
// ============================================================================

extern "C" {

// Thread-local error state
static thread_local CCspErrorCode s_lastError = CCSP_OK;
static thread_local char s_lastErrorMessage[256] = {0};

CCspErrorCode ccsp_get_last_error(void)
{
    return s_lastError;
}

const char * ccsp_get_last_error_message(void)
{
    return s_lastErrorMessage[0] ? s_lastErrorMessage : nullptr;
}

void ccsp_clear_error(void)
{
    s_lastError = CCSP_OK;
    s_lastErrorMessage[0] = '\0';
}

void ccsp_set_error( CCspErrorCode code, const char * message )
{
    s_lastError = code;
    if( message )
    {
        strncpy( s_lastErrorMessage, message, sizeof( s_lastErrorMessage ) - 1 );
        s_lastErrorMessage[sizeof( s_lastErrorMessage ) - 1] = '\0';
    }
    else
    {
        s_lastErrorMessage[0] = '\0';
    }
}

// ============================================================================
// Input Access Functions
// ============================================================================

int ccsp_input_is_valid( CCspInputHandle input )
{
    if( !input ) return 0;
    auto * provider = reinterpret_cast<csp::TimeSeriesProvider *>( input );
    return provider -> valid() ? 1 : 0;
}

int32_t ccsp_input_num_ticks( CCspInputHandle input )
{
    if( !input ) return 0;
    auto * provider = reinterpret_cast<csp::TimeSeriesProvider *>( input );
    return provider -> numTicks();
}

CCspType ccsp_input_get_type( CCspInputHandle input )
{
    if( !input ) return CCSP_TYPE_UNKNOWN;
    auto * provider = reinterpret_cast<csp::TimeSeriesProvider *>( input );

    // Map CspType to CCspType
    switch( provider -> type() -> type() )
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

CCspDateTime ccsp_input_get_last_time( CCspInputHandle input )
{
    if( !input ) return 0;
    auto * provider = reinterpret_cast<csp::TimeSeriesProvider *>( input );
    return provider -> lastTime().asNanoseconds();
}

CCspErrorCode ccsp_input_get_last_string( CCspInputHandle input,
                                          const char ** out_data,
                                          size_t * out_length )
{
    if( !input || !out_data || !out_length )
    {
        ccsp_set_error( CCSP_ERROR_NULL_POINTER, "null argument" );
        return CCSP_ERROR_NULL_POINTER;
    }

    auto * provider = reinterpret_cast<csp::TimeSeriesProvider *>( input );
    if( provider -> type() -> type() != csp::CspType::Type::STRING )
    {
        ccsp_set_error( CCSP_ERROR_TYPE_MISMATCH, "input is not a string type" );
        return CCSP_ERROR_TYPE_MISMATCH;
    }

    const std::string & value = provider -> lastValueTyped<std::string>();
    *out_data = value.data();
    *out_length = value.size();

    return CCSP_OK;
}

CCspErrorCode ccsp_input_get_last_int64( CCspInputHandle input, int64_t * out_value )
{
    if( !input || !out_value )
    {
        ccsp_set_error( CCSP_ERROR_NULL_POINTER, "null argument" );
        return CCSP_ERROR_NULL_POINTER;
    }

    auto * provider = reinterpret_cast<csp::TimeSeriesProvider *>( input );
    if( provider -> type() -> type() != csp::CspType::Type::INT64 )
    {
        ccsp_set_error( CCSP_ERROR_TYPE_MISMATCH, "input is not an int64 type" );
        return CCSP_ERROR_TYPE_MISMATCH;
    }

    *out_value = provider -> lastValueTyped<int64_t>();
    return CCSP_OK;
}

CCspErrorCode ccsp_input_get_last_double( CCspInputHandle input, double * out_value )
{
    if( !input || !out_value )
    {
        ccsp_set_error( CCSP_ERROR_NULL_POINTER, "null argument" );
        return CCSP_ERROR_NULL_POINTER;
    }

    auto * provider = reinterpret_cast<csp::TimeSeriesProvider *>( input );
    if( provider -> type() -> type() != csp::CspType::Type::DOUBLE )
    {
        ccsp_set_error( CCSP_ERROR_TYPE_MISMATCH, "input is not a double type" );
        return CCSP_ERROR_TYPE_MISMATCH;
    }

    *out_value = provider -> lastValueTyped<double>();
    return CCSP_OK;
}

CCspErrorCode ccsp_input_get_last_bool( CCspInputHandle input, int8_t * out_value )
{
    if( !input || !out_value )
    {
        ccsp_set_error( CCSP_ERROR_NULL_POINTER, "null argument" );
        return CCSP_ERROR_NULL_POINTER;
    }

    auto * provider = reinterpret_cast<csp::TimeSeriesProvider *>( input );
    if( provider -> type() -> type() != csp::CspType::Type::BOOL )
    {
        ccsp_set_error( CCSP_ERROR_TYPE_MISMATCH, "input is not a bool type" );
        return CCSP_ERROR_TYPE_MISMATCH;
    }

    *out_value = provider -> lastValueTyped<bool>() ? 1 : 0;
    return CCSP_OK;
}

CCspErrorCode ccsp_input_get_last_datetime( CCspInputHandle input, CCspDateTime * out_value )
{
    if( !input || !out_value )
    {
        ccsp_set_error( CCSP_ERROR_NULL_POINTER, "null argument" );
        return CCSP_ERROR_NULL_POINTER;
    }

    auto * provider = reinterpret_cast<csp::TimeSeriesProvider *>( input );
    if( provider -> type() -> type() != csp::CspType::Type::DATETIME )
    {
        ccsp_set_error( CCSP_ERROR_TYPE_MISMATCH, "input is not a datetime type" );
        return CCSP_ERROR_TYPE_MISMATCH;
    }

    *out_value = provider -> lastValueTyped<csp::DateTime>().asNanoseconds();
    return CCSP_OK;
}

// ============================================================================
// Engine Access Functions
// ============================================================================

CCspDateTime ccsp_engine_now( CCspEngineHandle engine )
{
    if( !engine ) return 0;
    auto * e = reinterpret_cast<csp::Engine *>( engine );
    return e -> rootEngine() -> now().asNanoseconds();
}

uint64_t ccsp_engine_cycle_count( CCspEngineHandle engine )
{
    if( !engine ) return 0;
    auto * e = reinterpret_cast<csp::Engine *>( engine );
    return e -> rootEngine() -> cycleCount();
}

} // extern "C"
