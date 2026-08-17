/*
 * Implementation of the C++ OutputAdapterExtern wrapper and C API functions.
 */
#include <csp/engine/OutputAdapterExtern.h>
#include <csp/engine/Engine.h>
#include <csp/engine/ExternBoundary.h>
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
    , m_vtable()
    , m_startTime( DateTime::NONE() )
    , m_endTime( DateTime::NONE() )
{
    if( !ccsp_vtable_adopt( &m_vtable, sizeof( m_vtable ), &vtable ) )
    {
        CSP_THROW( ValueError, "OutputAdapterExtern: vtable was not initialized with CCSP_VTABLE_INIT, "
                               "or was built against a newer csp ABI than this build supports" );
    }

    if( !m_vtable.execute )
    {
        CSP_THROW( ValueError, "OutputAdapterExtern: execute callback is required" );
    }
    if( !m_vtable.destroy )
    {
        CSP_THROW( ValueError, "OutputAdapterExtern: destroy callback is required" );
    }
}

OutputAdapterExtern::~OutputAdapterExtern()
{
    if( m_vtable.destroy )
    {
        // Destructors are implicitly noexcept, so a throwing callback would terminate the process
        try { m_vtable.destroy( m_vtable.user_data ); } catch( ... ) {}
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
        try
        {
            m_vtable.start( m_vtable.user_data, engineHandle,
                            m_startTime.asNanoseconds(), m_endTime.asNanoseconds() );
        }
        CSP_CATCH_EXTERN_CALLBACK( "output adapter start" )
    }
}

void OutputAdapterExtern::stop()
{
    if( m_vtable.stop )
    {
        try
        {
            m_vtable.stop( m_vtable.user_data );
        }
        CSP_CATCH_EXTERN_CALLBACK( "output adapter stop" )
    }
    OutputAdapter::stop();
}

void OutputAdapterExtern::executeImpl()
{
    CCspEngineHandle engineHandle = reinterpret_cast<CCspEngineHandle>( engine() );
    CCspInputHandle inputHandle = reinterpret_cast<CCspInputHandle>( input() );

    try
    {
        m_vtable.execute( m_vtable.user_data, engineHandle, inputHandle );
    }
    CSP_CATCH_EXTERN_CALLBACK( "output adapter execute" )
}

} // namespace csp

// ============================================================================
// C API Implementation
// ============================================================================

extern "C" {

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

    if( !provider -> valid() )
    {
        ccsp_set_error( CCSP_ERROR_VALUE, "input has not ticked" );
        return CCSP_ERROR_VALUE;
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

    if( !provider -> valid() )
    {
        ccsp_set_error( CCSP_ERROR_VALUE, "input has not ticked" );
        return CCSP_ERROR_VALUE;
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

    if( !provider -> valid() )
    {
        ccsp_set_error( CCSP_ERROR_VALUE, "input has not ticked" );
        return CCSP_ERROR_VALUE;
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

    if( !provider -> valid() )
    {
        ccsp_set_error( CCSP_ERROR_VALUE, "input has not ticked" );
        return CCSP_ERROR_VALUE;
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

    if( !provider -> valid() )
    {
        ccsp_set_error( CCSP_ERROR_VALUE, "input has not ticked" );
        return CCSP_ERROR_VALUE;
    }

    *out_value = provider -> lastValueTyped<csp::DateTime>().asNanoseconds();
    return CCSP_OK;
}

// ============================================================================
// Generic and indexed input access
// ============================================================================

/* Fills a CCspValue from a timeseries at a given buffer index. index 0 is the most recent. */
static CCspErrorCode inputValueAtIndex( CCspInputHandle input, int32_t index, CCspValue * out_value )
{
    if( !input || !out_value )
    {
        ccsp_set_error( CCSP_ERROR_NULL_POINTER, "null argument" );
        return CCSP_ERROR_NULL_POINTER;
    }

    auto * provider = reinterpret_cast<csp::TimeSeriesProvider *>( input );

    if( !provider -> valid() )
    {
        ccsp_set_error( CCSP_ERROR_VALUE, "input has not ticked" );
        return CCSP_ERROR_VALUE;
    }
    if( index < 0 || index >= provider -> numTicks() )
    {
        ccsp_set_error( CCSP_ERROR_OUT_OF_RANGE, "tick index out of range" );
        return CCSP_ERROR_OUT_OF_RANGE;
    }

    try
    {
        switch( provider -> type() -> type() )
        {
            case csp::CspType::Type::BOOL:
                ccsp_value_set_bool( out_value, provider -> valueAtIndex<bool>( index ) ? 1 : 0 );
                return CCSP_OK;
            case csp::CspType::Type::INT8:
                ccsp_value_set_int8( out_value, provider -> valueAtIndex<int8_t>( index ) );
                return CCSP_OK;
            case csp::CspType::Type::UINT8:
                ccsp_value_set_uint8( out_value, provider -> valueAtIndex<uint8_t>( index ) );
                return CCSP_OK;
            case csp::CspType::Type::INT16:
                ccsp_value_set_int16( out_value, provider -> valueAtIndex<int16_t>( index ) );
                return CCSP_OK;
            case csp::CspType::Type::UINT16:
                ccsp_value_set_uint16( out_value, provider -> valueAtIndex<uint16_t>( index ) );
                return CCSP_OK;
            case csp::CspType::Type::INT32:
                ccsp_value_set_int32( out_value, provider -> valueAtIndex<int32_t>( index ) );
                return CCSP_OK;
            case csp::CspType::Type::UINT32:
                ccsp_value_set_uint32( out_value, provider -> valueAtIndex<uint32_t>( index ) );
                return CCSP_OK;
            case csp::CspType::Type::INT64:
                ccsp_value_set_int64( out_value, provider -> valueAtIndex<int64_t>( index ) );
                return CCSP_OK;
            case csp::CspType::Type::UINT64:
                ccsp_value_set_uint64( out_value, provider -> valueAtIndex<uint64_t>( index ) );
                return CCSP_OK;
            case csp::CspType::Type::DOUBLE:
                ccsp_value_set_double( out_value, provider -> valueAtIndex<double>( index ) );
                return CCSP_OK;
            case csp::CspType::Type::DATETIME:
                ccsp_value_set_datetime( out_value, provider -> valueAtIndex<csp::DateTime>( index ).asNanoseconds() );
                return CCSP_OK;
            case csp::CspType::Type::TIMEDELTA:
                ccsp_value_set_timedelta( out_value, provider -> valueAtIndex<csp::TimeDelta>( index ).asNanoseconds() );
                return CCSP_OK;
            case csp::CspType::Type::STRING:
            {
                /* Borrowed: valid until the timeseries buffer moves on */
                const std::string & value = provider -> valueAtIndex<std::string>( index );
                ccsp_value_set_string_view( out_value, value.data(), value.size() );
                return CCSP_OK;
            }
            default:
                ccsp_set_error( CCSP_ERROR_NOT_IMPLEMENTED, "input type is not representable as a CCspValue" );
                return CCSP_ERROR_NOT_IMPLEMENTED;
        }
    }
    CCSP_CATCH_ERR
}

CCspErrorCode ccsp_input_get_last_value( CCspInputHandle input, CCspValue * out_value )
{
    return inputValueAtIndex( input, 0, out_value );
}

CCspErrorCode ccsp_input_get_value_at( CCspInputHandle input, int32_t index, CCspValue * out_value )
{
    return inputValueAtIndex( input, index, out_value );
}

CCspErrorCode ccsp_input_get_time_at( CCspInputHandle input, int32_t index, CCspDateTime * out_time )
{
    if( !input || !out_time )
    {
        ccsp_set_error( CCSP_ERROR_NULL_POINTER, "null argument" );
        return CCSP_ERROR_NULL_POINTER;
    }

    auto * provider = reinterpret_cast<csp::TimeSeriesProvider *>( input );

    if( !provider -> valid() )
    {
        ccsp_set_error( CCSP_ERROR_VALUE, "input has not ticked" );
        return CCSP_ERROR_VALUE;
    }
    if( index < 0 || index >= provider -> numTicks() )
    {
        ccsp_set_error( CCSP_ERROR_OUT_OF_RANGE, "tick index out of range" );
        return CCSP_ERROR_OUT_OF_RANGE;
    }

    try
    {
        *out_time = provider -> timeAtIndex( index ).asNanoseconds();
        return CCSP_OK;
    }
    CCSP_CATCH_ERR
}

// ============================================================================
// Output adapter creation
// ============================================================================

CCspOutputAdapterHandle ccsp_output_adapter_extern_create( CCspEngineHandle engine, CCspType input_type,
                                                           const CCspOutputAdapterVTable * vtable )
{
    if( !engine || !vtable )
    {
        ccsp_set_error( CCSP_ERROR_NULL_POINTER, "null engine or vtable" );
        return nullptr;
    }

    try
    {
        auto * eng = reinterpret_cast<csp::Engine *>( engine );

        csp::CspTypePtr cspType;
        switch( input_type )
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
            case CCSP_TYPE_DATE:      cspType = csp::CspType::DATE();      break;
            case CCSP_TYPE_TIME:      cspType = csp::CspType::TIME();      break;
            default:
                ccsp_set_error( CCSP_ERROR_INVALID_ARGUMENT, "unsupported input type" );
                return nullptr;
        }

        auto * adapter = eng -> createOwnedObject<csp::OutputAdapterExtern>( cspType, *vtable );
        return reinterpret_cast<CCspOutputAdapterHandle>( adapter );
    }
    CCSP_CATCH_RET( nullptr )
}

void ccsp_output_adapter_extern_destroy( CCspOutputAdapterHandle adapter )
{
    /* Engine-owned: the destroy callback runs from ~OutputAdapterExtern when the graph tears down */
    ( void ) adapter;
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
