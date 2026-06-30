/*
 * Example Output Adapter implementation in C
 *
 * This demonstrates how to implement an output adapter using the C ABI interface.
 */
#include <csp/engine/c/OutputAdapter.h>
#include <csp/engine/c/CspError.h>
#include <csp/engine/c/CspStruct.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "ExampleOutputAdapter.h"

/* Adapter state structure */
typedef struct {
    char * prefix;       /* Prefix to print before each value */
    int fd;              /* File descriptor to write to */
    int owns_prefix;     /* Whether we own the prefix memory */
} ExampleOutputAdapterState;

/* ============================================================================
 * Callback implementations
 * ============================================================================ */

static void example_output_start( void * user_data, CCspEngineHandle engine, CCspDateTime start_time, CCspDateTime end_time )
{
    ExampleOutputAdapterState * state = ( ExampleOutputAdapterState * ) user_data;
    ( void ) engine;

    dprintf( state -> fd, "[ExampleOutputAdapter] Started. Time range: %lld - %lld ns\n", ( long long ) start_time, ( long long ) end_time );
}

static void example_output_stop( void * user_data )
{
    ExampleOutputAdapterState * state = ( ExampleOutputAdapterState * ) user_data;
    dprintf( state -> fd, "[ExampleOutputAdapter] Stopped.\n" );
}

static void example_output_execute( void * user_data, CCspEngineHandle engine, CCspInputHandle input )
{
    ExampleOutputAdapterState * state = ( ExampleOutputAdapterState * ) user_data;
    CCspDateTime now = ccsp_engine_now( engine );
    CCspType type = ccsp_input_get_type( input );

    const char * prefix = state -> prefix ? state -> prefix : "";

    /* Print based on type */
    switch( type )
    {
        case CCSP_TYPE_BOOL:
        {
            int8_t val;
            if( ccsp_input_get_last_bool( input, &val ) == CCSP_OK )
            {
                dprintf( state -> fd, "%s[%lld] bool: %s\n", prefix, ( long long ) now, val ? "true" : "false" );
            }
            break;
        }
        case CCSP_TYPE_INT64:
        {
            int64_t val;
            if( ccsp_input_get_last_int64( input, &val ) == CCSP_OK )
            {
                dprintf( state -> fd, "%s[%lld] int64: %lld\n", prefix, ( long long ) now, ( long long ) val );
            }
            break;
        }
        case CCSP_TYPE_DOUBLE:
        {
            double val;
            if( ccsp_input_get_last_double( input, &val ) == CCSP_OK )
            {
                dprintf( state -> fd, "%s[%lld] double: %f\n", prefix, ( long long ) now, val );
            }
            break;
        }
        case CCSP_TYPE_STRING:
        {
            const char * data;
            size_t len;
            if( ccsp_input_get_last_string( input, &data, &len ) == CCSP_OK )
            {
                dprintf( state -> fd, "%s[%lld] string: %.*s\n", prefix, ( long long ) now, ( int ) len, data );
            }
            break;
        }
        case CCSP_TYPE_DATETIME:
        {
            CCspDateTime val;
            if( ccsp_input_get_last_datetime( input, &val ) == CCSP_OK )
            {
                dprintf( state -> fd, "%s[%lld] datetime: %lld ns\n", prefix, ( long long ) now, ( long long ) val );
            }
            break;
        }
        default:
            dprintf( state -> fd, "%s[%lld] <type %d>\n", prefix, ( long long ) now, ( int ) type );
            break;
    }
}

static void example_output_destroy( void * user_data )
{
    ExampleOutputAdapterState * state = ( ExampleOutputAdapterState * ) user_data;
    if( state )
    {
        if( state -> owns_prefix && state -> prefix )
        {
            free( state -> prefix );
        }
        free( state );
    }
}

/* ============================================================================
 * Public API
 * ============================================================================ */

CCspOutputAdapterVTable example_output_adapter_create( const char * prefix )
{
    return example_output_adapter_create_fd( STDOUT_FILENO, prefix );
}

CCspOutputAdapterVTable example_output_adapter_create_fd( int fd, const char * prefix )
{
    CCspOutputAdapterVTable vtable = {0};

    /* Allocate state */
    ExampleOutputAdapterState * state = ( ExampleOutputAdapterState * )malloc( sizeof( ExampleOutputAdapterState ) );
    if( !state )
    {
        /* Return an invalid vtable with NULL callbacks */
        return vtable;
    }

    state -> fd = fd;
    state -> owns_prefix = 0;
    state -> prefix = NULL;

    /* Copy prefix if provided */
    if( prefix )
    {
        size_t len = strlen( prefix );
        state -> prefix = ( char * )malloc( len + 1 );
        if( state -> prefix )
        {
            memcpy( state -> prefix, prefix, len + 1 );
            state -> owns_prefix = 1;
        }
    }

    /* Set up vtable */
    vtable.user_data = state;
    vtable.start = example_output_start;
    vtable.stop = example_output_stop;
    vtable.execute = example_output_execute;
    vtable.destroy = example_output_destroy;

    return vtable;
}
