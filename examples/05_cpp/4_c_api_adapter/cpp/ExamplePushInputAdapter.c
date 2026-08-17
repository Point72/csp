/*
 * Example Push Input Adapter implementation in C
 *
 * This demonstrates how to implement a push input adapter using the C ABI interface.
 * Values are produced on a background thread and pushed into the engine.
 */
#include <csp/engine/c/InputAdapter.h>
#include <csp/engine/c/CspError.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ExamplePushInputAdapter.h"

#ifdef _WIN32
#include <windows.h>
#include <process.h>
#else
#include <pthread.h>
#include <time.h>
#endif

/* ============================================================================
 * Portable worker thread
 *
 * The engine calls stop() on its own thread while the worker is running, so the stop state has
 * to be synchronized rather than a plain flag. Waiting on it also lets stop() interrupt the
 * sleep immediately instead of waiting out a full interval.
 * ============================================================================ */

#ifdef _WIN32
#define EXAMPLE_THREAD_RETURN unsigned __stdcall
#define EXAMPLE_THREAD_RESULT 0
typedef unsigned ( __stdcall * ExampleThreadFn )( void * );
#else
#define EXAMPLE_THREAD_RETURN void *
#define EXAMPLE_THREAD_RESULT NULL
typedef void * ( *ExampleThreadFn )( void * );
#endif

typedef struct {
    int started;
#ifdef _WIN32
    HANDLE thread;
    HANDLE stop_event;
#else
    pthread_t thread;
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    int stopping;
#endif
} ExampleWorker;

static int example_worker_start( ExampleWorker * w, ExampleThreadFn fn, void * arg )
{
    w -> started = 0;

#ifdef _WIN32
    w -> stop_event = CreateEvent( NULL, TRUE, FALSE, NULL );
    if( !w -> stop_event )
        return 0;

    w -> thread = ( HANDLE )_beginthreadex( NULL, 0, fn, arg, 0, NULL );
    if( !w -> thread )
    {
        CloseHandle( w -> stop_event );
        w -> stop_event = NULL;
        return 0;
    }
#else
    if( pthread_mutex_init( &w -> mutex, NULL ) != 0 )
        return 0;
    if( pthread_cond_init( &w -> cond, NULL ) != 0 )
    {
        pthread_mutex_destroy( &w -> mutex );
        return 0;
    }

    w -> stopping = 0;
    if( pthread_create( &w -> thread, NULL, fn, arg ) != 0 )
    {
        pthread_cond_destroy( &w -> cond );
        pthread_mutex_destroy( &w -> mutex );
        return 0;
    }
#endif

    w -> started = 1;
    return 1;
}

/* Waits up to timeout_ms; returns 1 once the worker has been asked to stop. */
static int example_worker_should_stop( ExampleWorker * w, int timeout_ms )
{
#ifdef _WIN32
    return WaitForSingleObject( w -> stop_event, ( DWORD )timeout_ms ) == WAIT_OBJECT_0;
#else
    struct timespec deadline;
    int stopping;

    clock_gettime( CLOCK_REALTIME, &deadline );
    deadline.tv_sec += timeout_ms / 1000;
    deadline.tv_nsec += ( long )( timeout_ms % 1000 ) * 1000000L;
    if( deadline.tv_nsec >= 1000000000L )
    {
        deadline.tv_sec += 1;
        deadline.tv_nsec -= 1000000000L;
    }

    pthread_mutex_lock( &w -> mutex );
    while( !w -> stopping )
    {
        int rc = pthread_cond_timedwait( &w -> cond, &w -> mutex, &deadline );
        if( rc == ETIMEDOUT )
            break;
        if( rc != 0 )
        {
            /* Stop rather than spin on an error we cannot recover from */
            w -> stopping = 1;
            break;
        }
    }
    stopping = w -> stopping;
    pthread_mutex_unlock( &w -> mutex );

    return stopping;
#endif
}

static void example_worker_stop( ExampleWorker * w )
{
    if( !w -> started )
        return;

#ifdef _WIN32
    SetEvent( w -> stop_event );
    WaitForSingleObject( w -> thread, INFINITE );
    CloseHandle( w -> thread );
    CloseHandle( w -> stop_event );
    w -> stop_event = NULL;
#else
    pthread_mutex_lock( &w -> mutex );
    w -> stopping = 1;
    pthread_cond_broadcast( &w -> cond );
    pthread_mutex_unlock( &w -> mutex );

    pthread_join( w -> thread, NULL );
    pthread_cond_destroy( &w -> cond );
    pthread_mutex_destroy( &w -> mutex );
#endif

    w -> started = 0;
}

/* Per-adapter generator, so two workers never share generator state */
static double example_next_random( uint64_t * seed )
{
    uint64_t x = *seed;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *seed = x;
    return ( double )( ( x * 2685821657736338717ULL ) >> 11 ) / ( double )( 1ULL << 53 );
}

/* ============================================================================
 * Integer adapter state
 * ============================================================================ */

typedef struct {
    int interval_ms;
    int64_t counter;
    CCspPushInputAdapterHandle adapter;
    ExampleWorker worker;
} IntAdapterState;

static EXAMPLE_THREAD_RETURN int_adapter_thread( void * arg )
{
    IntAdapterState * state = ( IntAdapterState * ) arg;

    for( ;; )
    {
        ccsp_push_input_adapter_push_int64( state -> adapter, state -> counter, NULL );
        state -> counter++;

        if( example_worker_should_stop( &state -> worker, state -> interval_ms ) )
            break;
    }

    return EXAMPLE_THREAD_RESULT;
}

static void int_adapter_start( void * user_data, CCspEngineHandle engine,
                               CCspPushInputAdapterHandle adapter,
                               CCspDateTime start_time, CCspDateTime end_time )
{
    IntAdapterState * state = ( IntAdapterState * ) user_data;
    ( void ) engine;
    ( void ) start_time;
    ( void ) end_time;

    state -> adapter = adapter;
    state -> counter = 0;

    if( !example_worker_start( &state -> worker, int_adapter_thread, state ) )
    {
        ccsp_set_error( CCSP_ERROR_RUNTIME, "failed to start example int adapter thread" );
        fprintf( stderr, "[ExampleIntInputAdapter] Failed to start worker thread\n" );
        return;
    }

    fprintf( stdout, "[ExampleIntInputAdapter] Started with interval %d ms\n", state -> interval_ms );
}

static void int_adapter_stop( void * user_data )
{
    IntAdapterState * state = ( IntAdapterState * ) user_data;

    example_worker_stop( &state -> worker );

    fprintf( stdout, "[ExampleIntInputAdapter] Stopped after %lld values\n", ( long long ) state -> counter );
}

static void int_adapter_destroy( void * user_data )
{
    IntAdapterState * state = ( IntAdapterState * ) user_data;

    /* stop() is normally called first, but an engine torn down mid-startup goes straight here */
    example_worker_stop( &state -> worker );
    free( state );
}

CCspPushInputAdapterVTable example_push_input_adapter_create_int( int interval_ms )
{
    CCspPushInputAdapterVTable vtable;
    CCSP_VTABLE_INIT( &vtable, CCspPushInputAdapterVTable );

    IntAdapterState * state = ( IntAdapterState * )malloc( sizeof( IntAdapterState ) );
    if( !state )
    {
        return vtable;
    }

    memset( state, 0, sizeof( IntAdapterState ) );
    state -> interval_ms = interval_ms > 0 ? interval_ms : 100;

    vtable.user_data = state;
    vtable.start = int_adapter_start;
    vtable.stop = int_adapter_stop;
    vtable.destroy = int_adapter_destroy;

    return vtable;
}

/* ============================================================================
 * Double adapter state
 * ============================================================================ */

typedef struct {
    int interval_ms;
    uint64_t seed;
    CCspPushInputAdapterHandle adapter;
    ExampleWorker worker;
} DoubleAdapterState;

static EXAMPLE_THREAD_RETURN double_adapter_thread( void * arg )
{
    DoubleAdapterState * state = ( DoubleAdapterState * ) arg;

    for( ;; )
    {
        double value = example_next_random( &state -> seed );
        ccsp_push_input_adapter_push_double( state -> adapter, value, NULL );

        if( example_worker_should_stop( &state -> worker, state -> interval_ms ) )
            break;
    }

    return EXAMPLE_THREAD_RESULT;
}

static void double_adapter_start( void * user_data, CCspEngineHandle engine,
                                  CCspPushInputAdapterHandle adapter,
                                  CCspDateTime start_time, CCspDateTime end_time )
{
    DoubleAdapterState * state = ( DoubleAdapterState * ) user_data;
    ( void ) engine;
    ( void ) start_time;
    ( void ) end_time;

    state -> adapter = adapter;

    if( !example_worker_start( &state -> worker, double_adapter_thread, state ) )
    {
        ccsp_set_error( CCSP_ERROR_RUNTIME, "failed to start example double adapter thread" );
        fprintf( stderr, "[ExampleDoubleInputAdapter] Failed to start worker thread\n" );
        return;
    }

    fprintf( stdout, "[ExampleDoubleInputAdapter] Started\n" );
}

static void double_adapter_stop( void * user_data )
{
    DoubleAdapterState * state = ( DoubleAdapterState * ) user_data;

    example_worker_stop( &state -> worker );

    fprintf( stdout, "[ExampleDoubleInputAdapter] Stopped\n" );
}

static void double_adapter_destroy( void * user_data )
{
    DoubleAdapterState * state = ( DoubleAdapterState * ) user_data;

    /* stop() is normally called first, but an engine torn down mid-startup goes straight here */
    example_worker_stop( &state -> worker );
    free( state );
}

CCspPushInputAdapterVTable example_push_input_adapter_create_double( int interval_ms )
{
    CCspPushInputAdapterVTable vtable;
    CCSP_VTABLE_INIT( &vtable, CCspPushInputAdapterVTable );

    DoubleAdapterState * state = ( DoubleAdapterState * ) malloc( sizeof( DoubleAdapterState ) );
    if( !state )
    {
        return vtable;
    }

    memset( state, 0, sizeof( DoubleAdapterState ) );
    state -> interval_ms = interval_ms > 0 ? interval_ms : 100;
    state -> seed = 0x9e3779b97f4a7c15ULL;

    vtable.user_data = state;
    vtable.start = double_adapter_start;
    vtable.stop = double_adapter_stop;
    vtable.destroy = double_adapter_destroy;

    return vtable;
}

/* ============================================================================
 * String adapter (callback-based)
 * ============================================================================ */

typedef struct {
    ExampleStringCallback callback;
    void * callback_data;
    CCspPushInputAdapterHandle adapter;
} StringAdapterState;

static void string_adapter_start( void * user_data, CCspEngineHandle engine,
                                  CCspPushInputAdapterHandle adapter,
                                  CCspDateTime start_time, CCspDateTime end_time )
{
    StringAdapterState * state = ( StringAdapterState * ) user_data;
    ( void ) engine;
    ( void ) start_time;
    ( void ) end_time;

    state -> adapter = adapter;
    fprintf( stdout, "[ExampleStringInputAdapter] Started\n" );
}

static void string_adapter_stop( void * user_data )
{
    ( void )user_data;
    fprintf( stdout, "[ExampleStringInputAdapter] Stopped\n" );
}

static void string_adapter_destroy( void * user_data )
{
    StringAdapterState * state = ( StringAdapterState * ) user_data;
    free( state );
}

CCspPushInputAdapterVTable example_push_input_adapter_create_string( ExampleStringCallback get_string, void * user_data )
{
    CCspPushInputAdapterVTable vtable;
    CCSP_VTABLE_INIT( &vtable, CCspPushInputAdapterVTable );

    StringAdapterState * state = ( StringAdapterState * ) malloc( sizeof( StringAdapterState ) );
    if( !state )
    {
        return vtable;
    }

    memset( state, 0, sizeof( StringAdapterState ) );
    state -> callback = get_string;
    state -> callback_data = user_data;

    vtable.user_data = state;
    vtable.start = string_adapter_start;
    vtable.stop = string_adapter_stop;
    vtable.destroy = string_adapter_destroy;

    return vtable;
}
