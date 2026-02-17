/*
 * Example Push Input Adapter implemented in C
 *
 * This demonstrates how to implement a push input adapter using the C ABI interface.
 * The adapter generates periodic values for testing.
 */
#ifndef _IN_CSP_ADAPTERS_C_EXAMPLE_PUSH_INPUT_ADAPTER_H
#define _IN_CSP_ADAPTERS_C_EXAMPLE_PUSH_INPUT_ADAPTER_H

#include <csp/engine/c/InputAdapter.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Create an example push input adapter that generates incrementing integers.
 *
 * @param interval_ms  Interval between pushes in milliseconds
 * @return             VTable structure to pass to ccsp_push_input_adapter_extern_create
 */
CCspPushInputAdapterVTable example_push_input_adapter_create_int( int interval_ms );

/*
 * Create an example push input adapter that generates random doubles.
 *
 * @param interval_ms  Interval between pushes in milliseconds
 * @return             VTable structure
 */
CCspPushInputAdapterVTable example_push_input_adapter_create_double( int interval_ms );

/*
 * Create an example push input adapter that echoes strings from a callback.
 *
 * @param get_string   Callback to get the next string to push (must be thread-safe)
 * @param user_data    User data passed to the callback
 * @return             VTable structure
 */
typedef const char * ( *ExampleStringCallback )( void * user_data );
CCspPushInputAdapterVTable example_push_input_adapter_create_string(
    ExampleStringCallback get_string, void * user_data );

#ifdef __cplusplus
}
#endif

#endif /* _IN_CSP_ADAPTERS_C_EXAMPLE_PUSH_INPUT_ADAPTER_H */

#endif /* _IN_CSP_ADAPTERS_C_EXAMPLE_PUSH_INPUT_ADAPTER_H */
