/*
 * Example Output Adapter implemented in C
 *
 * This demonstrates how to implement an output adapter using the C ABI interface.
 * The adapter simply prints received values to stdout.
 */
#ifndef _IN_CSP_ADAPTERS_C_EXAMPLE_OUTPUT_ADAPTER_H
#define _IN_CSP_ADAPTERS_C_EXAMPLE_OUTPUT_ADAPTER_H

#include <csp/engine/c/OutputAdapter.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Create an example output adapter.
 *
 * @param prefix  Prefix string to print before each value (can be NULL)
 * @return        VTable structure to pass to ccsp_output_adapter_extern_create
 */
CCspOutputAdapterVTable example_output_adapter_create( const char * prefix );

/*
 * Alternative: Get an adapter that logs to a specific file descriptor.
 *
 * @param fd      File descriptor to write to
 * @param prefix  Prefix string (can be NULL)
 * @return        VTable structure
 */
CCspOutputAdapterVTable example_output_adapter_create_fd( int fd, const char * prefix );

#ifdef __cplusplus
}
#endif

#endif /* _IN_CSP_ADAPTERS_C_EXAMPLE_OUTPUT_ADAPTER_H */

