/*
 * Example Managed Adapter using the CSP C API
 *
 * This example demonstrates:
 * - Creating an adapter manager that coordinates multiple adapters
 * - Managing lifecycle (start/stop) across adapters
 * - Status reporting
 * - Coordinated output adapters
 *
 * This is a more realistic example than ExampleOutputAdapter.c,
 * showing how real adapters like Kafka or WebSocket would be structured.
 */

#ifndef _IN_CSP_ADAPTERS_C_EXAMPLE_MANAGED_ADAPTER_H
#define _IN_CSP_ADAPTERS_C_EXAMPLE_MANAGED_ADAPTER_H

#include <csp/engine/c/AdapterManager.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * ManagedAdapterState - Shared state for managed adapters
 *
 * In a real adapter (like Kafka), this would contain:
 * - Connection handles
 * - Configuration
 * - Thread pools
 * - Message buffers
 */
typedef struct ManagedAdapterState {
    char name[64];
    int is_started;
    int message_count;
    CCspAdapterManagerHandle manager;
} ManagedAdapterState;

/*
 * ManagedOutputState - State for a single output adapter in the manager
 */
typedef struct ManagedOutputState {
    ManagedAdapterState * shared;
    char topic[64];       /* e.g., Kafka topic name */
    int messages_sent;
} ManagedOutputState;

/*
 * example_managed_adapter_create - Create managed adapter VTable
 *
 * Creates an adapter manager that can coordinate multiple output adapters.
 * Similar to how KafkaAdapterManager works.
 *
 * Parameters:
 *   name - Name for this adapter manager instance
 *
 * Returns:
 *   VTable for use with ccsp_adapter_manager_extern_create
 */
CCspAdapterManagerVTable example_managed_adapter_create( const char * name );

/*
 * example_managed_output_adapter_create - Create output adapter for manager
 *
 * Creates an output adapter that works with the managed adapter.
 *
 * Parameters:
 *   shared_state - Pointer to ManagedAdapterState from the manager
 *   topic        - Topic/channel name for this output
 *
 * Returns:
 *   VTable for use with ccsp_adapter_manager_create_output_adapter
 */
CCspOutputAdapterVTable example_managed_output_adapter_create(
    ManagedAdapterState * shared_state,
    const char * topic );

#ifdef __cplusplus
}
#endif

#endif /* _IN_CSP_ADAPTERS_C_EXAMPLE_MANAGED_ADAPTER_H */

#ifdef __cplusplus
}
#endif

#endif /* _IN_CSP_ADAPTERS_C_EXAMPLE_MANAGED_ADAPTER_H */
