/*
 * C API for CSP Adapter Manager
 *
 * Adapter managers coordinate the lifecycle of a group of related adapters.
 * They handle:
 * - Starting and stopping all managed adapters together
 * - Simulation time slicing for sim input adapters
 * - Status reporting
 * - Push group coordination
 *
 * This header provides the C ABI for external adapter managers.
 */

#ifndef _IN_CSP_ENGINE_C_ADAPTER_MANAGER_H
#define _IN_CSP_ENGINE_C_ADAPTER_MANAGER_H

#include <csp/engine/c/CspError.h>
#include <csp/engine/c/CspTime.h>
#include <csp/engine/c/OutputAdapter.h>
#include <csp/engine/c/InputAdapter.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Opaque Handles
 * ============================================================================ */

typedef struct CCspAdapterManagerImpl* CCspAdapterManagerHandle;
typedef struct CCspStatusAdapterImpl* CCspStatusAdapterHandle;
typedef struct CCspManagedSimInputAdapterImpl* CCspManagedSimInputAdapterHandle;

/* ============================================================================
 * Adapter Manager VTable
 * ============================================================================
 *
 * This struct defines the callbacks that external adapter managers must
 * implement. The CSP engine will call these functions at appropriate times.
 *
 * Lifecycle:
 *   1. Manager is created with ccsp_adapter_manager_extern_create()
 *   2. CSP calls start() when the graph starts
 *   3. For sim mode: CSP calls process_next_sim_time_slice() repeatedly
 *   4. CSP calls stop() when the graph stops
 *   5. CSP calls destroy() to clean up
 */

typedef struct CCspAdapterManagerVTable {
    /* User-defined data pointer passed to all callbacks */
    void* user_data;

    /* ========================================================================
     * Required Callbacks
     * ======================================================================== */

    /*
     * name - Return the name of this adapter manager
     *
     * Used for logging and debugging. Must return a valid C string that
     * remains valid for the lifetime of the adapter manager.
     *
     * Parameters:
     *   user_data   - The user_data pointer from this vtable
     *
     * Returns:
     *   A null-terminated string naming this adapter manager
     */
    const char* (*name)(void* user_data);

    /*
     * process_next_sim_time_slice - Process simulation data for a time slice
     *
     * Called repeatedly during simulation mode to process data. Should:
     *   1. Process all data with timestamp equal to 'time'
     *   2. Return the next available timestamp, or 0 if no more data
     *
     * The first call is made with start_time. Subsequent calls use the
     * previously returned timestamp.
     *
     * For realtime adapters that don't support simulation, return 0.
     *
     * Parameters:
     *   user_data   - The user_data pointer from this vtable
     *   time        - The current simulation time to process
     *
     * Returns:
     *   Next timestamp with available data, or 0 if no more data
     */
    CCspDateTime (*process_next_sim_time_slice)(void* user_data, CCspDateTime time);

    /*
     * destroy - Clean up adapter manager resources
     *
     * Called when the adapter manager is being destroyed. Must free all
     * resources allocated in user_data.
     *
     * Parameters:
     *   user_data   - The user_data pointer from this vtable
     */
    void (*destroy)(void* user_data);

    /* ========================================================================
     * Optional Callbacks (can be NULL)
     * ======================================================================== */

    /*
     * start - Called when the graph starts
     *
     * Initialize connections, open files, start threads, etc.
     *
     * Parameters:
     *   user_data   - The user_data pointer from this vtable
     *   manager     - Handle to this adapter manager (for creating adapters)
     *   start_time  - Graph start time
     *   end_time    - Graph end time
     */
    void (*start)(void* user_data, CCspAdapterManagerHandle manager,
                  CCspDateTime start_time, CCspDateTime end_time);

    /*
     * stop - Called when the graph stops
     *
     * Close connections, flush buffers, stop threads, etc.
     *
     * Parameters:
     *   user_data   - The user_data pointer from this vtable
     */
    void (*stop)(void* user_data);

} CCspAdapterManagerVTable;

/* ============================================================================
 * Adapter Manager Creation and Lifecycle
 * ============================================================================ */

/*
 * ccsp_adapter_manager_extern_create - Create an external adapter manager
 *
 * Creates a new adapter manager that wraps C callbacks.
 *
 * Parameters:
 *   engine   - Engine handle (from Python capsule or parent manager)
 *   vtable   - Pointer to vtable with callbacks (copied internally)
 *
 * Returns:
 *   Handle to the new adapter manager
 */
CCspAdapterManagerHandle ccsp_adapter_manager_extern_create(
    CCspEngineHandle engine,
    const CCspAdapterManagerVTable* vtable);

/*
 * ccsp_adapter_manager_extern_destroy - Destroy an external adapter manager
 *
 * Calls the destroy callback and frees internal resources.
 *
 * Parameters:
 *   manager  - Handle to the adapter manager
 */
void ccsp_adapter_manager_extern_destroy(CCspAdapterManagerHandle manager);

/* ============================================================================
 * Engine and Time Access
 * ============================================================================ */

/*
 * ccsp_adapter_manager_engine - Get the engine handle
 *
 * Parameters:
 *   manager  - Handle to the adapter manager
 *
 * Returns:
 *   Engine handle for use with other C API functions
 */
CCspEngineHandle ccsp_adapter_manager_engine(CCspAdapterManagerHandle manager);

/*
 * ccsp_adapter_manager_start_time - Get graph start time
 *
 * Only valid after start() has been called.
 *
 * Parameters:
 *   manager  - Handle to the adapter manager
 *
 * Returns:
 *   Start time in nanoseconds since epoch
 */
CCspDateTime ccsp_adapter_manager_start_time(CCspAdapterManagerHandle manager);

/*
 * ccsp_adapter_manager_end_time - Get graph end time
 *
 * Only valid after start() has been called.
 *
 * Parameters:
 *   manager  - Handle to the adapter manager
 *
 * Returns:
 *   End time in nanoseconds since epoch
 */
CCspDateTime ccsp_adapter_manager_end_time(CCspAdapterManagerHandle manager);

/* ============================================================================
 * Adapter Creation from Manager
 * ============================================================================
 *
 * These functions create adapters that are managed by the adapter manager.
 * The manager handles their lifecycle automatically.
 */

/*
 * ccsp_adapter_manager_create_output_adapter - Create a managed output adapter
 *
 * Creates an output adapter that will be started/stopped with the manager.
 *
 * Parameters:
 *   manager     - Handle to the adapter manager
 *   input_type  - Type of input data the adapter will receive
 *   vtable      - Pointer to output adapter vtable (copied internally)
 *
 * Returns:
 *   Handle to the new output adapter
 */
CCspOutputAdapterHandle ccsp_adapter_manager_create_output_adapter(
    CCspAdapterManagerHandle manager,
    CCspType input_type,
    const CCspOutputAdapterVTable* vtable);

/*
 * ccsp_adapter_manager_create_push_input_adapter - Create a managed push input adapter
 *
 * Creates a push input adapter that will be started/stopped with the manager.
 *
 * Parameters:
 *   manager     - Handle to the adapter manager
 *   type        - Type of data the adapter will push
 *   push_mode   - Push mode (LAST_VALUE, NON_COLLAPSING, or BURST)
 *   vtable      - Pointer to input adapter vtable (copied internally)
 *
 * Returns:
 *   Handle to the new push input adapter
 */
CCspPushInputAdapterHandle ccsp_adapter_manager_create_push_input_adapter(
    CCspAdapterManagerHandle manager,
    CCspType type,
    CCspPushMode push_mode,
    const CCspPushInputAdapterVTable* vtable);

/* ============================================================================
 * Status Reporting
 * ============================================================================
 *
 * Adapter managers can report status to the graph via a status adapter.
 */

/* Status levels (matching csp.StatusLevel) */
typedef enum {
    CCSP_STATUS_LEVEL_CRITICAL = 0,
    CCSP_STATUS_LEVEL_ERROR = 1,
    CCSP_STATUS_LEVEL_WARNING = 2,
    CCSP_STATUS_LEVEL_INFO = 3,
    CCSP_STATUS_LEVEL_DEBUG = 4
} CCspStatusLevel;

/*
 * ccsp_adapter_manager_push_status - Report status to the graph
 *
 * Pushes a status message that can be consumed by nodes in the graph.
 *
 * Parameters:
 *   manager    - Handle to the adapter manager
 *   level      - Status level (CRITICAL, ERROR, WARNING, INFO, DEBUG)
 *   err_code   - Application-specific error code
 *   message    - Status message (copied internally)
 *
 * Returns:
 *   CCSP_OK on success, error code on failure
 */
CCspErrorCode ccsp_adapter_manager_push_status(
    CCspAdapterManagerHandle manager,
    CCspStatusLevel level,
    int64_t err_code,
    const char* message);

/* ============================================================================
 * Managed Simulation Input Adapter
 * ============================================================================
 *
 * For adapters that need to provide data in simulation mode, use managed
 * simulation input adapters. The adapter manager coordinates time slicing.
 */

/*
 * ccsp_adapter_manager_create_managed_sim_input_adapter - Create a sim input adapter
 *
 * Creates an input adapter for simulation mode. Use this for adapters that
 * read from static data sources (files, databases).
 *
 * Parameters:
 *   manager     - Handle to the adapter manager
 *   type        - Type of data the adapter will provide
 *   push_mode   - Push mode for handling multiple ticks
 *
 * Returns:
 *   Handle to the managed sim input adapter
 */
CCspManagedSimInputAdapterHandle ccsp_adapter_manager_create_managed_sim_input_adapter(
    CCspAdapterManagerHandle manager,
    CCspType type,
    CCspPushMode push_mode);

/*
 * ccsp_managed_sim_input_adapter_push_* - Push data from simulation source
 *
 * These functions push typed data into the simulation input adapter.
 * Call these from process_next_sim_time_slice to provide data.
 *
 * Parameters:
 *   adapter - Handle to the managed sim input adapter
 *   value   - Value to push
 *
 * Returns:
 *   CCSP_OK on success, error code on failure
 */
CCspErrorCode ccsp_managed_sim_input_adapter_push_bool(
    CCspManagedSimInputAdapterHandle adapter, int8_t value);
CCspErrorCode ccsp_managed_sim_input_adapter_push_int64(
    CCspManagedSimInputAdapterHandle adapter, int64_t value);
CCspErrorCode ccsp_managed_sim_input_adapter_push_double(
    CCspManagedSimInputAdapterHandle adapter, double value);
CCspErrorCode ccsp_managed_sim_input_adapter_push_string(
    CCspManagedSimInputAdapterHandle adapter, const char* data, size_t length);
CCspErrorCode ccsp_managed_sim_input_adapter_push_datetime(
    CCspManagedSimInputAdapterHandle adapter, CCspDateTime value);

#ifdef __cplusplus
}
#endif

#endif /* _IN_CSP_ENGINE_C_ADAPTER_MANAGER_H */
