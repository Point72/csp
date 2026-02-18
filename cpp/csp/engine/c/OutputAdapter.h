/*
 * ABI-stable C Output Adapter interface for CSP Engine
 *
 * Output adapters receive data from the CSP graph and send it to external systems.
 * External adapters implement the CCspOutputAdapterVTable callbacks.
 *
 * Lifecycle:
 *   1. Adapter created via ccsp_output_adapter_extern_create()
 *   2. start() called when graph starts
 *   3. execute() called each time input has new value
 *   4. stop() called when graph stops
 *   5. destroy() called to clean up
 */
#ifndef _IN_CSP_ENGINE_C_OUTPUTADAPTER_H
#define _IN_CSP_ENGINE_C_OUTPUTADAPTER_H

#include <csp/engine/c/CspExport.h>
#include <csp/engine/c/CspType.h>
#include <csp/engine/c/CspValue.h>
#include <csp/engine/c/CspTime.h>
#include <csp/engine/c/CspError.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Opaque handle types
 * ============================================================================ */

/* Handle to the CSP engine (opaque) */
typedef struct CCspEngineImpl * CCspEngineHandle;

/* Handle to a time series input (opaque) */
typedef struct CCspInputImpl * CCspInputHandle;

/* Handle to the internal C++ OutputAdapter wrapper (opaque) */
typedef struct CCspOutputAdapterImpl * CCspOutputAdapterHandle;

/* ============================================================================
 * Input access functions (for use in execute callback)
 * ============================================================================ */

/*
 * Check if the input is valid (has ticked at least once)
 */
CSP_C_API_EXPORT int ccsp_input_is_valid( CCspInputHandle input );

/*
 * Get the number of ticks available in the input buffer
 */
CSP_C_API_EXPORT int32_t ccsp_input_num_ticks( CCspInputHandle input );

/*
 * Get the type of the input
 */
CSP_C_API_EXPORT CCspType ccsp_input_get_type( CCspInputHandle input );

/*
 * Get the last value from the input.
 * The value is borrowed - do not free it, and do not use after execute() returns.
 */
CSP_C_API_EXPORT CCspErrorCode ccsp_input_get_last_value( CCspInputHandle input, CCspValue* out_value );

/*
 * Get value at a specific index in the buffer.
 * Index 0 is the most recent, negative indices go back in history.
 */
CSP_C_API_EXPORT CCspErrorCode ccsp_input_get_value_at( CCspInputHandle input, int32_t index, CCspValue * out_value );

/*
 * Get the timestamp of the value at a specific index.
 */
CSP_C_API_EXPORT CCspErrorCode ccsp_input_get_time_at( CCspInputHandle input, int32_t index, CCspDateTime * out_time );

/*
 * Get the timestamp of the last value.
 */
CSP_C_API_EXPORT CCspDateTime ccsp_input_get_last_time( CCspInputHandle input );

/* ============================================================================
 * Engine access functions
 * ============================================================================ */

/*
 * Get current engine time
 */
CSP_C_API_EXPORT CCspDateTime ccsp_engine_now( CCspEngineHandle engine );

/*
 * Get current cycle count
 */
CSP_C_API_EXPORT uint64_t ccsp_engine_cycle_count( CCspEngineHandle engine );

/* ============================================================================
 * Output Adapter Callbacks (VTable)
 *
 * External adapters must implement these callbacks.
 * ============================================================================ */

typedef struct CCspOutputAdapterVTable {
    /*
     * User data pointer passed to all callbacks.
     * This is typically a pointer to your adapter's state structure.
     */
    void * user_data;

    /*
     * Called when the graph starts.
     * Optional - set to NULL if not needed.
     *
     * @param user_data  Your adapter's state
     * @param engine     Handle to the engine (for accessing time, etc.)
     * @param start_time Graph start time
     * @param end_time   Graph end time
     */
    void ( * start ) ( void * user_data, CCspEngineHandle engine,
                       CCspDateTime start_time, CCspDateTime end_time );

    /*
     * Called when the graph stops.
     * Optional - set to NULL if not needed.
     * Use this to flush buffers, close connections, etc.
     *
     * @param user_data  Your adapter's state
     */
    void ( * stop ) ( void * user_data );
        
    /*
     * Called each time the input has a new value.
     * REQUIRED - must not be NULL.
     *
     * @param user_data  Your adapter's state
     * @param engine     Handle to the engine
     * @param input      Handle to the input time series
     */
    void ( * execute ) ( void * user_data, CCspEngineHandle engine, CCspInputHandle input );

    /*
     * Called to destroy the adapter and free resources.
     * REQUIRED - must not be NULL (even if it does nothing).
     *
     * @param user_data  Your adapter's state
     */
    void ( * destroy ) ( void * user_data );

} CCspOutputAdapterVTable;

/* ============================================================================
 * Output Adapter Creation
 * ============================================================================ */

/*
 * Create an external output adapter.
 *
 * @param engine     Engine handle (from adapter manager)
 * @param input_type Type of the input time series (CCSP_TYPE_*)
 * @param vtable     Callback table (copied, caller can free after this returns)
 * @return           Handle to the adapter, or NULL on error
 *
 * Note: The returned handle should be returned to Python via capsule,
 * which will then be registered with the CSP graph.
 */
CSP_C_API_EXPORT CCspOutputAdapterHandle ccsp_output_adapter_extern_create( CCspEngineHandle engine, CCspType input_type,
                                                                            const CCspOutputAdapterVTable * vtable );

/*
 * Destroy an external output adapter.
 * This is typically called by CSP when the graph is destroyed.
 * The destroy callback in the vtable will be invoked.
 */
CSP_C_API_EXPORT void ccsp_output_adapter_extern_destroy( CCspOutputAdapterHandle adapter );

/* ============================================================================
 * Convenience functions for common output patterns
 * ============================================================================ */

/*
 * Get last value as string (convenience for string outputs).
 * Returns borrowed pointer valid until next execute() call.
 */
CSP_C_API_EXPORT CCspErrorCode ccsp_input_get_last_string( CCspInputHandle input, const char ** out_data,
                                                           size_t * out_length );

/*
 * Get last value as int64 (convenience for int outputs)
 */
CSP_C_API_EXPORT CCspErrorCode ccsp_input_get_last_int64( CCspInputHandle input, int64_t * out_value );

/*
 * Get last value as double (convenience for double outputs)
 */
CSP_C_API_EXPORT CCspErrorCode ccsp_input_get_last_double( CCspInputHandle input, double * out_value );

/*
 * Get last value as bool (convenience for bool outputs)
 */
CSP_C_API_EXPORT CCspErrorCode ccsp_input_get_last_bool( CCspInputHandle input, int8_t * out_value );

/*
 * Get last value as datetime (convenience for datetime outputs)
 */
CSP_C_API_EXPORT CCspErrorCode ccsp_input_get_last_datetime( CCspInputHandle input, CCspDateTime * out_value );

#ifdef __cplusplus
}
#endif

#endif /* _IN_CSP_ENGINE_C_OUTPUTADAPTER_H */
