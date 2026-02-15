/*
 * ABI-stable C Input Adapter interface for CSP Engine
 *
 * Input adapters push data into the CSP graph from external sources.
 * External adapters implement the CCspPushInputAdapterVTable callbacks.
 *
 * Lifecycle:
 *   1. Adapter created via ccsp_push_input_adapter_extern_create()
 *   2. start() called when graph starts - adapter can start its data source
 *   3. Adapter calls ccsp_push_input_adapter_push_*() to push data (thread-safe)
 *   4. stop() called when graph stops - adapter should stop its data source
 *   5. destroy() called to clean up
 *
 * Thread Safety:
 *   The push functions are thread-safe and can be called from any thread.
 *   All other functions must be called from the engine thread.
 */
#ifndef _IN_CSP_ENGINE_C_INPUTADAPTER_H
#define _IN_CSP_ENGINE_C_INPUTADAPTER_H

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

/* Handle to the CSP engine (opaque) - same as in OutputAdapter.h */
typedef struct CCspEngineImpl* CCspEngineHandle;

/* Handle to the internal C++ PushInputAdapter wrapper (opaque) */
typedef struct CCspPushInputAdapterImpl* CCspPushInputAdapterHandle;

/* Handle to a push batch for grouping events (opaque) */
typedef struct CCspPushBatchImpl* CCspPushBatchHandle;

/* Handle to a push group for synchronizing adapters (opaque) */
typedef struct CCspPushGroupImpl* CCspPushGroupHandle;

/* ============================================================================
 * Push Mode
 * ============================================================================ */

typedef enum {
    /* Only the last value per engine cycle is kept (values collapse) */
    CCSP_PUSH_MODE_LAST_VALUE = 0,

    /* Every value creates a separate engine cycle (no collapsing) */
    CCSP_PUSH_MODE_NON_COLLAPSING = 1,

    /* Values are batched into a vector per engine cycle */
    CCSP_PUSH_MODE_BURST = 2
} CCspPushMode;

/* ============================================================================
 * Push Input Adapter Callbacks (VTable)
 *
 * External adapters must implement these callbacks.
 * ============================================================================ */

typedef struct CCspPushInputAdapterVTable {
    /*
     * User data pointer passed to all callbacks.
     * This is typically a pointer to your adapter's state structure.
     */
    void* user_data;

    /*
     * Called when the graph starts.
     * Use this to start your data source (threads, connections, etc.)
     * Optional - set to NULL if not needed.
     *
     * @param user_data  Your adapter's state
     * @param engine     Handle to the engine
     * @param adapter    Handle to this adapter (for pushing data)
     * @param start_time Graph start time
     * @param end_time   Graph end time
     */
    void (*start)(void* user_data, CCspEngineHandle engine,
                  CCspPushInputAdapterHandle adapter,
                  CCspDateTime start_time, CCspDateTime end_time);

    /*
     * Called when the graph stops.
     * Use this to stop your data source.
     * Optional - set to NULL if not needed.
     *
     * @param user_data  Your adapter's state
     */
    void (*stop)(void* user_data);

    /*
     * Called to destroy the adapter and free resources.
     * REQUIRED - must not be NULL (even if it does nothing).
     *
     * @param user_data  Your adapter's state
     */
    void (*destroy)(void* user_data);

} CCspPushInputAdapterVTable;

/* ============================================================================
 * Push Input Adapter Creation
 * ============================================================================ */

/*
 * Create an external push input adapter.
 *
 * @param engine     Engine handle (from adapter manager)
 * @param type       Type of data this adapter will push
 * @param push_mode  How to handle multiple values per cycle
 * @param group      Optional push group for synchronization (can be NULL)
 * @param vtable     Callback table (copied, caller can free after this returns)
 * @return           Handle to the adapter, or NULL on error
 */
CCspPushInputAdapterHandle ccsp_push_input_adapter_extern_create(
    CCspEngineHandle engine,
    CCspType type,
    CCspPushMode push_mode,
    CCspPushGroupHandle group,
    const CCspPushInputAdapterVTable* vtable
);

/*
 * Destroy an external push input adapter.
 * This is typically called by CSP when the graph is destroyed.
 */
void ccsp_push_input_adapter_extern_destroy(CCspPushInputAdapterHandle adapter);

/* ============================================================================
 * Push Functions (Thread-Safe)
 *
 * Call these from your data source thread to push data into the graph.
 * ============================================================================ */

/*
 * Push a generic value.
 * The value is copied; caller retains ownership of the CCspValue.
 *
 * @param adapter  Handle to the adapter
 * @param value    Value to push
 * @param batch    Optional batch handle (can be NULL for unbatched push)
 */
CCspErrorCode ccsp_push_input_adapter_push_value(
    CCspPushInputAdapterHandle adapter,
    const CCspValue* value,
    CCspPushBatchHandle batch
);

/* Type-specific push functions (more efficient, avoid CCspValue overhead) */

CCspErrorCode ccsp_push_input_adapter_push_bool(
    CCspPushInputAdapterHandle adapter, int8_t value, CCspPushBatchHandle batch);

CCspErrorCode ccsp_push_input_adapter_push_int8(
    CCspPushInputAdapterHandle adapter, int8_t value, CCspPushBatchHandle batch);

CCspErrorCode ccsp_push_input_adapter_push_uint8(
    CCspPushInputAdapterHandle adapter, uint8_t value, CCspPushBatchHandle batch);

CCspErrorCode ccsp_push_input_adapter_push_int16(
    CCspPushInputAdapterHandle adapter, int16_t value, CCspPushBatchHandle batch);

CCspErrorCode ccsp_push_input_adapter_push_uint16(
    CCspPushInputAdapterHandle adapter, uint16_t value, CCspPushBatchHandle batch);

CCspErrorCode ccsp_push_input_adapter_push_int32(
    CCspPushInputAdapterHandle adapter, int32_t value, CCspPushBatchHandle batch);

CCspErrorCode ccsp_push_input_adapter_push_uint32(
    CCspPushInputAdapterHandle adapter, uint32_t value, CCspPushBatchHandle batch);

CCspErrorCode ccsp_push_input_adapter_push_int64(
    CCspPushInputAdapterHandle adapter, int64_t value, CCspPushBatchHandle batch);

CCspErrorCode ccsp_push_input_adapter_push_uint64(
    CCspPushInputAdapterHandle adapter, uint64_t value, CCspPushBatchHandle batch);

CCspErrorCode ccsp_push_input_adapter_push_double(
    CCspPushInputAdapterHandle adapter, double value, CCspPushBatchHandle batch);

CCspErrorCode ccsp_push_input_adapter_push_datetime(
    CCspPushInputAdapterHandle adapter, CCspDateTime value, CCspPushBatchHandle batch);

CCspErrorCode ccsp_push_input_adapter_push_timedelta(
    CCspPushInputAdapterHandle adapter, CCspTimeDelta value, CCspPushBatchHandle batch);

/*
 * Push a string value.
 * The string data is copied internally.
 */
CCspErrorCode ccsp_push_input_adapter_push_string(
    CCspPushInputAdapterHandle adapter,
    const char* data, size_t length,
    CCspPushBatchHandle batch
);

/*
 * Push a struct value.
 * The struct is copied internally.
 */
CCspErrorCode ccsp_push_input_adapter_push_struct(
    CCspPushInputAdapterHandle adapter,
    CCspStructHandle value,
    CCspPushBatchHandle batch
);

/* ============================================================================
 * Push Batch Management
 *
 * Batches group multiple push events to be processed atomically.
 * ============================================================================ */

/*
 * Create a push batch.
 * Events added to a batch are held until the batch is flushed.
 *
 * @param engine  Engine handle
 * @return        Handle to the batch, or NULL on error
 */
CCspPushBatchHandle ccsp_push_batch_create(CCspEngineHandle engine);

/*
 * Flush a push batch, releasing all pending events to the engine.
 * The batch can be reused after flushing.
 */
void ccsp_push_batch_flush(CCspPushBatchHandle batch);

/*
 * Destroy a push batch.
 * Any unflushed events are flushed before destruction.
 */
void ccsp_push_batch_destroy(CCspPushBatchHandle batch);

/* ============================================================================
 * Push Group Management
 *
 * Groups synchronize multiple input adapters so they don't get out of sync.
 * ============================================================================ */

/*
 * Create a push group.
 *
 * @return  Handle to the group, or NULL on error
 */
CCspPushGroupHandle ccsp_push_group_create(void);

/*
 * Destroy a push group.
 */
void ccsp_push_group_destroy(CCspPushGroupHandle group);

#ifdef __cplusplus
}
#endif

#endif /* _IN_CSP_ENGINE_C_INPUTADAPTER_H */
