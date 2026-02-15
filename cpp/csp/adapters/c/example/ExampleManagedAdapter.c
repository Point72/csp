/*
 * Example Managed Adapter Implementation
 *
 * Demonstrates how to build an adapter manager that coordinates
 * multiple input/output adapters, similar to KafkaAdapterManager.
 */

#include "ExampleManagedAdapter.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ============================================================================
 * Adapter Manager Callbacks
 * ============================================================================ */

static const char* managed_adapter_name(void* user_data)
{
    ManagedAdapterState* state = (ManagedAdapterState*)user_data;
    return state->name;
}

static void managed_adapter_start(void* user_data, CCspAdapterManagerHandle manager,
                                  CCspDateTime start_time, CCspDateTime end_time)
{
    ManagedAdapterState* state = (ManagedAdapterState*)user_data;
    state->manager = manager;
    state->is_started = 1;
    state->message_count = 0;

    /* Calculate time in seconds for display */
    double start_sec = (double)start_time / 1000000000.0;
    double end_sec = (double)end_time / 1000000000.0;

    printf("[%s] Manager started (start=%.3f, end=%.3f)\n",
           state->name, start_sec, end_sec);

    /* Report status to the graph */
    ccsp_adapter_manager_push_status(manager, CCSP_STATUS_LEVEL_INFO, 0,
                                     "Manager started successfully");
}

static void managed_adapter_stop(void* user_data)
{
    ManagedAdapterState* state = (ManagedAdapterState*)user_data;

    printf("[%s] Manager stopped. Total messages: %d\n",
           state->name, state->message_count);

    state->is_started = 0;
}

static CCspDateTime managed_adapter_process_next_sim_time_slice(void* user_data,
                                                                  CCspDateTime time)
{
    /* This example is realtime-only, so we return 0 (no more sim data) */
    (void)user_data;
    (void)time;
    return 0;
}

static void managed_adapter_destroy(void* user_data)
{
    ManagedAdapterState* state = (ManagedAdapterState*)user_data;
    printf("[%s] Manager destroyed\n", state->name);
    free(state);
}

CCspAdapterManagerVTable example_managed_adapter_create(const char* name)
{
    ManagedAdapterState* state = (ManagedAdapterState*)malloc(sizeof(ManagedAdapterState));
    if (!state) {
        CCspAdapterManagerVTable empty = {0};
        return empty;
    }

    memset(state, 0, sizeof(ManagedAdapterState));
    if (name) {
        strncpy(state->name, name, sizeof(state->name) - 1);
    } else {
        strncpy(state->name, "ExampleManagedAdapter", sizeof(state->name) - 1);
    }

    CCspAdapterManagerVTable vtable;
    vtable.user_data = state;
    vtable.name = managed_adapter_name;
    vtable.start = managed_adapter_start;
    vtable.stop = managed_adapter_stop;
    vtable.process_next_sim_time_slice = managed_adapter_process_next_sim_time_slice;
    vtable.destroy = managed_adapter_destroy;

    return vtable;
}

/* ============================================================================
 * Managed Output Adapter Callbacks
 * ============================================================================ */

static void managed_output_start(void* user_data, CCspEngineHandle engine,
                                  CCspDateTime start_time, CCspDateTime end_time)
{
    ManagedOutputState* state = (ManagedOutputState*)user_data;
    (void)engine;
    (void)start_time;
    (void)end_time;

    printf("  [%s/%s] Output adapter started\n",
           state->shared->name, state->topic);
    state->messages_sent = 0;
}

static void managed_output_stop(void* user_data)
{
    ManagedOutputState* state = (ManagedOutputState*)user_data;
    printf("  [%s/%s] Output adapter stopped. Messages sent: %d\n",
           state->shared->name, state->topic, state->messages_sent);
}

static void managed_output_execute(void* user_data, CCspEngineHandle engine,
                                    CCspInputHandle input)
{
    ManagedOutputState* state = (ManagedOutputState*)user_data;

    if (!ccsp_input_is_valid(input)) {
        return;
    }

    /* Get current engine time */
    CCspDateTime now = ccsp_engine_now(engine);
    double now_sec = (double)now / 1000000000.0;

    /* Get the input type and value */
    CCspType type = ccsp_input_get_type(input);

    printf("  [%s/%s] t=%.3f -> ", state->shared->name, state->topic, now_sec);

    switch (type) {
        case CCSP_TYPE_INT64: {
            int64_t value;
            if (ccsp_input_get_last_int64(input, &value) == CCSP_OK) {
                printf("int64: %lld\n", (long long)value);
            }
            break;
        }
        case CCSP_TYPE_DOUBLE: {
            double value;
            if (ccsp_input_get_last_double(input, &value) == CCSP_OK) {
                printf("double: %.6f\n", value);
            }
            break;
        }
        case CCSP_TYPE_STRING: {
            const char* data;
            size_t length;
            if (ccsp_input_get_last_string(input, &data, &length) == CCSP_OK) {
                printf("string: \"%.*s\"\n", (int)length, data);
            }
            break;
        }
        default:
            printf("(type %d)\n", type);
            break;
    }

    state->messages_sent++;
    state->shared->message_count++;
}

static void managed_output_destroy(void* user_data)
{
    ManagedOutputState* state = (ManagedOutputState*)user_data;
    printf("  [%s/%s] Output adapter destroyed\n",
           state->shared->name, state->topic);
    free(state);
}

CCspOutputAdapterVTable example_managed_output_adapter_create(
    ManagedAdapterState* shared_state,
    const char* topic)
{
    CCspOutputAdapterVTable vtable = {0};

    if (!shared_state) {
        return vtable;
    }

    ManagedOutputState* state = (ManagedOutputState*)malloc(sizeof(ManagedOutputState));
    if (!state) {
        return vtable;
    }

    memset(state, 0, sizeof(ManagedOutputState));
    state->shared = shared_state;
    if (topic) {
        strncpy(state->topic, topic, sizeof(state->topic) - 1);
    } else {
        strncpy(state->topic, "default", sizeof(state->topic) - 1);
    }

    vtable.user_data = state;
    vtable.start = managed_output_start;
    vtable.stop = managed_output_stop;
    vtable.execute = managed_output_execute;
    vtable.destroy = managed_output_destroy;

    return vtable;
}
