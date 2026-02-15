/*
 * Example Push Input Adapter implementation in C
 *
 * This demonstrates how to implement a push input adapter using the C ABI interface.
 * Note: This is a simplified example. A real adapter would use proper threading.
 */
#include "ExamplePushInputAdapter.h"
#include <csp/engine/c/InputAdapter.h>
#include <csp/engine/c/CspError.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <pthread.h>
#include <unistd.h>
#endif

/* ============================================================================
 * Integer adapter state
 * ============================================================================ */

typedef struct {
    int interval_ms;
    int64_t counter;
    int running;
    CCspPushInputAdapterHandle adapter;
#ifdef _WIN32
    HANDLE thread;
#else
    pthread_t thread;
#endif
} IntAdapterState;

static void* int_adapter_thread(void* arg)
{
    IntAdapterState* state = (IntAdapterState*)arg;

    while (state->running) {
        /* Push the current counter value */
        ccsp_push_input_adapter_push_int64(state->adapter, state->counter, NULL);
        state->counter++;

        /* Sleep for the interval */
#ifdef _WIN32
        Sleep(state->interval_ms);
#else
        usleep(state->interval_ms * 1000);
#endif
    }

    return NULL;
}

static void int_adapter_start(void* user_data, CCspEngineHandle engine,
                               CCspPushInputAdapterHandle adapter,
                               CCspDateTime start_time, CCspDateTime end_time)
{
    IntAdapterState* state = (IntAdapterState*)user_data;
    (void)engine;
    (void)start_time;
    (void)end_time;

    state->adapter = adapter;
    state->running = 1;
    state->counter = 0;

#ifdef _WIN32
    state->thread = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)int_adapter_thread,
                                  state, 0, NULL);
#else
    pthread_create(&state->thread, NULL, int_adapter_thread, state);
#endif

    fprintf(stdout, "[ExampleIntInputAdapter] Started with interval %d ms\n",
            state->interval_ms);
}

static void int_adapter_stop(void* user_data)
{
    IntAdapterState* state = (IntAdapterState*)user_data;
    state->running = 0;

#ifdef _WIN32
    WaitForSingleObject(state->thread, INFINITE);
    CloseHandle(state->thread);
#else
    pthread_join(state->thread, NULL);
#endif

    fprintf(stdout, "[ExampleIntInputAdapter] Stopped after %lld values\n",
            (long long)state->counter);
}

static void int_adapter_destroy(void* user_data)
{
    IntAdapterState* state = (IntAdapterState*)user_data;
    free(state);
}

CCspPushInputAdapterVTable example_push_input_adapter_create_int(int interval_ms)
{
    CCspPushInputAdapterVTable vtable = {0};

    IntAdapterState* state = (IntAdapterState*)malloc(sizeof(IntAdapterState));
    if (!state) {
        return vtable;
    }

    memset(state, 0, sizeof(IntAdapterState));
    state->interval_ms = interval_ms > 0 ? interval_ms : 100;

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
    int running;
    CCspPushInputAdapterHandle adapter;
#ifdef _WIN32
    HANDLE thread;
#else
    pthread_t thread;
#endif
} DoubleAdapterState;

static void* double_adapter_thread(void* arg)
{
    DoubleAdapterState* state = (DoubleAdapterState*)arg;

    while (state->running) {
        /* Generate a random double between 0 and 1 */
        double value = (double)rand() / (double)RAND_MAX;
        ccsp_push_input_adapter_push_double(state->adapter, value, NULL);

#ifdef _WIN32
        Sleep(state->interval_ms);
#else
        usleep(state->interval_ms * 1000);
#endif
    }

    return NULL;
}

static void double_adapter_start(void* user_data, CCspEngineHandle engine,
                                  CCspPushInputAdapterHandle adapter,
                                  CCspDateTime start_time, CCspDateTime end_time)
{
    DoubleAdapterState* state = (DoubleAdapterState*)user_data;
    (void)engine;
    (void)start_time;
    (void)end_time;

    state->adapter = adapter;
    state->running = 1;

#ifdef _WIN32
    state->thread = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)double_adapter_thread,
                                  state, 0, NULL);
#else
    pthread_create(&state->thread, NULL, double_adapter_thread, state);
#endif

    fprintf(stdout, "[ExampleDoubleInputAdapter] Started\n");
}

static void double_adapter_stop(void* user_data)
{
    DoubleAdapterState* state = (DoubleAdapterState*)user_data;
    state->running = 0;

#ifdef _WIN32
    WaitForSingleObject(state->thread, INFINITE);
    CloseHandle(state->thread);
#else
    pthread_join(state->thread, NULL);
#endif

    fprintf(stdout, "[ExampleDoubleInputAdapter] Stopped\n");
}

static void double_adapter_destroy(void* user_data)
{
    DoubleAdapterState* state = (DoubleAdapterState*)user_data;
    free(state);
}

CCspPushInputAdapterVTable example_push_input_adapter_create_double(int interval_ms)
{
    CCspPushInputAdapterVTable vtable = {0};

    DoubleAdapterState* state = (DoubleAdapterState*)malloc(sizeof(DoubleAdapterState));
    if (!state) {
        return vtable;
    }

    memset(state, 0, sizeof(DoubleAdapterState));
    state->interval_ms = interval_ms > 0 ? interval_ms : 100;

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
    void* callback_data;
    CCspPushInputAdapterHandle adapter;
} StringAdapterState;

static void string_adapter_start(void* user_data, CCspEngineHandle engine,
                                  CCspPushInputAdapterHandle adapter,
                                  CCspDateTime start_time, CCspDateTime end_time)
{
    StringAdapterState* state = (StringAdapterState*)user_data;
    (void)engine;
    (void)start_time;
    (void)end_time;

    state->adapter = adapter;
    fprintf(stdout, "[ExampleStringInputAdapter] Started\n");
}

static void string_adapter_stop(void* user_data)
{
    (void)user_data;
    fprintf(stdout, "[ExampleStringInputAdapter] Stopped\n");
}

static void string_adapter_destroy(void* user_data)
{
    StringAdapterState* state = (StringAdapterState*)user_data;
    free(state);
}

CCspPushInputAdapterVTable example_push_input_adapter_create_string(
    ExampleStringCallback get_string, void* user_data)
{
    CCspPushInputAdapterVTable vtable = {0};

    StringAdapterState* state = (StringAdapterState*)malloc(sizeof(StringAdapterState));
    if (!state) {
        return vtable;
    }

    memset(state, 0, sizeof(StringAdapterState));
    state->callback = get_string;
    state->callback_data = user_data;

    vtable.user_data = state;
    vtable.start = string_adapter_start;
    vtable.stop = string_adapter_stop;
    vtable.destroy = string_adapter_destroy;

    return vtable;
}
