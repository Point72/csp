# CSP C API Roadmap

## Overview

This document outlines the roadmap for completing the C API that enables CSP adapters (Kafka, Parquet, WebSocket) to be compiled and distributed separately from CSP core.

## Current State Analysis

### What Has Been Done

| File | Status | Description |
|------|--------|-------------|
| `cpp/csp/engine/c/CspType.h` | ✅ Basic | CCspType enum with all basic type definitions |
| `cpp/csp/engine/c/OutputAdapter.h` | 🔶 Stub | Empty opaque OutputAdapter struct |
| `cpp/csp/engine/OutputAdapterExtern.h` | 🔶 Stub | C++ wrapper class, no implementation |
| `cpp/csp/python/c/PyOutputAdapter.h` | ✅ Partial | Python capsule creation for C adapters |
| `cpp/csp/adapters/c/example/*` | 🔶 Stub | Empty example adapter files |
| `cpp/csp/python/adapters/c/exampleadapterimpl.c` | 🔶 Stub | Basic Python module structure |
| `csp/adapters/c_example.py` | ✅ Complete | Python wiring definitions |

### Adapter Dependencies Analysis

Analyzing the Kafka, Parquet, and WebSocket adapters reveals these CSP core dependencies:

#### Output Adapter Dependencies
- `Engine*` - for scheduling
- `TimeSeriesProvider::lastValueTyped<T>()` - reading input values
- `CspType` - type information
- `Dictionary` - configuration properties
- `Struct` - structured data access
- `DateTime/TimeDelta` - time handling

#### Input Adapter Dependencies
- `Engine*` - for scheduling
- `PushInputAdapter::pushTick<T>()` - pushing data into graph
- `PushBatch/PushGroup` - batching and synchronization
- `CspType` - type information
- `StructMeta/StructField` - struct metadata access
- `DateTime/TimeDelta` - time handling

#### Adapter Manager Dependencies
- `AdapterManager` base class - lifecycle (start/stop)
- `RootEngine` - root engine access
- `StatusAdapter` - status reporting
- `Dictionary` - configuration

---

## Phase 1: Core C Interface Types

**Goal:** Define ABI-stable C representations of all CSP types

### 1.1 Basic Value Types (`cpp/csp/engine/c/CspValue.h`)

```c
// Already defined: CCspType enum

// Value container for passing data across ABI
typedef struct {
    CCspType type;
    union {
        int8_t   bool_val;  // CCSP_TYPE_BOOL
        int8_t   int8_val;
        uint8_t  uint8_val;
        int16_t  int16_val;
        uint16_t uint16_val;
        int32_t  int32_val;
        uint32_t uint32_val;
        int64_t  int64_val;
        uint64_t uint64_val;
        double   double_val;
        struct { const char* data; size_t length; } string_val;
        int64_t  datetime_val;   // nanoseconds since epoch
        int64_t  timedelta_val;  // nanoseconds
        int32_t  date_val;       // days since epoch
        int64_t  time_val;       // nanoseconds since midnight
        int32_t  enum_val;       // enum ordinal
        void*    struct_ptr;     // opaque pointer + metadata
        void*    dialect_ptr;    // opaque dialect object
        struct { void* data; size_t length; CCspType elem_type; } array_val;
    };
} CCspValue;
```

### 1.2 Time Types (`cpp/csp/engine/c/CspTime.h`)

```c
typedef int64_t CCspDateTime;   // nanoseconds since Unix epoch
typedef int64_t CCspTimeDelta;  // nanoseconds duration
typedef int32_t CCspDate;       // days since Unix epoch
typedef int64_t CCspTime;       // nanoseconds since midnight

// Conversion functions
CCspDateTime ccsp_datetime_now();
CCspDateTime ccsp_datetime_from_parts(int year, int month, int day, int hour, int min, int sec, int nsec);
CCspTimeDelta ccsp_timedelta_from_seconds(double seconds);
```

### 1.3 String Type (`cpp/csp/engine/c/CspString.h`)

```c
typedef struct {
    const char* data;
    size_t length;
    int is_owned;  // 1 if the C side owns the memory
} CCspString;

CCspString ccsp_string_create(const char* data, size_t length);
void ccsp_string_free(CCspString* str);
```

### 1.4 Type Metadata (`cpp/csp/engine/c/CspTypeMeta.h`)

```c
// Opaque handles
typedef struct CCspTypeInfo* CCspTypeHandle;
typedef struct CCspStructMeta* CCspStructMetaHandle;
typedef struct CCspStructField* CCspStructFieldHandle;

// Type introspection
CCspType ccsp_type_get_kind(CCspTypeHandle type);
int ccsp_type_is_array(CCspTypeHandle type);
CCspTypeHandle ccsp_type_array_elem_type(CCspTypeHandle type);

// Struct introspection
const char* ccsp_struct_meta_name(CCspStructMetaHandle meta);
size_t ccsp_struct_meta_field_count(CCspStructMetaHandle meta);
CCspStructFieldHandle ccsp_struct_meta_field_at(CCspStructMetaHandle meta, size_t index);
CCspStructFieldHandle ccsp_struct_meta_field_by_name(CCspStructMetaHandle meta, const char* name);

// Field introspection
const char* ccsp_struct_field_name(CCspStructFieldHandle field);
CCspTypeHandle ccsp_struct_field_type(CCspStructFieldHandle field);
```

---

## Phase 2: Output Adapter C Interface

**Goal:** Complete C interface for output adapters

### 2.1 Output Adapter Interface (`cpp/csp/engine/c/OutputAdapter.h`)

```c
// Forward declarations
typedef struct CCspEngine CCspEngine;
typedef struct CCspTimeSeriesProvider CCspTimeSeriesProvider;

// Output adapter lifecycle callbacks (implemented by external adapter)
typedef struct {
    void* user_data;

    // Called when adapter should execute (input has new value)
    void (*execute)(void* user_data, CCspTimeSeriesProvider* input, CCspDateTime now);

    // Called on graph start
    void (*start)(void* user_data, CCspDateTime start_time, CCspDateTime end_time);

    // Called on graph stop/cleanup
    void (*stop)(void* user_data);

    // Destructor
    void (*destroy)(void* user_data);

} CCspOutputAdapterCallbacks;

// Registration
typedef struct CCspOutputAdapter CCspOutputAdapter;

CCspOutputAdapter* ccsp_output_adapter_create(
    CCspEngine* engine,
    CCspTypeHandle input_type,
    CCspOutputAdapterCallbacks callbacks
);

void ccsp_output_adapter_destroy(CCspOutputAdapter* adapter);
```

### 2.2 Input Value Access (`cpp/csp/engine/c/CspInput.h`)

```c
// Get last value from input time series
int ccsp_input_get_value(CCspTimeSeriesProvider* input, CCspValue* out_value);

// Check if input is valid (has ticked)
int ccsp_input_is_valid(CCspTimeSeriesProvider* input);

// Get tick count
int ccsp_input_num_ticks(CCspTimeSeriesProvider* input);

// Historical access
int ccsp_input_get_value_at_index(CCspTimeSeriesProvider* input, int32_t index, CCspValue* out_value);
CCspDateTime ccsp_input_get_time_at_index(CCspTimeSeriesProvider* input, int32_t index);
```

### 2.3 C++ Wrapper Implementation (`cpp/csp/engine/OutputAdapterExtern.cpp`)

```cpp
class OutputAdapterExtern : public OutputAdapter {
public:
    OutputAdapterExtern(Engine* engine, const CspTypePtr& type,
                        CCspOutputAdapterCallbacks callbacks);
    ~OutputAdapterExtern() override;

    void executeImpl() override {
        CCspTimeSeriesProvider* c_input = wrapTimeSeriesProvider(input());
        m_callbacks.execute(m_callbacks.user_data, c_input, now().asNanoseconds());
    }

    void start() override {
        if (m_callbacks.start) {
            m_callbacks.start(m_callbacks.user_data,
                             m_startTime.asNanoseconds(),
                             m_endTime.asNanoseconds());
        }
    }

    void stop() override {
        if (m_callbacks.stop) {
            m_callbacks.stop(m_callbacks.user_data);
        }
    }

private:
    CCspOutputAdapterCallbacks m_callbacks;
};
```

---

## Phase 3: Input Adapter C Interface

**Goal:** Complete C interface for push input adapters

### 3.1 Push Input Adapter Interface (`cpp/csp/engine/c/InputAdapter.h`)

```c
typedef struct CCspPushInputAdapter CCspPushInputAdapter;
typedef struct CCspPushBatch CCspPushBatch;
typedef struct CCspPushGroup CCspPushGroup;

// Input adapter lifecycle callbacks (implemented by external adapter)
typedef struct {
    void* user_data;

    // Called on graph start
    void (*start)(void* user_data, CCspDateTime start_time, CCspDateTime end_time);

    // Called on graph stop
    void (*stop)(void* user_data);

    // Destructor
    void (*destroy)(void* user_data);

} CCspPushInputAdapterCallbacks;

// Creation
CCspPushInputAdapter* ccsp_push_input_adapter_create(
    CCspEngine* engine,
    CCspTypeHandle type,
    int push_mode,  // 0=LAST_VALUE, 1=NON_COLLAPSING, 2=BURST
    CCspPushGroup* group,  // can be NULL
    CCspPushInputAdapterCallbacks callbacks
);

void ccsp_push_input_adapter_destroy(CCspPushInputAdapter* adapter);

// Push data into the graph (thread-safe, called from adapter thread)
void ccsp_push_input_adapter_push_bool(CCspPushInputAdapter* adapter, int8_t value, CCspPushBatch* batch);
void ccsp_push_input_adapter_push_int64(CCspPushInputAdapter* adapter, int64_t value, CCspPushBatch* batch);
void ccsp_push_input_adapter_push_double(CCspPushInputAdapter* adapter, double value, CCspPushBatch* batch);
void ccsp_push_input_adapter_push_string(CCspPushInputAdapter* adapter, const char* data, size_t len, CCspPushBatch* batch);
void ccsp_push_input_adapter_push_datetime(CCspPushInputAdapter* adapter, CCspDateTime value, CCspPushBatch* batch);
void ccsp_push_input_adapter_push_struct(CCspPushInputAdapter* adapter, void* struct_ptr, CCspPushBatch* batch);
// ... additional type-specific push functions

// Generic push with CCspValue
void ccsp_push_input_adapter_push_value(CCspPushInputAdapter* adapter, const CCspValue* value, CCspPushBatch* batch);

// Batch management
CCspPushBatch* ccsp_push_batch_create(CCspEngine* engine);
void ccsp_push_batch_flush(CCspPushBatch* batch);
void ccsp_push_batch_destroy(CCspPushBatch* batch);

// Group management
CCspPushGroup* ccsp_push_group_create();
void ccsp_push_group_destroy(CCspPushGroup* group);
```

### 3.2 C++ Wrapper Implementation (`cpp/csp/engine/PushInputAdapterExtern.cpp`)

```cpp
class PushInputAdapterExtern : public PushInputAdapter {
public:
    PushInputAdapterExtern(Engine* engine, const CspTypePtr& type,
                           PushMode pushMode, PushGroup* group,
                           CCspPushInputAdapterCallbacks callbacks);
    ~PushInputAdapterExtern() override;

    void start(DateTime start, DateTime end) override {
        if (m_callbacks.start) {
            m_callbacks.start(m_callbacks.user_data,
                             start.asNanoseconds(),
                             end.asNanoseconds());
        }
    }

    void stop() override {
        if (m_callbacks.stop) {
            m_callbacks.stop(m_callbacks.user_data);
        }
    }

    // C API will call these methods
    template<typename T>
    void pushFromC(const T& value, PushBatch* batch) {
        this->pushTick(value, batch);
    }

private:
    CCspPushInputAdapterCallbacks m_callbacks;
};
```

---

## Phase 4: Adapter Manager C Interface

**Goal:** Enable external adapter managers with proper lifecycle

### 4.1 Adapter Manager Interface (`cpp/csp/engine/c/AdapterManager.h`)

```c
typedef struct CCspAdapterManager CCspAdapterManager;

// Adapter manager lifecycle callbacks
typedef struct {
    void* user_data;

    // Required: name of the adapter manager
    const char* (*get_name)(void* user_data);

    // Called when graph starts
    void (*start)(void* user_data, CCspDateTime start_time, CCspDateTime end_time);

    // Called when graph stops
    void (*stop)(void* user_data);

    // For sim adapters: return next sim time or 0 if none
    CCspDateTime (*process_next_sim_time_slice)(void* user_data, CCspDateTime current_time);

    // Destructor
    void (*destroy)(void* user_data);

} CCspAdapterManagerCallbacks;

// Creation
CCspAdapterManager* ccsp_adapter_manager_create(
    CCspEngine* engine,
    CCspAdapterManagerCallbacks callbacks
);

void ccsp_adapter_manager_destroy(CCspAdapterManager* adapter_manager);

// Get root engine for scheduling
CCspEngine* ccsp_adapter_manager_engine(CCspAdapterManager* manager);
```

---

## Phase 5: Dictionary/Configuration C Interface

**Goal:** Enable passing configuration to external adapters

### 5.1 Dictionary Interface (`cpp/csp/engine/c/CspDictionary.h`)

```c
typedef struct CCspDictionary CCspDictionary;

// Creation and destruction
CCspDictionary* ccsp_dictionary_create();
void ccsp_dictionary_destroy(CCspDictionary* dict);

// Check existence
int ccsp_dictionary_exists(const CCspDictionary* dict, const char* key);

// Getters (return 0 on success, non-zero on error)
int ccsp_dictionary_get_bool(const CCspDictionary* dict, const char* key, int* out_value);
int ccsp_dictionary_get_int64(const CCspDictionary* dict, const char* key, int64_t* out_value);
int ccsp_dictionary_get_double(const CCspDictionary* dict, const char* key, double* out_value);
int ccsp_dictionary_get_string(const CCspDictionary* dict, const char* key, const char** out_data, size_t* out_len);
int ccsp_dictionary_get_datetime(const CCspDictionary* dict, const char* key, CCspDateTime* out_value);
int ccsp_dictionary_get_dict(const CCspDictionary* dict, const char* key, CCspDictionary** out_dict);

// Getters with defaults
int64_t ccsp_dictionary_get_int64_or(const CCspDictionary* dict, const char* key, int64_t default_val);
double ccsp_dictionary_get_double_or(const CCspDictionary* dict, const char* key, double default_val);
const char* ccsp_dictionary_get_string_or(const CCspDictionary* dict, const char* key, const char* default_val);

// Iteration
size_t ccsp_dictionary_size(const CCspDictionary* dict);
typedef struct CCspDictIterator CCspDictIterator;
CCspDictIterator* ccsp_dictionary_iter_create(const CCspDictionary* dict);
int ccsp_dictionary_iter_next(CCspDictIterator* iter, const char** out_key, CCspValue* out_value);
void ccsp_dictionary_iter_destroy(CCspDictIterator* iter);
```

---

## Phase 6: Struct Access C Interface

**Goal:** Enable reading/writing struct fields from C

### 6.1 Struct Access (`cpp/csp/engine/c/CspStruct.h`)

```c
typedef struct CCspStruct CCspStruct;

// Create a new struct instance
CCspStruct* ccsp_struct_create(CCspStructMetaHandle meta);
void ccsp_struct_destroy(CCspStruct* s);

// Clone a struct
CCspStruct* ccsp_struct_clone(const CCspStruct* s);

// Field access by name (slower but more convenient)
int ccsp_struct_get_field(const CCspStruct* s, const char* field_name, CCspValue* out_value);
int ccsp_struct_set_field(CCspStruct* s, const char* field_name, const CCspValue* value);

// Field access by handle (faster, for hot paths)
int ccsp_struct_get_field_by_handle(const CCspStruct* s, CCspStructFieldHandle field, CCspValue* out_value);
int ccsp_struct_set_field_by_handle(CCspStruct* s, CCspStructFieldHandle field, const CCspValue* value);

// Check if field is set
int ccsp_struct_is_field_set(const CCspStruct* s, CCspStructFieldHandle field);

// Validation
int ccsp_struct_validate(const CCspStruct* s);
```

---

## Phase 7: Error Handling

**Goal:** Consistent error reporting across the ABI

### 7.1 Error Handling (`cpp/csp/engine/c/CspError.h`)

```c
typedef enum {
    CCSP_OK = 0,
    CCSP_ERROR_NULL_POINTER,
    CCSP_ERROR_TYPE_MISMATCH,
    CCSP_ERROR_KEY_NOT_FOUND,
    CCSP_ERROR_INVALID_ARGUMENT,
    CCSP_ERROR_OUT_OF_MEMORY,
    CCSP_ERROR_RUNTIME,
    CCSP_ERROR_VALUE,
    CCSP_ERROR_UNKNOWN
} CCspError;

// Thread-local error state
CCspError ccsp_get_last_error();
const char* ccsp_get_last_error_message();
void ccsp_clear_error();

// Set error (for adapter implementations)
void ccsp_set_error(CCspError code, const char* message);
```

---

## Phase 8: Engine Access

**Goal:** Allow adapters to interact with the engine

### 8.1 Engine Interface (`cpp/csp/engine/c/CspEngine.h`)

```c
// Get current time
CCspDateTime ccsp_engine_now(CCspEngine* engine);

// Get cycle count
uint64_t ccsp_engine_cycle_count(CCspEngine* engine);

// Schedule a callback
typedef void (*CCspCallback)(void* user_data);
void ccsp_engine_schedule_callback(CCspEngine* engine, CCspDateTime time, CCspCallback callback, void* user_data);

// Status reporting
typedef struct CCspStatusAdapter CCspStatusAdapter;
void ccsp_status_adapter_push_status(CCspStatusAdapter* adapter, int level, const char* message);
```

---

## Phase 9: Build System and Packaging

**Goal:** Enable separate compilation and distribution

### 9.1 CMake Configuration

```cmake
# cpp/csp/engine/c/CMakeLists.txt
add_library(csp_c_api SHARED
    CspValue.cpp
    CspTime.cpp
    CspDictionary.cpp
    CspStruct.cpp
    CspError.cpp
    CspEngine.cpp
    OutputAdapter.cpp
    InputAdapter.cpp
    AdapterManager.cpp
)

target_include_directories(csp_c_api PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Install headers and library
install(TARGETS csp_c_api
    EXPORT csp_c_api-targets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

install(DIRECTORY include/csp/c
    DESTINATION include/csp
)

# Generate pkg-config file
configure_file(csp_c_api.pc.in csp_c_api.pc @ONLY)
install(FILES ${CMAKE_BINARY_DIR}/csp_c_api.pc DESTINATION lib/pkgconfig)
```

### 9.2 External Adapter Build Example

```cmake
# External adapter CMakeLists.txt
find_package(csp_c_api REQUIRED)

add_library(my_kafka_adapter SHARED
    MyKafkaAdapter.c
)

target_link_libraries(my_kafka_adapter
    csp_c_api::csp_c_api
    rdkafka
)
```

---

## Phase 10: Reference Implementation

**Goal:** Implement one adapter (e.g., simple WebSocket output) to validate the API

### 10.1 Complete Example Implementation

See [cpp/csp/adapters/c/example/](../cpp/csp/adapters/c/example/) for a working example that demonstrates:

1. Output adapter that logs to console
2. Push input adapter that reads from a simple source
3. Adapter manager with proper lifecycle

---

## Implementation Priority

### Must Have (Phase 1-4)
1. **Phase 1**: Core C types - foundation for everything
2. **Phase 2**: Output adapter - simplest case, validates design
3. **Phase 4**: Adapter manager - required for any real adapter
4. **Phase 5**: Dictionary - configuration is essential

### Should Have (Phase 5-7)
5. **Phase 3**: Input adapter - needed for Kafka/WebSocket input
6. **Phase 6**: Struct access - needed for structured data
7. **Phase 7**: Error handling - production quality

### Nice to Have (Phase 8-10)
8. **Phase 8**: Engine access - advanced functionality
9. **Phase 9**: Build system - distribution
10. **Phase 10**: Reference implementation - documentation

---

## Estimated Effort

| Phase | Effort (days) | Dependencies |
|-------|---------------|--------------|
| Phase 1 | 3-5 | None |
| Phase 2 | 5-7 | Phase 1 |
| Phase 3 | 5-7 | Phase 1 |
| Phase 4 | 3-5 | Phase 1, 2 |
| Phase 5 | 3-5 | Phase 1 |
| Phase 6 | 5-7 | Phase 1, 5 |
| Phase 7 | 2-3 | None |
| Phase 8 | 3-5 | Phase 1, 4 |
| Phase 9 | 2-3 | All above |
| Phase 10 | 5-7 | All above |

**Total: ~35-50 days**

---

## Success Criteria

The C API is complete when:

1. ✅ **Kafka adapter** can be compiled as a separate shared library that:
   - Receives messages via push input adapter
   - Sends messages via output adapter
   - Manages its lifecycle via adapter manager
   - Uses Dictionary for configuration

2. ✅ **Parquet adapter** can be compiled separately with:
   - Output adapter for writing
   - Input adapter for reading (sim mode)
   - Struct field access for column mapping

3. ✅ **WebSocket adapter** can be compiled separately with:
   - Bidirectional communication
   - Connection lifecycle management
   - String/binary message handling

4. ✅ **ABI Stability**:
   - Adapters compiled with version N work with CSP version N+1
   - No C++ types cross the ABI boundary
   - All pointers are opaque handles

5. ✅ **Documentation**:
   - Complete API reference
   - Migration guide for existing adapters
   - Example code for new adapters

---

## Open Questions

1. **Struct Metadata Sharing**: How do we share struct definitions between CSP and external adapters?
   - Option A: Adapters define their own structs, CSP marshals
   - Option B: CSP provides struct creation API
   - Option C: Use a common serialization format (e.g., Arrow)

2. **Memory Ownership**: Who owns memory for strings, arrays, structs passed across the ABI?
   - Proposed: CSP owns internal data; adapters must copy if they need to retain

3. **Thread Safety**: Which functions are thread-safe?
   - Proposed: Only push_* functions are thread-safe; all others require single-thread access

4. **Versioning**: How do we version the C API?
   - Proposed: Version number in header, runtime check function

5. **Python Integration**: How do external adapters integrate with Python?
   - Current: PyCapsule mechanism looks correct
   - Need: Complete the Python wrapper layer

---

## Next Steps

1. Review this roadmap with stakeholders
2. Finalize decisions on open questions
3. Begin Phase 1 implementation
4. Set up CI for testing C API compatibility
