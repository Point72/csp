# CSP C API Reference

This document provides a complete reference for the CSP C API, which allows adapters to be written in C (or any language with C FFI) and compiled separately from CSP.

## Table of Contents

- [Overview](#overview)
- [Header Files](#header-files)
- [Error Handling](#error-handling)
- [Type System](#type-system)
- [Time Types](#time-types)
- [String Types](#string-types)
- [Value Container](#value-container)
- [Dictionary Access](#dictionary-access)
- [Output Adapters](#output-adapters)
- [Push Input Adapters](#push-input-adapters)
- [Engine Access](#engine-access)
- [Input Access](#input-access)
- [Adapter Managers](#adapter-managers)

______________________________________________________________________

## Overview

The C API provides ABI-stable interfaces for:

- **Output Adapters**: Receive data from the CSP graph and send to external systems
- **Push Input Adapters**: Push data from external sources into the CSP graph
- **Adapter Managers**: Coordinate multiple adapters sharing resources and lifecycle

All types are designed to be stable across CSP versions, using:

- Fixed-size integer types (`int32_t`, `int64_t`, etc.)
- Opaque handle pointers for internal CSP objects
- VTable pattern for polymorphism

### Including the Headers

```c
#include <csp/engine/c/CspError.h>     // Error handling
#include <csp/engine/c/CspType.h>      // Type enumeration
#include <csp/engine/c/CspTime.h>      // Time types
#include <csp/engine/c/CspString.h>    // String types
#include <csp/engine/c/CspValue.h>     // Value container
#include <csp/engine/c/CspDictionary.h> // Dictionary access
#include <csp/engine/c/OutputAdapter.h> // Output adapter API
#include <csp/engine/c/InputAdapter.h>  // Input adapter API
#include <csp/engine/c/AdapterManager.h> // Adapter manager API
```

______________________________________________________________________

## Error Handling

**Header:** `<csp/engine/c/CspError.h>`

### Error Codes

```c
typedef enum {
    CCSP_OK = 0,                    // Success
    CCSP_ERROR_NULL_POINTER,        // NULL argument provided
    CCSP_ERROR_TYPE_MISMATCH,       // Type does not match expected
    CCSP_ERROR_KEY_NOT_FOUND,       // Dictionary key not found
    CCSP_ERROR_INVALID_ARGUMENT,    // Invalid argument value
    CCSP_ERROR_OUT_OF_MEMORY,       // Memory allocation failed
    CCSP_ERROR_OUT_OF_RANGE,        // Index out of range
    CCSP_ERROR_RUNTIME,             // Runtime error
    CCSP_ERROR_VALUE,               // Value error
    CCSP_ERROR_NOT_IMPLEMENTED,     // Feature not implemented
    CCSP_ERROR_UNKNOWN              // Unknown error
} CCspErrorCode;
```

### Functions

#### `ccsp_get_last_error`

```c
CCspErrorCode ccsp_get_last_error(void);
```

Returns the last error code for the current thread.

#### `ccsp_get_last_error_message`

```c
const char* ccsp_get_last_error_message(void);
```

Returns the last error message for the current thread, or `NULL` if no message is set.

#### `ccsp_clear_error`

```c
void ccsp_clear_error(void);
```

Clears the last error for the current thread.

#### `ccsp_set_error`

```c
void ccsp_set_error(CCspErrorCode code, const char* message);
```

Sets an error code and message. The message is copied internally.

### Macros

```c
// Return from function if expression returns error
#define CCSP_RETURN_IF_ERROR(expr)

// Return NULL from function if expression returns error
#define CCSP_RETURN_NULL_IF_ERROR(expr)
```

______________________________________________________________________

## Type System

**Header:** `<csp/engine/c/CspType.h>`

### Type Enumeration

```c
typedef enum {
    CCSP_TYPE_UNKNOWN = 0,
    CCSP_TYPE_BOOL,
    CCSP_TYPE_INT8,
    CCSP_TYPE_UINT8,
    CCSP_TYPE_INT16,
    CCSP_TYPE_UINT16,
    CCSP_TYPE_INT32,
    CCSP_TYPE_UINT32,
    CCSP_TYPE_INT64,
    CCSP_TYPE_UINT64,
    CCSP_TYPE_DOUBLE,
    CCSP_TYPE_STRING,
    CCSP_TYPE_DATETIME,
    CCSP_TYPE_TIMEDELTA,
    CCSP_TYPE_DATE,
    CCSP_TYPE_TIME,
    CCSP_TYPE_ENUM,
    CCSP_TYPE_STRUCT,
    CCSP_TYPE_ARRAY,
    CCSP_TYPE_DIALECT_GENERIC
} CCspType;
```

### Type Mapping

| CCspType                    | C Type                   | CSP C++ Type         |
| --------------------------- | ------------------------ | -------------------- |
| `CCSP_TYPE_BOOL`            | `int8_t`                 | `bool`               |
| `CCSP_TYPE_INT8`            | `int8_t`                 | `int8_t`             |
| `CCSP_TYPE_UINT8`           | `uint8_t`                | `uint8_t`            |
| `CCSP_TYPE_INT16`           | `int16_t`                | `int16_t`            |
| `CCSP_TYPE_UINT16`          | `uint16_t`               | `uint16_t`           |
| `CCSP_TYPE_INT32`           | `int32_t`                | `int32_t`            |
| `CCSP_TYPE_UINT32`          | `uint32_t`               | `uint32_t`           |
| `CCSP_TYPE_INT64`           | `int64_t`                | `int64_t`            |
| `CCSP_TYPE_UINT64`          | `uint64_t`               | `uint64_t`           |
| `CCSP_TYPE_DOUBLE`          | `double`                 | `double`             |
| `CCSP_TYPE_STRING`          | `const char*` + `size_t` | `std::string`        |
| `CCSP_TYPE_DATETIME`        | `int64_t` (nanoseconds)  | `csp::DateTime`      |
| `CCSP_TYPE_TIMEDELTA`       | `int64_t` (nanoseconds)  | `csp::TimeDelta`     |
| `CCSP_TYPE_DATE`            | `int32_t` (days)         | `csp::Date`          |
| `CCSP_TYPE_TIME`            | `int64_t` (nanoseconds)  | `csp::Time`          |
| `CCSP_TYPE_ENUM`            | `int32_t`                | `csp::CspEnum`       |
| `CCSP_TYPE_STRUCT`          | opaque handle            | `csp::StructPtr`     |
| `CCSP_TYPE_ARRAY`           | opaque handle            | `std::vector<T>`     |
| `CCSP_TYPE_DIALECT_GENERIC` | opaque pointer           | `PyObject*` (Python) |

______________________________________________________________________

## Time Types

**Header:** `<csp/engine/c/CspTime.h>`

### Type Definitions

```c
typedef int64_t CCspDateTime;   // Nanoseconds since Unix epoch
typedef int64_t CCspTimeDelta;  // Duration in nanoseconds
typedef int32_t CCspDate;       // Days since Unix epoch
typedef int64_t CCspTime;       // Nanoseconds since midnight
```

### Constants

```c
#define CCSP_NANOSECONDS_PER_SECOND      1000000000LL
#define CCSP_NANOSECONDS_PER_MILLISECOND 1000000LL
#define CCSP_NANOSECONDS_PER_MICROSECOND 1000LL
#define CCSP_SECONDS_PER_DAY             86400LL

#define CCSP_DATETIME_MIN INT64_MIN
#define CCSP_DATETIME_MAX INT64_MAX
#define CCSP_TIMEDELTA_ZERO 0LL
```

### DateTime Functions

```c
// Construction
CCspDateTime ccsp_datetime_from_nanoseconds(int64_t nanoseconds);
CCspDateTime ccsp_datetime_from_seconds(int64_t seconds);
CCspDateTime ccsp_datetime_from_milliseconds(int64_t milliseconds);
CCspDateTime ccsp_datetime_from_parts(
    int year, int month, int day,
    int hour, int minute, int second, int nanosecond);

// Extraction
int64_t ccsp_datetime_to_nanoseconds(CCspDateTime dt);
int64_t ccsp_datetime_to_seconds(CCspDateTime dt);
int64_t ccsp_datetime_to_milliseconds(CCspDateTime dt);
void ccsp_datetime_to_parts(CCspDateTime dt,
    int* year, int* month, int* day,
    int* hour, int* minute, int* second, int* nanosecond);

// Arithmetic
CCspDateTime ccsp_datetime_add(CCspDateTime dt, CCspTimeDelta delta);
CCspTimeDelta ccsp_datetime_diff(CCspDateTime a, CCspDateTime b);
```

### TimeDelta Functions

```c
// Construction
CCspTimeDelta ccsp_timedelta_from_nanoseconds(int64_t nanoseconds);
CCspTimeDelta ccsp_timedelta_from_microseconds(int64_t microseconds);
CCspTimeDelta ccsp_timedelta_from_milliseconds(int64_t milliseconds);
CCspTimeDelta ccsp_timedelta_from_seconds(double seconds);
CCspTimeDelta ccsp_timedelta_from_minutes(double minutes);
CCspTimeDelta ccsp_timedelta_from_hours(double hours);
CCspTimeDelta ccsp_timedelta_from_days(double days);

// Extraction
double ccsp_timedelta_to_seconds(CCspTimeDelta td);
int64_t ccsp_timedelta_to_nanoseconds(CCspTimeDelta td);
```

### Date Functions

```c
CCspDate ccsp_date_from_days(int32_t days_since_epoch);
CCspDate ccsp_date_from_parts(int year, int month, int day);
int32_t ccsp_date_to_days(CCspDate date);
void ccsp_date_to_parts(CCspDate date, int* year, int* month, int* day);
```

### Time Functions

```c
CCspTime ccsp_time_from_nanoseconds(int64_t nanoseconds_since_midnight);
CCspTime ccsp_time_from_parts(int hour, int minute, int second, int nanosecond);
int64_t ccsp_time_to_nanoseconds(CCspTime time);
void ccsp_time_to_parts(CCspTime time, int* hour, int* minute, int* second, int* nanosecond);
```

______________________________________________________________________

## String Types

**Header:** `<csp/engine/c/CspString.h>`

### String View (Non-owning)

```c
typedef struct {
    const char* data;   // Pointer to string data
    size_t length;      // Length in bytes
} CCspStringView;
```

Use string views for passing strings into CSP functions. The data must remain valid for the duration of the call.

### Owned String

```c
typedef struct {
    char* data;         // Pointer to string data
    size_t length;      // Length in bytes
    size_t capacity;    // Allocated capacity
} CCspString;
```

Owned strings are returned from CSP functions and must be freed with `ccsp_string_free()`.

### Functions

```c
// Create views
CCspStringView ccsp_string_view_from_cstr(const char* cstr);
CCspStringView ccsp_string_view_from_data(const char* data, size_t length);

// Create owned strings
CCspString ccsp_string_create(const char* data, size_t length);
CCspString ccsp_string_create_from_cstr(const char* cstr);
CCspString ccsp_string_create_with_capacity(size_t capacity);

// Free owned string
void ccsp_string_free(CCspString* str);

// Convert owned to view
CCspStringView ccsp_string_as_view(const CCspString* str);

// Check empty
static inline int ccsp_string_view_is_empty(CCspStringView view);
static inline int ccsp_string_is_empty(const CCspString* str);
```

______________________________________________________________________

## Value Container

**Header:** `<csp/engine/c/CspValue.h>`

### CCspValue Structure

`CCspValue` is a tagged union that can hold any CSP type:

```c
typedef struct {
    CCspType type;      // Type tag
    union {
        int8_t bool_val;
        int8_t int8_val;
        uint8_t uint8_val;
        int16_t int16_val;
        uint16_t uint16_val;
        int32_t int32_val;
        uint32_t uint32_val;
        int64_t int64_val;
        uint64_t uint64_val;
        double double_val;
        CCspStringValue string_val;
        CCspDateTime datetime_val;
        CCspTimeDelta timedelta_val;
        CCspDate date_val;
        CCspTime time_val;
        CCspEnumValue enum_val;
        CCspStructHandle struct_val;
        CCspArrayValue array_val;
        CCspDialectValue dialect_val;
    };
} CCspValue;
```

### Lifecycle Functions

```c
void ccsp_value_init(CCspValue* value);          // Initialize to unknown
void ccsp_value_free(CCspValue* value);          // Free owned memory
CCspErrorCode ccsp_value_copy(CCspValue* dest, const CCspValue* src);
void ccsp_value_move(CCspValue* dest, CCspValue* src);
```

### Setters

```c
void ccsp_value_set_bool(CCspValue* value, int8_t v);
void ccsp_value_set_int8(CCspValue* value, int8_t v);
void ccsp_value_set_uint8(CCspValue* value, uint8_t v);
void ccsp_value_set_int16(CCspValue* value, int16_t v);
void ccsp_value_set_uint16(CCspValue* value, uint16_t v);
void ccsp_value_set_int32(CCspValue* value, int32_t v);
void ccsp_value_set_uint32(CCspValue* value, uint32_t v);
void ccsp_value_set_int64(CCspValue* value, int64_t v);
void ccsp_value_set_uint64(CCspValue* value, uint64_t v);
void ccsp_value_set_double(CCspValue* value, double v);
void ccsp_value_set_datetime(CCspValue* value, CCspDateTime v);
void ccsp_value_set_timedelta(CCspValue* value, CCspTimeDelta v);
void ccsp_value_set_date(CCspValue* value, CCspDate v);
void ccsp_value_set_time(CCspValue* value, CCspTime v);

// String (copies data)
CCspErrorCode ccsp_value_set_string(CCspValue* value, const char* data, size_t length);
CCspErrorCode ccsp_value_set_string_cstr(CCspValue* value, const char* cstr);

// String view (does not copy)
void ccsp_value_set_string_view(CCspValue* value, const char* data, size_t length);

// Struct and enum
void ccsp_value_set_struct(CCspValue* value, CCspStructHandle s);
void ccsp_value_set_enum(CCspValue* value, int32_t ordinal, CCspEnumMetaHandle meta);
```

### Getters

All getters return `CCspErrorCode` and write to an output parameter:

```c
CCspErrorCode ccsp_value_get_bool(const CCspValue* value, int8_t* out);
CCspErrorCode ccsp_value_get_int8(const CCspValue* value, int8_t* out);
CCspErrorCode ccsp_value_get_uint8(const CCspValue* value, uint8_t* out);
CCspErrorCode ccsp_value_get_int16(const CCspValue* value, int16_t* out);
CCspErrorCode ccsp_value_get_uint16(const CCspValue* value, uint16_t* out);
CCspErrorCode ccsp_value_get_int32(const CCspValue* value, int32_t* out);
CCspErrorCode ccsp_value_get_uint32(const CCspValue* value, uint32_t* out);
CCspErrorCode ccsp_value_get_int64(const CCspValue* value, int64_t* out);
CCspErrorCode ccsp_value_get_uint64(const CCspValue* value, uint64_t* out);
CCspErrorCode ccsp_value_get_double(const CCspValue* value, double* out);
CCspErrorCode ccsp_value_get_datetime(const CCspValue* value, CCspDateTime* out);
CCspErrorCode ccsp_value_get_timedelta(const CCspValue* value, CCspTimeDelta* out);
CCspErrorCode ccsp_value_get_date(const CCspValue* value, CCspDate* out);
CCspErrorCode ccsp_value_get_time(const CCspValue* value, CCspTime* out);
CCspErrorCode ccsp_value_get_string(const CCspValue* value, const char** out_data, size_t* out_length);
CCspErrorCode ccsp_value_get_struct(const CCspValue* value, CCspStructHandle* out);
CCspErrorCode ccsp_value_get_enum(const CCspValue* value, int32_t* out_ordinal, CCspEnumMetaHandle* out_meta);
```

### Type Checking

```c
static inline int ccsp_value_is_type(const CCspValue* value, CCspType type);
static inline int ccsp_value_is_valid(const CCspValue* value);
int ccsp_value_is_numeric(const CCspValue* value);
int ccsp_value_is_integer(const CCspValue* value);
```

______________________________________________________________________

## Dictionary Access

**Header:** `<csp/engine/c/CspDictionary.h>`

The Dictionary API provides read-only access to CSP dictionaries, which are used to pass configuration from Python to C adapters.

### Handles

```c
typedef void* CCspDictionaryHandle;    // Opaque handle to csp::Dictionary
typedef void* CCspDictIteratorHandle;  // Opaque iterator handle
```

### Value Types

```c
typedef enum {
    CCSP_DICT_TYPE_NONE = 0,      // std::monostate
    CCSP_DICT_TYPE_BOOL,          // bool
    CCSP_DICT_TYPE_INT32,         // int32_t
    CCSP_DICT_TYPE_UINT32,        // uint32_t
    CCSP_DICT_TYPE_INT64,         // int64_t
    CCSP_DICT_TYPE_UINT64,        // uint64_t
    CCSP_DICT_TYPE_DOUBLE,        // double
    CCSP_DICT_TYPE_STRING,        // std::string
    CCSP_DICT_TYPE_DATETIME,      // csp::DateTime
    CCSP_DICT_TYPE_TIMEDELTA,     // csp::TimeDelta
    CCSP_DICT_TYPE_STRUCT_META,   // StructMetaPtr
    CCSP_DICT_TYPE_DIALECT,       // DialectGenericType
    CCSP_DICT_TYPE_DICTIONARY,    // nested Dictionary
    CCSP_DICT_TYPE_VECTOR,        // Vector
    CCSP_DICT_TYPE_DATA           // shared_ptr<Data>
} CCspDictValueType;
```

### Basic Operations

```c
// Get number of entries
size_t ccsp_dictionary_size(CCspDictionaryHandle dict);

// Check if key exists
int ccsp_dictionary_has_key(CCspDictionaryHandle dict, const char* key);

// Get value type for key
CCspDictValueType ccsp_dictionary_get_type(CCspDictionaryHandle dict, const char* key);
```

### Type-Safe Getters

These functions return `CCSP_OK` on success or an error code on failure:

```c
CCspErrorCode ccsp_dictionary_get_bool(CCspDictionaryHandle dict, const char* key, int8_t* out_value);
CCspErrorCode ccsp_dictionary_get_int32(CCspDictionaryHandle dict, const char* key, int32_t* out_value);
CCspErrorCode ccsp_dictionary_get_uint32(CCspDictionaryHandle dict, const char* key, uint32_t* out_value);
CCspErrorCode ccsp_dictionary_get_int64(CCspDictionaryHandle dict, const char* key, int64_t* out_value);
CCspErrorCode ccsp_dictionary_get_uint64(CCspDictionaryHandle dict, const char* key, uint64_t* out_value);
CCspErrorCode ccsp_dictionary_get_double(CCspDictionaryHandle dict, const char* key, double* out_value);
CCspErrorCode ccsp_dictionary_get_datetime(CCspDictionaryHandle dict, const char* key, CCspDateTime* out_value);
CCspErrorCode ccsp_dictionary_get_timedelta(CCspDictionaryHandle dict, const char* key, CCspTimeDelta* out_value);

// Returns pointer to internal string data (valid while dictionary exists)
CCspErrorCode ccsp_dictionary_get_string(CCspDictionaryHandle dict, const char* key, 
                                         const char** out_data, size_t* out_length);

// Returns handle to nested dictionary (must NOT be freed - owned by parent)
CCspErrorCode ccsp_dictionary_get_dict(CCspDictionaryHandle dict, const char* key,
                                       CCspDictionaryHandle* out_dict);
```

### Getters with Default Values

These functions return the value directly, or the provided default if key is missing:

```c
int8_t ccsp_dictionary_get_bool_or(CCspDictionaryHandle dict, const char* key, int8_t default_value);
int32_t ccsp_dictionary_get_int32_or(CCspDictionaryHandle dict, const char* key, int32_t default_value);
uint32_t ccsp_dictionary_get_uint32_or(CCspDictionaryHandle dict, const char* key, uint32_t default_value);
int64_t ccsp_dictionary_get_int64_or(CCspDictionaryHandle dict, const char* key, int64_t default_value);
uint64_t ccsp_dictionary_get_uint64_or(CCspDictionaryHandle dict, const char* key, uint64_t default_value);
double ccsp_dictionary_get_double_or(CCspDictionaryHandle dict, const char* key, double default_value);
CCspDateTime ccsp_dictionary_get_datetime_or(CCspDictionaryHandle dict, const char* key, CCspDateTime default_value);
CCspTimeDelta ccsp_dictionary_get_timedelta_or(CCspDictionaryHandle dict, const char* key, CCspTimeDelta default_value);

// Returns pointer to string or default_value if key missing (NULL-safe)
const char* ccsp_dictionary_get_string_or(CCspDictionaryHandle dict, const char* key, const char* default_value);
```

### Iteration

Iterate over all key-value pairs in a dictionary:

```c
// Create iterator
CCspDictIteratorHandle ccsp_dictionary_iter_create(CCspDictionaryHandle dict);

// Destroy iterator
void ccsp_dictionary_iter_destroy(CCspDictIteratorHandle iter);

// Advance and get next key (returns 0 when exhausted)
int ccsp_dictionary_iter_next(CCspDictIteratorHandle iter, const char** out_key);

// Get type of current value
CCspDictValueType ccsp_dictionary_iter_value_type(CCspDictIteratorHandle iter);

// Get current value (type-specific)
CCspErrorCode ccsp_dictionary_iter_get_bool(CCspDictIteratorHandle iter, int8_t* out_value);
CCspErrorCode ccsp_dictionary_iter_get_int32(CCspDictIteratorHandle iter, int32_t* out_value);
// ... similar for other types
CCspErrorCode ccsp_dictionary_iter_get_dict(CCspDictIteratorHandle iter, CCspDictionaryHandle* out_dict);
```

### Example Usage

```c
void process_config(CCspDictionaryHandle config)
{
    // Direct access with defaults
    int32_t port = ccsp_dictionary_get_int32_or(config, "port", 9092);
    const char* host = ccsp_dictionary_get_string_or(config, "host", "localhost");
    
    // Type-safe access with error handling
    const char* topic_data = NULL;
    size_t topic_len = 0;
    if (ccsp_dictionary_get_string(config, "topic", &topic_data, &topic_len) != CCSP_OK) {
        // Handle missing required field
    }
    
    // Iteration
    CCspDictIteratorHandle iter = ccsp_dictionary_iter_create(config);
    const char* key;
    while (ccsp_dictionary_iter_next(iter, &key)) {
        CCspDictValueType type = ccsp_dictionary_iter_value_type(iter);
        printf("Key: %s, Type: %d\n", key, type);
    }
    ccsp_dictionary_iter_destroy(iter);
}
```

______________________________________________________________________

## Struct Access

**Header:** `<csp/engine/c/CspStruct.h>`

The Struct API provides access to CSP Structs, which are structured data types with named, typed fields. This API allows C code to:

- Inspect struct type metadata (fields, types)
- Read field values from struct instances
- Write field values to struct instances
- Create and copy struct instances

### Opaque Handles

```c
typedef void* CCspStructMetaHandle;   // Handle to csp::StructMeta (type info)
typedef CCspStructImpl* CCspStructHandle;  // Handle to csp::Struct (instance)
typedef void* CCspStructFieldHandle;  // Handle to csp::StructField (field info)
```

### StructMeta Functions (Type Information)

```c
// Get struct type name
const char* ccsp_struct_meta_name(CCspStructMetaHandle meta);

// Get number of fields
size_t ccsp_struct_meta_field_count(CCspStructMetaHandle meta);

// Get field by index (0-based)
CCspStructFieldHandle ccsp_struct_meta_field_by_index(CCspStructMetaHandle meta, size_t index);

// Get field by name
CCspStructFieldHandle ccsp_struct_meta_field_by_name(CCspStructMetaHandle meta, const char* name);

// Get field name by index
const char* ccsp_struct_meta_field_name_by_index(CCspStructMetaHandle meta, size_t index);

// Check if struct type is strict
int ccsp_struct_meta_is_strict(CCspStructMetaHandle meta);
```

### StructField Functions (Field Metadata)

```c
// Get field name
const char* ccsp_struct_field_name(CCspStructFieldHandle field);

// Get field type (returns CCspType enum)
CCspType ccsp_struct_field_type(CCspStructFieldHandle field);

// Check if field is optional
int ccsp_struct_field_is_optional(CCspStructFieldHandle field);
```

### Struct Instance Functions

```c
// Get struct's meta (type info)
CCspStructMetaHandle ccsp_struct_meta(CCspStructHandle s);

// Check if field is set
int ccsp_struct_field_is_set(CCspStructHandle s, CCspStructFieldHandle field);

// Check if field is None
int ccsp_struct_field_is_none(CCspStructHandle s, CCspStructFieldHandle field);
```

### Field Value Getters

All getters return `CCSP_OK` on success or an error code on failure:

```c
// Primitive types
CCspErrorCode ccsp_struct_get_bool(CCspStructHandle s, CCspStructFieldHandle field, int8_t* out_value);
CCspErrorCode ccsp_struct_get_int8(CCspStructHandle s, CCspStructFieldHandle field, int8_t* out_value);
CCspErrorCode ccsp_struct_get_uint8(CCspStructHandle s, CCspStructFieldHandle field, uint8_t* out_value);
CCspErrorCode ccsp_struct_get_int16(CCspStructHandle s, CCspStructFieldHandle field, int16_t* out_value);
CCspErrorCode ccsp_struct_get_uint16(CCspStructHandle s, CCspStructFieldHandle field, uint16_t* out_value);
CCspErrorCode ccsp_struct_get_int32(CCspStructHandle s, CCspStructFieldHandle field, int32_t* out_value);
CCspErrorCode ccsp_struct_get_uint32(CCspStructHandle s, CCspStructFieldHandle field, uint32_t* out_value);
CCspErrorCode ccsp_struct_get_int64(CCspStructHandle s, CCspStructFieldHandle field, int64_t* out_value);
CCspErrorCode ccsp_struct_get_uint64(CCspStructHandle s, CCspStructFieldHandle field, uint64_t* out_value);
CCspErrorCode ccsp_struct_get_double(CCspStructHandle s, CCspStructFieldHandle field, double* out_value);

// Time types
CCspErrorCode ccsp_struct_get_datetime(CCspStructHandle s, CCspStructFieldHandle field, CCspDateTime* out_value);
CCspErrorCode ccsp_struct_get_timedelta(CCspStructHandle s, CCspStructFieldHandle field, CCspTimeDelta* out_value);

// String (returns pointer to internal data, valid while struct exists)
CCspErrorCode ccsp_struct_get_string(CCspStructHandle s, CCspStructFieldHandle field,
                                     const char** out_data, size_t* out_length);

// Enum (returns ordinal value)
CCspErrorCode ccsp_struct_get_enum(CCspStructHandle s, CCspStructFieldHandle field, int32_t* out_ordinal);
```

### Convenience Getters (by Name)

Access fields directly by name:

```c
CCspErrorCode ccsp_struct_get_bool_by_name(CCspStructHandle s, const char* name, int8_t* out_value);
CCspErrorCode ccsp_struct_get_int32_by_name(CCspStructHandle s, const char* name, int32_t* out_value);
CCspErrorCode ccsp_struct_get_int64_by_name(CCspStructHandle s, const char* name, int64_t* out_value);
CCspErrorCode ccsp_struct_get_double_by_name(CCspStructHandle s, const char* name, double* out_value);
CCspErrorCode ccsp_struct_get_datetime_by_name(CCspStructHandle s, const char* name, CCspDateTime* out_value);
CCspErrorCode ccsp_struct_get_string_by_name(CCspStructHandle s, const char* name,
                                             const char** out_data, size_t* out_length);
```

### Field Value Setters

All setters return `CCSP_OK` on success or an error code on failure:

```c
// Primitive types
CCspErrorCode ccsp_struct_set_bool(CCspStructHandle s, CCspStructFieldHandle field, int8_t value);
CCspErrorCode ccsp_struct_set_int8(CCspStructHandle s, CCspStructFieldHandle field, int8_t value);
CCspErrorCode ccsp_struct_set_uint8(CCspStructHandle s, CCspStructFieldHandle field, uint8_t value);
CCspErrorCode ccsp_struct_set_int16(CCspStructHandle s, CCspStructFieldHandle field, int16_t value);
CCspErrorCode ccsp_struct_set_uint16(CCspStructHandle s, CCspStructFieldHandle field, uint16_t value);
CCspErrorCode ccsp_struct_set_int32(CCspStructHandle s, CCspStructFieldHandle field, int32_t value);
CCspErrorCode ccsp_struct_set_uint32(CCspStructHandle s, CCspStructFieldHandle field, uint32_t value);
CCspErrorCode ccsp_struct_set_int64(CCspStructHandle s, CCspStructFieldHandle field, int64_t value);
CCspErrorCode ccsp_struct_set_uint64(CCspStructHandle s, CCspStructFieldHandle field, uint64_t value);
CCspErrorCode ccsp_struct_set_double(CCspStructHandle s, CCspStructFieldHandle field, double value);

// Time types
CCspErrorCode ccsp_struct_set_datetime(CCspStructHandle s, CCspStructFieldHandle field, CCspDateTime value);
CCspErrorCode ccsp_struct_set_timedelta(CCspStructHandle s, CCspStructFieldHandle field, CCspTimeDelta value);

// String (copies the data)
CCspErrorCode ccsp_struct_set_string(CCspStructHandle s, CCspStructFieldHandle field,
                                     const char* data, size_t length);
```

### Struct Lifecycle

```c
// Create a new struct instance (must call ccsp_struct_destroy when done)
CCspStructHandle ccsp_struct_create(CCspStructMetaHandle meta);

// Destroy a struct instance
void ccsp_struct_destroy(CCspStructHandle s);

// Create a deep copy of a struct
CCspStructHandle ccsp_struct_copy(CCspStructHandle s);
```

### Example Usage

```c
void process_struct(CCspStructHandle s)
{
    // Get struct meta (type info)
    CCspStructMetaHandle meta = ccsp_struct_meta(s);
    printf("Struct type: %s\n", ccsp_struct_meta_name(meta));
    
    // Iterate over fields
    size_t field_count = ccsp_struct_meta_field_count(meta);
    for (size_t i = 0; i < field_count; i++) {
        CCspStructFieldHandle field = ccsp_struct_meta_field_by_index(meta, i);
        const char* name = ccsp_struct_field_name(field);
        CCspType type = ccsp_struct_field_type(field);
        
        // Check if field is set
        if (!ccsp_struct_field_is_set(s, field)) {
            printf("  %s: <unset>\n", name);
            continue;
        }
        
        // Print based on type
        switch (type) {
            case CCSP_TYPE_INT64: {
                int64_t val;
                ccsp_struct_get_int64(s, field, &val);
                printf("  %s: %lld\n", name, (long long)val);
                break;
            }
            case CCSP_TYPE_STRING: {
                const char* data;
                size_t len;
                ccsp_struct_get_string(s, field, &data, &len);
                printf("  %s: %.*s\n", name, (int)len, data);
                break;
            }
            // ... handle other types
        }
    }
    
    // Direct access by name
    double price;
    if (ccsp_struct_get_double_by_name(s, "price", &price) == CCSP_OK) {
        printf("Price: %f\n", price);
    }
}

// Creating a new struct
void create_order(CCspStructMetaHandle order_meta)
{
    CCspStructHandle order = ccsp_struct_create(order_meta);
    if (!order) {
        // Handle error
        return;
    }
    
    // Set fields
    CCspStructFieldHandle symbol_field = ccsp_struct_meta_field_by_name(order_meta, "symbol");
    ccsp_struct_set_string(order, symbol_field, "AAPL", 4);
    
    CCspStructFieldHandle qty_field = ccsp_struct_meta_field_by_name(order_meta, "quantity");
    ccsp_struct_set_int64(order, qty_field, 100);
    
    // Use struct...
    
    // Cleanup
    ccsp_struct_destroy(order);
}
```

______________________________________________________________________

## Output Adapters

**Header:** `<csp/engine/c/OutputAdapter.h>`

Output adapters receive data from the CSP graph and send it to external systems.

### Opaque Handles

```c
typedef struct CCspEngineImpl* CCspEngineHandle;
typedef struct CCspInputImpl* CCspInputHandle;
typedef struct CCspOutputAdapterImpl* CCspOutputAdapterHandle;
```

### VTable Structure

```c
typedef struct CCspOutputAdapterVTable {
    void* user_data;

    void (*start)(void* user_data, CCspEngineHandle engine,
                  CCspDateTime start_time, CCspDateTime end_time);

    void (*stop)(void* user_data);

    void (*execute)(void* user_data, CCspEngineHandle engine,
                    CCspInputHandle input);

    void (*destroy)(void* user_data);

} CCspOutputAdapterVTable;
```

#### Callbacks

| Callback  | Required     | Description                                                |
| --------- | ------------ | ---------------------------------------------------------- |
| `start`   | Optional     | Called when graph starts. Initialize resources here.       |
| `stop`    | Optional     | Called when graph stops. Flush buffers, close connections. |
| `execute` | **Required** | Called when input has new value. Process the data here.    |
| `destroy` | **Required** | Called to clean up. Free all allocated memory.             |

### Creation

```c
CCspOutputAdapterHandle ccsp_output_adapter_extern_create(
    CCspEngineHandle engine,
    CCspType input_type,
    const CCspOutputAdapterVTable* vtable
);

void ccsp_output_adapter_extern_destroy(CCspOutputAdapterHandle adapter);
```

______________________________________________________________________

## Push Input Adapters

**Header:** `<csp/engine/c/InputAdapter.h>`

Push input adapters push data from external sources into the CSP graph.

### Push Mode

```c
typedef enum {
    CCSP_PUSH_MODE_LAST_VALUE = 0,      // Values collapse per cycle
    CCSP_PUSH_MODE_NON_COLLAPSING = 1,  // Each value = separate cycle
    CCSP_PUSH_MODE_BURST = 2            // Values batched into vector
} CCspPushMode;
```

### VTable Structure

```c
typedef struct CCspPushInputAdapterVTable {
    void* user_data;

    void (*start)(void* user_data, CCspEngineHandle engine,
                  CCspPushInputAdapterHandle adapter,
                  CCspDateTime start_time, CCspDateTime end_time);

    void (*stop)(void* user_data);

    void (*destroy)(void* user_data);

} CCspPushInputAdapterVTable;
```

**Note:** The `start` callback receives the adapter handle, which is needed for pushing data.

### Creation

```c
CCspPushInputAdapterHandle ccsp_push_input_adapter_extern_create(
    CCspEngineHandle engine,
    CCspType type,
    CCspPushMode push_mode,
    CCspPushGroupHandle group,  // Can be NULL
    const CCspPushInputAdapterVTable* vtable
);

void ccsp_push_input_adapter_extern_destroy(CCspPushInputAdapterHandle adapter);
```

### Push Functions (Thread-Safe)

These functions can be called from any thread:

```c
// Generic push
CCspErrorCode ccsp_push_input_adapter_push_value(
    CCspPushInputAdapterHandle adapter,
    const CCspValue* value,
    CCspPushBatchHandle batch);

// Type-specific push (more efficient)
CCspErrorCode ccsp_push_input_adapter_push_bool(CCspPushInputAdapterHandle adapter, int8_t value, CCspPushBatchHandle batch);
CCspErrorCode ccsp_push_input_adapter_push_int8(CCspPushInputAdapterHandle adapter, int8_t value, CCspPushBatchHandle batch);
CCspErrorCode ccsp_push_input_adapter_push_uint8(CCspPushInputAdapterHandle adapter, uint8_t value, CCspPushBatchHandle batch);
CCspErrorCode ccsp_push_input_adapter_push_int16(CCspPushInputAdapterHandle adapter, int16_t value, CCspPushBatchHandle batch);
CCspErrorCode ccsp_push_input_adapter_push_uint16(CCspPushInputAdapterHandle adapter, uint16_t value, CCspPushBatchHandle batch);
CCspErrorCode ccsp_push_input_adapter_push_int32(CCspPushInputAdapterHandle adapter, int32_t value, CCspPushBatchHandle batch);
CCspErrorCode ccsp_push_input_adapter_push_uint32(CCspPushInputAdapterHandle adapter, uint32_t value, CCspPushBatchHandle batch);
CCspErrorCode ccsp_push_input_adapter_push_int64(CCspPushInputAdapterHandle adapter, int64_t value, CCspPushBatchHandle batch);
CCspErrorCode ccsp_push_input_adapter_push_uint64(CCspPushInputAdapterHandle adapter, uint64_t value, CCspPushBatchHandle batch);
CCspErrorCode ccsp_push_input_adapter_push_double(CCspPushInputAdapterHandle adapter, double value, CCspPushBatchHandle batch);
CCspErrorCode ccsp_push_input_adapter_push_datetime(CCspPushInputAdapterHandle adapter, CCspDateTime value, CCspPushBatchHandle batch);
CCspErrorCode ccsp_push_input_adapter_push_timedelta(CCspPushInputAdapterHandle adapter, CCspTimeDelta value, CCspPushBatchHandle batch);
CCspErrorCode ccsp_push_input_adapter_push_string(CCspPushInputAdapterHandle adapter, const char* data, size_t length, CCspPushBatchHandle batch);
CCspErrorCode ccsp_push_input_adapter_push_struct(CCspPushInputAdapterHandle adapter, CCspStructHandle value, CCspPushBatchHandle batch);
```

### Batch Management

Batches group multiple events to be processed atomically:

```c
CCspPushBatchHandle ccsp_push_batch_create(CCspEngineHandle engine);
void ccsp_push_batch_flush(CCspPushBatchHandle batch);
void ccsp_push_batch_destroy(CCspPushBatchHandle batch);
```

### Group Management

Groups synchronize multiple input adapters:

```c
CCspPushGroupHandle ccsp_push_group_create(void);
void ccsp_push_group_destroy(CCspPushGroupHandle group);
```

______________________________________________________________________

## Engine Access

**Header:** `<csp/engine/c/OutputAdapter.h>`

### Functions

```c
// Get current engine time
CCspDateTime ccsp_engine_now(CCspEngineHandle engine);

// Get current cycle count
uint64_t ccsp_engine_cycle_count(CCspEngineHandle engine);
```

______________________________________________________________________

## Input Access

**Header:** `<csp/engine/c/OutputAdapter.h>`

Functions for reading input values in output adapter `execute` callbacks:

### Status Functions

```c
int ccsp_input_is_valid(CCspInputHandle input);
int32_t ccsp_input_num_ticks(CCspInputHandle input);
CCspType ccsp_input_get_type(CCspInputHandle input);
CCspDateTime ccsp_input_get_last_time(CCspInputHandle input);
```

### Value Access

```c
// Generic value access
CCspErrorCode ccsp_input_get_last_value(CCspInputHandle input, CCspValue* out_value);
CCspErrorCode ccsp_input_get_value_at(CCspInputHandle input, int32_t index, CCspValue* out_value);
CCspErrorCode ccsp_input_get_time_at(CCspInputHandle input, int32_t index, CCspDateTime* out_time);

// Convenience functions (more efficient for common types)
CCspErrorCode ccsp_input_get_last_string(CCspInputHandle input, const char** out_data, size_t* out_length);
CCspErrorCode ccsp_input_get_last_int64(CCspInputHandle input, int64_t* out_value);
CCspErrorCode ccsp_input_get_last_double(CCspInputHandle input, double* out_value);
CCspErrorCode ccsp_input_get_last_bool(CCspInputHandle input, int8_t* out_value);
CCspErrorCode ccsp_input_get_last_datetime(CCspInputHandle input, CCspDateTime* out_value);
```

______________________________________________________________________

## Adapter Managers

**Header:** `<csp/engine/c/AdapterManager.h>`

Adapter managers coordinate a group of related adapters, handling shared lifecycle, status reporting, and resource management.

### Opaque Handles

```c
typedef struct CCspAdapterManagerImpl* CCspAdapterManagerHandle;
typedef struct CCspStatusAdapterImpl* CCspStatusAdapterHandle;
typedef struct CCspManagedSimInputAdapterImpl* CCspManagedSimInputAdapterHandle;
```

### VTable Structure

```c
typedef struct CCspAdapterManagerVTable {
    void* user_data;

    // REQUIRED: Return manager name (for logging)
    const char* (*name)(void* user_data);

    // REQUIRED: Process simulation time slice
    // Return next timestamp, or 0 if no more data
    CCspDateTime (*process_next_sim_time_slice)(void* user_data, CCspDateTime time);

    // REQUIRED: Clean up manager resources
    void (*destroy)(void* user_data);

    // OPTIONAL: Called when graph starts
    void (*start)(void* user_data, CCspAdapterManagerHandle manager,
                  CCspDateTime start_time, CCspDateTime end_time);

    // OPTIONAL: Called when graph stops
    void (*stop)(void* user_data);

} CCspAdapterManagerVTable;
```

#### Callbacks

| Callback                      | Required     | Description                                          |
| ----------------------------- | ------------ | ---------------------------------------------------- |
| `name`                        | **Required** | Returns manager name for logging/debugging           |
| `process_next_sim_time_slice` | **Required** | Processes sim data, returns next timestamp or 0      |
| `destroy`                     | **Required** | Frees all allocated resources                        |
| `start`                       | Optional     | Called when graph starts. Initialize resources here. |
| `stop`                        | Optional     | Called when graph stops. Clean up resources.         |

### Creation and Lifecycle

```c
// Create an adapter manager
CCspAdapterManagerHandle ccsp_adapter_manager_extern_create(
    CCspEngineHandle engine,
    const CCspAdapterManagerVTable* vtable);

// Destroy is automatic - engine handles cleanup
void ccsp_adapter_manager_extern_destroy(CCspAdapterManagerHandle manager);
```

### Engine and Time Access

```c
// Get engine handle for use with other C API functions
CCspEngineHandle ccsp_adapter_manager_engine(CCspAdapterManagerHandle manager);

// Get graph start time (valid after start() called)
CCspDateTime ccsp_adapter_manager_start_time(CCspAdapterManagerHandle manager);

// Get graph end time (valid after start() called)
CCspDateTime ccsp_adapter_manager_end_time(CCspAdapterManagerHandle manager);
```

### Adapter Creation from Manager

Adapters created via the manager share its lifecycle:

```c
// Create a managed output adapter
CCspOutputAdapterHandle ccsp_adapter_manager_create_output_adapter(
    CCspAdapterManagerHandle manager,
    CCspType input_type,
    const CCspOutputAdapterVTable* vtable);

// Create a managed push input adapter
CCspPushInputAdapterHandle ccsp_adapter_manager_create_push_input_adapter(
    CCspAdapterManagerHandle manager,
    CCspType type,
    CCspPushMode push_mode,
    const CCspPushInputAdapterVTable* vtable);
```

### Status Reporting

```c
typedef enum {
    CCSP_STATUS_LEVEL_CRITICAL = 0,
    CCSP_STATUS_LEVEL_ERROR = 1,
    CCSP_STATUS_LEVEL_WARNING = 2,
    CCSP_STATUS_LEVEL_INFO = 3,
    CCSP_STATUS_LEVEL_DEBUG = 4
} CCspStatusLevel;

// Push a status message to the graph
CCspErrorCode ccsp_adapter_manager_push_status(
    CCspAdapterManagerHandle manager,
    CCspStatusLevel level,
    int64_t err_code,
    const char* message);
```

### Managed Simulation Input Adapter

For adapters that provide data in simulation mode:

```c
// Create a managed sim input adapter
CCspManagedSimInputAdapterHandle ccsp_adapter_manager_create_managed_sim_input_adapter(
    CCspAdapterManagerHandle manager,
    CCspType type,
    CCspPushMode push_mode);

// Push data (call from process_next_sim_time_slice)
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
```

______________________________________________________________________

## Thread Safety

| Function Category                                 | Thread Safety                                         |
| ------------------------------------------------- | ----------------------------------------------------- |
| Push functions (`ccsp_push_input_adapter_push_*`) | **Thread-safe**                                       |
| Batch functions                                   | Thread-safe                                           |
| Input access functions                            | **Not thread-safe** (call only from execute callback) |
| Engine access functions                           | Not thread-safe                                       |
| Error functions                                   | Thread-safe (thread-local storage)                    |

______________________________________________________________________

## Memory Ownership

| Type                                      | Ownership                                          |
| ----------------------------------------- | -------------------------------------------------- |
| `CCspValue` with string                   | If `is_owned=1`, you must call `ccsp_value_free()` |
| Strings from `ccsp_input_get_last_string` | Borrowed from CSP, valid until next tick           |
| VTable `user_data`                        | You own this memory, free in `destroy` callback    |
| Opaque handles                            | CSP owns these, do not free                        |

______________________________________________________________________

## See Also

- [Write C API Adapters](../how-tos/Write-C-API-Adapters.md) - How-to guide
- [Example Adapters](../../cpp/csp/adapters/c/example/) - Reference implementations
