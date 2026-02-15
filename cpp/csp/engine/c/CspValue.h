/*
 * ABI-stable C Value Container for CSP Engine
 *
 * CCspValue is a tagged union that can hold any CSP type.
 * It is used for passing values across the C API boundary.
 *
 * Memory ownership:
 * - For primitive types (bool, int, double, datetime, etc.), values are copied.
 * - For strings, the CCspValue may own or borrow the data (check is_owned).
 * - For structs and arrays, values are opaque pointers managed by CSP.
 */
#ifndef _IN_CSP_ENGINE_C_CSPVALUE_H
#define _IN_CSP_ENGINE_C_CSPVALUE_H

#include <csp/engine/c/CspType.h>
#include <csp/engine/c/CspTime.h>
#include <csp/engine/c/CspString.h>
#include <csp/engine/c/CspError.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Forward declarations for opaque types */
typedef struct CCspStructImpl* CCspStructHandle;
typedef struct CCspArrayImpl* CCspArrayHandle;
typedef struct CCspEnumMetaImpl* CCspEnumMetaHandle;
typedef struct CCspTypeInfoImpl* CCspTypeHandle;

/*
 * String value with ownership flag
 */
typedef struct {
    const char* data;
    size_t length;
    int is_owned;       /* 1 if CCspValue owns the memory, 0 if borrowed */
} CCspStringValue;

/*
 * Array value (for CCSP_TYPE_ARRAY)
 */
typedef struct {
    void* data;         /* Pointer to array data */
    size_t length;      /* Number of elements */
    CCspType elem_type; /* Element type */
    int is_owned;       /* 1 if CCspValue owns the memory */
} CCspArrayValue;

/*
 * Enum value (for CCSP_TYPE_ENUM)
 */
typedef struct {
    int32_t ordinal;            /* Enum ordinal value */
    CCspEnumMetaHandle meta;    /* Enum metadata (for string conversion) */
} CCspEnumValue;

/*
 * Dialect generic value (for CCSP_TYPE_DIALECT_GENERIC)
 * This is an opaque pointer that the dialect (e.g., Python) knows how to handle.
 */
typedef struct {
    void* ptr;          /* Opaque pointer to dialect-specific object */
    int type_id;        /* Dialect-specific type identifier */
} CCspDialectValue;

/*
 * Main value container - a tagged union for any CSP type
 */
typedef struct {
    CCspType type;      /* Type tag */
    union {
        int8_t bool_val;                /* CCSP_TYPE_BOOL */
        int8_t int8_val;                /* CCSP_TYPE_INT8 */
        uint8_t uint8_val;              /* CCSP_TYPE_UINT8 */
        int16_t int16_val;              /* CCSP_TYPE_INT16 */
        uint16_t uint16_val;            /* CCSP_TYPE_UINT16 */
        int32_t int32_val;              /* CCSP_TYPE_INT32 */
        uint32_t uint32_val;            /* CCSP_TYPE_UINT32 */
        int64_t int64_val;              /* CCSP_TYPE_INT64 */
        uint64_t uint64_val;            /* CCSP_TYPE_UINT64 */
        double double_val;              /* CCSP_TYPE_DOUBLE */
        CCspStringValue string_val;     /* CCSP_TYPE_STRING */
        CCspDateTime datetime_val;      /* CCSP_TYPE_DATETIME */
        CCspTimeDelta timedelta_val;    /* CCSP_TYPE_TIMEDELTA */
        CCspDate date_val;              /* CCSP_TYPE_DATE */
        CCspTime time_val;              /* CCSP_TYPE_TIME */
        CCspEnumValue enum_val;         /* CCSP_TYPE_ENUM */
        CCspStructHandle struct_val;    /* CCSP_TYPE_STRUCT (opaque handle) */
        CCspArrayValue array_val;       /* CCSP_TYPE_ARRAY */
        CCspDialectValue dialect_val;   /* CCSP_TYPE_DIALECT_GENERIC */
    };
} CCspValue;

/*
 * Initialize a CCspValue to unknown/invalid state
 */
void ccsp_value_init(CCspValue* value);

/*
 * Free any owned memory in a CCspValue.
 * Safe to call multiple times or on uninitialized values.
 */
void ccsp_value_free(CCspValue* value);

/*
 * Copy a CCspValue.
 * For owned strings/arrays, this creates a deep copy.
 */
CCspErrorCode ccsp_value_copy(CCspValue* dest, const CCspValue* src);

/*
 * Move a CCspValue (transfers ownership, source becomes invalid)
 */
void ccsp_value_move(CCspValue* dest, CCspValue* src);

/* ============================================================================
 * Type-safe setters
 * ============================================================================ */

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

/*
 * Set string value (copies the data, CCspValue owns the copy)
 */
CCspErrorCode ccsp_value_set_string(CCspValue* value, const char* data, size_t length);

/*
 * Set string value from null-terminated C string (copies the data)
 */
CCspErrorCode ccsp_value_set_string_cstr(CCspValue* value, const char* cstr);

/*
 * Set string value as a view (does NOT copy, caller must ensure data outlives value)
 */
void ccsp_value_set_string_view(CCspValue* value, const char* data, size_t length);

/*
 * Set struct value (opaque handle, CSP manages the struct)
 */
void ccsp_value_set_struct(CCspValue* value, CCspStructHandle s);

/*
 * Set enum value
 */
void ccsp_value_set_enum(CCspValue* value, int32_t ordinal, CCspEnumMetaHandle meta);

/* ============================================================================
 * Type-safe getters (return error if type mismatch)
 * ============================================================================ */

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

/*
 * Get string value (returns pointer to internal data, do not free)
 */
CCspErrorCode ccsp_value_get_string(const CCspValue* value, const char** out_data, size_t* out_length);

/*
 * Get struct handle
 */
CCspErrorCode ccsp_value_get_struct(const CCspValue* value, CCspStructHandle* out);

/*
 * Get enum value
 */
CCspErrorCode ccsp_value_get_enum(const CCspValue* value, int32_t* out_ordinal, CCspEnumMetaHandle* out_meta);

/* ============================================================================
 * Type checking
 * ============================================================================ */

/* Check if value is of a specific type */
static inline int ccsp_value_is_type(const CCspValue* value, CCspType type) {
    return value != NULL && value->type == type;
}

/* Check if value is valid (not UNKNOWN) */
static inline int ccsp_value_is_valid(const CCspValue* value) {
    return value != NULL && value->type != CCSP_TYPE_UNKNOWN;
}

/* Check if value is a numeric type */
int ccsp_value_is_numeric(const CCspValue* value);

/* Check if value is an integer type (signed or unsigned) */
int ccsp_value_is_integer(const CCspValue* value);

#ifdef __cplusplus
}
#endif

#endif /* _IN_CSP_ENGINE_C_CSPVALUE_H */
