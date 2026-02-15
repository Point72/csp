/*
 * C API for CSP Struct Access
 *
 * This header provides C-compatible access to CSP Structs and StructMeta.
 * Structs are CSP's structured data type with named, typed fields.
 *
 * Key concepts:
 * - StructMeta: Type information (field names, types, offsets)
 * - Struct: Instance data (actual field values)
 * - StructField: Individual field access (get/set values)
 *
 * All pointers returned are borrowed references owned by the parent object.
 * Do not free them unless explicitly documented.
 */

#ifndef _IN_CSP_ENGINE_C_CSPSTRUCT_H
#define _IN_CSP_ENGINE_C_CSPSTRUCT_H

#include <csp/engine/c/CspError.h>
#include <csp/engine/c/CspType.h>
#include <csp/engine/c/CspTime.h>
#include <csp/engine/c/CspValue.h>  /* For CCspStructHandle */
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Opaque Handle Types
 * ============================================================================ */

/* Handle to csp::StructMeta (type information) */
typedef void * CCspStructMetaHandle;

/* CCspStructHandle is defined in CspValue.h as it's used for struct values */

/* Handle to csp::StructField (field metadata) */
typedef void * CCspStructFieldHandle;

/* ============================================================================
 * StructMeta Functions (Type Information)
 * ============================================================================ */

/*
 * ccsp_struct_meta_name - Get the name of a struct type
 *
 * Parameters:
 *   meta - Handle to StructMeta
 *
 * Returns:
 *   Pointer to null-terminated string (borrowed, do not free)
 *   NULL if meta is NULL
 */
const char * ccsp_struct_meta_name( CCspStructMetaHandle meta );

/*
 * ccsp_struct_meta_field_count - Get the number of fields in a struct type
 *
 * Parameters:
 *   meta - Handle to StructMeta
 *
 * Returns:
 *   Number of fields, or 0 if meta is NULL
 */
size_t ccsp_struct_meta_field_count( CCspStructMetaHandle meta );

/*
 * ccsp_struct_meta_field_by_index - Get field handle by index
 *
 * Parameters:
 *   meta  - Handle to StructMeta
 *   index - Field index (0-based)
 *
 * Returns:
 *   Handle to StructField, or NULL if index out of range
 */
CCspStructFieldHandle ccsp_struct_meta_field_by_index( CCspStructMetaHandle meta, size_t index );

/*
 * ccsp_struct_meta_field_by_name - Get field handle by name
 *
 * Parameters:
 *   meta - Handle to StructMeta
 *   name - Field name (null-terminated)
 *
 * Returns:
 *   Handle to StructField, or NULL if not found
 */
CCspStructFieldHandle ccsp_struct_meta_field_by_name( CCspStructMetaHandle meta, const char * name );

/*
 * ccsp_struct_meta_field_name_by_index - Get field name by index
 *
 * Parameters:
 *   meta  - Handle to StructMeta
 *   index - Field index (0-based)
 *
 * Returns:
 *   Pointer to null-terminated string (borrowed, do not free)
 *   NULL if index out of range
 */
const char * ccsp_struct_meta_field_name_by_index( CCspStructMetaHandle meta, size_t index );

/*
 * ccsp_struct_meta_is_strict - Check if struct type is strict
 *
 * Strict structs require all fields to be set.
 *
 * Parameters:
 *   meta - Handle to StructMeta
 *
 * Returns:
 *   1 if strict, 0 otherwise
 */
int ccsp_struct_meta_is_strict( CCspStructMetaHandle meta );

/* ============================================================================
 * StructField Functions (Field Metadata)
 * ============================================================================ */

/*
 * ccsp_struct_field_name - Get the name of a field
 *
 * Parameters:
 *   field - Handle to StructField
 *
 * Returns:
 *   Pointer to null-terminated string (borrowed, do not free)
 *   NULL if field is NULL
 */
const char * ccsp_struct_field_name( CCspStructFieldHandle field );

/*
 * ccsp_struct_field_type - Get the CSP type of a field
 *
 * Parameters:
 *   field - Handle to StructField
 *
 * Returns:
 *   CCspType enum value, or CCSP_TYPE_UNKNOWN if field is NULL
 */
CCspType ccsp_struct_field_type( CCspStructFieldHandle field );

/*
 * ccsp_struct_field_is_optional - Check if a field is optional
 *
 * Parameters:
 *   field - Handle to StructField
 *
 * Returns:
 *   1 if optional, 0 otherwise
 */
int ccsp_struct_field_is_optional( CCspStructFieldHandle field );

/* ============================================================================
 * Struct Instance Functions
 * ============================================================================ */

/*
 * ccsp_struct_meta - Get the StructMeta for a struct instance
 *
 * Parameters:
 *   s - Handle to Struct
 *
 * Returns:
 *   Handle to StructMeta, or NULL if s is NULL
 */
CCspStructMetaHandle ccsp_struct_meta( CCspStructHandle s );

/*
 * ccsp_struct_field_is_set - Check if a field is set on a struct instance
 *
 * Parameters:
 *   s     - Handle to Struct
 *   field - Handle to StructField
 *
 * Returns:
 *   1 if set, 0 if not set or on error
 */
int ccsp_struct_field_is_set( CCspStructHandle s, CCspStructFieldHandle field );

/*
 * ccsp_struct_field_is_none - Check if an optional field is explicitly None
 *
 * Parameters:
 *   s     - Handle to Struct
 *   field - Handle to StructField
 *
 * Returns:
 *   1 if None, 0 otherwise
 */
int ccsp_struct_field_is_none( CCspStructHandle s, CCspStructFieldHandle field );

/* ============================================================================
 * Field Value Getters
 *
 * All getters return CCSP_OK on success or an error code.
 * CCSP_ERROR_KEY_NOT_FOUND is returned if the field is not set.
 * CCSP_ERROR_TYPE_MISMATCH is returned if the field type doesn't match.
 * ============================================================================ */

CCspErrorCode ccsp_struct_get_bool( CCspStructHandle s, CCspStructFieldHandle field, int8_t * out_value );
CCspErrorCode ccsp_struct_get_int8( CCspStructHandle s, CCspStructFieldHandle field, int8_t * out_value );
CCspErrorCode ccsp_struct_get_uint8( CCspStructHandle s, CCspStructFieldHandle field, uint8_t * out_value );
CCspErrorCode ccsp_struct_get_int16( CCspStructHandle s, CCspStructFieldHandle field, int16_t * out_value );
CCspErrorCode ccsp_struct_get_uint16( CCspStructHandle s, CCspStructFieldHandle field, uint16_t * out_value );
CCspErrorCode ccsp_struct_get_int32( CCspStructHandle s, CCspStructFieldHandle field, int32_t * out_value );
CCspErrorCode ccsp_struct_get_uint32( CCspStructHandle s, CCspStructFieldHandle field, uint32_t * out_value );
CCspErrorCode ccsp_struct_get_int64( CCspStructHandle s, CCspStructFieldHandle field, int64_t * out_value );
CCspErrorCode ccsp_struct_get_uint64( CCspStructHandle s, CCspStructFieldHandle field, uint64_t * out_value );
CCspErrorCode ccsp_struct_get_double( CCspStructHandle s, CCspStructFieldHandle field, double * out_value );
CCspErrorCode ccsp_struct_get_datetime( CCspStructHandle s, CCspStructFieldHandle field, CCspDateTime * out_value );
CCspErrorCode ccsp_struct_get_timedelta( CCspStructHandle s, CCspStructFieldHandle field, CCspTimeDelta * out_value );

/*
 * ccsp_struct_get_string - Get a string field value
 *
 * Parameters:
 *   s          - Handle to Struct
 *   field      - Handle to StructField
 *   out_data   - Output pointer to string data (borrowed, valid while struct exists)
 *   out_length - Output string length in bytes
 *
 * Returns:
 *   CCSP_OK on success, error code on failure
 */
CCspErrorCode ccsp_struct_get_string( CCspStructHandle s, CCspStructFieldHandle field,
                                      const char ** out_data, size_t * out_length );

/*
 * ccsp_struct_get_enum - Get an enum field value (as ordinal)
 *
 * Parameters:
 *   s           - Handle to Struct
 *   field       - Handle to StructField
 *   out_ordinal - Output enum ordinal value
 *
 * Returns:
 *   CCSP_OK on success, error code on failure
 */
CCspErrorCode ccsp_struct_get_enum( CCspStructHandle s, CCspStructFieldHandle field, int32_t * out_ordinal );

/*
 * ccsp_struct_get_struct - Get a nested struct field value
 *
 * Parameters:
 *   s          - Handle to Struct
 *   field      - Handle to StructField
 *   out_struct - Output handle to nested struct (borrowed, do not free)
 *
 * Returns:
 *   CCSP_OK on success, error code on failure
 */
CCspErrorCode ccsp_struct_get_struct( CCspStructHandle s, CCspStructFieldHandle field,
                                      CCspStructHandle * out_struct );

/* ============================================================================
 * Field Value Getters by Name (Convenience)
 *
 * These combine field lookup and value access in one call.
 * Less efficient than caching the field handle for repeated access.
 * ============================================================================ */

CCspErrorCode ccsp_struct_get_bool_by_name( CCspStructHandle s, const char * name, int8_t * out_value );
CCspErrorCode ccsp_struct_get_int32_by_name( CCspStructHandle s, const char * name, int32_t * out_value );
CCspErrorCode ccsp_struct_get_int64_by_name( CCspStructHandle s, const char * name, int64_t * out_value );
CCspErrorCode ccsp_struct_get_double_by_name( CCspStructHandle s, const char * name, double * out_value );
CCspErrorCode ccsp_struct_get_datetime_by_name( CCspStructHandle s, const char * name, CCspDateTime * out_value );
CCspErrorCode ccsp_struct_get_string_by_name( CCspStructHandle s, const char * name,
                                              const char ** out_data, size_t * out_length );

/* ============================================================================
 * Field Value Setters
 *
 * All setters return CCSP_OK on success or an error code.
 * CCSP_ERROR_TYPE_MISMATCH is returned if the field type doesn't match.
 * ============================================================================ */

CCspErrorCode ccsp_struct_set_bool( CCspStructHandle s, CCspStructFieldHandle field, int8_t value );
CCspErrorCode ccsp_struct_set_int8( CCspStructHandle s, CCspStructFieldHandle field, int8_t value );
CCspErrorCode ccsp_struct_set_uint8( CCspStructHandle s, CCspStructFieldHandle field, uint8_t value );
CCspErrorCode ccsp_struct_set_int16( CCspStructHandle s, CCspStructFieldHandle field, int16_t value );
CCspErrorCode ccsp_struct_set_uint16( CCspStructHandle s, CCspStructFieldHandle field, uint16_t value );
CCspErrorCode ccsp_struct_set_int32( CCspStructHandle s, CCspStructFieldHandle field, int32_t value );
CCspErrorCode ccsp_struct_set_uint32( CCspStructHandle s, CCspStructFieldHandle field, uint32_t value );
CCspErrorCode ccsp_struct_set_int64( CCspStructHandle s, CCspStructFieldHandle field, int64_t value );
CCspErrorCode ccsp_struct_set_uint64( CCspStructHandle s, CCspStructFieldHandle field, uint64_t value );
CCspErrorCode ccsp_struct_set_double( CCspStructHandle s, CCspStructFieldHandle field, double value );
CCspErrorCode ccsp_struct_set_datetime( CCspStructHandle s, CCspStructFieldHandle field, CCspDateTime value );
CCspErrorCode ccsp_struct_set_timedelta( CCspStructHandle s, CCspStructFieldHandle field, CCspTimeDelta value );

/*
 * ccsp_struct_set_string - Set a string field value
 *
 * The string data is copied into the struct.
 *
 * Parameters:
 *   s      - Handle to Struct
 *   field  - Handle to StructField
 *   data   - String data
 *   length - String length in bytes
 *
 * Returns:
 *   CCSP_OK on success, error code on failure
 */
CCspErrorCode ccsp_struct_set_string( CCspStructHandle s, CCspStructFieldHandle field,
                                      const char * data, size_t length );

/*
 * ccsp_struct_set_enum - Set an enum field value (by ordinal)
 *
 * Parameters:
 *   s       - Handle to Struct
 *   field   - Handle to StructField
 *   ordinal - Enum ordinal value
 *
 * Returns:
 *   CCSP_OK on success, error code on failure
 */
CCspErrorCode ccsp_struct_set_enum( CCspStructHandle s, CCspStructFieldHandle field, int32_t ordinal );

/* ============================================================================
 * Struct Creation (if needed by adapters)
 * ============================================================================ */

/*
 * ccsp_struct_create - Create a new struct instance from StructMeta
 *
 * The returned struct must be freed with ccsp_struct_destroy when done.
 *
 * Parameters:
 *   meta - Handle to StructMeta
 *
 * Returns:
 *   Handle to new Struct, or NULL on error
 */
CCspStructHandle ccsp_struct_create( CCspStructMetaHandle meta );

/*
 * ccsp_struct_destroy - Destroy a struct created with ccsp_struct_create
 *
 * Do NOT use this on structs obtained from other sources (e.g., from
 * ccsp_struct_get_struct or from input values).
 *
 * Parameters:
 *   s - Handle to Struct to destroy
 */
void ccsp_struct_destroy( CCspStructHandle s );

/*
 * ccsp_struct_copy - Create a deep copy of a struct
 *
 * The returned struct must be freed with ccsp_struct_destroy.
 *
 * Parameters:
 *   s - Handle to Struct to copy
 *
 * Returns:
 *   Handle to new Struct copy, or NULL on error
 */
CCspStructHandle ccsp_struct_copy( CCspStructHandle s );

#ifdef __cplusplus
}
#endif

#endif /* _IN_CSP_ENGINE_C_CSPSTRUCT_H */
