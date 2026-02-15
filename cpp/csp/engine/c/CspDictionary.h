/*
 * ABI-stable C Dictionary interface for CSP Engine
 *
 * Dictionary is used to pass configuration from Python to C adapters.
 * This API provides read-only access to dictionary values.
 *
 * The Dictionary is passed to adapters via their creation functions.
 * All pointers returned from getters are borrowed and valid only while
 * the dictionary exists.
 */
#ifndef _IN_CSP_ENGINE_C_CSPDICTIONARY_H
#define _IN_CSP_ENGINE_C_CSPDICTIONARY_H

#include <csp/engine/c/CspError.h>
#include <csp/engine/c/CspTime.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Opaque Handles
 * ============================================================================ */

/* Handle to a CSP Dictionary (opaque) */
typedef struct CCspDictionaryImpl * CCspDictionaryHandle;

/* Handle to a Dictionary iterator (opaque) */
typedef struct CCspDictIteratorImpl * CCspDictIteratorHandle;

/* ============================================================================
 * Dictionary Type Enumeration
 * ============================================================================
 *
 * These correspond to the types that can be stored in a Dictionary.
 */

typedef enum {
    CCSP_DICT_TYPE_NONE = 0,      /* monostate / not found */
    CCSP_DICT_TYPE_BOOL,
    CCSP_DICT_TYPE_INT32,
    CCSP_DICT_TYPE_UINT32,
    CCSP_DICT_TYPE_INT64,
    CCSP_DICT_TYPE_UINT64,
    CCSP_DICT_TYPE_DOUBLE,
    CCSP_DICT_TYPE_STRING,
    CCSP_DICT_TYPE_DATETIME,
    CCSP_DICT_TYPE_TIMEDELTA,
    CCSP_DICT_TYPE_STRUCT_META,   /* StructMetaPtr - opaque */
    CCSP_DICT_TYPE_DIALECT,       /* DialectGenericType - opaque */
    CCSP_DICT_TYPE_DICTIONARY,    /* Nested dictionary */
    CCSP_DICT_TYPE_VECTOR,        /* Vector of values */
    CCSP_DICT_TYPE_DATA           /* Raw data pointer */
} CCspDictValueType;

/* ============================================================================
 * Dictionary Basic Operations
 * ============================================================================ */

/*
 * ccsp_dictionary_exists - Check if a key exists in the dictionary
 *
 * Parameters:
 *   dict   - Dictionary handle
 *   key    - Key to look up (null-terminated string)
 *
 * Returns:
 *   1 if the key exists, 0 if not (or if dict is NULL)
 */
int ccsp_dictionary_exists( CCspDictionaryHandle dict, const char * key );

/*
 * ccsp_dictionary_size - Get the number of entries in the dictionary
 *
 * Parameters:
 *   dict   - Dictionary handle
 *
 * Returns:
 *   Number of entries, or 0 if dict is NULL
 */
size_t ccsp_dictionary_size( CCspDictionaryHandle dict );

/*
 * ccsp_dictionary_is_empty - Check if the dictionary is empty
 *
 * Parameters:
 *   dict   - Dictionary handle
 *
 * Returns:
 *   1 if empty (or NULL), 0 if has entries
 */
int ccsp_dictionary_is_empty( CCspDictionaryHandle dict );

/*
 * ccsp_dictionary_get_type - Get the type of a value in the dictionary
 *
 * Parameters:
 *   dict   - Dictionary handle
 *   key    - Key to look up
 *
 * Returns:
 *   CCspDictValueType indicating the type, or CCSP_DICT_TYPE_NONE if not found
 */
CCspDictValueType ccsp_dictionary_get_type( CCspDictionaryHandle dict, const char * key );

/* ============================================================================
 * Type-Safe Getters (return error if type mismatch or key not found)
 * ============================================================================ */

CCspErrorCode ccsp_dictionary_get_bool( CCspDictionaryHandle dict, const char * key, int8_t * out_value );
CCspErrorCode ccsp_dictionary_get_int32( CCspDictionaryHandle dict, const char * key, int32_t * out_value );
CCspErrorCode ccsp_dictionary_get_uint32( CCspDictionaryHandle dict, const char * key, uint32_t * out_value );
CCspErrorCode ccsp_dictionary_get_int64( CCspDictionaryHandle dict, const char * key, int64_t * out_value );
CCspErrorCode ccsp_dictionary_get_uint64( CCspDictionaryHandle dict, const char * key, uint64_t * out_value );
CCspErrorCode ccsp_dictionary_get_double( CCspDictionaryHandle dict, const char * key, double * out_value );
CCspErrorCode ccsp_dictionary_get_datetime( CCspDictionaryHandle dict, const char * key, CCspDateTime * out_value );
CCspErrorCode ccsp_dictionary_get_timedelta( CCspDictionaryHandle dict, const char * key, CCspTimeDelta * out_value );

/*
 * ccsp_dictionary_get_string - Get a string value
 *
 * The returned pointer is borrowed and valid only while the dictionary exists.
 *
 * Parameters:
 *   dict       - Dictionary handle
 *   key        - Key to look up
 *   out_data   - Output: pointer to string data (NOT null-terminated in general)
 *   out_length - Output: length of the string
 *
 * Returns:
 *   CCSP_OK on success, error code on failure
 */
CCspErrorCode ccsp_dictionary_get_string( CCspDictionaryHandle dict, const char * key,
                                          const char ** out_data, size_t * out_length );

/*
 * ccsp_dictionary_get_dict - Get a nested dictionary
 *
 * The returned handle is borrowed and valid only while the parent dictionary exists.
 *
 * Parameters:
 *   dict     - Dictionary handle
 *   key      - Key to look up
 *   out_dict - Output: handle to the nested dictionary
 *
 * Returns:
 *   CCSP_OK on success, error code on failure
 */
CCspErrorCode ccsp_dictionary_get_dict( CCspDictionaryHandle dict, const char * key,
                                        CCspDictionaryHandle * out_dict );

/* ============================================================================
 * Getters with Default Values (do not error on missing keys)
 * ============================================================================
 *
 * These return the default value if the key doesn't exist or has the wrong type.
 */

int8_t   ccsp_dictionary_get_bool_or( CCspDictionaryHandle dict, const char * key, int8_t default_value );
int32_t  ccsp_dictionary_get_int32_or( CCspDictionaryHandle dict, const char * key, int32_t default_value );
uint32_t ccsp_dictionary_get_uint32_or( CCspDictionaryHandle dict, const char * key, uint32_t default_value );
int64_t  ccsp_dictionary_get_int64_or( CCspDictionaryHandle dict, const char * key, int64_t default_value );
uint64_t ccsp_dictionary_get_uint64_or( CCspDictionaryHandle dict, const char * key, uint64_t default_value );
double   ccsp_dictionary_get_double_or( CCspDictionaryHandle dict, const char * key, double default_value );
CCspDateTime ccsp_dictionary_get_datetime_or( CCspDictionaryHandle dict, const char * key, CCspDateTime default_value );
CCspTimeDelta ccsp_dictionary_get_timedelta_or( CCspDictionaryHandle dict, const char * key, CCspTimeDelta default_value );

/*
 * ccsp_dictionary_get_string_or - Get a string with default
 *
 * Note: Returns pointer to the default_value if key not found, so the
 * default_value must remain valid if you use the returned pointer.
 *
 * Parameters:
 *   dict          - Dictionary handle
 *   key           - Key to look up
 *   default_value - Default value (null-terminated C string)
 *   out_length    - Output: length of the returned string (can be NULL if not needed)
 *
 * Returns:
 *   Pointer to string data (borrowed from dict or default_value)
 */
const char * ccsp_dictionary_get_string_or( CCspDictionaryHandle dict, const char * key,
                                            const char * default_value, size_t * out_length );

/* ============================================================================
 * Dictionary Iteration
 * ============================================================================
 *
 * Iterate over all key-value pairs in the dictionary.
 *
 * Example:
 *   CCspDictIteratorHandle iter = ccsp_dictionary_iter_create( dict );
 *   const char * key;
 *   while( ccsp_dictionary_iter_next( iter, &key ) )
 *   {
 *       CCspDictValueType type = ccsp_dictionary_iter_value_type( iter );
 *       // ... use key and type ...
 *   }
 *   ccsp_dictionary_iter_destroy( iter );
 */

/*
 * ccsp_dictionary_iter_create - Create an iterator for the dictionary
 *
 * Parameters:
 *   dict - Dictionary handle
 *
 * Returns:
 *   Iterator handle, or NULL on error
 */
CCspDictIteratorHandle ccsp_dictionary_iter_create( CCspDictionaryHandle dict );

/*
 * ccsp_dictionary_iter_destroy - Destroy an iterator
 *
 * Parameters:
 *   iter - Iterator handle
 */
void ccsp_dictionary_iter_destroy( CCspDictIteratorHandle iter );

/*
 * ccsp_dictionary_iter_next - Advance to the next entry
 *
 * Parameters:
 *   iter    - Iterator handle
 *   out_key - Output: pointer to the key string (borrowed, valid until next call)
 *
 * Returns:
 *   1 if there is a next entry, 0 if iteration is complete
 */
int ccsp_dictionary_iter_next( CCspDictIteratorHandle iter, const char ** out_key );

/*
 * ccsp_dictionary_iter_value_type - Get the type of the current value
 *
 * Must be called after ccsp_dictionary_iter_next returns 1.
 *
 * Parameters:
 *   iter - Iterator handle
 *
 * Returns:
 *   Type of the current value
 */
CCspDictValueType ccsp_dictionary_iter_value_type( CCspDictIteratorHandle iter );

/* Type-safe value getters for current iterator position */
CCspErrorCode ccsp_dictionary_iter_get_bool( CCspDictIteratorHandle iter, int8_t * out_value );
CCspErrorCode ccsp_dictionary_iter_get_int32( CCspDictIteratorHandle iter, int32_t * out_value );
CCspErrorCode ccsp_dictionary_iter_get_uint32( CCspDictIteratorHandle iter, uint32_t * out_value );
CCspErrorCode ccsp_dictionary_iter_get_int64( CCspDictIteratorHandle iter, int64_t * out_value );
CCspErrorCode ccsp_dictionary_iter_get_uint64( CCspDictIteratorHandle iter, uint64_t * out_value );
CCspErrorCode ccsp_dictionary_iter_get_double( CCspDictIteratorHandle iter, double * out_value );
CCspErrorCode ccsp_dictionary_iter_get_datetime( CCspDictIteratorHandle iter, CCspDateTime * out_value );
CCspErrorCode ccsp_dictionary_iter_get_timedelta( CCspDictIteratorHandle iter, CCspTimeDelta * out_value );
CCspErrorCode ccsp_dictionary_iter_get_string( CCspDictIteratorHandle iter, const char ** out_data, size_t * out_length );
CCspErrorCode ccsp_dictionary_iter_get_dict( CCspDictIteratorHandle iter, CCspDictionaryHandle * out_dict );

#ifdef __cplusplus
}
#endif

#endif /* _IN_CSP_ENGINE_C_CSPDICTIONARY_H */
