/*
 * ABI-stable C String Types for CSP Engine
 *
 * Strings crossing the ABI boundary use a length-prefixed format
 * to avoid issues with null-terminated strings containing embedded nulls
 * (important for binary data / bytes type).
 */
#ifndef _IN_CSP_ENGINE_C_CSPSTRING_H
#define _IN_CSP_ENGINE_C_CSPSTRING_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * String view (non-owning reference to string data)
 * Use this for passing strings into CSP functions.
 * The data pointer must remain valid for the duration of the call.
 */
typedef struct {
    const char* data;   /* Pointer to string data (may contain embedded nulls) */
    size_t length;      /* Length in bytes (not including any null terminator) */
} CCspStringView;

/*
 * Owned string (CSP owns the memory)
 * Use this for strings returned from CSP functions.
 * Must be freed with ccsp_string_free().
 */
typedef struct {
    char* data;         /* Pointer to string data */
    size_t length;      /* Length in bytes */
    size_t capacity;    /* Allocated capacity (internal use) */
} CCspString;

/*
 * Create a string view from a null-terminated C string.
 * The original string must outlive the view.
 */
CCspStringView ccsp_string_view_from_cstr(const char* cstr);

/*
 * Create a string view from data and length.
 * The original data must outlive the view.
 */
CCspStringView ccsp_string_view_from_data(const char* data, size_t length);

/*
 * Create an owned string by copying the given data.
 * Returns empty string on allocation failure.
 */
CCspString ccsp_string_create(const char* data, size_t length);

/*
 * Create an owned string from a null-terminated C string.
 * Returns empty string on allocation failure.
 */
CCspString ccsp_string_create_from_cstr(const char* cstr);

/*
 * Create an empty owned string with the given capacity.
 * Useful when you need to build a string incrementally.
 */
CCspString ccsp_string_create_with_capacity(size_t capacity);

/*
 * Free an owned string's memory.
 * Safe to call on an already-freed or zero-initialized string.
 */
void ccsp_string_free(CCspString* str);

/*
 * Get a view of an owned string.
 * The view is only valid while the owned string is not modified or freed.
 */
CCspStringView ccsp_string_as_view(const CCspString* str);

/*
 * Check if a string view is empty.
 */
static inline int ccsp_string_view_is_empty(CCspStringView view) {
    return view.length == 0 || view.data == NULL;
}

/*
 * Check if an owned string is empty.
 */
static inline int ccsp_string_is_empty(const CCspString* str) {
    return str == NULL || str->length == 0 || str->data == NULL;
}

#ifdef __cplusplus
}
#endif

#endif /* _IN_CSP_ENGINE_C_CSPSTRING_H */
