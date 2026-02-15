/*
 * ABI-stable C Error Handling for CSP Engine
 *
 * This provides consistent error reporting across the C API boundary.
 * Errors are stored in thread-local storage for retrieval.
 */
#ifndef _IN_CSP_ENGINE_C_CSPERROR_H
#define _IN_CSP_ENGINE_C_CSPERROR_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Error codes */
typedef enum {
    CCSP_OK = 0,
    CCSP_ERROR_NULL_POINTER,
    CCSP_ERROR_TYPE_MISMATCH,
    CCSP_ERROR_KEY_NOT_FOUND,
    CCSP_ERROR_INVALID_ARGUMENT,
    CCSP_ERROR_OUT_OF_MEMORY,
    CCSP_ERROR_OUT_OF_RANGE,
    CCSP_ERROR_RUNTIME,
    CCSP_ERROR_VALUE,
    CCSP_ERROR_NOT_IMPLEMENTED,
    CCSP_ERROR_UNKNOWN
} CCspErrorCode;

/* Get the last error code for the current thread */
CCspErrorCode ccsp_get_last_error(void);

/* Get the last error message for the current thread (may be NULL) */
const char* ccsp_get_last_error_message(void);

/* Clear the last error for the current thread */
void ccsp_clear_error(void);

/*
 * Set an error (for adapter implementations).
 * The message is copied internally.
 */
void ccsp_set_error(CCspErrorCode code, const char* message);

/*
 * Helper macro for checking and returning on error
 */
#define CCSP_RETURN_IF_ERROR(expr) \
    do { \
        CCspErrorCode _err = (expr); \
        if (_err != CCSP_OK) return _err; \
    } while(0)

#define CCSP_RETURN_NULL_IF_ERROR(expr) \
    do { \
        CCspErrorCode _err = (expr); \
        if (_err != CCSP_OK) return NULL; \
    } while(0)

#ifdef __cplusplus
}
#endif

#endif /* _IN_CSP_ENGINE_C_CSPERROR_H */
