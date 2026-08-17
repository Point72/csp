/*
 * ABI-stable C Error Handling for CSP Engine
 *
 * This provides consistent error reporting across the C API boundary.
 * Errors are stored in thread-local storage for retrieval.
 */
#ifndef _IN_CSP_ENGINE_C_CSPERROR_H
#define _IN_CSP_ENGINE_C_CSPERROR_H

#include <csp/engine/c/CspExport.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Error codes */
typedef enum {
    CCSP_OK = 0,
    CCSP_ERROR_NULL_POINTER = 1,
    CCSP_ERROR_TYPE_MISMATCH = 2,
    CCSP_ERROR_KEY_NOT_FOUND = 3,
    CCSP_ERROR_INVALID_ARGUMENT = 4,
    CCSP_ERROR_OUT_OF_MEMORY = 5,
    CCSP_ERROR_OUT_OF_RANGE = 6,
    CCSP_ERROR_RUNTIME = 7,
    CCSP_ERROR_VALUE = 8,
    CCSP_ERROR_NOT_IMPLEMENTED = 9,
    CCSP_ERROR_UNKNOWN = 10
} CCspErrorCode;

/* Get the last error code for the current thread */
CSP_C_API_EXPORT CCspErrorCode ccsp_get_last_error( void );

/* Get the last error message for the current thread (may be NULL) */
CSP_C_API_EXPORT const char * ccsp_get_last_error_message( void );

/* Clear the last error for the current thread */
CSP_C_API_EXPORT void ccsp_clear_error( void );

/*
 * Set an error (for adapter implementations).
 * The message is copied internally.
 */
CSP_C_API_EXPORT void ccsp_set_error( CCspErrorCode code, const char * message );

/*
 * Helper macro for checking and returning on error
 */
#define CCSP_RETURN_IF_ERROR( expr ) \
    do { \
        CCspErrorCode _err = ( expr ); \
        if ( _err != CCSP_OK ) return _err; \
    } while(0)

#define CCSP_RETURN_NULL_IF_ERROR( expr ) \
    do { \
        CCspErrorCode _err = ( expr ); \
        if ( _err != CCSP_OK ) return NULL; \
    } while(0)

#ifdef __cplusplus
}
#endif

#endif /* _IN_CSP_ENGINE_C_CSPERROR_H */
