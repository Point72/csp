#ifndef _IN_CSP_ENGINE_EXTERNBOUNDARY_H
#define _IN_CSP_ENGINE_EXTERNBOUNDARY_H

#include <csp/core/Exception.h>
#include <csp/engine/c/CspError.h>

/*
 * Exception barriers for the two directions of the C API boundary.
 *
 * Exceptions must never unwind through a frame with C linkage, so both an engine callback into
 * user C code and a user C call into the engine have to be contained where they cross over.
 */

/* Engine code invoking a user-supplied callback: report the failure as an engine error.
 * Destructors and other teardown paths cannot use this; they swallow with a bare catch instead. */
#define CSP_CATCH_EXTERN_CALLBACK( desc )                                                         \
    catch( const std::exception & e )                                                             \
    {                                                                                             \
        CSP_THROW( csp::RuntimeException, "C API " desc " callback failed: " << e.what() );       \
    }                                                                                             \
    catch( ... )                                                                                  \
    {                                                                                             \
        CSP_THROW( csp::RuntimeException, "C API " desc " callback threw an unknown exception" ); \
    }

/* User C code calling into the engine, for entry points returning CCspErrorCode */
#define CCSP_CATCH_ERR                                                                            \
    catch( const std::exception & ccsp_ex_ )                                                      \
    {                                                                                             \
        ccsp_set_error( CCSP_ERROR_RUNTIME, ccsp_ex_.what() );                                    \
        return CCSP_ERROR_RUNTIME;                                                                \
    }                                                                                             \
    catch( ... )                                                                                  \
    {                                                                                             \
        ccsp_set_error( CCSP_ERROR_UNKNOWN, "unknown error" );                                    \
        return CCSP_ERROR_UNKNOWN;                                                                \
    }

/* Same, for entry points whose return type cannot carry an error code */
#define CCSP_CATCH_RET( errval )                                                                  \
    catch( const std::exception & ccsp_ex_ )                                                      \
    {                                                                                             \
        ccsp_set_error( CCSP_ERROR_RUNTIME, ccsp_ex_.what() );                                    \
        return errval;                                                                            \
    }                                                                                             \
    catch( ... )                                                                                  \
    {                                                                                             \
        ccsp_set_error( CCSP_ERROR_UNKNOWN, "unknown error" );                                    \
        return errval;                                                                            \
    }

#endif
