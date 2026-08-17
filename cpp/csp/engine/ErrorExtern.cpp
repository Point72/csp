/*
 * Thread-local error state for the C API.
 *
 * Kept separate from the adapter shims so the value, string and time layers can be built and
 * tested without pulling in the engine.
 */

#include <csp/engine/c/CspError.h>

#include <string.h>

extern "C" {

static thread_local CCspErrorCode s_lastError = CCSP_OK;
static thread_local char s_lastErrorMessage[ 256 ] = { 0 };

CCspErrorCode ccsp_get_last_error( void )
{
    return s_lastError;
}

const char * ccsp_get_last_error_message( void )
{
    return s_lastErrorMessage[ 0 ] ? s_lastErrorMessage : nullptr;
}

void ccsp_clear_error( void )
{
    s_lastError = CCSP_OK;
    s_lastErrorMessage[ 0 ] = '\0';
}

void ccsp_set_error( CCspErrorCode code, const char * message )
{
    s_lastError = code;
    if( message )
    {
        strncpy( s_lastErrorMessage, message, sizeof( s_lastErrorMessage ) - 1 );
        s_lastErrorMessage[ sizeof( s_lastErrorMessage ) - 1 ] = '\0';
    }
    else
    {
        s_lastErrorMessage[ 0 ] = '\0';
    }
}

}
