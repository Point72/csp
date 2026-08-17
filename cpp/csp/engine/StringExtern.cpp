/*
 * Implementation of the C API string types.
 *
 * Owned strings are allocated with malloc so a C consumer can reason about them, and always keep
 * a null terminator one byte past the length so the data can also be used as a C string when it
 * contains no embedded nulls.
 */

#include <csp/engine/c/CspString.h>

#include <stdlib.h>
#include <string.h>

extern "C" {

CCspStringView ccsp_string_view_from_cstr( const char * cstr )
{
    CCspStringView view;
    view.data = cstr;
    view.length = cstr ? strlen( cstr ) : 0;
    return view;
}

CCspStringView ccsp_string_view_from_data( const char * data, size_t length )
{
    CCspStringView view;
    view.data = data;
    view.length = data ? length : 0;
    return view;
}

CCspString ccsp_string_create( const char * data, size_t length )
{
    CCspString str;
    str.data = NULL;
    str.length = 0;
    str.capacity = 0;

    if( !data && length > 0 )
        return str;

    /* Guard the +1 for the terminator */
    if( length == SIZE_MAX )
        return str;

    char * buffer = ( char * ) malloc( length + 1 );
    if( !buffer )
        return str;

    if( length > 0 )
        memcpy( buffer, data, length );
    buffer[ length ] = '\0';

    str.data = buffer;
    str.length = length;
    str.capacity = length + 1;
    return str;
}

CCspString ccsp_string_create_from_cstr( const char * cstr )
{
    return ccsp_string_create( cstr, cstr ? strlen( cstr ) : 0 );
}

CCspString ccsp_string_create_with_capacity( size_t capacity )
{
    CCspString str;
    str.data = NULL;
    str.length = 0;
    str.capacity = 0;

    if( capacity == SIZE_MAX )
        return str;

    char * buffer = ( char * ) malloc( capacity + 1 );
    if( !buffer )
        return str;

    buffer[ 0 ] = '\0';
    str.data = buffer;
    str.length = 0;
    str.capacity = capacity + 1;
    return str;
}

void ccsp_string_free( CCspString * str )
{
    if( !str )
        return;

    free( str -> data );
    str -> data = NULL;
    str -> length = 0;
    str -> capacity = 0;
}

CCspStringView ccsp_string_as_view( const CCspString * str )
{
    CCspStringView view;
    view.data = str ? str -> data : NULL;
    view.length = str ? str -> length : 0;
    return view;
}

}
