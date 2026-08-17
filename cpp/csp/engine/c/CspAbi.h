/*
 * ABI versioning for the C API callback tables.
 *
 * Every callback table begins with these two fields, in this order:
 *
 *     uint32_t abi_version;   set to CCSP_ABI_VERSION
 *     uint32_t struct_size;   set to sizeof( the table )
 *
 * This lets CSP accept a table from an adapter built against an older header: it copies only the
 * bytes that adapter actually provided and zero-fills the callbacks it does not know about.
 */
#ifndef _IN_CSP_ENGINE_C_CSPABI_H
#define _IN_CSP_ENGINE_C_CSPABI_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Bump when a field is appended to any callback table */
#define CCSP_ABI_VERSION 1u

/* Zero a callback table and stamp its ABI fields */
#define CCSP_VTABLE_INIT( vtable_ptr, vtable_type )                         \
    do {                                                                    \
        memset( ( vtable_ptr ), 0, sizeof( vtable_type ) );                 \
        ( vtable_ptr ) -> abi_version = CCSP_ABI_VERSION;                   \
        ( vtable_ptr ) -> struct_size = ( uint32_t ) sizeof( vtable_type ); \
    } while( 0 )

/*
 * Copy a caller-supplied callback table into a full-size table of this build.
 *
 * Returns 0 if the table cannot be interpreted, which happens when it was produced by a newer
 * header than this build understands, or when the header fields are not filled in at all.
 */
static inline int ccsp_vtable_adopt( void * dest, size_t dest_size, const void * src )
{
    uint32_t abi_version = 0;
    uint32_t struct_size = 0;

    if( !dest || !src || dest_size < 2 * sizeof( uint32_t ) )
        return 0;

    memcpy( &abi_version, src, sizeof( abi_version ) );
    memcpy( &struct_size, ( const char * ) src + sizeof( uint32_t ), sizeof( struct_size ) );

    if( abi_version == 0 || abi_version > CCSP_ABI_VERSION )
        return 0;
    if( struct_size < 2 * sizeof( uint32_t ) )
        return 0;

    memset( dest, 0, dest_size );
    memcpy( dest, src, struct_size < dest_size ? struct_size : dest_size );

    /* Record the size this build actually holds, not the caller's */
    struct_size = ( uint32_t ) dest_size;
    memcpy( ( char * ) dest + sizeof( uint32_t ), &struct_size, sizeof( struct_size ) );

    return 1;
}

#ifdef __cplusplus
}
#endif

#endif /* _IN_CSP_ENGINE_C_CSPABI_H */
