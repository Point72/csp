/*
 * Python integration helpers for C ABI adapter managers
 *
 * This provides utilities for creating Python capsules that wrap C adapter managers.
 */
#ifndef _IN_CSP_PYTHON_C_PYADAPTERMANAGER_H
#define _IN_CSP_PYTHON_C_PYADAPTERMANAGER_H

#include "Python.h"
#include <csp/engine/c/AdapterManager.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Capsule name for C adapter managers */
static const char * const CSP_C_ADAPTER_MANAGER_CAPSULE_NAME = "csp.c.AdapterManagerCapsule";

/*
 * Capsule context marking that user_data ownership has moved to the engine-owned manager.
 * The capsule still owns the VTable copy and must free it with its own allocator, but must not
 * run the destroy callback a second time.
 */
#ifndef CCSP_PY_CAPSULE_TRANSFERRED
#define CCSP_PY_CAPSULE_TRANSFERRED ( ( void * ) 1 )
#endif

/*
 * Extract a VTable from a Python capsule.
 *
 * @param capsule  Python capsule object
 * @return         Pointer to VTable, or NULL on error
 */
static inline CCspAdapterManagerVTable * ccsp_py_get_adapter_manager_vtable( PyObject * capsule )
{
    if( !PyCapsule_IsValid( capsule, CSP_C_ADAPTER_MANAGER_CAPSULE_NAME ) )
    {
        PyErr_SetString( PyExc_TypeError, "expected adapter manager capsule" );
        return NULL;
    }
    return ( CCspAdapterManagerVTable * ) PyCapsule_GetPointer( capsule, CSP_C_ADAPTER_MANAGER_CAPSULE_NAME );
}

/*
 * Destructor for adapter manager capsules that frees the VTable copy, and calls the destroy
 * callback unless ownership of user_data has already been transferred.
 */
static inline void ccsp_py_adapter_manager_capsule_destructor( PyObject * capsule )
{
    CCspAdapterManagerVTable * vtable = ( CCspAdapterManagerVTable * ) PyCapsule_GetPointer( capsule, CSP_C_ADAPTER_MANAGER_CAPSULE_NAME );
    if( vtable )
    {
        if( vtable -> destroy && PyCapsule_GetContext( capsule ) != CCSP_PY_CAPSULE_TRANSFERRED )
        {
            vtable -> destroy( vtable -> user_data );
        }
        free( vtable );
    }
}

/*
 * Create a Python capsule wrapping an adapter manager VTable with automatic cleanup.
 * The VTable is copied internally and the destroy callback will be invoked
 * when the capsule is garbage collected.
 *
 * @param vtable  Pointer to the VTable structure
 * @return        Python capsule object, or NULL on error
 */
static inline PyObject * ccsp_py_create_adapter_manager_capsule_owned( const CCspAdapterManagerVTable * vtable )
{
    if( !vtable )
    {
        PyErr_SetString( PyExc_ValueError, "vtable cannot be NULL" );
        return NULL;
    }

    /* Allocate a copy of the vtable */
    CCspAdapterManagerVTable * vtable_copy = ( CCspAdapterManagerVTable * ) malloc( sizeof( CCspAdapterManagerVTable ) );
    if( !vtable_copy )
    {
        PyErr_NoMemory();
        return NULL;
    }

    if( !ccsp_vtable_adopt( vtable_copy, sizeof( CCspAdapterManagerVTable ), vtable ) )
    {
        free( vtable_copy );
        PyErr_SetString( PyExc_ValueError, "vtable was not initialized with CCSP_VTABLE_INIT, or has an unsupported ABI version" );
        return NULL;
    }

    PyObject * capsule = PyCapsule_New( vtable_copy, CSP_C_ADAPTER_MANAGER_CAPSULE_NAME, ccsp_py_adapter_manager_capsule_destructor );
    if( !capsule )
    {
        free( vtable_copy );
        return NULL;
    }
    return capsule;
}

#ifdef __cplusplus
}
#endif

#endif /* _IN_CSP_PYTHON_C_PYADAPTERMANAGER_H */
