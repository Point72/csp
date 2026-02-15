/*
 * Python integration helpers for C ABI push input adapters
 *
 * This provides utilities for creating Python capsules that wrap C input adapters.
 */
#ifndef _IN_CSP_PYTHON_C_PYINPUTADAPTER_H
#define _IN_CSP_PYTHON_C_PYINPUTADAPTER_H

#include "Python.h"
#include <csp/engine/c/InputAdapter.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Capsule name for C push input adapters */
static const char * const CSP_C_INPUT_ADAPTER_CAPSULE_NAME = "csp.c.InputAdapterCapsule";

/*
 * Create a Python capsule wrapping a push input adapter VTable.
 * The VTable is copied internally.
 *
 * @param vtable  Pointer to the VTable structure
 * @return        Python capsule object, or NULL on error
 */
static inline PyObject * ccsp_py_create_input_adapter_capsule( const CCspPushInputAdapterVTable * vtable )
{
    if( !vtable )
    {
        PyErr_SetString( PyExc_ValueError, "vtable cannot be NULL" );
        return NULL;
    }

    /* Allocate a copy of the vtable */
    CCspPushInputAdapterVTable * vtable_copy = ( CCspPushInputAdapterVTable * )malloc( sizeof( CCspPushInputAdapterVTable ) );
    if( !vtable_copy )
    {
        PyErr_NoMemory();
        return NULL;
    }
    *vtable_copy = *vtable;

    return PyCapsule_New( vtable_copy, CSP_C_INPUT_ADAPTER_CAPSULE_NAME, NULL );
}

/*
 * Extract a VTable from a Python capsule.
 *
 * @param capsule  Python capsule object
 * @return         Pointer to VTable, or NULL on error
 */
static inline CCspPushInputAdapterVTable * ccsp_py_get_input_adapter_vtable( PyObject * capsule )
{
    if( !PyCapsule_IsValid( capsule, CSP_C_INPUT_ADAPTER_CAPSULE_NAME ) )
    {
        PyErr_SetString( PyExc_TypeError, "expected input adapter capsule" );
        return NULL;
    }
    return ( CCspPushInputAdapterVTable * )PyCapsule_GetPointer( capsule, CSP_C_INPUT_ADAPTER_CAPSULE_NAME );
}

/*
 * Destructor for input adapter capsules that cleans up the VTable copy
 * and calls the destroy callback.
 */
static inline void ccsp_py_input_adapter_capsule_destructor( PyObject * capsule )
{
    CCspPushInputAdapterVTable * vtable = ( CCspPushInputAdapterVTable * )PyCapsule_GetPointer( capsule, CSP_C_INPUT_ADAPTER_CAPSULE_NAME );
    if( vtable )
    {
        if( vtable -> destroy )
        {
            vtable -> destroy( vtable -> user_data );
        }
        free( vtable );
    }
}

/*
 * Create a Python capsule wrapping a push input adapter VTable with automatic cleanup.
 * The VTable is copied internally and the destroy callback will be invoked
 * when the capsule is garbage collected.
 *
 * @param vtable  Pointer to the VTable structure
 * @return        Python capsule object, or NULL on error
 */
static inline PyObject * ccsp_py_create_input_adapter_capsule_owned( const CCspPushInputAdapterVTable * vtable )
{
    if( !vtable )
    {
        PyErr_SetString( PyExc_ValueError, "vtable cannot be NULL" );
        return NULL;
    }

    /* Allocate a copy of the vtable */
    CCspPushInputAdapterVTable * vtable_copy = ( CCspPushInputAdapterVTable * )malloc( sizeof( CCspPushInputAdapterVTable ) );
    if( !vtable_copy )
    {
        PyErr_NoMemory();
        return NULL;
    }
    *vtable_copy = *vtable;

    return PyCapsule_New( vtable_copy, CSP_C_INPUT_ADAPTER_CAPSULE_NAME, ccsp_py_input_adapter_capsule_destructor );
}

#ifdef __cplusplus
}
#endif

#endif /* _IN_CSP_PYTHON_C_PYINPUTADAPTER_H */
