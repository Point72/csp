/*
 * Python integration helpers for C ABI output adapters
 *
 * This provides utilities for creating Python capsules that wrap C output adapters.
 */
#ifndef _IN_CSP_PYTHON_C_PYOUTPUTADAPTER_H
#define _IN_CSP_PYTHON_C_PYOUTPUTADAPTER_H

#include "Python.h"
#include <csp/engine/c/OutputAdapter.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Capsule name for C output adapters */
static const char * const CSP_C_OUTPUT_ADAPTER_CAPSULE_NAME = "csp.c.OutputAdapterCapsule";

/*
 * Create a Python capsule wrapping an output adapter VTable.
 * The VTable is copied internally.
 *
 * @param vtable  Pointer to the VTable structure
 * @return        Python capsule object, or NULL on error
 */
static inline PyObject * ccsp_py_create_output_adapter_capsule( const CCspOutputAdapterVTable * vtable )
{
    if( !vtable )
    {
        PyErr_SetString( PyExc_ValueError, "vtable cannot be NULL" );
        return NULL;
    }

    /* Allocate a copy of the vtable */
    CCspOutputAdapterVTable * vtable_copy = ( CCspOutputAdapterVTable * )malloc( sizeof( CCspOutputAdapterVTable ) );
    if( !vtable_copy )
    {
        PyErr_NoMemory();
        return NULL;
    }
    *vtable_copy = *vtable;

    return PyCapsule_New( vtable_copy, CSP_C_OUTPUT_ADAPTER_CAPSULE_NAME, NULL );
}

/*
 * Extract a VTable from a Python capsule.
 *
 * @param capsule  Python capsule object
 * @return         Pointer to VTable, or NULL on error
 */
static inline CCspOutputAdapterVTable * ccsp_py_get_output_adapter_vtable( PyObject * capsule )
{
    if( !PyCapsule_IsValid( capsule, CSP_C_OUTPUT_ADAPTER_CAPSULE_NAME ) )
    {
        PyErr_SetString( PyExc_TypeError, "expected output adapter capsule" );
        return NULL;
    }
    return ( CCspOutputAdapterVTable * )PyCapsule_GetPointer( capsule, CSP_C_OUTPUT_ADAPTER_CAPSULE_NAME );
}

/*
 * Destructor for output adapter capsules that cleans up the VTable copy
 * and calls the destroy callback.
 */
static inline void ccsp_py_output_adapter_capsule_destructor( PyObject * capsule )
{
    CCspOutputAdapterVTable * vtable = ( CCspOutputAdapterVTable * )PyCapsule_GetPointer( capsule, CSP_C_OUTPUT_ADAPTER_CAPSULE_NAME );
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
 * Create a Python capsule wrapping an output adapter VTable with automatic cleanup.
 * The VTable is copied internally and the destroy callback will be invoked
 * when the capsule is garbage collected.
 *
 * @param vtable  Pointer to the VTable structure
 * @return        Python capsule object, or NULL on error
 */
static inline PyObject * ccsp_py_create_output_adapter_capsule_owned( const CCspOutputAdapterVTable * vtable )
{
    if( !vtable )
    {
        PyErr_SetString( PyExc_ValueError, "vtable cannot be NULL" );
        return NULL;
    }

    /* Allocate a copy of the vtable */
    CCspOutputAdapterVTable * vtable_copy = ( CCspOutputAdapterVTable * )malloc( sizeof( CCspOutputAdapterVTable ) );
    if( !vtable_copy )
    {
        PyErr_NoMemory();
        return NULL;
    }
    *vtable_copy = *vtable;

    return PyCapsule_New( vtable_copy, CSP_C_OUTPUT_ADAPTER_CAPSULE_NAME, ccsp_py_output_adapter_capsule_destructor );
}

#ifdef __cplusplus
}
}
#endif

#endif /* _IN_CSP_PYTHON_C_PYOUTPUTADAPTER_H */

