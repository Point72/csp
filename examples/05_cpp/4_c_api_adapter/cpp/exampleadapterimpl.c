/*
 * Python bindings for the example C adapters
 */
#include <stddef.h>
#include <stdlib.h>
#include "Python.h"

#include <csp/engine/c/OutputAdapter.h>
#include <csp/engine/c/InputAdapter.h>
#include <csp/engine/c/AdapterManager.h>
#include <csp/engine/c/CspStruct.h>
#include <csp/engine/c/CspError.h>
#include <csp/python/c/PyOutputAdapter.h>
#include <csp/python/c/PyInputAdapter.h>
#include <csp/python/c/PyAdapterManager.h>

#include "ExampleOutputAdapter.h"
#include "ExamplePushInputAdapter.h"
#include "ExampleManagedAdapter.h"

/*
 * _example_adapter_manager(engine, properties: dict) -> capsule
 *
 * Create an example adapter manager.
 */
static PyObject * create_adapter_manager_py( PyObject * self, PyObject * args )
{
    ( void )self;

    PyObject * engine_capsule = NULL;
    PyObject * properties = NULL;

    if( !PyArg_ParseTuple( args, "OO", &engine_capsule, &properties ) )
    {
        return NULL;
    }

    /* Extract prefix from properties dict */
    const char * prefix = "";
    if( properties && PyDict_Check( properties ) )
    {
        PyObject * prefix_obj = PyDict_GetItemString( properties, "prefix" );
        if( prefix_obj && PyUnicode_Check( prefix_obj ) )
        {
            prefix = PyUnicode_AsUTF8( prefix_obj );
        }
    }

    CCspAdapterManagerVTable vtable = example_managed_adapter_create( prefix );
    if( !vtable.name || !vtable.destroy )
    {
        PyErr_SetString( PyExc_MemoryError, "Failed to create adapter manager" );
        return NULL;
    }

    return ccsp_py_create_adapter_manager_capsule_owned( &vtable );
}

/*
 * _example_input_adapter(mgr, engine, pytype, push_mode, scalars) -> capsule
 *
 * Create an example input adapter that generates incrementing integers.
 * Called by input_adapter_def wiring layer with standard arguments.
 */
static PyObject * create_input_adapter_py( PyObject * self, PyObject * args, PyObject * kwargs )
{
    ( void )self;
    ( void )kwargs;  /* Not used when called from wiring layer */

    /* When called from input_adapter_def:
     * args = (mgr, engine, pytype, push_mode, scalars)
     * scalars is a tuple containing the kwargs defined in the adapter_def
     */
    PyObject * mgr = NULL;
    PyObject * engine = NULL;
    PyObject * pytype = NULL;
    PyObject * push_mode = NULL;
    PyObject * scalars = NULL;

    if( !PyArg_ParseTuple( args, "OOOOO", &mgr, &engine, &pytype, &push_mode, &scalars ) )
    {
        /* Fall back to old behavior for direct calls */
        PyErr_Clear();
        static char * kwlist[] = { "interval_ms", NULL };
        int interval_ms = 100;

        if( !PyArg_ParseTupleAndKeywords( args, kwargs, "|i", kwlist, &interval_ms ) )
        {
            return NULL;
        }

        CCspPushInputAdapterVTable vtable = example_push_input_adapter_create_int( interval_ms );
        if( !vtable.destroy )
        {
            PyErr_SetString( PyExc_MemoryError, "Failed to create input adapter" );
            return NULL;
        }

        return ccsp_py_create_input_adapter_capsule_owned( &vtable );
    }

    /* Extract interval_ms from scalars tuple */
    int interval_ms = 100;
    if( scalars && PyTuple_Check( scalars ) && PyTuple_Size( scalars ) > 0 )
    {
        PyObject * interval_obj = PyTuple_GetItem( scalars, 0 );
        if( interval_obj && PyLong_Check( interval_obj ) )
        {
            interval_ms = ( int )PyLong_AsLong( interval_obj );
        }
    }

    CCspPushInputAdapterVTable vtable = example_push_input_adapter_create_int( interval_ms );
    if( !vtable.destroy )
    {
        PyErr_SetString( PyExc_MemoryError, "Failed to create input adapter" );
        return NULL;
    }

    return ccsp_py_create_input_adapter_capsule_owned( &vtable );
}

/*
 * _example_output_adapter(mgr, engine, scalars) -> capsule
 *
 * Create an example output adapter that prints values to stdout.
 * Called by output_adapter_def wiring layer with standard arguments.
 */
static PyObject * create_output_adapter_py( PyObject * self, PyObject * args, PyObject * kwargs )
{
    ( void ) self;
    ( void ) kwargs;  /* Not used when called from wiring layer */

    /* When called from output_adapter_def:
     * args = (mgr, engine, scalars)
     * scalars is a tuple containing the kwargs defined in the adapter_def
     */
    PyObject * mgr = NULL;
    PyObject * engine = NULL;
    PyObject * scalars = NULL;

    if( !PyArg_ParseTuple( args, "OOO", &mgr, &engine, &scalars ) )
    {
        /* Fall back to old behavior for direct calls */
        PyErr_Clear();
        static char * kwlist[] = { "prefix", NULL };
        const char * prefix = NULL;

        if( !PyArg_ParseTupleAndKeywords( args, kwargs, "|s", kwlist, &prefix ) )
        {
            return NULL;
        }

        CCspOutputAdapterVTable vtable = example_output_adapter_create( prefix );
        if( !vtable.execute || !vtable.destroy )
        {
            PyErr_SetString( PyExc_MemoryError, "Failed to create output adapter" );
            return NULL;
        }

        return ccsp_py_create_output_adapter_capsule_owned( &vtable );
    }

    /* Extract prefix from scalars tuple */
    const char * prefix = NULL;
    if( scalars && PyTuple_Check( scalars ) && PyTuple_Size( scalars ) > 0 )
    {
        PyObject * prefix_obj = PyTuple_GetItem( scalars, 0 );
        if( prefix_obj && prefix_obj != Py_None && PyUnicode_Check( prefix_obj ) )
        {
            prefix = PyUnicode_AsUTF8( prefix_obj );
        }
    }

    CCspOutputAdapterVTable vtable = example_output_adapter_create( prefix );
    if( !vtable.execute || !vtable.destroy )
    {
        PyErr_SetString( PyExc_MemoryError, "Failed to create output adapter" );
        return NULL;
    }

    return ccsp_py_create_output_adapter_capsule_owned( &vtable );
}

/*
 * _example_inspect_struct_type(struct_type) -> dict
 *
 * Demonstrates using the C struct API to inspect a struct type's fields.
 * Returns a dict with field information.
 */
static PyObject * inspect_struct_type_py( PyObject * self, PyObject * args )
{
    ( void )self;

    PyObject * struct_type = NULL;

    if( !PyArg_ParseTuple( args, "O", &struct_type ) )
    {
        return NULL;
    }

    /* Check if this is a struct type (has structMeta attribute via PyStructMeta) */
    if( !PyType_Check( struct_type ) )
    {
        PyErr_SetString( PyExc_TypeError, "Expected a csp.Struct subclass" );
        return NULL;
    }

    /* Try to get the structMeta capsule from the type's __dict__ or via attribute */
    PyObject * struct_meta_capsule = PyObject_GetAttrString( struct_type, "_struct_meta_capsule" );
    if( !struct_meta_capsule )
    {
        PyErr_Clear();
        /* Fall back to checking if it looks like a struct type */
        PyErr_SetString( PyExc_TypeError, "Not a valid csp.Struct type (no _struct_meta_capsule)" );
        return NULL;
    }

    if( !PyCapsule_CheckExact( struct_meta_capsule ) )
    {
        Py_DECREF( struct_meta_capsule );
        PyErr_SetString( PyExc_TypeError, "_struct_meta_capsule is not a capsule" );
        return NULL;
    }

    CCspStructMetaHandle meta = ( CCspStructMetaHandle )PyCapsule_GetPointer( struct_meta_capsule, NULL );
    Py_DECREF( struct_meta_capsule );

    if( !meta )
    {
        PyErr_SetString( PyExc_ValueError, "Invalid struct meta capsule" );
        return NULL;
    }

    /* Build result dictionary */
    PyObject * result = PyDict_New();
    if( !result )
    {
        return NULL;
    }

    /* Get struct name */
    const char * name = ccsp_struct_meta_name( meta );
    if( name )
    {
        PyObject * name_str = PyUnicode_FromString( name );
        PyDict_SetItemString( result, "name", name_str );
        Py_DECREF( name_str );
    }

    /* Get field count */
    size_t field_count = ccsp_struct_meta_field_count( meta );
    PyObject * count_int = PyLong_FromSize_t( field_count );
    PyDict_SetItemString( result, "field_count", count_int );
    Py_DECREF( count_int );

    /* Check if strict */
    int is_strict = ccsp_struct_meta_is_strict( meta );
    PyDict_SetItemString( result, "is_strict", is_strict ? Py_True : Py_False );

    /* Build list of field info */
    PyObject * fields_list = PyList_New( ( Py_ssize_t )field_count );
    if( !fields_list )
    {
        Py_DECREF( result );
        return NULL;
    }

    for( size_t i = 0; i < field_count; i++ )
    {
        CCspStructFieldHandle field = ccsp_struct_meta_field_by_index( meta, i );
        if( !field )
        {
            continue;
        }

        PyObject * field_dict = PyDict_New();
        if( !field_dict )
        {
            Py_DECREF( fields_list );
            Py_DECREF( result );
            return NULL;
        }

        const char * field_name = ccsp_struct_field_name( field );
        if( field_name )
        {
            PyObject * name_str = PyUnicode_FromString( field_name );
            PyDict_SetItemString( field_dict, "name", name_str );
            Py_DECREF( name_str );
        }

        CCspType field_type = ccsp_struct_field_type( field );
        PyObject * type_int = PyLong_FromLong( ( long )field_type );
        PyDict_SetItemString( field_dict, "type", type_int );
        Py_DECREF( type_int );

        int is_optional = ccsp_struct_field_is_optional( field );
        PyDict_SetItemString( field_dict, "is_optional", is_optional ? Py_True : Py_False );

        PyList_SET_ITEM( fields_list, ( Py_ssize_t )i, field_dict );
    }

    PyDict_SetItemString( result, "fields", fields_list );
    Py_DECREF( fields_list );

    return result;
}

static PyMethodDef exampleadapter_methods[] = {
    { "_example_adapter_manager", ( PyCFunction )create_adapter_manager_py, METH_VARARGS,
      "Create an example adapter manager." },
    { "_example_input_adapter", ( PyCFunction )create_input_adapter_py, METH_VARARGS | METH_KEYWORDS,
      "Create an example input adapter that generates incrementing integers." },
    { "_example_output_adapter", ( PyCFunction )create_output_adapter_py, METH_VARARGS | METH_KEYWORDS,
      "Create an example output adapter that prints values to stdout." },
    { "_example_inspect_struct_type", ( PyCFunction )inspect_struct_type_py, METH_VARARGS,
      "Inspect a csp.Struct type's fields using the C struct API." },
    { NULL, NULL, 0, NULL }
};

static PyModuleDef exampleadapter_module = {
    PyModuleDef_HEAD_INIT,
    "_exampleadapterimpl",
    "Example C adapter implementations for CSP",
    -1,
    exampleadapter_methods
};

PyMODINIT_FUNC PyInit__exampleadapterimpl( void )
{
    return PyModule_Create( &exampleadapter_module );
}