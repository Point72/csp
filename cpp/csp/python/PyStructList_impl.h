#ifndef _IN_CSP_PYTHON_PYSTRUCTLIST_IMPL_H
#define _IN_CSP_PYTHON_PYSTRUCTLIST_IMPL_H

#include <csp/engine/PartialSwitchCspType.h>
#include <csp/python/Conversions.h>
#include <csp/python/InitHelper.h>
#include <csp/python/PyCspType.h>
#include <csp/python/PyStructList.h>
#include <csp/python/VectorWrapper.h>
#include <algorithm>

namespace csp::python
{

template<typename StorageT>
inline StorageT PyStructList<StorageT>::fromPythonValue( PyObject * value ) const
{
    return static_cast<StorageT>( fromPython<ElemT>( value, *elemType() ) );
}

template<typename StorageT>
static PyObject * PyStructList_Append( PyStructList<StorageT> * self, PyObject * args )
{
    CSP_BEGIN_METHOD;

    PyObject * value;
    if( !PyArg_ParseTuple( args, "O", &value ) )
        return NULL;
    
    if( PyList_Append( ( PyObject * ) self, value ) < 0 )
        return NULL;

    // Append the value to the vector stored in the struct field
    self -> vector.append( self -> fromPythonValue( value ) );

    CSP_RETURN_NONE;
}

template<typename StorageT>
static PyObject * PyStructList_Insert( PyStructList<StorageT> * self, PyObject * args )
{
    CSP_BEGIN_METHOD;

    Py_ssize_t index;
    PyObject * value;
    if( !PyArg_ParseTuple( args, "nO", &index, &value ) )
        return NULL;

    if( PyList_Insert( ( PyObject * ) self, index, value ) < 0 )
        return NULL;

    // Insert the value in the vector stored in the struct field
    self -> vector.insert( self -> fromPythonValue( value ), index );

    CSP_RETURN_NONE;
}

template<typename StorageT>
static PyObject * PyStructList_Pop( PyStructList<StorageT> * self, PyObject * args )
{
    CSP_BEGIN_METHOD;

    Py_ssize_t index = -1;
    if( !PyArg_ParseTuple( args, "|n", &index) )
        return NULL;

    PyObjectPtr func = PyObjectPtr::own( PyObject_GetAttrString( ( PyObject * ) &PyList_Type, "pop" ) );
    PyObjectPtr result = PyObjectPtr::check( PyObject_CallFunction( func.ptr(), "On", self, index ) );

    // Pop the value from the vector stored in the struct field
    self -> vector.pop( index );

    return result.release();

    CSP_RETURN_NULL;
}

template<typename StorageT>
static PyObject * PyStructList_Reverse( PyStructList<StorageT> * self, PyObject * Py_UNUSED( ignored ) )
{
    CSP_BEGIN_METHOD;

    if( PyList_Reverse( ( PyObject * ) self ) < 0 )
        return NULL;

    // Reverse the vector stored in the struct field
    self -> vector.reverse();

    CSP_RETURN_NONE;
}

template<typename StorageT>
static PyObject * PyStructList_Sort( PyStructList<StorageT> * self, PyObject * args, PyObject * kwargs )
{
    CSP_BEGIN_METHOD;

    if( args && PyObject_Length( args ) > 0 )
    {
        PyErr_SetString( PyExc_TypeError, "sort() takes no positional arguments" );
        return NULL;
    }

    // We could have implemented sorting using VectorWrapper method, however, decided not to because of:
    // - When the 'key' argument is present, fuguring out C++ sort is too complex, so still need to fall back to Python method in this case.
    // - In order to implement sort using VectorWrapper method, we would need to implement comparison ops for all CSP types, even dummy ones for types like DialectGeneric.
    // - Even with all that, benchmark showed only 3x speed up for using C++ sort.
    // Because of that, the ultimate decision was to use Python sort and convert.
    PyObjectPtr func = PyObjectPtr::own( PyObject_GetAttrString( ( PyObject * ) &PyList_Type, "sort" ) );
    PyObjectPtr arguments = PyObjectPtr::own( PyTuple_Pack( 1, ( PyObject * ) self ) );
    PyObjectPtr result = PyObjectPtr::check( PyObject_Call( func.ptr(), arguments.ptr(), kwargs ) );   

    // Copy sorted list into the vector stored in the struct field
    typename VectorWrapper<StorageT>::IndexType sz = self -> vector.size();
    for( typename VectorWrapper<StorageT>::IndexType index = 0; index < sz; ++index )
    {
        PyObject * value = PyList_GET_ITEM( ( PyObject * ) self, index );
        self -> vector[ index ] = self -> fromPythonValue( value );
    }

    CSP_RETURN_NONE;
}

template<typename StorageT>
static PyObject * PyStructList_Extend( PyStructList<StorageT> * self, PyObject * args )
{
    CSP_BEGIN_METHOD;

    PyObject * iterable;
    if( !PyArg_ParseTuple( args, "O", &iterable ) )
        return NULL;

    PyObjectPtr func = PyObjectPtr::own( PyObject_GetAttrString( ( PyObject * ) &PyList_Type, "extend" ) );
    PyObjectPtr result = PyObjectPtr::check( PyObject_CallFunctionObjArgs( func.ptr(), self, iterable, NULL ) );

    std::vector<StorageT> v = FromPython<std::vector<StorageT>>::impl( iterable, self -> arrayType );

    self -> vector.extend( v );

    CSP_RETURN_NONE;
}

template<typename StorageT>
static PyObject * PyStructList_Remove( PyStructList<StorageT> * self, PyObject * args )
{
    CSP_BEGIN_METHOD;
    
    PyObject * value;
    if( !PyArg_ParseTuple( args, "O", &value) )
        return NULL;

    PyObjectPtr func = PyObjectPtr::own( PyObject_GetAttrString( ( PyObject * ) &PyList_Type, "remove" ) );
    PyObjectPtr result = PyObjectPtr::check( PyObject_CallFunctionObjArgs( func.ptr(), self, value, NULL ) );   

    // Remove the value from the vector stored in the struct field
    self -> vector.remove( self -> fromPythonValue( value ) );

    CSP_RETURN_NONE;
}

template<typename StorageT>
static PyObject * PyStructList_Clear( PyStructList<StorageT> * self, PyObject * Py_UNUSED( ignored ) )
{
    CSP_BEGIN_METHOD;
    
    PyObjectPtr func = PyObjectPtr::own( PyObject_GetAttrString( ( PyObject * ) &PyList_Type, "clear" ) );
    PyObjectPtr result = PyObjectPtr::check( PyObject_CallFunctionObjArgs( func.ptr(), self, NULL ) );   

    // Clear the vector stored in the struct field
    self -> vector.clear();

    CSP_RETURN_NONE;
}

template<typename StorageT>
static int py_struct_list_ass_item( PyObject * sself, Py_ssize_t index, PyObject * value )
{
    CSP_BEGIN_METHOD;

    // Deal with Python list indices that can be negative
    index = ( ( PyStructList<StorageT> * ) sself ) -> vector.normalizeIndex( index );
    
    PyObjectPtr result;
    // The value is not NULL -> assign it to vector[ index ]
    if( value != NULL )
    {
        Py_INCREF( value );
        if( PyList_SetItem( sself, index, value ) < 0 )
            return -1;
    }
    // The value is NULL -> erase vector[ index ]
    else
    {
        PyObjectPtr func = PyObjectPtr::own( PyObject_GetAttrString( ( PyObject * ) &PyList_Type, "__delitem__" ) );
        PyObjectPtr arguments = PyObjectPtr::own( PyTuple_Pack( 2, sself, PyLong_FromSsize_t( index ) ) );
        result = PyObjectPtr::own( PyObject_Call( func.ptr(), arguments.ptr(), NULL ) );
        if( !result.ptr() )
            return -1;
    }

    // Set the value in the vector stored in the struct field
    // The value is not NULL -> assign it to vector[ index ]
    if( value != NULL )
        ( ( PyStructList<StorageT> * ) sself ) -> vector[ index ] = ( ( PyStructList<StorageT> * ) sself ) -> fromPythonValue( value );
    // The value is NULL -> erase vector[ index ]
    else
        ( ( PyStructList<StorageT> * ) sself ) -> vector.eraseItem( index );
    
    CSP_RETURN_INT;
}

template<typename StorageT>
static int py_struct_list_ass_subscript( PyObject * sself, PyObject * item, PyObject * value )
{
    CSP_BEGIN_METHOD;
    
    // The item is the individual index
    if( !PySlice_Check( item ) )
    {
        Py_ssize_t index = PyNumber_AsSsize_t( item, PyExc_IndexError );
        if( index == -1 && PyErr_Occurred() )
            return -1;

        return py_struct_list_ass_item<StorageT>( sself, index, value );
    }

    // The item is a slice
    Py_ssize_t start, stop, step;
    if( PySlice_Unpack( item, &start, &stop, &step ) < 0 )
        return -1;
    PyObjectPtr result;
    // The value is not NULL -> assign it to vector[ slice ]
    if( value != NULL )
    {
        PyObjectPtr func = PyObjectPtr::own( PyObject_GetAttrString( ( PyObject * ) &PyList_Type, "__setitem__" ) );
        PyObjectPtr arguments = PyObjectPtr::own( PyTuple_Pack( 3, sself, item, value ) );
        result = PyObjectPtr::own( PyObject_Call( func.ptr(), arguments.ptr(), NULL ) );
    }
    // The value is NULL -> erase vector[ slice ]
    else
    {
        PyObjectPtr func = PyObjectPtr::own( PyObject_GetAttrString( ( PyObject * ) &PyList_Type, "__delitem__" ) );
        PyObjectPtr arguments = PyObjectPtr::own( PyTuple_Pack( 2, sself, item ) );
        result = PyObjectPtr::own( PyObject_Call( func.ptr(), arguments.ptr(), NULL ) );
    }
    if( !result.ptr() )
        return -1;
    
    // Set the slice in the vector stored in the struct field
    // The value is not NULL -> assign it to vector[ slice ]
    if( value != NULL )
    {
        // Check that all elements to assign to the slice are of correct type
        if( !PySequence_Check( value ) )
        {
            PyErr_SetString( PyExc_TypeError, "can only assign an iterable" );
            return -1;
        }
        std::vector<StorageT> v = FromPython<std::vector<StorageT>>::impl( value, ( ( PyStructList<StorageT> * ) sself ) -> arrayType );
        ( ( PyStructList<StorageT> * ) sself ) -> vector.setSlice( v, start, stop, step );
    }
    // The value is NULL -> erase vector[ slice ]
    else
    {
        ( ( PyStructList<StorageT> * ) sself ) -> vector.eraseSlice( start, stop, step );
    }

    CSP_RETURN_INT;
}

template<typename StorageT>
static PyObject * py_struct_list_inplace_concat( PyObject * sself, PyObject * other )
{    
    CSP_BEGIN_METHOD;
    
    PyObjectPtr arguments = PyObjectPtr::own( PyTuple_Pack( 1, other ) );
    PyObjectPtr result = PyObjectPtr::check( PyStructList_Extend<StorageT>( ( PyStructList<StorageT> * ) sself, arguments.ptr() ) );
    Py_INCREF( sself );
    return sself;

    CSP_RETURN_NULL;
}

template<typename StorageT>
static PyObject * py_struct_list_inplace_repeat( PyObject * sself, Py_ssize_t n )
{    
    CSP_BEGIN_METHOD;
    
    PyObjectPtr func = PyObjectPtr::own( PyObject_GetAttrString( ( PyObject * ) &PyList_Type, "__imul__" ) );
    PyObjectPtr result = PyObjectPtr::check( PyObject_CallFunction( func.ptr(), "On", ( PyStructList<StorageT> * ) sself, n ) );   

    // Emulate repeating on the vector stored in the struct field
    ( ( PyStructList<StorageT> * ) sself ) -> vector.repeat( n );
    
    Py_INCREF( sself );
    return sself;

    CSP_RETURN_NULL;
}

template<typename StorageT>
static PyObject * PyStructList_reduce( PyStructList<StorageT> * self, PyObject * Py_UNUSED( ignored ) )
{
    CSP_BEGIN_METHOD;
    
    PyObjectPtr list = PyObjectPtr::own( toPython( self -> vector.getVector(), self -> arrayType ) );
    PyObject * result = Py_BuildValue( "O(O)", &PyList_Type, list.ptr() );
    return result;

    CSP_RETURN_NULL;
}

template<typename StorageT>
static PyMethodDef PyStructList_methods[] = {
    { "append",     ( PyCFunction ) PyStructList_Append<StorageT>,   METH_VARARGS,                  NULL },
    { "insert",     ( PyCFunction ) PyStructList_Insert<StorageT>,   METH_VARARGS,                  NULL },
    { "pop",        ( PyCFunction ) PyStructList_Pop<StorageT>,      METH_VARARGS,                  NULL },
    { "reverse",    ( PyCFunction ) PyStructList_Reverse<StorageT>,  METH_NOARGS,                   NULL },
    { "sort",       ( PyCFunction ) PyStructList_Sort<StorageT>,     METH_VARARGS | METH_KEYWORDS,  NULL },
    { "extend",     ( PyCFunction ) PyStructList_Extend<StorageT>,   METH_VARARGS,                  NULL },
    { "remove",     ( PyCFunction ) PyStructList_Remove<StorageT>,   METH_VARARGS,                  NULL },
    { "clear",      ( PyCFunction ) PyStructList_Clear<StorageT>,    METH_NOARGS,                   NULL },
    {"__reduce__",  ( PyCFunction ) PyStructList_reduce<StorageT>,   METH_NOARGS,                   NULL },
    { NULL},
};

template<typename StorageT>
static PySequenceMethods py_struct_list_as_sequence = {
    PyList_Type.tp_as_sequence -> sq_length,                                /* sq_length */
    PyList_Type.tp_as_sequence -> sq_concat,                                /* sq_concat */
    PyList_Type.tp_as_sequence -> sq_repeat,                                /* sq_repeat */
    PyList_Type.tp_as_sequence -> sq_item,                                  /* sq_item */
    0,                                                                      /* sq_slice */
    py_struct_list_ass_item<StorageT>,                                      /* sq_ass_item */
    0,                                                                      /* sq_ass_slice */
    PyList_Type.tp_as_sequence -> sq_contains,                              /* sq_contains */
    py_struct_list_inplace_concat<StorageT>,                                /* sq_inplace_concat */
    py_struct_list_inplace_repeat<StorageT>                                 /* sq_inplace_repeat */
};

template<typename StorageT>
static PyMappingMethods py_struct_list_as_mapping = {
    PyList_Type.tp_as_mapping -> mp_length,
    PyList_Type.tp_as_mapping -> mp_subscript,
    py_struct_list_ass_subscript<StorageT>
};

static PyObject * PyStructList_new( PyTypeObject *type, PyObject *args, PyObject *kwds )
{
    // Since the PyStructList has no real meaning when created from Python, we can reconstruct the PSL's value
    // by just treating it as a list. Thus, we simply override the tp_new behaviour to return a list object here.
    // Again, since we don't have tp_init for the PSL, we need to rely on the Python list's tp_init function.

    return PyObject_Call( ( PyObject * ) &PyList_Type, args, kwds ); // Calls both tp_new and tp_init for a Python list
}

template<typename StorageT>
static int PyStructList_tp_clear( PyStructList<StorageT> * self )
{
    Py_CLEAR( self -> pystruct );
    PyObject * pself = ( PyObject * ) self;
    Py_TYPE( pself ) -> tp_base -> tp_clear( pself );
    return 0;
}

template<typename StorageT>
static int PyStructList_tp_traverse( PyStructList<StorageT> * self, visitproc visit, void * arg )
{
    Py_VISIT( self -> pystruct );
    PyObject * pself = ( PyObject * ) self;
    Py_TYPE( pself ) -> tp_base -> tp_traverse( pself, visit, arg );
    return 0;
}

template<typename StorageT>
static void PyStructList_tp_dealloc( PyStructList<StorageT> * self )
{    
    PyObject_GC_UnTrack( self );
    Py_CLEAR( self -> pystruct );
    PyObject * pself = ( PyObject * ) self;
    Py_TYPE( pself ) -> tp_base -> tp_dealloc( pself );
}

template<typename StorageT>
PyTypeObject PyStructList<StorageT>::PyType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "_cspimpl.PyStructList",   /* tp_name */
    sizeof(PyStructList<StorageT>), /* tp_basicsize */
    0,                         /* tp_itemsize */
    ( destructor ) PyStructList_tp_dealloc<StorageT>, /* tp_dealloc */
    0,                         /* tp_print */
    0,                         /* tp_getattr */
    0,                         /* tp_setattr */
    0,                         /* tp_reserved */
    0,                         /* tp_repr */
    0,                         /* tp_as_number */
    &py_struct_list_as_sequence<StorageT>,   /* tp_as_sequence */
    &py_struct_list_as_mapping<StorageT>,     /* tp_as_mapping */
    0,                         /* tp_hash  */
    0,                         /* tp_call */
    0,                         /* tp_str */
    0,                         /* tp_getattro */
    0,                         /* tp_setattro */
    0,                         /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC | Py_TPFLAGS_LIST_SUBCLASS, /* tp_flags */
    "",                        /* tp_doc */
    ( traverseproc ) PyStructList_tp_traverse<StorageT>,         /* tp_traverse */
    ( inquiry ) PyStructList_tp_clear<StorageT>,             /* tp_clear */
    0,                         /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    0,                         /* tp_iter */
    0,                         /* tp_iternext */
    PyStructList_methods<StorageT>, /* tp_methods */
    0,                         /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    0,                         /* tp_init */
    PyType_GenericAlloc,       /* tp_alloc */
    PyStructList_new,          /* tp_new */
    PyObject_GC_Del,           /* tp_free */
};

}

#endif
