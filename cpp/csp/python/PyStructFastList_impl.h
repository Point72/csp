#ifndef _IN_CSP_PYTHON_PYSTRUCTFASTLIST_IMPL_H
#define _IN_CSP_PYTHON_PYSTRUCTFASTLIST_IMPL_H

#include <csp/engine/PartialSwitchCspType.h>
#include <csp/python/Conversions.h>
#include <csp/python/InitHelper.h>
#include <csp/python/PyCspType.h>
#include <csp/python/PyStruct.h>
#include <csp/python/PyStructFastList.h>
#include <csp/python/VectorWrapper.h>
#include <algorithm>


namespace csp::python
{

template<typename StorageT>
struct PyStructFastListIterator;

template<typename StorageT>
struct PyStructFastListReverseIterator;

template<typename StorageT>
inline PyObject * PyStructFastList<StorageT>::toPythonValue( const StorageT & value ) const
{
    return toPython<ElemT>( value, *elemType() );
}

template<typename StorageT>
inline StorageT PyStructFastList<StorageT>::fromPythonValue( PyObject * value ) const
{
    return static_cast<StorageT>( fromPython<ElemT>( value, *elemType() ) );
}

template<typename StorageT>
static Py_ssize_t py_struct_fast_list_len( PyObject * sself )
{
    CSP_BEGIN_METHOD;

    return ( ( PyStructFastList<StorageT> * ) sself ) -> vector.size();

    CSP_RETURN_INT;
}

template<typename StorageT>
static PyObject * PyStructFastList_representation_helper( PyStructFastList<StorageT> * self, bool show_unset )
{
    static thread_local std::string tl_repr;

    // Each PyStructFastList is responsible for clearing the TLS string after
    size_t offset = tl_repr.size();
    repr_array<StorageT>( self -> vector.getVector(), *self -> elemType(), tl_repr, false );

    PyObject * rv = PyUnicode_FromString( tl_repr.c_str() + offset );
    tl_repr.erase( offset );
    return rv;
}

template<typename StorageT>
static PyObject * PyStructFastList_Repr( PyStructFastList<StorageT> * self )
{
    CSP_BEGIN_METHOD;

    return PyStructFastList_representation_helper( self, false );

    CSP_RETURN_NULL;
}

template<typename StorageT>
static PyObject * PyStructFastList_Str( PyStructFastList<StorageT> * self )
{
    CSP_BEGIN_METHOD;

    return PyStructFastList_representation_helper( self, true );

    CSP_RETURN_NULL;
}

template<typename StorageT>
static PyObject * py_struct_fast_list_item( PyObject * sself, Py_ssize_t index )
{
    CSP_BEGIN_METHOD;

    // Return the value from the vector stored in the struct field
    return ( ( PyStructFastList<StorageT> * ) sself ) -> toPythonValue( ( ( PyStructFastList<StorageT> * ) sself ) -> vector[ index ] );

    CSP_RETURN_NULL;
}

template<typename StorageT>
static PyObject * py_struct_fast_list_subscript( PyObject * sself, PyObject * item )
{
    CSP_BEGIN_METHOD;
    
    // The item is the individual index
    if( !PySlice_Check( item ) )
    {
        Py_ssize_t index = PyNumber_AsSsize_t( item, PyExc_IndexError );
        if( index == -1 && PyErr_Occurred() )
            return NULL;

        return py_struct_fast_list_item<StorageT>( sself, index );
    }

    // The item is a slice
    Py_ssize_t start, stop, step;
    if( PySlice_Unpack( item, &start, &stop, &step ) < 0 )
        return NULL;
    
    // Return the slice from the vector stored in the struct field
    return toPython( ( ( PyStructFastList<StorageT> * ) sself ) -> vector.getSlice( start, stop, step ), ( ( PyStructFastList<StorageT> * ) sself ) -> arrayType );

    CSP_RETURN_NULL;
}

template<typename StorageT>
static PyObject * PyStructFastList_Copy( PyStructFastList<StorageT> * self, PyObject * Py_UNUSED( ignored ) )
{
    CSP_BEGIN_METHOD;

    PyObject * list = toPython( self -> vector.getVector(), self -> arrayType );
    return list;

    CSP_RETURN_NONE;
}

template<typename StorageT>
static PyObject * PyStructFastList_Index( PyStructFastList<StorageT> * self, PyObject * args )
{
    CSP_BEGIN_METHOD;
    
    typename VectorWrapper<StorageT>::IndexType sz = self -> vector.size();
    Py_ssize_t start_index = 0;
    Py_ssize_t stop_index = sz;
    PyObject * value;
    if( !PyArg_ParseTuple( args, "O|nn", &value, &start_index, &stop_index ) )
        return NULL;

    // Return the index of the the value in the vector stored in the struct field
    typename VectorWrapper<StorageT>::IndexType index;
    try
    {
        index = self -> vector.index( self -> fromPythonValue( value ), start_index, stop_index );
    }
    catch( TypeError const& )
    {
        CSP_THROW( ValueError, "Value not found." );
    }
    return PyLong_FromSsize_t(index);

    CSP_RETURN_NONE;
}

template<typename StorageT>
static PyObject * PyStructFastList_Count( PyStructFastList<StorageT> * self, PyObject * args )
{
    CSP_BEGIN_METHOD;
    
    PyObject * value;
    if( !PyArg_ParseTuple( args, "O", &value) )
        return NULL;

    // Return the count of the value in the vector stored in the struct field
    typename VectorWrapper<StorageT>::IndexType index;
    try
    {
        index = self -> vector.count( self -> fromPythonValue( value ) );
    }
    catch( TypeError const& )
    {
        index = 0;
    }

    return PyLong_FromSsize_t(index);

    CSP_RETURN_NONE;
}

template<typename StorageT>
static PyObject * PyStructFastList_Append( PyStructFastList<StorageT> * self, PyObject * args )
{
    CSP_BEGIN_METHOD;

    PyObject * value;
    if( !PyArg_ParseTuple( args, "O", &value ) )
        return NULL;
    
    // Append the value to the vector stored in the struct field
    self -> vector.append( self -> fromPythonValue( value ) );

    CSP_RETURN_NONE;
}

template<typename StorageT>
static PyObject * PyStructFastList_Insert( PyStructFastList<StorageT> * self, PyObject * args )
{
    CSP_BEGIN_METHOD;

    Py_ssize_t index;
    PyObject * value;
    if( !PyArg_ParseTuple( args, "nO", &index, &value ) )
        return NULL;

    // Insert the value in the vector stored in the struct field
    self -> vector.insert( self -> fromPythonValue( value ), index );

    CSP_RETURN_NONE;
}

template<typename StorageT>
static PyObject * PyStructFastList_Pop( PyStructFastList<StorageT> * self, PyObject * args )
{
    CSP_BEGIN_METHOD;

    Py_ssize_t index = -1;
    if( !PyArg_ParseTuple( args, "|n", &index ) )
        return NULL;

    // Pop the value from the vector stored in the struct field
    return self -> toPythonValue( self -> vector.pop( index ) );

    CSP_RETURN_NULL;
}

template<typename StorageT>
static PyObject * PyStructFastList_Reverse( PyStructFastList<StorageT> * self, PyObject * Py_UNUSED( ignored ) )
{
    CSP_BEGIN_METHOD;

    // Reverse the vector stored in the struct field
    self -> vector.reverse();

    CSP_RETURN_NONE;
}

template<typename StorageT>
static PyObject * PyStructFastList_Sort( PyStructFastList<StorageT> * self, PyObject * args, PyObject * kwargs )
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
    PyObjectPtr list = PyObjectPtr::own( toPython( self -> vector.getVector(), self -> arrayType ) );
    PyObjectPtr func = PyObjectPtr::own( PyObject_GetAttrString( ( PyObject * ) &PyList_Type, "sort" ) );
    PyObjectPtr arguments = PyObjectPtr::own( PyTuple_Pack( 1, list.ptr() ) );
    PyObjectPtr result = PyObjectPtr::check( PyObject_Call( func.ptr(), arguments.ptr(), kwargs ) );

    // Copy sorted list into the vector stored in the struct field
    typename VectorWrapper<StorageT>::IndexType sz = self -> vector.size();
    for( typename VectorWrapper<StorageT>::IndexType index = 0; index < sz; ++index )
    {
        PyObject * value = PyList_GET_ITEM( list.ptr(), index );
        self -> vector[ index ] = self -> fromPythonValue( value );
    }

    CSP_RETURN_NONE;
}

template<typename StorageT>
static PyObject * PyStructFastList_Extend( PyStructFastList<StorageT> * self, PyObject * args )
{
    CSP_BEGIN_METHOD;

    PyObject * iterable;
    if( !PyArg_ParseTuple( args, "O", &iterable ) )
        return NULL;

    std::vector<StorageT> v = FromPython<std::vector<StorageT>>::impl( iterable, self -> arrayType );
    self -> vector.extend( v );

    CSP_RETURN_NONE;
}

template<typename StorageT>
static PyObject * PyStructFastList_Remove( PyStructFastList<StorageT> * self, PyObject * args )
{
    CSP_BEGIN_METHOD;
    
    PyObject * value;
    if( !PyArg_ParseTuple( args, "O", &value) )
        return NULL;

    // Remove the value from the vector stored in the struct field
    self -> vector.remove( self -> fromPythonValue( value ) );

    CSP_RETURN_NONE;
}

template<typename StorageT>
static PyObject * PyStructFastList_Clear( PyStructFastList<StorageT> * self, PyObject * Py_UNUSED( ignored ) )
{
    CSP_BEGIN_METHOD;

    // Clear the vector stored in the struct field
    self -> vector.clear();

    CSP_RETURN_NONE;
}

template<typename StorageT>
static int py_struct_fast_list_ass_item( PyObject * sself, Py_ssize_t index, PyObject * value )
{
    CSP_BEGIN_METHOD;
    
    // Set the value in the vector stored in the struct field
    // The value is not NULL -> assign it to vector[ index ]
    if( value != NULL )
        ( ( PyStructFastList<StorageT> * ) sself ) -> vector[ index ] = ( ( PyStructFastList<StorageT> * ) sself ) -> fromPythonValue( value );
    // The value is NULL -> erase vector[ index ]
    else
        ( ( PyStructFastList<StorageT> * ) sself ) -> vector.eraseItem( index );
    
    CSP_RETURN_INT;
}

template<typename StorageT>
static int py_struct_fast_list_ass_subscript( PyObject * sself, PyObject * item, PyObject * value )
{
    CSP_BEGIN_METHOD;
    
    // The item is the individual index
    if( !PySlice_Check( item ) )
    {
        Py_ssize_t index = PyNumber_AsSsize_t( item, PyExc_IndexError );
        if( index == -1 && PyErr_Occurred() )
            return -1;

        return py_struct_fast_list_ass_item<StorageT>( sself, index, value );
    }

    // The item is a slice
    Py_ssize_t start, stop, step;
    if( PySlice_Unpack( item, &start, &stop, &step ) < 0 )
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
        std::vector<StorageT> v = FromPython<std::vector<StorageT>>::impl( value, ( ( PyStructFastList<StorageT> * ) sself ) -> arrayType );
        ( ( PyStructFastList<StorageT> * ) sself ) -> vector.setSlice( v, start, stop, step );
    }
    // The value is NULL -> erase vector[ slice ]
    else
    {
        ( ( PyStructFastList<StorageT> * ) sself ) -> vector.eraseSlice( start, stop, step );
    }

    CSP_RETURN_INT;
}

template<typename StorageT>
static PyObject * py_struct_fast_list_inplace_concat( PyObject * sself, PyObject * other )
{
    CSP_BEGIN_METHOD;

    // Emulate concatenation on the vector stored in the struct field
    std::vector<StorageT> v = FromPython<std::vector<StorageT>>::impl( other, ( ( PyStructFastList<StorageT> * ) sself ) -> arrayType );
    ( ( PyStructFastList<StorageT> * ) sself ) -> vector.extend( v );

    Py_INCREF( sself );
    return sself;

    CSP_RETURN_NULL;
}

template<typename StorageT>
static PyObject * py_struct_fast_list_concat( PyObject * sself, PyObject * other )
{
    CSP_BEGIN_METHOD;

    // Parse iterable provided and check that all elements of iterable are of correct type
    if( ( !PyList_Check( other ) ) && ( Py_TYPE( other ) != &PyStructFastList<StorageT>::PyType ) )
    {
        PyErr_SetString( PyExc_TypeError, "can only concatenate typed list or _cspimpl.PyStructFastList to _cspimpl.PyStructFastList with the same type" );
        return NULL;
    }

    // Convert the operands to Python lists and use Python list concatenation
    PyObjectPtr list = PyObjectPtr::own( toPython( ( ( PyStructFastList<StorageT> * ) sself ) -> vector.getVector(), ( ( PyStructFastList<StorageT> * ) sself ) -> arrayType ) );
    PyObjectPtr other_list = PyObjectPtr::incref( other );

    if( !PyList_Check( other ))
        other_list = PyObjectPtr::own( toPython( ( ( PyStructFastList<StorageT> * ) other ) -> vector.getVector(), ( ( PyStructFastList<StorageT> * ) other ) -> arrayType ) );

    PyObjectPtr result = PyObjectPtr::check( PySequence_Concat( list.ptr(), other_list.ptr() ) );

    return result.release();

    CSP_RETURN_NULL;
}

template<typename StorageT>
static PyObject * py_struct_fast_list_inplace_repeat( PyObject * sself, Py_ssize_t n )
{
    CSP_BEGIN_METHOD;

    // Emulate repeating on the vector stored in the struct field
    ( ( PyStructFastList<StorageT> * ) sself ) -> vector.repeat( n );

    Py_INCREF( sself );
    return sself;

    CSP_RETURN_NULL;
}

template<typename StorageT>
static PyObject * py_struct_fast_list_repeat( PyObject * sself, Py_ssize_t n )
{
    CSP_BEGIN_METHOD;

    // Convert the operand to Python list and use Python list repeating
    PyObjectPtr list = PyObjectPtr::own( toPython( ( ( PyStructFastList<StorageT> * ) sself ) -> vector.getVector(), ( ( PyStructFastList<StorageT> * ) sself ) -> arrayType ) );

    PyObjectPtr result = PyObjectPtr::check( PySequence_Repeat( list.ptr(), n ) );

    return result.release();

    CSP_RETURN_NULL;
}

template<typename StorageT>
static int py_struct_fast_list_contains( PyObject * sself, PyObject * value )
{
    CSP_BEGIN_METHOD;

    if( ( ( PyStructFastList<StorageT> * ) sself ) -> vector.contains( ( ( PyStructFastList<StorageT> * ) sself ) -> fromPythonValue( value ) ))
        return 1;

    return 0;

    CSP_RETURN_INT;
}

template<typename StorageT>
static PyObject * py_struct_fast_list_richcompare( PyObject * sself, PyObject * other, int op )
{
    CSP_BEGIN_METHOD;

    if( !PyList_Check( other ) && ( Py_TYPE( other ) != &PyStructFastList<StorageT>::PyType ) )
        Py_RETURN_NOTIMPLEMENTED;

    // We could have implemented rich comparison using VectorWrapper, however, decided not to because of:
    // - In order to implement rich comparison using VectorWrapper, we would need to implement comparison ops for all CSP types, even dummy ones for types like DialectGeneric.
    // - Rich comparison is not very frequently used.
    // Because of that, the ultimate decision was to use Python rich comparison and convert.
    PyObjectPtr list = PyObjectPtr::own( toPython( ( ( PyStructFastList<StorageT> * ) sself ) -> vector.getVector(), ( ( PyStructFastList<StorageT> * ) sself ) -> arrayType ) );
    PyObjectPtr other_list = PyObjectPtr::incref( other );
    if( !PyList_Check( other ) )
        other_list = PyObjectPtr::own( toPython( ( ( PyStructFastList<StorageT> * ) other ) -> vector.getVector(), ( ( PyStructFastList<StorageT> * ) other ) -> arrayType ) );
    PyObjectPtr result = PyObjectPtr::check( PyObject_RichCompare( list.ptr(), other_list.ptr(), op ) );

    return result.release();

    CSP_RETURN_NULL;
}

template<typename StorageT>
static PyObject * PyStructFastList_Sizeof( PyStructFastList<StorageT> * self, PyObject * Py_UNUSED( ignored ) )
{
    CSP_BEGIN_METHOD;

    PyObject * pself = ( PyObject * ) self;
    size_t result = ( size_t ) Py_TYPE( pself ) -> tp_basicsize;
    // In Python, sizeof list growth with the number of elements, here we mimic that
    result += sizeof( self -> vector ) + self -> vector.size() * sizeof( StorageT );
    return PyLong_FromSize_t( result );

    CSP_RETURN_NONE;
}

template<typename StorageT>
static PyObject * py_struct_fast_list_iter( PyObject * sself )
{
    CSP_BEGIN_METHOD;

    return PyIterator<PyStructFastListIterator<StorageT>>::create( PyStructFastListIterator<StorageT>( ( PyStructFastList<StorageT> * ) sself ) );

    CSP_RETURN_NULL;
}

template<typename StorageT>
static PyObject * PyStructFastList_Reversed( PyStructFastList<StorageT> * self, PyObject * Py_UNUSED( ignored ) )
{
    CSP_BEGIN_METHOD;

    return PyIterator<PyStructFastListReverseIterator<StorageT>>::create( PyStructFastListReverseIterator<StorageT>( self ) );

    CSP_RETURN_NULL;
}

template<typename StorageT>
static PyObject * PyStructFastList_reduce( PyStructFastList<StorageT> * self, PyObject * Py_UNUSED( ignored) )
{
    CSP_BEGIN_METHOD;
    
    PyObjectPtr list = PyObjectPtr::own( toPython( self -> vector.getVector(), self -> arrayType ) );
    PyObject * result = Py_BuildValue( "O(O)", &PyList_Type, list.ptr() );
    return result;

    CSP_RETURN_NULL;
}

template<typename StorageT>
static PyMethodDef PyStructFastList_methods[] = {
    { "__getitem__",   ( PyCFunction ) py_struct_fast_list_subscript<StorageT>,  METH_VARARGS,                  NULL },
    { "copy",          ( PyCFunction ) PyStructFastList_Copy<StorageT>,          METH_NOARGS,                   NULL },
    { "index",         ( PyCFunction ) PyStructFastList_Index<StorageT>,         METH_VARARGS,                  NULL },
    { "count",         ( PyCFunction ) PyStructFastList_Count<StorageT>,         METH_VARARGS,                  NULL },
    { "__sizeof__",    ( PyCFunction ) PyStructFastList_Sizeof<StorageT>,        METH_NOARGS,                   NULL },
    { "__reversed__",  ( PyCFunction ) PyStructFastList_Reversed<StorageT>,      METH_NOARGS,                   NULL },
    { "append",        ( PyCFunction ) PyStructFastList_Append<StorageT>,        METH_VARARGS,                  NULL },
    { "insert",        ( PyCFunction ) PyStructFastList_Insert<StorageT>,        METH_VARARGS,                  NULL },
    { "pop",           ( PyCFunction ) PyStructFastList_Pop<StorageT>,           METH_VARARGS,                  NULL },
    { "reverse",       ( PyCFunction ) PyStructFastList_Reverse<StorageT>,       METH_NOARGS,                   NULL },
    { "sort",          ( PyCFunction ) PyStructFastList_Sort<StorageT>,          METH_VARARGS | METH_KEYWORDS,  NULL },
    { "extend",        ( PyCFunction ) PyStructFastList_Extend<StorageT>,        METH_VARARGS,                  NULL },
    { "remove",        ( PyCFunction ) PyStructFastList_Remove<StorageT>,        METH_VARARGS,                  NULL },
    { "clear",         ( PyCFunction ) PyStructFastList_Clear<StorageT>,         METH_NOARGS,                   NULL },
    {"__reduce__",     ( PyCFunction ) PyStructFastList_reduce<StorageT>,        METH_NOARGS,                   NULL },
    { NULL},
};

template<typename StorageT>
static PySequenceMethods py_struct_fast_list_as_sequence = {
    py_struct_fast_list_len<StorageT>,                                /* sq_length */
    py_struct_fast_list_concat<StorageT>,                             /* sq_concat */
    py_struct_fast_list_repeat<StorageT>,                             /* sq_repeat */
    py_struct_fast_list_item<StorageT>,                               /* sq_item */
    0,                                                                /* sq_slice */
    py_struct_fast_list_ass_item<StorageT>,                           /* sq_ass_item */
    0,                                                                /* sq_ass_slice */
    py_struct_fast_list_contains<StorageT>,                           /* sq_contains */
    py_struct_fast_list_inplace_concat<StorageT>,                     /* sq_inplace_concat */
    py_struct_fast_list_inplace_repeat<StorageT>                      /* sq_inplace_repeat */
};

template<typename StorageT>
static PyMappingMethods py_struct_fast_list_as_mapping = {
    py_struct_fast_list_len<StorageT>,
    py_struct_fast_list_subscript<StorageT>,
    py_struct_fast_list_ass_subscript<StorageT>
};

static PyObject * PyStructFastList_new( PyTypeObject *type, PyObject *args, PyObject *kwds )
{
    // Since the PyStructFastList has no real meaning when created from Python, we can reconstruct the FL's value
    // by just treating it as a list. Thus, we simply override the tp_new behaviour to return a list object here.
    // Again, since we don't have tp_init for the FL, we need to rely on the Python list's tp_init function.

    return PyObject_Call( ( PyObject * ) &PyList_Type, args, kwds ); // Calls both tp_new and tp_init for a Python list
}

template<typename StorageT>
static int PyStructFastList_tp_clear( PyStructFastList<StorageT> * self )
{
    Py_CLEAR( self -> pystruct );
    return 0;
}

template<typename StorageT>
static int PyStructFastList_tp_traverse( PyStructFastList<StorageT> * self, visitproc visit, void * arg )
{
    Py_VISIT( self -> pystruct );
    return 0;
}

template<typename StorageT>
static void PyStructFastList_tp_dealloc( PyStructFastList<StorageT> * self )
{    
    PyObject_GC_UnTrack( self );
    Py_CLEAR( self -> pystruct );
}

template<typename StorageT>
PyTypeObject  PyStructFastList<StorageT>::PyType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "_cspimpl.PyStructFastList",   /* tp_name */
    sizeof(PyStructFastList<StorageT>), /* tp_basicsize */
    0,                         /* tp_itemsize */
    ( destructor ) PyStructFastList_tp_dealloc<StorageT>, /* tp_dealloc */
    0,                         /* tp_print */
    0,                         /* tp_getattr */
    0,                         /* tp_setattr */
    0,                         /* tp_reserved */
    ( reprfunc ) PyStructFastList_Repr<StorageT>, /* tp_repr */
    0,                         /* tp_as_number */
    &py_struct_fast_list_as_sequence<StorageT>, /* tp_as_sequence */
    &py_struct_fast_list_as_mapping<StorageT>, /* tp_as_mapping */
    PyObject_HashNotImplemented, /* tp_hash  */
    0,                         /* tp_call */
    ( reprfunc ) PyStructFastList_Str<StorageT>, /* tp_str */
    PyObject_GenericGetAttr,   /* tp_getattro */
    0,                         /* tp_setattro */
    0,                         /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC | Py_TPFLAGS_SEQUENCE, /* tp_flags */
    "",                        /* tp_doc */
    ( traverseproc ) PyStructFastList_tp_traverse<StorageT>, /* tp_traverse */
    ( inquiry ) PyStructFastList_tp_clear<StorageT>, /* tp_clear */
    py_struct_fast_list_richcompare<StorageT>, /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    py_struct_fast_list_iter<StorageT>, /* tp_iter */
    0,                         /* tp_iternext */
    PyStructFastList_methods<StorageT>, /* tp_methods */
    0,                         /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    0,                         /* tp_init */
    PyType_GenericAlloc,       /* tp_alloc */
    PyStructFastList_new,      /* tp_new */
    PyObject_GC_Del,           /* tp_free */
};

template<typename StorageT>
struct PyStructFastListIterator : public PyObject
{
    PyStructFastListIterator( PyStructFastList<StorageT> * v ) : m_v( v ), m_index( 0 )
    {
        Py_INCREF( m_v );
    }
    ~PyStructFastListIterator()
    {
        Py_DECREF( m_v );
    }
    PyObject * iternext()
    {
        if( m_index < py_struct_fast_list_len<StorageT>( m_v ) )
            return py_struct_fast_list_item<StorageT>( m_v, m_index++ );
        PyErr_SetString( PyExc_StopIteration, "" );
        return NULL;
    }
protected:
    PyStructFastList<StorageT> * m_v;
    typename VectorWrapper<StorageT>::IndexType m_index;
};

template<typename StorageT>
struct PyStructFastListReverseIterator : public PyObject
{
    PyStructFastListReverseIterator( PyStructFastList<StorageT> * v ) : m_v( v ), m_index( py_struct_fast_list_len<StorageT>( v ) - 1 )
    {
        Py_INCREF( m_v );
    }
    ~PyStructFastListReverseIterator()
    {
        Py_DECREF( m_v );
    }
    PyObject * iternext()
    {
        if( m_index >= 0 )
            return py_struct_fast_list_item<StorageT>( m_v, m_index-- );
        PyErr_SetString( PyExc_StopIteration, "" );
        return NULL;
    }
protected:
    PyStructFastList<StorageT> * m_v;
    typename VectorWrapper<StorageT>::IndexType m_index;
};

}

#endif
