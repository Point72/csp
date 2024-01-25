#ifndef _IN_PY_ITERATOR_H
#define _IN_PY_ITERATOR_H

#include <Python.h>

//helper python iterator wrapper for any kind of iterator type
namespace csp::python
{

template<typename ITER_IMPL>
struct PyIterator : public PyObject
{

    PyIterator( const ITER_IMPL & iter ) : m_pyiter( iter )
    {
        //force usage for init hook
        s_typeRegister = true;
    }

    static PyObject * static_iternext( PyIterator * pyiter ) { return pyiter -> m_pyiter.iternext(); }
    static PyObject * static_iter( PyIterator * pyiter )     { Py_INCREF( pyiter ); return pyiter; }
    
    static PyIterator * create( const ITER_IMPL & iter )
    {
        PyIterator * pyiter = static_cast<PyIterator *>( PyType.tp_alloc( &PyType, 0 ) );
        new ( pyiter ) PyIterator( iter );
        return pyiter;
    }

    static PyTypeObject PyType;

    static bool s_typeRegister;

private:
    ITER_IMPL m_pyiter;
};

template<typename ITER_IMPL> PyTypeObject PyIterator<ITER_IMPL>::PyType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "_cspimpl.PyIterator",     /* tp_name */
    sizeof(PyIterator<ITER_IMPL>),        /* tp_basicsize */
    0,                         /* tp_itemsize */
    0,                         /* tp_dealloc */
    0,                         /* tp_print */
    0,                         /* tp_getattr */
    0,                         /* tp_setattr */
    0,                         /* tp_reserved */
    0,                         /* tp_repr */
    0,                         /* tp_as_number */
    0,                         /* tp_as_sequence */
    0,                         /* tp_as_mapping */
    0,                         /* tp_hash  */
    0,                         /* tp_call */
    0,                         /* tp_str */
    0,                         /* tp_getattro */
    0,                         /* tp_setattro */
    0,                         /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,        /* tp_flags */
    "csp generic iterator",    /* tp_doc */
    0,                         /* tp_traverse */
    0,                         /* tp_clear */
    0,                         /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    ( getiterfunc) PyIterator<ITER_IMPL>::static_iter,   /* tp_iter */
    ( iternextfunc ) PyIterator<ITER_IMPL>::static_iternext,   /* tp_iternext */
    0,                         /* tp_methods */
    0,                         /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    0,                         /* tp_init */
    0,                         /* tp_alloc */
    PyType_GenericNew,         /* tp_new */
};

//TODO figure out why this doesnt work
template<typename ITER_IMPL> bool PyIterator<ITER_IMPL>::s_typeRegister = InitHelper::instance().registerCallback( 
    InitHelper::typeInitCallback( &PyIterator<ITER_IMPL>::PyType, "" ) );

}

#endif
