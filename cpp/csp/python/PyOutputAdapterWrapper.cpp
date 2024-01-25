#include <csp/engine/OutputAdapter.h>
#include <csp/python/Conversions.h>
#include <csp/python/Exception.h>
#include <csp/python/InitHelper.h>
#include <csp/python/PyEngine.h>
#include <csp/python/PyInputAdapterWrapper.h>
#include <csp/python/PyNodeWrapper.h>
#include <csp/python/PyOutputAdapterWrapper.h>
#include <csp/python/PyAdapterManagerWrapper.h>

namespace csp::python
{

PyObject * PyOutputAdapterWrapper::createAdapter( Creator creator, PyObject * args )
{
    CSP_BEGIN_METHOD;

    PyObject * pyAdapterManager = nullptr;

    PyEngine * pyEngine   = nullptr;
    PyObject * pyArgs     = nullptr;


    if( !PyArg_ParseTuple( args, "OO!O!", 
                           &pyAdapterManager,
                           &PyEngine::PyType, &pyEngine, 
                           &PyTuple_Type, &pyArgs ) )
        CSP_THROW( PythonPassthrough, "" );

    csp::AdapterManager *adapterMgr  = nullptr;
    if( PyCapsule_CheckExact( pyAdapterManager ) )
        adapterMgr = PyAdapterManagerWrapper::extractAdapterManager( pyAdapterManager );

    auto adapter = creator( adapterMgr, pyEngine, pyArgs );

    PyOutputAdapterWrapper * object = ( PyOutputAdapterWrapper * ) PyOutputAdapterWrapper::PyType.tp_alloc( &PyOutputAdapterWrapper::PyType, 0 ); 
    new( object ) PyOutputAdapterWrapper( adapter );
    return object;

    CSP_RETURN_NULL;
}

static PyObject * PyOutputAdapterWrapper_linkFrom( PyOutputAdapterWrapper * self, PyObject * args )
{
    CSP_BEGIN_METHOD;

    int outputIdx, outputBasketIdx, inputIdx, inputBasketIdx;
    PyObject * source;
    if( !PyArg_ParseTuple( args, "Oiiii", &source, 
                           &outputIdx, &outputBasketIdx,
                           &inputIdx, &inputBasketIdx ) )
        return nullptr;

    OutputId outputId( outputIdx, outputBasketIdx );

    if( PyType_IsSubtype( Py_TYPE( source ), &PyNodeWrapper::PyType ) )
    {
        self -> adapter() -> link( static_cast<PyNodeWrapper * >( source ) -> node() -> output( outputId ) );
    }
    else if( PyType_IsSubtype( Py_TYPE( source ), &PyInputAdapterWrapper::PyType ) )
    {
        self -> adapter() -> link( static_cast<PyInputAdapterWrapper *>( source ) -> adapter() );
    }
    else
        CSP_THROW( TypeError, "link_from expected PyNode or PyInputAdapter as source, got " << Py_TYPE( source ) -> tp_name );

    CSP_RETURN_NONE;
}

static PyMethodDef PyOutputAdapterWrapper_methods[] = {
    { "link_from",         (PyCFunction) PyOutputAdapterWrapper_linkFrom, METH_VARARGS, "links node's output to target output adapter" },
    {NULL}
};

PyTypeObject PyOutputAdapterWrapper::PyType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "_cspimpl.PyOutputAdapterWrapper", /* tp_name */
    sizeof(PyOutputAdapterWrapper),    /* tp_basicsize */
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
    "csp output adapter wrapper", /* tp_doc */
    0,                         /* tp_traverse */
    0,                         /* tp_clear */
    0,                         /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    0,                         /* tp_iter */
    0,                         /* tp_iternext */
    PyOutputAdapterWrapper_methods,    /* tp_methods */
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
    0,                         /* tp_free */ /* Low-level free-memory routine */
    0,                         /* tp_is_gc */ /* For PyObject_IS_GC */
    0,                         /* tp_bases */
    0,                         /* tp_mro */ /* method resolution order */
    0,                         /* tp_cache */
    0,                         /* tp_subclasses */
    0,                         /* tp_weaklist */
    0,                         /* tp_del */
    0,                         /* tp_version_tag */
    0                          /* tp_finalize */
};

REGISTER_TYPE_INIT( &PyOutputAdapterWrapper::PyType, "PyOutputAdapterWrapper" )

}
