#include <csp/engine/InputAdapter.h>
#include <csp/python/Conversions.h>
#include <csp/python/Exception.h>
#include <csp/python/InitHelper.h>
#include <csp/python/PyEngine.h>
#include <csp/python/PyInputAdapterWrapper.h>
#include <csp/python/PyAdapterManagerWrapper.h>

namespace csp::python
{

PyObject * PyInputAdapterWrapper::createAdapter( Creator creator, PyObject * args )
{
    CSP_BEGIN_METHOD;


    PyObject * pyAdapterManager = nullptr;

    PyEngine * pyEngine   = nullptr;
    PyObject * pyType     = nullptr;
    PyObject * pyArgs     = nullptr;
    int        pushmode   = -1;

    if( !PyArg_ParseTuple( args, "OO!OiO!", 
                           &pyAdapterManager,
                           &PyEngine::PyType, &pyEngine, 
                           &pyType,
                           &pushmode,
                           &PyTuple_Type, &pyArgs ) )
        CSP_THROW( PythonPassthrough, "" );

    if( pushmode == PushMode::UNKNOWN || pushmode >= PushMode::NUM_TYPES )
        CSP_THROW( ValueError, "invalid pushmode " << pushmode );

    csp::AdapterManager *adapterMgr  = nullptr;
    if( PyCapsule_CheckExact( pyAdapterManager ) )
        adapterMgr = PyAdapterManagerWrapper::extractAdapterManager( pyAdapterManager );

    auto adapter = creator( adapterMgr, pyEngine, pyType, PushMode( pushmode ), pyArgs );

    return create( adapter );
    CSP_RETURN_NULL;
}

PyObject * PyInputAdapterWrapper::create( InputAdapter * adapter )
{
    CSP_BEGIN_METHOD;
    PyInputAdapterWrapper * object = ( PyInputAdapterWrapper * ) PyInputAdapterWrapper::PyType.tp_alloc( &PyInputAdapterWrapper::PyType, 0 ); 
    new( object ) PyInputAdapterWrapper( adapter );
    return object;
    CSP_RETURN_NULL;
}

PyTypeObject PyInputAdapterWrapper::PyType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "_cspimpl.PyInputAdapterWrapper", /* tp_name */
    sizeof(PyInputAdapterWrapper),    /* tp_basicsize */
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
    "csp input adapter wrapper", /* tp_doc */
};

REGISTER_TYPE_INIT( &PyInputAdapterWrapper::PyType, "PyInputAdapterWrapper" )

}
