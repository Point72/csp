#include <csp/python/Conversions.h>
#include <csp/python/Exception.h>
#include <csp/python/InitHelper.h>
#include <csp/python/PyOutputProxy.h>

namespace csp::python
{

PyOutputProxy::PyOutputProxy( PyObject * pyType, Node * node, OutputId id ) : m_pyType( PyObjectPtr::incref( pyType ) ),
                                                                              m_node( node ),
                                                                              m_id( id )
{
}

PyOutputProxy * PyOutputProxy::create( PyObject * pyType, Node * node, OutputId id )
{
    PyOutputProxy * proxy = ( PyOutputProxy * ) PyType.tp_new( &PyType, nullptr, nullptr );
    new( proxy ) PyOutputProxy( pyType, node, id );
    return proxy;
}

static void PyOutputProxy_dealloc( PyOutputProxy * self )
{
    CSP_BEGIN_METHOD;

    ( self ) -> ~PyOutputProxy();
    PyOutputProxy::PyType.tp_free( self ); 

    CSP_RETURN;
}

static PyObject * PyOutputProxy_output( PyOutputProxy * proxy, PyObject * value )
{
    CSP_BEGIN_METHOD;

    proxy -> outputTick( value );
    CSP_RETURN_NONE;
}

static PyNumberMethods PyOutputProxy_NumberMethods = {
    (binaryfunc ) PyOutputProxy_output, /* binaryfunc nb_add */
    0, /* binaryfunc nb_subtract */
    0, /* binaryfunc nb_multiply */
    0, /* binaryfunc nb_remainder */
    0, /* binaryfunc nb_divmod */
    0, /* ternaryfunc nb_power */
    0, /* unaryfunc nb_negative */
    0, /* unaryfunc nb_positive */
    0, /* unaryfunc nb_absolute */
    0, /* inquiry nb_nonzero */ 
    0, /* unaryfunc nb_invert */
    0, /* binaryfunc nb_lshift */
    0, /* binaryfunc nb_rshift */
    0, /* binaryfunc nb_and */
    0, /* binaryfunc nb_xor */
    0, /* binaryfunc nb_or */
    0, /* unaryfunc nb_int */
    0, /* void * reserved */
    0  /* unaryfunc nb_float */
};

PyTypeObject PyOutputProxy::PyType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "_cspimpl.PyOutputProxy",   /* tp_name */
    sizeof(PyOutputProxy),      /* tp_basicsize */
    0,                         /* tp_itemsize */
    ( destructor ) PyOutputProxy_dealloc, /* tp_dealloc */
    0,                         /* tp_print */
    0,                         /* tp_getattr */
    0,                         /* tp_setattr */
    0,                         /* tp_reserved */
    0,                         /* tp_repr */
    &PyOutputProxy_NumberMethods,/* tp_as_number */
    0,                         /* tp_as_sequence */
    0,                         /* tp_as_mapping */
    0,                         /* tp_hash  */
    0,                         /* tp_call */
    0,                         /* tp_str */
    0,                         /* tp_getattro */
    0,                         /* tp_setattro */
    0,                         /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,        /* tp_flags */
    "csp output proxy",        /* tp_doc */
    0,                         /* tp_traverse */
    0,                         /* tp_clear */
    0,                         /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    0,                         /* tp_iter */
    0,                         /* tp_iternext */
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

REGISTER_TYPE_INIT( &PyOutputProxy::PyType, "PyOutputProxy" );

}
