#include <csp/engine/AdapterManager.h>
#include <csp/python/Conversions.h>
#include <csp/python/PyAdapterManager.h>
#include <csp/python/PyEngine.h>
#include <csp/python/Exception.h>
#include <csp/python/InitHelper.h>
#include <csp/python/PyObjectPtr.h>
#include <memory>

namespace csp::python
{

class PyAdapterManager : public AdapterManager
{
public:
    PyAdapterManager( Engine * engine, PyObjectPtr self ) : AdapterManager( engine ),
                                                            m_self( self )
    {
    }

    const char * name() const override
    {
        return Py_TYPE( m_self.ptr() ) -> tp_name;
    }

    void start( DateTime start, DateTime end ) override
    {
        PyObjectPtr rv = PyObjectPtr::own( PyObject_CallMethod( m_self.ptr(), "start", "OO", 
                                                                PyObjectPtr::own( toPython( start ) ).ptr(),
                                                                PyObjectPtr::own( toPython( end ) ).ptr() ) );
        if( !rv.ptr() )
            CSP_THROW( PythonPassthrough, "" );

        AdapterManager::start( start, end );
    }

    void stop() override
    {
        PyObjectPtr rv = PyObjectPtr::own( PyObject_CallMethod( m_self.ptr(), "stop", nullptr ) ); 

        if( !rv.ptr() )
            CSP_THROW( PythonPassthrough, "" );
    }

    DateTime processNextSimTimeSlice( DateTime time ) override
    {
        PyObjectPtr rv = PyObjectPtr::own( PyObject_CallMethod( m_self.ptr(), "process_next_sim_timeslice", "O", 
                                                                PyObjectPtr::own( toPython( time ) ).ptr() ) );

        if( !rv.ptr() )
        {
            if( PyErr_Occurred() == PyExc_KeyboardInterrupt )
            {
                rootEngine() -> shutdown();
                return DateTime::NONE();
            }

            CSP_THROW( PythonPassthrough, "" );
        }
        
        return rv.ptr() == Py_None ? DateTime::NONE() : fromPython<DateTime>( rv.ptr() );
    }

private:
    PyObjectPtr m_self;
};

static PyObject * PyAdapterManager_PyObject_starttime( PyAdapterManager_PyObject * self ) { return toPython( self -> manager -> starttime() ); }
static PyObject * PyAdapterManager_PyObject_endtime( PyAdapterManager_PyObject * self )   { return toPython( self -> manager -> endtime() ); }

static PyObject * PyAdapterManager_PyObject_shutdown_engine( PyAdapterManager_PyObject * self, PyObject * pyException )
{
    CSP_BEGIN_METHOD;
    
    self -> manager -> rootEngine() -> shutdown( PyEngine_shutdown_make_exception( pyException ) );

    CSP_RETURN_NONE;
}

static int PyAdapterManager_init( PyAdapterManager_PyObject *self, PyObject *args, PyObject *kwds )
{
    CSP_BEGIN_METHOD;

    PyEngine * pyEngine;
    
    if( !PyArg_ParseTuple( args, "O!", 
                           &PyEngine::PyType, &pyEngine ) )
        CSP_THROW( PythonPassthrough, "" );

    self -> manager = pyEngine -> engine() -> createOwnedObject<PyAdapterManager>( PyObjectPtr::incref( ( PyObject * ) self ) );
    CSP_RETURN_INT;
}

static PyMethodDef PyAdapterManager_methods[] = {
    { "starttime",          (PyCFunction) PyAdapterManager_PyObject_starttime, METH_NOARGS, "starttime" },
    { "endtime",            (PyCFunction) PyAdapterManager_PyObject_endtime,    METH_NOARGS, "endtime" },
    { "shutdown_engine",    (PyCFunction) PyAdapterManager_PyObject_shutdown_engine,  METH_O, "shutdown_engine" },
    {NULL}
};

PyTypeObject PyAdapterManager_PyObject::PyType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "_cspimpl.PyAdapterManager", /* tp_name */
    sizeof(PyAdapterManager_PyObject),    /* tp_basicsize */
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
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
    "csp adapter manager",     /* tp_doc */
    0,                         /* tp_traverse */
    0,                         /* tp_clear */
    0,                         /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    0,                         /* tp_iter */
    0,                         /* tp_iternext */
    PyAdapterManager_methods,    /* tp_methods */
    0,                         /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    ( initproc ) PyAdapterManager_init, /* tp_init */
    0,
    PyType_GenericNew,
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

REGISTER_TYPE_INIT( &PyAdapterManager_PyObject::PyType, "PyAdapterManager" );

}
