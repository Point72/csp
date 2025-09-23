#include <csp/engine/DynamicEngine.h>
#include <csp/python/Conversions.h>
#include <csp/python/InitHelper.h>
#include <csp/python/PyEngine.h>
#include <csp/python/PyNode.h>
#include <frameobject.h>
#include <traceback.h>

namespace csp::python
{

static PyObject * _csp_now( PyObject*, PyObject * nodeptr )
{
    CSP_BEGIN_METHOD;

    csp::Node * node = reinterpret_cast<csp::Node *>( fromPython<uint64_t>( nodeptr ) );
    return toPython( node -> now() );

    CSP_RETURN_NULL;
}

static PyObject * _engine_start_time( PyObject*, PyObject * nodeptr )
{
    CSP_BEGIN_METHOD;

    csp::Node * node = reinterpret_cast<csp::Node *>( fromPython<uint64_t>( nodeptr ) );
    return toPython( node -> rootEngine() -> startTime() );

    CSP_RETURN_NULL;
}

static PyObject * _engine_stats( PyObject*, PyObject * nodeptr )
{
    CSP_BEGIN_METHOD;

    csp::Node * node = reinterpret_cast<csp::Node *>( fromPython<uint64_t>( nodeptr ) );
    return toPython( node -> rootEngine() -> engine_stats() );

    CSP_RETURN_NULL;
}


static PyObject * _engine_end_time( PyObject*, PyObject * nodeptr )
{
    CSP_BEGIN_METHOD;

    csp::Node * node = reinterpret_cast<csp::Node *>( fromPython<uint64_t>( nodeptr ) );
    return toPython( node -> rootEngine() -> endTime() );

    CSP_RETURN_NULL;
}

static PyObject * _csp_stop_engine( PyObject*, PyObject * args, PyObject * kwargs )
{
    CSP_BEGIN_METHOD;
    int dynamicOnly = 0;
    uint64_t nodeptr;
    static const char * kwlist[] = { "node", "dynamic", nullptr };
    if( !PyArg_ParseTupleAndKeywords( args, kwargs, "L|p", ( char ** ) kwlist, &nodeptr, &dynamicOnly ) )
        CSP_THROW( PythonPassthrough, "" );

    csp::Node * node = reinterpret_cast<csp::Node *>( nodeptr );
    if( dynamicOnly && !node -> engine() -> isRootEngine() )
        static_cast<DynamicEngine *>( node -> engine() ) -> shutdown();
    else
        node -> rootEngine() -> shutdown();
    CSP_RETURN_NONE;
}

//Ingloriously stolen from Python 3.7.0!
static PyObject *_create_traceback( PyObject *, PyObject * args )
{
    CSP_BEGIN_METHOD;

    PyObject *next;
    PyFrameObject *frame;
    int lasti;
    int lineno;

    if( !PyArg_ParseTuple( args, "OO!ii", 
                           &next,
                           &PyFrame_Type, &frame, &lasti, &lineno ) )
        CSP_THROW( PythonPassthrough, "" );

    if( next == Py_None )
        next = nullptr;
    else if( !PyTraceBack_Check( next ) )
        CSP_THROW( TypeError, "expected traceback type" );

    PyTracebackObject *tb;
    tb = PyObject_GC_New( PyTracebackObject, &PyTraceBack_Type );
    if( tb != NULL )
    {
        Py_XINCREF(next);
        tb -> tb_next = (PyTracebackObject * ) next;
        Py_XINCREF(frame);
        tb -> tb_frame = frame;
        tb -> tb_lasti = lasti;
        tb -> tb_lineno = lineno;
        PyObject_GC_Track( tb );
    }

    return (PyObject *)tb;

    CSP_RETURN_NULL
}

static PyObject *_set_capture_cpp_backtrace( PyObject *, PyObject *args )
{
    CSP_BEGIN_METHOD;
    int value;

    if( !PyArg_ParseTuple( args, "p", &value ) )
        CSP_THROW( PythonPassthrough, "" );
    capture_cpp_exception_trace_flag() = bool( value );
    CSP_RETURN_NONE;
}


static PyMethodDef _cspimpl_methods[] = {
    {"_csp_now",                    (PyCFunction) _csp_now,                   METH_O, "current engine time"},
    {"_csp_engine_start_time",      (PyCFunction) _engine_start_time,         METH_O, "engine start time"},
    {"_csp_engine_end_time",        (PyCFunction) _engine_end_time,           METH_O, "engine end time"},
    {"_csp_stop_engine",            (PyCFunction) _csp_stop_engine,           METH_VARARGS | METH_KEYWORDS, "stop engine"},
    {"create_traceback",            (PyCFunction) _create_traceback,          METH_VARARGS,   "internal"},
    {"_csp_engine_stats",           (PyCFunction) _engine_stats,              METH_O, "engine statistics"},
    {"set_capture_cpp_backtrace",   (PyCFunction) _set_capture_cpp_backtrace, METH_VARARGS,   "internal"},
    {nullptr}
};

static PyModuleDef _cspimpl_module = {
    PyModuleDef_HEAD_INIT,
    "_cspimpl",
    "_cspimpl c++ module",
    -1,
    _cspimpl_methods, NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC PyInit__cspimpl(void)
{
    PyObject* m;

    m = PyModule_Create( &_cspimpl_module);

    if( m == NULL )
        return NULL;

    if( !InitHelper::instance().execute( m ) )
        return NULL;

    return m;
}

}
