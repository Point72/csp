#include <Python.h>
#include <csp/engine/CppNode.h>
#include <csp/engine/PushInputAdapter.h>
#include <csp/python/Conversions.h>
#include <csp/python/Exception.h>
#include <csp/python/InitHelper.h>
#include <csp/python/PyAdapterManagerWrapper.h>
#include <csp/python/PyCppNode.h>
#include <csp/python/PyEngine.h>
#include <csp/python/PyInputAdapterWrapper.h>
#include <csp/python/PyObjectPtr.h>
#include <csp/python/PyOutputAdapterWrapper.h>

namespace csp::cppnodes
{

// Expose C++ nodes for testing in Python
// Keep nodes for a specific test to a single namespace under testing

namespace testing
{

namespace stop_start_test
{

using namespace csp::python;

void setStatus( const DialectGenericType & obj_, const std::string & name )
{
    PyObjectPtr obj = PyObjectPtr::own( toPython( obj_ ) );
    PyObjectPtr attr = PyObjectPtr::own( PyUnicode_FromString( name.c_str() ) );
    PyObject_SetAttr( obj.get(), attr.get(), Py_True );
}

DECLARE_CPPNODE( start_n1_set_value )
{
    INIT_CPPNODE( start_n1_set_value ) {}

    SCALAR_INPUT( DialectGenericType, obj_ );

    START()
    {
       setStatus( obj_, "n1_started" );
    }
    INVOKE() {}

    STOP()
    {
        setStatus( obj_, "n1_stopped" );
    }
};
EXPORT_CPPNODE( start_n1_set_value );

DECLARE_CPPNODE( start_n2_throw )
{
    INIT_CPPNODE( start_n2_throw ) {}

    SCALAR_INPUT( DialectGenericType, obj_ );

    START()
    {
        CSP_THROW( ValueError, "n2 start failed" );
    }
    INVOKE() {}

    STOP()
    {
        setStatus( obj_, "n2_stopped" );
    }
};
EXPORT_CPPNODE( start_n2_throw );

}

namespace interrupt_stop_test
{

using namespace csp::python;

void setStatus( const DialectGenericType & obj_, int64_t idx )
{
    PyObjectPtr obj = PyObjectPtr::own( toPython( obj_ ) );
    PyObjectPtr list = PyObjectPtr::own( PyObject_GetAttrString( obj.get(), "stopped" ) );
    PyList_SET_ITEM( list.get(), idx, Py_True );
}

DECLARE_CPPNODE( set_stop_index )
{
    INIT_CPPNODE( set_stop_index ) {}

    SCALAR_INPUT( DialectGenericType, obj_ );
    SCALAR_INPUT( int64_t, idx );

    START() {}
    INVOKE() {}

    STOP()
    {
       setStatus( obj_, idx );
    }
};
EXPORT_CPPNODE( set_stop_index );

}

}

}

namespace csp::cppadapters
{

namespace testing
{

using namespace csp::python;

// Local copy of PyPushBatch so this module doesn't depend on the PyPushBatch::PyType
// symbol from _cspimpl (which causes linker errors on Windows).
struct CustomPyPushBatch
{
    PyObject_HEAD
    PushBatch batch;

    static PyTypeObject PyType;
};

static int CustomPyPushBatch_init( CustomPyPushBatch * self, PyObject * args, PyObject * kwargs )
{
    CSP_BEGIN_METHOD;
    if( PyTuple_GET_SIZE( args ) != 1 )
        CSP_THROW( TypeError, "PushBatch expected engine as single positional argument" );

    PyObject * pyengine = PyTuple_GET_ITEM( args, 0 );
    if( pyengine -> ob_type != &PyEngine::PyType )
        CSP_THROW( TypeError, "PushBatch expected engine as single positional argument" );

    csp::RootEngine * engine = ( ( PyEngine * ) pyengine ) -> rootEngine();
    new( &self -> batch ) PushBatch( engine );
    CSP_RETURN_INT;
}

static void CustomPyPushBatch_dealloc( CustomPyPushBatch * self )
{
    CSP_BEGIN_METHOD;
    self -> batch.~PushBatch();
    Py_TYPE( self ) -> tp_free( self );
    CSP_RETURN;
}

static PyObject * CustomPyPushBatch_enter( PyObject * self )
{
    Py_INCREF( self );
    return self;
}

static PyObject * CustomPyPushBatch_exit( CustomPyPushBatch * self, PyObject * args )
{
    PyObject * exception = PyTuple_GET_ITEM( args, 0 );
    if( exception != Py_None )
        self -> batch.clear();

    self -> batch.flush();
    Py_RETURN_NONE;
}

static PyMethodDef CustomPyPushBatch_methods[] = {
    { "__enter__", (PyCFunction) CustomPyPushBatch_enter, METH_NOARGS,  "enter context" },
    { "__exit__",  (PyCFunction) CustomPyPushBatch_exit,  METH_VARARGS, "exit context" },
    {NULL}
};

PyTypeObject CustomPyPushBatch::PyType = {
    .ob_base = PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "_csptestlibimpl.CustomPushBatch",
    .tp_basicsize = sizeof(CustomPyPushBatch),
    .tp_dealloc = ( destructor ) CustomPyPushBatch_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "csp push batch",
    .tp_methods = CustomPyPushBatch_methods,
    .tp_init = ( initproc ) CustomPyPushBatch_init,
    .tp_new = PyType_GenericNew,
};

// Test push adapter that accepts Python callables instead of final values.
// On the engine thread, transformRawEvent invokes the callable to produce
// the actual tick value (a DialectGenericType), preserving group properties
// and deleting the raw event.
class CallablePyPushInputAdapter: public PushInputAdapter
{
public:
    CallablePyPushInputAdapter( Engine * engine, AdapterManager * manager, PyObjectPtr pyadapter, PyObject * pyType,
                             PushMode pushMode, PyObjectPtr pyPushGroup, PushGroup * pushGroup, PyObjectPtr pyEngine ):
        PushInputAdapter( engine, pyTypeAsCspType( pyType ), pushMode, pushGroup, true ),
        m_pyadapter( pyadapter ),
        m_pyType( PyObjectPtr::incref( ( PyObject * ) pyType ) ),
        m_pyPushGroup( pyPushGroup ),
        m_pyEngine( pyEngine )
    {
    }

    void start( DateTime start, DateTime end ) override
    {
        PyObjectPtr rv = PyObjectPtr::own( PyObject_CallMethod( m_pyadapter.ptr(), "start", "OO", 
                                                                PyObjectPtr::own( toPython( start ) ).ptr(),
                                                                PyObjectPtr::own( toPython( end ) ).ptr() ) );
        if( !rv.ptr() )
            CSP_THROW( PythonPassthrough, "" );
    }

    void stop() override
    {
        PyObjectPtr rv = PyObjectPtr::own( PyObject_CallMethod( m_pyadapter.ptr(), "stop", nullptr ) ); 

        if( !rv.ptr() )
        {
            if( PyErr_Occurred() == PyExc_KeyboardInterrupt )
            {
                PyErr_Clear();
                rv = PyObjectPtr::own( PyObject_CallMethod( m_pyadapter.ptr(), "stop", nullptr ) );
            }

            if( !rv.ptr() )
                CSP_THROW( PythonPassthrough, "" );
        }
    }

    void pushPyTick( PyObject * value, PushBatch * batch )
    {
        PyObjectPtr ptr = PyObjectPtr::incref( value );
        pushTick<PyObjectPtr>( std::move( ptr ), batch );
    }

    PushEvent * transformRawEvent( PushEvent * event ) override
    {
        csp::TypedPushEvent<PyObjectPtr> *tevent  = static_cast<csp::TypedPushEvent<PyObjectPtr> *>( event );
        PyObjectPtr result = PyObjectPtr::own( PyObject_CallNoArgs( (tevent -> data).get() ) );
        if (result.get() == NULL) {
            CSP_THROW( RuntimeException, "Error while invoking callable" );
        }
        auto value = csp::python::fromPython<DialectGenericType>( result.get() );
        PushEvent * new_event = new csp::TypedPushEvent<DialectGenericType>( this, std::forward<DialectGenericType>(value) );

        // Maintain group properties from the raw event
        if( event -> isGroupEnd() )
            new_event -> flagGroupEnd();

        // Delete the raw event
        delete tevent;

        return new_event;
    }

    PyObject * pyEngine() { return m_pyEngine.ptr(); }

private:
    PyObjectPtr m_pyadapter;
    PyObjectPtr m_pyType;
    PyObjectPtr m_pyPushGroup;
    PyObjectPtr m_pyEngine;
};

// Python-exposed type wrapping CallablePyPushInputAdapter.
// Exposes push_tick (with optional PushBatch), engine(), and shutdown_engine to Python.
struct CallablePyPushInputAdapter_PyObject
{
    PyObject_HEAD
    CallablePyPushInputAdapter * adapter;

    static PyObject * pushTick( CallablePyPushInputAdapter_PyObject * self, PyObject * args, PyObject **kwargs )
    {
        CSP_BEGIN_METHOD;
        PyObject *pyvalue = nullptr;

        Py_ssize_t len = PyTuple_GET_SIZE( args );
        if( len < 1 || len > 2 )
            CSP_THROW( TypeError, "push_tick takes value and optional batch as positional arguments" );

        pyvalue = PyTuple_GET_ITEM( args, 0 );

        PushBatch * batch = nullptr;
        if( len == 2 )
        {
            PyObject * pybatch = PyTuple_GET_ITEM( args, 1 );
            if( pybatch -> ob_type != &CustomPyPushBatch::PyType )
                CSP_THROW( TypeError, "push_tick expected PushBatch type as second argument, got " << pybatch -> ob_type -> tp_name );

            batch = &( ( CustomPyPushBatch * ) pybatch ) -> batch;
        }

        self -> adapter -> pushPyTick( pyvalue, batch );

        CSP_RETURN_NONE;
    }

    static PyObject * engine( CallablePyPushInputAdapter_PyObject * self )
    {
        PyObject * pyEngine = self -> adapter -> pyEngine();
        Py_INCREF( pyEngine );
        return pyEngine;
    }

    static PyObject * shutdown_engine( CallablePyPushInputAdapter_PyObject * self, PyObject * pyException )
    {
        CSP_BEGIN_METHOD;

        self -> adapter -> rootEngine() -> shutdown( PyEngine_shutdown_make_exception( pyException ) );

        CSP_RETURN_NONE;
    }

    static PyTypeObject PyType;
};

static PyMethodDef CallablePyPushInputAdapter_PyObject_methods[] = {
    { "push_tick",          (PyCFunction) CallablePyPushInputAdapter_PyObject::pushTick, METH_VARARGS, "push new tick" },
    { "engine",             (PyCFunction) CallablePyPushInputAdapter_PyObject::engine,   METH_NOARGS,  "get engine" },
    { "shutdown_engine",    (PyCFunction) CallablePyPushInputAdapter_PyObject::shutdown_engine, METH_O, "shutdown_engine" },
    {NULL}
};

PyTypeObject CallablePyPushInputAdapter_PyObject::PyType = {
    .ob_base = PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "_csptestlibimpl.CallablePyPushInputAdapter",
    .tp_basicsize = sizeof(CallablePyPushInputAdapter_PyObject),
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_doc = "test csp push input adapter",
    .tp_methods = CallablePyPushInputAdapter_PyObject_methods,
    .tp_new = PyType_GenericNew,
};

static InputAdapter * callablepypushinputadapter_creator( csp::AdapterManager * manager, PyEngine * pyengine,
                                                  PyObject * pyType, PushMode pushMode, PyObject * args )
{
    PyTypeObject * pyAdapterType = nullptr;
    PyObject * adapterArgs = nullptr;
    PyObject * pyPushGroup;

    if( !PyArg_ParseTuple( args, "O!OO!", &PyType_Type, &pyAdapterType, &pyPushGroup, &PyTuple_Type, &adapterArgs ) )
        CSP_THROW( PythonPassthrough, "" );

    if( !PyType_IsSubtype( pyAdapterType, &CallablePyPushInputAdapter_PyObject::PyType ) )
        CSP_THROW( TypeError, "Expected PushInputAdapter derived type, got " << pyAdapterType -> tp_name );

    csp::PushGroup *pushGroup = nullptr;
    if( pyPushGroup != Py_None )
    {
        pushGroup = ( csp::PushGroup * ) PyCapsule_GetPointer( pyPushGroup, nullptr );
        if( !pushGroup )
        {
            PyErr_Clear();
            CSP_THROW( TypeError, "Expected PushGroup instance for push group, got: " << PyObjectPtr::incref( pyPushGroup ) );
        }
    }

    CallablePyPushInputAdapter_PyObject * pyAdapter = ( CallablePyPushInputAdapter_PyObject * ) PyObject_Call( ( PyObject * ) pyAdapterType, adapterArgs, nullptr );
    if( !pyAdapter )
        CSP_THROW( PythonPassthrough, "" );

    pyAdapter -> adapter = pyengine -> engine() -> createOwnedObject<CallablePyPushInputAdapter>(
        manager, PyObjectPtr::own( ( PyObject * ) pyAdapter ), pyType, pushMode, PyObjectPtr::incref( pyPushGroup ), pushGroup,
        PyObjectPtr::incref( ( PyObject * ) pyengine ) );
    return pyAdapter -> adapter;
}

REGISTER_TYPE_INIT( &CustomPyPushBatch::PyType,                   "CustomPushBatch" );
REGISTER_TYPE_INIT( &CallablePyPushInputAdapter_PyObject::PyType, "CallablePyPushInputAdapter" );

REGISTER_INPUT_ADAPTER( _callablepushadapter, callablepypushinputadapter_creator );

}

}

// Test nodes
REGISTER_CPPNODE( csp::cppnodes::testing::stop_start_test, start_n1_set_value );
REGISTER_CPPNODE( csp::cppnodes::testing::stop_start_test, start_n2_throw );
REGISTER_CPPNODE( csp::cppnodes::testing::interrupt_stop_test, set_stop_index );

static PyModuleDef _csptestlibimpl_module = {
    PyModuleDef_HEAD_INIT,
    "_csptestlibimpl",
    "_csptestlibimpl c++ module",
    -1,
    NULL, NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC PyInit__csptestlibimpl(void)
{
    PyObject* m;

    m = PyModule_Create( &_csptestlibimpl_module);
    if( m == NULL )
        return NULL;

    if( !csp::python::InitHelper::instance().execute( m ) )
        return NULL;

    return m;
}
