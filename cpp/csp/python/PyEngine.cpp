#include <csp/engine/TimeSeriesProvider.h>
#include <csp/python/Common.h>
#include <csp/python/Conversions.h>
#include <csp/python/InitHelper.h>
#include <csp/python/PyEngine.h>
#include <csp/python/PyGraphOutputAdapter.h>
#include <csp/python/PyNode.h>


//define engine-level csp switch types here
#include <csp/python/PyCspType.h>

namespace csp::python
{

//The actual engine object
PythonEngine::PythonEngine( PyEngine * parent, const Dictionary & settings ) : RootEngine( settings ),
                                                                               m_parent( parent ),
                                                                               m_pyThreadState( nullptr )
{
    m_outputNumpy = settings.get<bool>( "output_numpy", false );
}

void PythonEngine::dialectUnlockGIL() noexcept
{
    assert( !m_pyThreadState );
    m_pyThreadState = PyEval_SaveThread();
}

void PythonEngine::dialectLockGIL() noexcept
{
    PyEval_RestoreThread( m_pyThreadState );
    m_pyThreadState = nullptr;
}

//The engine python wrapper object
PyEngine::PyEngine( const Dictionary & settings )
{
    m_ownEngine = true;
    m_engine = new PythonEngine( this, settings );
}

PyEngine::PyEngine( Engine * engine )
{
    m_ownEngine = false;
    m_engine = engine;
}

PyEngine::~PyEngine()
{
    if( m_ownEngine )
        delete m_engine;
    m_engine = nullptr;
}

PyObject * PyEngine::collectOutputs()
{
    CSP_BEGIN_METHOD;

    if( m_engine -> graphOutputKeys().empty() || rootEngine() -> interrupted() )
        Py_RETURN_NONE;

    PyObjectPtr out = PyObjectPtr::own( PyDict_New() );

    for( auto & key : m_engine -> graphOutputKeys() )
    {
        //Not sure if there is a more efficient way to do this other than converting per element ( aside from using numpy arrays which would
        //impose a dep on numpy )
        auto * adapter = static_cast<PyGraphOutputAdapter*>( m_engine -> graphOutput( key ) );
        auto & pykey = reinterpret_cast<const PyObjectPtr &>( key );
        if( PyDict_SetItem( out.ptr(), pykey.ptr(), adapter -> result().ptr() ) < 0 )
            CSP_THROW( PythonPassthrough, "" );
    }

    return out.release();
    CSP_RETURN_NONE;
}

PyEngine * PyEngine::create( Engine * engine )
{
    PyEngine * pyengine = ( PyEngine * ) PyType.tp_new( &PyType, nullptr, nullptr );
    new ( pyengine ) PyEngine( engine );
    return pyengine;

}

static int PyEngine_init( PyEngine * self, PyObject * args, PyObject * kwargs )
{
    CSP_BEGIN_METHOD;
    new( self ) PyEngine( fromPython<Dictionary>( kwargs ) );
    CSP_RETURN_INT;
}

static void PyEngine_dealloc( PyEngine * self )
{
    CSP_BEGIN_METHOD;
    self -> ~PyEngine();
    Py_TYPE( self ) -> tp_free( self ); 
    CSP_RETURN;
}

static PyObject * PyEngine_run( PyEngine * self, PyObject * args )
{
    CSP_BEGIN_METHOD;

    PyObject * pyStart;
    PyObject * pyEnd;
    if( !PyArg_ParseTuple( args, "OO", &pyStart, &pyEnd ) ) 
        return nullptr;

    auto start = fromPython<DateTime>( pyStart );
    auto end   = fromPython<DateTime>( pyEnd);

    CSP_TRUE_OR_THROW_RUNTIME( self -> engine() -> isRootEngine(), "engine is not root engine" );
    self -> rootEngine() -> run( start, end );

    return self -> collectOutputs();
    CSP_RETURN_NONE;
}

static PyMethodDef PyEngine_methods[] = {
    { "run",             (PyCFunction) PyEngine_run,            METH_VARARGS, "start and run engine" },
    { NULL }
};

PyTypeObject PyEngine::PyType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "_cspimpl.PyEngine",       /* tp_name */
    sizeof(PyEngine),          /* tp_basicsize */
    0,                         /* tp_itemsize */
    ( destructor ) PyEngine_dealloc, /* tp_dealloc */
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
    "csp engine",              /* tp_doc */
    0,                         /* tp_traverse */
    0,                         /* tp_clear */
    0,                         /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    0,                         /* tp_iter */
    0,                         /* tp_iternext */
    PyEngine_methods,          /* tp_methods */
    0,                         /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    ( initproc )PyEngine_init, /* tp_init */
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

REGISTER_TYPE_INIT( &PyEngine::PyType, "PyEngine" );

}
