#include <csp/engine/PushPullInputAdapter.h>
#include <csp/python/Common.h>
#include <csp/python/Conversions.h>
#include <csp/python/Exception.h>
#include <csp/python/PyInputAdapterWrapper.h>
#include <csp/python/PyPushInputAdapter.h>
#include <csp/python/PyEngine.h>
#include <csp/python/PyObjectPtr.h>

namespace csp::python
{

class PyPushPullInputAdapter : public PushPullInputAdapter
{
public:
    PyPushPullInputAdapter( Engine * engine, AdapterManager * manager,
                            PyObjectPtr pyadapter, PyObject * pyType,
                            PushMode pushMode, PushGroup * pushGroup ) : PushPullInputAdapter( engine, pyTypeAsCspType( pyType ), pushMode, pushGroup ),
                                                                         m_pyadapter( pyadapter ),
                                                                         m_pyType( PyObjectPtr::incref( ( PyObject * ) pyType ) )
    {
    }

    //override nextPullEvent so we can release GIL while we wait
    PushPullEvent * nextPullEvent() override
    {
        ReleaseGIL release;
        return PushPullInputAdapter::nextPullEvent();
    }

    void start( DateTime start, DateTime end ) override
    {
        PyObjectPtr rv = PyObjectPtr::own( PyObject_CallMethod( m_pyadapter.ptr(), "start", "OO", 
                                                                PyObjectPtr::own( toPython( start ) ).ptr(),
                                                                PyObjectPtr::own( toPython( end ) ).ptr() ) );
        if( !rv.ptr() )
            CSP_THROW( PythonPassthrough, "" );

        PushPullInputAdapter::start( start, end );
    }

    void stop() override
    {
        PushPullInputAdapter::stop();

        PyObjectPtr rv = PyObjectPtr::own( PyObject_CallMethod( m_pyadapter.ptr(), "stop", nullptr ) ); 

        if( !rv.ptr() )
            CSP_THROW( PythonPassthrough, "" );
    }

    virtual void pushPyTick( bool live, PyObject * time, PyObject * value, PushBatch * batch ) = 0;

protected:
    PyObjectPtr m_pyadapter;
    PyObjectPtr m_pyType;
};

template<typename T>
class TypedPyPushPullInputAdapter : public PyPushPullInputAdapter
{
public:
    TypedPyPushPullInputAdapter( Engine * engine, AdapterManager * manager, PyObjectPtr pyadapter, PyObject * pyType,
                                 PushMode pushMode, PyObjectPtr pyPushGroup, PushGroup * pushGroup ):
        PyPushPullInputAdapter( engine, manager, pyadapter, pyType, pushMode, pushGroup ),
        m_pyPushGroup( pyPushGroup )
    {
    }

    void pushPyTick( bool live, PyObject * time, PyObject * value, PushBatch * batch ) override
    {
        DateTime t = fromPython<DateTime>( time );
        try
        {
            if( !validatePyType( this -> dataType(), m_pyType.ptr(), value ) )
                CSP_THROW( TypeError, "" );
            pushTick<T>( live, t, std::move( fromPython<T>( value, *this -> dataType() ) ), batch );
        }
        catch( const TypeError & )
        {
            CSP_THROW( TypeError, "\"" << Py_TYPE( m_pyadapter.ptr() ) -> tp_name << "\" push adapter expected output type to be of type \"" 
                       << pyTypeToString( m_pyType.ptr() ) << "\" got type \"" << Py_TYPE( value ) -> tp_name << "\"" );
        }
    }

private:
    PyObjectPtr m_pyPushGroup;
};

struct PyPushPullInputAdapter_PyObject
{
    PyObject_HEAD
    PyPushPullInputAdapter * adapter;

    static PyObject * pushTick( PyPushPullInputAdapter_PyObject * self, PyObject * args, PyObject **kwargs )
    {
        CSP_BEGIN_METHOD;
        PyObject *pylive;
        PyObject *pytime;
        PyObject *pyvalue;
        PyObject *pybatch = nullptr;

        if( !PyArg_UnpackTuple( args, "push_tick", 3, 4, &pylive, &pytime, &pyvalue, &pybatch ) )
            CSP_THROW( PythonPassthrough, "" );

        PushBatch * batch = nullptr;
        if( pybatch )
        {
            if( pybatch -> ob_type != &PyPushBatch::PyType )
                CSP_THROW( TypeError, "push_tick expected PushBatch type as second argument, got " << pybatch -> ob_type -> tp_name );

            batch = &( ( PyPushBatch * ) pybatch ) -> batch;
        }

        self -> adapter -> pushPyTick( fromPython<bool>( pylive ), pytime, pyvalue, batch );

        CSP_RETURN_NONE;
    }
    
    static PyObject * flagReplayComplete( PyPushPullInputAdapter_PyObject * self, PyObject * args, PyObject **kwargs )
    {
        CSP_BEGIN_METHOD;
        self -> adapter -> flagReplayComplete();
        CSP_RETURN_NONE;
    }

    static PyObject * shutdown_engine( PyPushPullInputAdapter_PyObject * self, PyObject * pyException )
    {
        CSP_BEGIN_METHOD;
        
        self -> adapter -> rootEngine() -> shutdown( PyEngine_shutdown_make_exception( pyException ) );

        CSP_RETURN_NONE;
    }

    static PyTypeObject PyType;
};

static PyMethodDef PyPushPullInputAdapter_PyObject_methods[] = {
    { "push_tick",              (PyCFunction) PyPushPullInputAdapter_PyObject::pushTick, METH_VARARGS, "push new tick" },
    { "flag_replay_complete",   (PyCFunction) PyPushPullInputAdapter_PyObject::flagReplayComplete, METH_VARARGS, "finish replay ticks" },
    { "shutdown_engine",        (PyCFunction) PyPushPullInputAdapter_PyObject::shutdown_engine, METH_O, "shutdown engine" },
    {NULL}
};

PyTypeObject PyPushPullInputAdapter_PyObject::PyType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "_cspimpl.PyPushPullInputAdapter", /* tp_name */
    sizeof(PyPushPullInputAdapter_PyObject),    /* tp_basicsize */
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
    "csp push input adapter",  /* tp_doc */
    0,                         /* tp_traverse */
    0,                         /* tp_clear */
    0,                         /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    0,                         /* tp_iter */
    0,                         /* tp_iternext */
    PyPushPullInputAdapter_PyObject_methods,    /* tp_methods */
    0,                         /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    0,                         /* tp_init */
    0,
    PyType_GenericNew,
};

static InputAdapter * pypushpullinputadapter_creator( csp::AdapterManager * manager, PyEngine * pyengine, 
                                                      PyObject * pyType, PushMode pushMode, PyObject * args )
{
    PyTypeObject * pyAdapterType = nullptr;
    PyObject * adapterArgs = nullptr;
    PyObject * pyPushGroup;

    if( !PyArg_ParseTuple( args, "O!OO!", &PyType_Type, &pyAdapterType, &pyPushGroup, &PyTuple_Type, &adapterArgs ) )
        CSP_THROW( PythonPassthrough, "" );

    if( !PyType_IsSubtype( pyAdapterType, &PyPushPullInputAdapter_PyObject::PyType ) )
        CSP_THROW( TypeError, "Expected PushPullInputAdapter derived type, got " << pyAdapterType -> tp_name );

    csp::PushGroup *pushGroup = nullptr;
    if( pyPushGroup != Py_None )
    {
        pushGroup = ( csp::PushGroup * ) PyCapsule_GetPointer( pyPushGroup, nullptr );
        if( !pushGroup )
            CSP_THROW( PythonPassthrough, "" );
    }

    PyPushPullInputAdapter_PyObject * pyAdapter = ( PyPushPullInputAdapter_PyObject * ) PyObject_Call( ( PyObject * ) pyAdapterType, adapterArgs, nullptr );
    if( !pyAdapter )
        CSP_THROW( PythonPassthrough, "" );

    switchPyType( pyType,
                  [&]( auto tag )
                  {
                      pyAdapter -> adapter = pyengine -> engine() -> createOwnedObject<TypedPyPushPullInputAdapter<typename decltype(tag)::type>>( 
                          manager, PyObjectPtr::own( ( PyObject * ) pyAdapter ), pyType, pushMode, PyObjectPtr::incref( pyPushGroup ), pushGroup );
                  } );
    
    return pyAdapter -> adapter;
}

REGISTER_TYPE_INIT( &PyPushPullInputAdapter_PyObject::PyType, "PyPushPullInputAdapter" );
REGISTER_INPUT_ADAPTER( _pushpulladapter, pypushpullinputadapter_creator );

}

