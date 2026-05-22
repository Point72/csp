#include <csp/python/PyEventLoop.h>
#include <csp/python/Conversions.h>
#include <csp/python/Exception.h>
#include <csp/python/InitHelper.h>
#include <csp/python/PyObjectPtr.h>
#include <csp/core/Time.h>
#include <thread>

namespace csp::python
{

PyEventLoopAdapter::PyEventLoopAdapter( PythonEngine * engine ) : m_engine( engine ),
                                                                   m_nextCallbackId( 1 ),
                                                                   m_stopRequested( false )
{
}

PyEventLoopAdapter::~PyEventLoopAdapter()
{
    // Clean up any remaining callbacks
    for( auto & entry : m_callbacks )
        cleanupEntry( &entry );

    m_callbacks.clear();

    // Clean up threadsafe queue
    std::lock_guard<std::mutex> lock( m_threadSafeMutex );
    for( auto & item : m_threadSafeQueue )
    {
        Py_XDECREF( std::get<0>( item ) );
        Py_XDECREF( std::get<1>( item ) );
        Py_XDECREF( std::get<2>( item ) );
    }
    m_threadSafeQueue.clear();
}

double PyEventLoopAdapter::time() const
{
    DateTime now = m_engine -> now();
    if( now == DateTime::NONE() )
    {
        // If engine hasn't started, return current wall time
        now = DateTime::now();
    }
    // Convert to seconds since Unix epoch
    return static_cast<double>( now.asNanoseconds() ) / 1e9;
}

uint64_t PyEventLoopAdapter::scheduleCallback( DateTime time, PyObject * callback,
                                               PyObject * args, PyObject * context )
{
    uint64_t callbackId = m_nextCallbackId++;

    // Hold references
    Py_INCREF( callback );
    Py_XINCREF( args );
    Py_XINCREF( context );

    CallbackEntry entry;
    entry.callback  = callback;
    entry.args      = args;
    entry.context   = context;
    entry.id        = callbackId;
    entry.cancelled = false;

    auto it = m_callbacks.insert( m_callbacks.end(), entry );

    // Schedule with CSP's scheduler
    auto handle = m_engine -> scheduleCallback(
        time,
        [this, it]() -> const InputAdapter *
        {
            if( !it -> cancelled )
                executeCallback( &( *it ) );

            cleanupEntry( &( *it ) );
            m_callbacks.erase( it );
            return nullptr;
        }
    );

    it -> handle = handle;

    return callbackId;
}

uint64_t PyEventLoopAdapter::callSoon( PyObject * callback, PyObject * args, PyObject * context )
{
    // Schedule for "now" - will execute in the next cycle
    return scheduleCallback( m_engine -> now(), callback, args, context );
}

uint64_t PyEventLoopAdapter::callLater( double delay, PyObject * callback,
                                        PyObject * args, PyObject * context )
{
    if( delay < 0 )
        delay = 0;

    TimeDelta delta      = TimeDelta::fromSeconds( delay );
    DateTime  targetTime = m_engine -> now() + delta;

    return scheduleCallback( targetTime, callback, args, context );
}

uint64_t PyEventLoopAdapter::callAt( double when, PyObject * callback,
                                     PyObject * args, PyObject * context )
{
    // Convert seconds since epoch to DateTime
    int64_t  nanos      = static_cast<int64_t>( when * 1e9 );
    DateTime targetTime = DateTime::fromNanoseconds( nanos );

    return scheduleCallback( targetTime, callback, args, context );
}

bool PyEventLoopAdapter::cancelCallback( uint64_t callbackId )
{
    for( auto & entry : m_callbacks )
    {
        if( entry.id == callbackId && !entry.cancelled )
        {
            entry.cancelled = true;
            m_engine -> cancelCallback( entry.handle );
            return true;
        }
    }
    return false;
}

bool PyEventLoopAdapter::isCallbackPending( uint64_t callbackId ) const
{
    for( const auto & entry : m_callbacks )
    {
        if( entry.id == callbackId && !entry.cancelled )
            return true;
    }
    return false;
}

uint64_t PyEventLoopAdapter::callSoonThreadsafe( PyObject * callback, PyObject * args, PyObject * context )
{
    // Add to threadsafe queue
    Py_INCREF( callback );
    Py_XINCREF( args );
    Py_XINCREF( context );

    {
        std::lock_guard<std::mutex> lock( m_threadSafeMutex );
        m_threadSafeQueue.emplace_back( callback, args, context );
    }

    // The engine will pick these up on the next cycle
    // In a real implementation, we'd wake up the engine here
    return 0;  // No ID for threadsafe callbacks currently
}

void PyEventLoopAdapter::processPendingThreadsafeCallbacks()
{
    std::vector<std::tuple<PyObject *, PyObject *, PyObject *>> pending;

    {
        std::lock_guard<std::mutex> lock( m_threadSafeMutex );
        pending.swap( m_threadSafeQueue );
    }

    for( auto & item : pending )
    {
        callSoon( std::get<0>( item ), std::get<1>( item ), std::get<2>( item ) );
        // callSoon will incref, so decref the ones we added in callSoonThreadsafe
        Py_DECREF( std::get<0>( item ) );
        Py_XDECREF( std::get<1>( item ) );
        Py_XDECREF( std::get<2>( item ) );
    }
}

void PyEventLoopAdapter::stop()
{
    m_stopRequested = true;
    m_engine -> shutdown();
}

void PyEventLoopAdapter::executeCallback( CallbackEntry * entry )
{
    PyObject * result = nullptr;

    if( entry -> context )
    {
        // Run in context
        PyObject * contextRun = PyObject_GetAttrString( entry -> context, "run" );
        if( contextRun )
        {
            if( entry -> args )
            {
                // Prepend callback to args
                Py_ssize_t argsSize = PyTuple_Size( entry -> args );
                PyObjectPtr newArgs = PyObjectPtr::own( PyTuple_New( argsSize + 1 ) );
                Py_INCREF( entry -> callback );
                PyTuple_SET_ITEM( newArgs.ptr(), 0, entry -> callback );
                for( Py_ssize_t i = 0; i < argsSize; ++i )
                {
                    PyObject * item = PyTuple_GET_ITEM( entry -> args, i );
                    Py_INCREF( item );
                    PyTuple_SET_ITEM( newArgs.ptr(), i + 1, item );
                }
                result = PyObject_Call( contextRun, newArgs.ptr(), nullptr );
            }
            else
            {
                PyObjectPtr args = PyObjectPtr::own( PyTuple_Pack( 1, entry -> callback ) );
                result = PyObject_Call( contextRun, args.ptr(), nullptr );
            }
            Py_DECREF( contextRun );
        }
    }
    else
    {
        // Direct call
        if( entry -> args )
            result = PyObject_Call( entry -> callback, entry -> args, nullptr );
        else
            result = PyObject_CallNoArgs( entry -> callback );
    }

    if( result )
        Py_DECREF( result );
    else
        // Handle exception - for now just print it
        PyErr_Print();
}

void PyEventLoopAdapter::cleanupEntry( CallbackEntry * entry )
{
    Py_DECREF( entry -> callback );
    Py_XDECREF( entry -> args );
    Py_XDECREF( entry -> context );
}

static void PyAsyncioHandle_dealloc( PyAsyncioHandle * self )
{
    CSP_BEGIN_METHOD;
    Py_XDECREF( self -> callback );
    Py_XDECREF( ( PyObject * ) self -> loop );
    Py_TYPE( self ) -> tp_free( ( PyObject * ) self );
    CSP_RETURN;
}

static PyObject * PyAsyncioHandle_cancel( PyAsyncioHandle * self )
{
    CSP_BEGIN_METHOD;
    if( !self -> cancelled && self -> loop && self -> loop -> adapter )
    {
        self -> loop -> adapter -> cancelCallback( self -> callback_id );
        self -> cancelled = true;
    }
    Py_RETURN_NONE;
    CSP_RETURN_NONE;
}

static PyObject * PyAsyncioHandle_cancelled( PyAsyncioHandle * self )
{
    if( self -> cancelled )
        Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

static PyObject * PyAsyncioHandle_repr( PyAsyncioHandle * self )
{
    return PyUnicode_FromFormat( "<Handle callback=%R cancelled=%s>",
                                 self -> callback,
                                 self -> cancelled ? "True" : "False" );
}

static PyMethodDef PyAsyncioHandle_methods[] = {
    { "cancel",    ( PyCFunction ) PyAsyncioHandle_cancel,    METH_NOARGS, "Cancel the callback." },
    { "cancelled", ( PyCFunction ) PyAsyncioHandle_cancelled, METH_NOARGS, "Return True if the callback was cancelled." },
    { NULL }
};

PyTypeObject PyAsyncioHandle::PyType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "_cspimpl.AsyncioHandle",                /* tp_name */
    sizeof(PyAsyncioHandle),                 /* tp_basicsize */
    0,                                       /* tp_itemsize */
    ( destructor ) PyAsyncioHandle_dealloc,  /* tp_dealloc */
    0,                                       /* tp_print */
    0,                                       /* tp_getattr */
    0,                                       /* tp_setattr */
    0,                                       /* tp_reserved */
    ( reprfunc ) PyAsyncioHandle_repr,       /* tp_repr */
    0,                                       /* tp_as_number */
    0,                                       /* tp_as_sequence */
    0,                                       /* tp_as_mapping */
    0,                                       /* tp_hash  */
    0,                                       /* tp_call */
    0,                                       /* tp_str */
    0,                                       /* tp_getattro */
    0,                                       /* tp_setattro */
    0,                                       /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,                      /* tp_flags */
    "asyncio Handle wrapper",                /* tp_doc */
    0,                                       /* tp_traverse */
    0,                                       /* tp_clear */
    0,                                       /* tp_richcompare */
    0,                                       /* tp_weaklistoffset */
    0,                                       /* tp_iter */
    0,                                       /* tp_iternext */
    PyAsyncioHandle_methods,                 /* tp_methods */
    0,                                       /* tp_members */
    0,                                       /* tp_getset */
    0,                                       /* tp_base */
    0,                                       /* tp_dict */
    0,                                       /* tp_descr_get */
    0,                                       /* tp_descr_set */
    0,                                       /* tp_dictoffset */
    0,                                       /* tp_init */
    0,                                       /* tp_alloc */
    PyType_GenericNew,                       /* tp_new */
};

static void PyAsyncioTimerHandle_dealloc( PyAsyncioTimerHandle * self )
{
    CSP_BEGIN_METHOD;
    Py_XDECREF( self -> callback );
    Py_XDECREF( ( PyObject * ) self -> loop );
    Py_TYPE( self ) -> tp_free( ( PyObject * ) self );
    CSP_RETURN;
}

static PyObject * PyAsyncioTimerHandle_cancel( PyAsyncioTimerHandle * self )
{
    CSP_BEGIN_METHOD;
    if( !self -> cancelled && self -> loop && self -> loop -> adapter )
    {
        self -> loop -> adapter -> cancelCallback( self -> callback_id );
        self -> cancelled = true;
    }
    Py_RETURN_NONE;
    CSP_RETURN_NONE;
}

static PyObject * PyAsyncioTimerHandle_cancelled( PyAsyncioTimerHandle * self )
{
    if( self -> cancelled )
        Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

static PyObject * PyAsyncioTimerHandle_when( PyAsyncioTimerHandle * self )
{
    return PyFloat_FromDouble( self -> when );
}

static PyObject * PyAsyncioTimerHandle_repr( PyAsyncioTimerHandle * self )
{
    return PyUnicode_FromFormat( "<TimerHandle callback=%R when=%R cancelled=%s>",
                                 self -> callback,
                                 PyFloat_FromDouble( self -> when ),
                                 self -> cancelled ? "True" : "False" );
}

static PyMethodDef PyAsyncioTimerHandle_methods[] = {
    { "cancel",    ( PyCFunction ) PyAsyncioTimerHandle_cancel,    METH_NOARGS, "Cancel the callback." },
    { "cancelled", ( PyCFunction ) PyAsyncioTimerHandle_cancelled, METH_NOARGS, "Return True if the callback was cancelled." },
    { "when",      ( PyCFunction ) PyAsyncioTimerHandle_when,      METH_NOARGS, "Return scheduled callback time." },
    { NULL }
};

PyTypeObject PyAsyncioTimerHandle::PyType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "_cspimpl.AsyncioTimerHandle",                /* tp_name */
    sizeof(PyAsyncioTimerHandle),                 /* tp_basicsize */
    0,                                            /* tp_itemsize */
    ( destructor ) PyAsyncioTimerHandle_dealloc,  /* tp_dealloc */
    0,                                            /* tp_print */
    0,                                            /* tp_getattr */
    0,                                            /* tp_setattr */
    0,                                            /* tp_reserved */
    ( reprfunc ) PyAsyncioTimerHandle_repr,       /* tp_repr */
    0,                                            /* tp_as_number */
    0,                                            /* tp_as_sequence */
    0,                                            /* tp_as_mapping */
    0,                                            /* tp_hash  */
    0,                                            /* tp_call */
    0,                                            /* tp_str */
    0,                                            /* tp_getattro */
    0,                                            /* tp_setattro */
    0,                                            /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,                           /* tp_flags */
    "asyncio TimerHandle wrapper",                /* tp_doc */
    0,                                            /* tp_traverse */
    0,                                            /* tp_clear */
    0,                                            /* tp_richcompare */
    0,                                            /* tp_weaklistoffset */
    0,                                            /* tp_iter */
    0,                                            /* tp_iternext */
    PyAsyncioTimerHandle_methods,                 /* tp_methods */
    0,                                            /* tp_members */
    0,                                            /* tp_getset */
    0,                                            /* tp_base */
    0,                                            /* tp_dict */
    0,                                            /* tp_descr_get */
    0,                                            /* tp_descr_set */
    0,                                            /* tp_dictoffset */
    0,                                            /* tp_init */
    0,                                            /* tp_alloc */
    PyType_GenericNew,                            /* tp_new */
};

// Register types
REGISTER_TYPE_INIT( &PyAsyncioHandle::PyType, "AsyncioHandle" );
REGISTER_TYPE_INIT( &PyAsyncioTimerHandle::PyType, "AsyncioTimerHandle" );

}
