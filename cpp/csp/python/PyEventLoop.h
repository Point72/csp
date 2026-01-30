#ifndef _IN_CSP_PYTHON_PYEVENTLOOP_H
#define _IN_CSP_PYTHON_PYEVENTLOOP_H

#include <csp/engine/RootEngine.h>
#include <csp/python/PyEngine.h>
#include <Python.h>
#include <functional>
#include <list>
#include <mutex>
#include <vector>

namespace csp::python
{

//PyEventLoopAdapter provides the bridge between CSP's scheduler and Python's asyncio.
//It allows asyncio callbacks to be scheduled and executed within CSP's event loop.
class PyEventLoopAdapter
{
public:
    struct CallbackEntry
    {
        PyObject          * callback;
        PyObject          * args;      // Can be nullptr
        PyObject          * context;   // Can be nullptr
        Scheduler::Handle   handle;
        uint64_t            id;
        bool                cancelled;
    };

    PyEventLoopAdapter( PythonEngine * engine );
    ~PyEventLoopAdapter();

    // Schedule a callback to be called as soon as possible
    // Returns a unique callback ID that can be used for cancellation
    uint64_t callSoon( PyObject * callback, PyObject * args, PyObject * context );

    // Schedule a callback to be called at a specific time
    // delay is in seconds
    uint64_t callLater( double delay, PyObject * callback, PyObject * args, PyObject * context );

    // Schedule a callback to be called at a specific absolute time
    // when is in seconds since epoch (loop.time() compatible)
    uint64_t callAt( double when, PyObject * callback, PyObject * args, PyObject * context );

    // Cancel a scheduled callback
    bool cancelCallback( uint64_t callbackId );

    // Check if a callback is still pending
    bool isCallbackPending( uint64_t callbackId ) const;

    // Get the current loop time in seconds
    double time() const;

    // Thread-safe version of callSoon (wakes up the engine if needed)
    uint64_t callSoonThreadsafe( PyObject * callback, PyObject * args, PyObject * context );

    // Process any pending threadsafe callbacks (called from engine thread)
    void processPendingThreadsafeCallbacks();

    // Stop the loop
    void stop();

    // Check if stop was requested
    bool stopRequested() const { return m_stopRequested; }

private:
    uint64_t scheduleCallback( DateTime time, PyObject * callback, PyObject * args, PyObject * context );
    void executeCallback( CallbackEntry * entry );
    void cleanupEntry( CallbackEntry * entry );

    PythonEngine              * m_engine;
    uint64_t                    m_nextCallbackId;
    std::list<CallbackEntry>    m_callbacks;

    // Thread-safe callback queue
    std::mutex                                                  m_threadSafeMutex;
    std::vector<std::tuple<PyObject *, PyObject *, PyObject *>> m_threadSafeQueue;

    bool m_stopRequested;
};

//PyAsyncioLoop is the Python wrapper for the asyncio event loop.
struct PyAsyncioLoop
{
    PyObject_HEAD
    PyEventLoopAdapter * adapter;
    PyEngine           * pyengine;
    bool                 running;
    bool                 closed;
    bool                 debug;
    PyObject           * exception_handler;
    PyObject           * task_factory;
    PyObject           * ready;  // deque of ready callbacks
    uint64_t             thread_id;

    static PyTypeObject PyType;
};

// Handle wrapper for Python
struct PyAsyncioHandle
{
    PyObject_HEAD
    PyAsyncioLoop * loop;
    uint64_t        callback_id;
    PyObject      * callback;  // For repr
    bool            cancelled;

    static PyTypeObject PyType;
};

// TimerHandle wrapper for Python
struct PyAsyncioTimerHandle
{
    PyObject_HEAD
    PyAsyncioLoop * loop;
    uint64_t        callback_id;
    PyObject      * callback;  // For repr
    double          when;      // Scheduled time
    bool            cancelled;

    static PyTypeObject PyType;
};

}

#endif
