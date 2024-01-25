#include <csp/core/Time.h>
#include <csp/engine/OutputAdapter.h>
#include <csp/python/Conversions.h>
#include <csp/python/Exception.h>
#include <csp/python/PyEngine.h>
#include <csp/python/PyInputAdapterWrapper.h>
#include <csp/python/PyObjectPtr.h>
#include <csp/python/PyOutputAdapterWrapper.h>
#include <csp/python/PyNodeWrapper.h>

namespace csp::python
{

class PyOutputAdapter : public OutputAdapter
{
public:
    PyOutputAdapter(
        Engine * engine,
        PyObjectPtr pyadapter
    )
        : OutputAdapter( engine ),
          m_pyadapter( pyadapter )
    {}

    void executeImpl() override;
    void start() override;
    void stop() override;
    
    const char * name() const override { return "PyOutputAdapter"; }
private:
    PyObjectPtr m_pyadapter;
};

void PyOutputAdapter::executeImpl()
{
    const TimeSeriesProvider *inputs = input();

    PyObject* last_time = toPython(inputs->lastTime());
    PyObject* last_value = lastValueToPython(inputs);

    PyObjectPtr rv = PyObjectPtr::own(
        PyObject_CallMethod(
            m_pyadapter.ptr(),
            "on_tick",
            "OO",
            PyObjectPtr::own(last_time).ptr(),
            PyObjectPtr::own(last_value).ptr()
        )
    );
    if( !rv.ptr() )
        CSP_THROW( PythonPassthrough, "" );
}

void PyOutputAdapter::start()
{
    PyObjectPtr rv = PyObjectPtr::own( PyObject_CallMethod( m_pyadapter.ptr(), "start", nullptr ) );
    if( !rv.ptr() )
        CSP_THROW( PythonPassthrough, "" );
}

void PyOutputAdapter::stop()
{
    PyObjectPtr rv = PyObjectPtr::own( PyObject_CallMethod( m_pyadapter.ptr(), "stop", nullptr ) ); 
    if( !rv.ptr() )
        CSP_THROW( PythonPassthrough, "" );
}

static OutputAdapter * pyoutputadapter_creator( csp::AdapterManager * manager, PyEngine * pyengine, PyObject * args )
{
    PyTypeObject * pyAdapterType = nullptr;
    PyObject * adapterArgs = nullptr;
 
     if( !PyArg_ParseTuple( args, "O!O!", &PyType_Type, &pyAdapterType,
                           &PyTuple_Type, &adapterArgs ) )
        CSP_THROW( PythonPassthrough, "" );

    PyObject * pyAdapter = PyObject_Call( ( PyObject * ) pyAdapterType, adapterArgs, nullptr );
    if( !pyAdapter )
        CSP_THROW( PythonPassthrough, "" );

    return pyengine -> engine() -> createOwnedObject<PyOutputAdapter>( PyObjectPtr::own( pyAdapter ) );
}

// NOTE: no python object is exported as its not needed
// at this time.

REGISTER_OUTPUT_ADAPTER( _outputadapter, pyoutputadapter_creator );

}
