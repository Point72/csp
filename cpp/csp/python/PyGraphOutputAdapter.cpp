#include <csp/engine/DynamicEngine.h>
#include <csp/python/Conversions.h>
#include <csp/python/PyEngine.h>
#include <csp/python/PyGraphOutputAdapter.h>
#include <csp/python/Exception.h>
#include <csp/python/PyOutputAdapterWrapper.h>
#include <csp/python/NumpyConversions.h>

namespace csp::python
{

PyObjectPtr PyGraphOutputAdapter::result()
{
    if( !m_result )
        processResults();

    return m_result;
}

void PyGraphOutputAdapter::processResults()
{
    auto * ts = input();
    auto len = tickCount() > 0 ? std::min( tickCount(), ts -> numTicks() ) : ts -> numTicks();
    //NOTE we may want to limit ticks to requested history as well... TBD?

    if( static_cast<PythonEngine *>( rootEngine() ) -> outputNumpy() )
    {
        auto res = valuesAtIndexToNumpy( ValueType::TIMESTAMP_VALUE_TUPLE,
            ts, len - 1, 0, autogen::TimeIndexPolicy::INCLUSIVE, autogen::TimeIndexPolicy::INCLUSIVE );
        m_result = PyObjectPtr::check( res );
    }
    else
    {
        m_result = PyObjectPtr::check( PyList_New( len ) );
        switchCspType( ts -> type(),
                       [ this, ts, len ]( auto tag )
                       {
                           Py_ssize_t   idx   = 0;
                           for( int32_t index = len - 1; index >= 0; --index )
                           {
                               PyObjectPtr pydt    = PyObjectPtr::own( toPython( ts -> timeAtIndex( index ) ) );
                               PyObjectPtr pyvalue = PyObjectPtr::own(
                                   toPython( ts -> valueAtIndex<typename decltype(tag)::type>( index ), *ts -> type() ) );
                               PyObjectPtr entry   = PyObjectPtr::check( PyTuple_Pack( 2, pydt.ptr(), pyvalue.ptr() ) );
                               PyList_SET_ITEM( m_result.ptr(), idx++, entry.release() );
                           }
                       } );
    }
}

//csp.add_graph_output
static OutputAdapter * creator( csp::AdapterManager * manager, PyEngine * pyengine, PyObject * args )
{
    PyObject * pyKey  = nullptr;
    int        tickCount = -1;
    PyObject * pyTickHistory = nullptr;
 
    if( !PyArg_ParseTuple( args, "OiO", &pyKey, &tickCount, &pyTickHistory ) )
        CSP_THROW( PythonPassthrough, "" );

    auto * engine = pyengine -> engine();
    auto key = fromPython<DialectGenericType>( pyKey );
    auto adapter = std::make_shared<PyGraphOutputAdapter>( engine, tickCount, fromPython<TimeDelta>( pyTickHistory ) );
    engine -> registerGraphOutput( key, adapter );
    return adapter.get();
}

//for returning __outputs__ from a graph ( as opposed to csp.add_graph_output calls )
static OutputAdapter * return_creator( csp::AdapterManager * manager, PyEngine * pyengine, PyObject * args )
{
    PyObject * pyKey  = nullptr;

    if( !PyArg_ParseTuple( args, "O", &pyKey ) )
        CSP_THROW( PythonPassthrough, "" );

    auto * engine = pyengine -> engine();
    if( engine -> isRootEngine() )
    {
        PyObjectPtr argsEx = PyObjectPtr::check( Py_BuildValue( "OiN",
                                                                pyKey,
                                                                -1,
                                                                toPython( TimeDelta::NONE() ) ) );
        return creator( manager, pyengine, argsEx.get() );
    }

    //dynamic graph output
    auto * adapter = engine -> createOwnedObject<DynamicEngine::GraphOutput>();

    //for graph outputs keys should always be strings, unnamed single should be "".  For historical reasons unnamed is passed as 0
    std::string key;
    if( PyUnicode_Check( pyKey ) )
        key = fromPython<std::string>( pyKey );
    else
    {
        assert( fromPython<int64_t>( pyKey ) == 0 );
    }

    static_cast<DynamicEngine *>( engine ) -> registerGraphOutput( key, adapter );
    return adapter;
}

REGISTER_OUTPUT_ADAPTER( _graph_output_adapter, creator );
REGISTER_OUTPUT_ADAPTER( _graph_return_adapter, return_creator );

}
