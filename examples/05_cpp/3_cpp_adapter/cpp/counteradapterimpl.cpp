/**
 * counteradapterimpl.cpp - Python bindings for the Counter adapter example
 *
 * This file demonstrates how to expose C++ adapters to Python using CSP's
 * registration macros. It provides the glue between Python and the C++
 * CounterAdapterManager, CounterInputAdapter, and CounterOutputAdapter.
 */

#include "CounterAdapterManager.h"
#include <csp/engine/PushInputAdapter.h>
#include <csp/python/Conversions.h>
#include <csp/python/Exception.h>
#include <csp/python/InitHelper.h>
#include <csp/python/PyAdapterManagerWrapper.h>
#include <csp/python/PyEngine.h>
#include <csp/python/PyInputAdapterWrapper.h>
#include <csp/python/PyOutputAdapterWrapper.h>

using namespace csp::adapters::counter;

namespace csp::python
{

/**
 * Factory function for creating the CounterAdapterManager.
 * Called by the Python adapter manager wrapper when the graph starts.
 *
 * @param engine The PyEngine wrapper
 * @param properties Dictionary of configuration properties
 * @return The created AdapterManager
 */
csp::AdapterManager * create_counter_adapter_manager( PyEngine * engine, const Dictionary & properties )
{
    return engine -> engine() -> createOwnedObject<CounterAdapterManager>( properties );
}

/**
 * Factory function for creating the CounterInputAdapter.
 * Called when the Python code creates an input adapter instance.
 *
 * @param manager The parent AdapterManager
 * @param pyengine The PyEngine wrapper
 * @param pyType The CSP type of the adapter output
 * @param pushMode Push mode (LAST_VALUE, NON_COLLAPSING, etc.)
 * @param args Python tuple of additional arguments
 * @return The created InputAdapter
 */
static InputAdapter * create_counter_input_adapter( csp::AdapterManager * manager, PyEngine * pyengine,
                                                    PyObject * pyType, PushMode pushMode, PyObject * args )
{
    auto & cspType = pyTypeAsCspType( pyType );

    PyObject * pyProperties;
    PyObject * type;

    auto * counterManager = dynamic_cast<CounterAdapterManager *>( manager );
    if( !counterManager )
        CSP_THROW( TypeError, "Expected CounterAdapterManager" );

    if( !PyArg_ParseTuple( args, "O!O!",
                           &PyType_Type, &type,
                           &PyDict_Type, &pyProperties ) )
        CSP_THROW( PythonPassthrough, "" );

    return counterManager -> getInputAdapter( cspType, pushMode, fromPython<Dictionary>( pyProperties ) );
}

/**
 * Factory function for creating the CounterOutputAdapter.
 * Called when the Python code creates an output adapter instance.
 *
 * @param manager The parent AdapterManager
 * @param pyengine The PyEngine wrapper
 * @param args Python tuple of additional arguments
 * @return The created OutputAdapter
 */
static OutputAdapter * create_counter_output_adapter( csp::AdapterManager * manager, PyEngine * pyengine, PyObject * args )
{
    PyObject * pyProperties;
    PyObject * pyType;

    auto * counterManager = dynamic_cast<CounterAdapterManager *>( manager );
    if( !counterManager )
        CSP_THROW( TypeError, "Expected CounterAdapterManager" );

    if( !PyArg_ParseTuple( args, "OO!",
                           &pyType,
                           &PyDict_Type, &pyProperties ) )
        CSP_THROW( PythonPassthrough, "" );

    auto & cspType = pyTypeAsCspType( pyType );

    return counterManager -> getOutputAdapter( cspType, fromPython<Dictionary>( pyProperties ) );
}

// Register the adapter manager and adapters with CSP
// These macros make the C++ code available to Python
REGISTER_ADAPTER_MANAGER( _counter_adapter_manager, create_counter_adapter_manager );
REGISTER_INPUT_ADAPTER(   _counter_input_adapter,   create_counter_input_adapter );
REGISTER_OUTPUT_ADAPTER(  _counter_output_adapter,  create_counter_output_adapter );

// Python module definition
static PyModuleDef _counteradapterimpl_module = {
    PyModuleDef_HEAD_INIT,
    "_counteradapter",                       // Module name
    "Counter adapter example C++ module",   // Module docstring
    -1,
    NULL, NULL, NULL, NULL, NULL
};

// Module initialization function
PyMODINIT_FUNC PyInit__counteradapter(void)
{
    PyObject* m;

    m = PyModule_Create( &_counteradapterimpl_module );
    if( m == NULL )
        return NULL;

    // Execute all registered initializers (adapter manager, input/output adapters)
    if( !InitHelper::instance().execute( m ) )
        return NULL;

    return m;
}

}
