#include <csp/adapters/websocket/ClientAdapterManager.h>
#include <csp/engine/PushInputAdapter.h>
#include <csp/python/Conversions.h>
#include <csp/python/Exception.h>
#include <csp/python/InitHelper.h>
#include <csp/python/PyAdapterManagerWrapper.h>
#include <csp/python/PyEngine.h>
#include <csp/python/PyInputAdapterWrapper.h>
#include <csp/python/PyOutputAdapterWrapper.h>

using namespace csp::adapters::websocket;

namespace csp::python
{

using namespace csp;
using namespace csp::python;

//AdapterManager
csp::AdapterManager * create_websocket_adapter_manager( PyEngine * engine, const Dictionary & properties )
{
    return engine -> engine() -> createOwnedObject<ClientAdapterManager>( properties );
}

static InputAdapter * create_websocket_input_adapter( csp::AdapterManager * manager, PyEngine * pyengine, PyObject * pyType, PushMode pushMode, PyObject * args )
{
    auto & cspType = pyTypeAsCspType( pyType );

    PyObject * pyProperties;
    PyObject * type;
    auto * websocketManager = dynamic_cast<ClientAdapterManager*>( manager );
    if( !websocketManager )
        CSP_THROW( TypeError, "Expected WebsocketClientAdapterManager" );

    if( !PyArg_ParseTuple( args, "O!O!",
                           &PyType_Type, &type,
                           &PyDict_Type, &pyProperties ) )
        CSP_THROW( PythonPassthrough, "" );

    return websocketManager -> getInputAdapter( cspType, pushMode, fromPython<Dictionary>( pyProperties ) );
}

static OutputAdapter * create_websocket_output_adapter( csp::AdapterManager * manager, PyEngine * pyengine, PyObject * args )
{
    auto * websocketManager = dynamic_cast<ClientAdapterManager*>( manager );
    if( !websocketManager )
        CSP_THROW( TypeError, "Expected WebsocketClientAdapterManager" );
    return websocketManager -> getOutputAdapter();
}

static OutputAdapter * create_websocket_header_update_adapter( csp::AdapterManager * manager, PyEngine * pyengine, PyObject * args )
{
    auto * websocketManager = dynamic_cast<ClientAdapterManager*>( manager );
    if( !websocketManager )
        CSP_THROW( TypeError, "Expected WebsocketClientAdapterManager" );
    return websocketManager -> getHeaderUpdateAdapter();
}

REGISTER_ADAPTER_MANAGER( _websocket_adapter_manager, create_websocket_adapter_manager );
REGISTER_INPUT_ADAPTER(   _websocket_input_adapter,   create_websocket_input_adapter );
REGISTER_OUTPUT_ADAPTER(  _websocket_output_adapter,  create_websocket_output_adapter );
REGISTER_OUTPUT_ADAPTER(  _websocket_header_update_adapter,  create_websocket_header_update_adapter);

static PyModuleDef _websocketadapterimpl_module = {
        PyModuleDef_HEAD_INIT,
        "_websocketadapterimpl",
        "_websocketadapterimpl c++ module",
        -1,
        NULL, NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC PyInit__websocketadapterimpl(void)
{
    PyObject* m;

    m = PyModule_Create( &_websocketadapterimpl_module);
    if( m == NULL )
        return NULL;

    if( !InitHelper::instance().execute( m ) )
        return NULL;

    return m;
}

}
