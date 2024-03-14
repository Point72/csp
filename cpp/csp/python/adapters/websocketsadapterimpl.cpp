#include <csp/adapters/websockets/ClientAdapterManager.h>
#include <csp/engine/PushInputAdapter.h>
#include <csp/python/Conversions.h>
#include <csp/python/Exception.h>
#include <csp/python/InitHelper.h>
#include <csp/python/PyAdapterManagerWrapper.h>
#include <csp/python/PyEngine.h>
#include <csp/python/PyInputAdapterWrapper.h>
#include <csp/python/PyOutputAdapterWrapper.h>

using namespace csp::adapters::websockets;

namespace csp::python
{

using namespace csp;
using namespace csp::python;

//AdapterManager
csp::AdapterManager * create_websockets_adapter_manager( PyEngine * engine, const Dictionary & properties )
{
    return engine -> engine() -> createOwnedObject<ClientAdapterManager>( properties );
}

static InputAdapter * create_websockets_input_adapter( csp::AdapterManager * manager, PyEngine * pyengine, PyObject * pyType, PushMode pushMode, PyObject * args )
{
    auto & cspType = pyTypeAsCspType( pyType );

    PyObject * pyProperties;
    PyObject * type;
    auto * websocketsManager = dynamic_cast<ClientAdapterManager*>( manager );
    if( !websocketsManager )
        CSP_THROW( TypeError, "Expected ClientAdapterManager" );

    if( !PyArg_ParseTuple( args, "O!O!",
                           &PyType_Type, &type,
                           &PyDict_Type, &pyProperties ) )
        CSP_THROW( PythonPassthrough, "" );

    return websocketsManager -> getInputAdapter( cspType, pushMode, fromPython<Dictionary>( pyProperties ) );
}

static OutputAdapter * create_websockets_output_adapter( csp::AdapterManager * manager, PyEngine * pyengine, PyObject * args )
{
    auto * websocketsManager = dynamic_cast<ClientAdapterManager*>( manager );
    if( !websocketsManager )
        CSP_THROW( TypeError, "Expected ClientAdapterManager" );
    return websocketsManager -> getOutputAdapter();
}

static OutputAdapter * create_websockets_header_update_adapter( csp::AdapterManager * manager, PyEngine * pyengine, PyObject * args )
{
    auto * websocketsManager = dynamic_cast<ClientAdapterManager*>( manager );
    if( !websocketsManager )
        CSP_THROW( TypeError, "Expected ClientAdapterManager" );
    return websocketsManager -> getHeaderUpdateAdapter();
}

REGISTER_ADAPTER_MANAGER( _websockets_adapter_manager, create_websockets_adapter_manager );
REGISTER_INPUT_ADAPTER(   _websockets_input_adapter,   create_websockets_input_adapter );
REGISTER_OUTPUT_ADAPTER(  _websockets_output_adapter,  create_websockets_output_adapter );
REGISTER_OUTPUT_ADAPTER(  _websockets_header_update_adapter,  create_websockets_header_update_adapter);

static PyModuleDef _websocketsadapterimpl_module = {
        PyModuleDef_HEAD_INIT,
        "_websocketsadapterimpl",
        "_websocketsadapterimpl c++ module",
        -1,
        NULL, NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC PyInit__websocketsadapterimpl(void)
{
    PyObject* m;

    m = PyModule_Create( &_websocketsadapterimpl_module);
    if( m == NULL )
        return NULL;

    if( !InitHelper::instance().execute( m ) )
        return NULL;

    return m;
}

}
