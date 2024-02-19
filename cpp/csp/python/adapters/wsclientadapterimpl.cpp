#include <csp/adapters/websocket_client/WSClientAdapterManager.h>
#include <csp/engine/PushInputAdapter.h>
#include <csp/python/Conversions.h>
#include <csp/python/Exception.h>
#include <csp/python/InitHelper.h>
#include <csp/python/PyAdapterManagerWrapper.h>
#include <csp/python/PyEngine.h>
#include <csp/python/PyInputAdapterWrapper.h>
#include <csp/python/PyOutputAdapterWrapper.h>

using namespace csp::adapters::wsclient;

namespace csp::python
{

//AdapterManager
csp::AdapterManager * create_wsclient_adapter_manager( PyEngine * engine, const Dictionary & properties )
{
    return engine -> engine() -> createOwnedObject<WSClientAdapterManager>( properties );
}

static InputAdapter * create_wsclient_input_adapter( csp::AdapterManager * manager, PyEngine * pyengine, PyObject * pyType, PushMode pushMode, PyObject * args )
{
    auto & cspType = pyTypeAsCspType( pyType );

    PyObject * pyProperties;
    PyObject * type;

    auto * wsclientManager = dynamic_cast<WSClientAdapterManager*>( manager );
    if( !wsclientManager )
        CSP_THROW( TypeError, "Expected WSClientAdapterManager" );

    if( !PyArg_ParseTuple( args, "O!O!",
                           &PyType_Type, &type,
                           &PyDict_Type, &pyProperties ) )
        CSP_THROW( PythonPassthrough, "" );

    return wsclientManager -> getInputAdapter( cspType, pushMode, fromPython<Dictionary>( pyProperties ) );
}

static OutputAdapter * create_wsclient_output_adapter( csp::AdapterManager * manager, PyEngine * pyengine, PyObject * args )
{
    // PyObject * pyProperties;
    // PyObject * pyType;

    auto * wsclientManager = dynamic_cast<WSClientAdapterManager*>( manager );
    if( !wsclientManager )
        CSP_THROW( TypeError, "Expected WSClientAdapterManager" );

    // if( !PyArg_ParseTuple( args, "OO!",
    //                        &pyType,
    //                        &PyDict_Type, &pyProperties ) )
    //     CSP_THROW( PythonPassthrough, "" );

    // auto & cspType = pyTypeAsCspType( pyType );

    return wsclientManager -> getOutputAdapter();
}

REGISTER_ADAPTER_MANAGER( _wsclient_adapter_manager, create_wsclient_adapter_manager );
REGISTER_INPUT_ADAPTER(   _wsclient_input_adapter,   create_wsclient_input_adapter );
REGISTER_OUTPUT_ADAPTER(  _wsclient_output_adapter,  create_wsclient_output_adapter );

static PyModuleDef _wsclientadapterimpl_module = {
        PyModuleDef_HEAD_INIT,
        "_wsclientadapterimpl",
        "_wsclientadapterimpl c++ module",
        -1,
        NULL, NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC PyInit__wsclientadapterimpl(void)
{
    PyObject* m;

    m = PyModule_Create( &_wsclientadapterimpl_module);
    if( m == NULL )
        return NULL;

    if( !InitHelper::instance().execute( m ) )
        return NULL;

    return m;
}

}
