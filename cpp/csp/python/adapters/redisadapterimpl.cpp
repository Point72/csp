#include <csp/adapters/redis/RedisAdapterManager.h>
#include <csp/engine/PushInputAdapter.h>
#include <csp/python/Conversions.h>
#include <csp/python/Exception.h>
#include <csp/python/InitHelper.h>
#include <csp/python/PyAdapterManagerWrapper.h>
#include <csp/python/PyEngine.h>
#include <csp/python/PyInputAdapterWrapper.h>
#include <csp/python/PyOutputAdapterWrapper.h>

using namespace csp::adapters::redis;

namespace csp::python
{

//AdapterManager
csp::AdapterManager * create_redis_adapter_manager( PyEngine * engine, const Dictionary & properties )
{
    return engine -> engine() -> createOwnedObject<RedisAdapterManager>( properties );
}

static InputAdapter * create_redis_input_adapter( csp::AdapterManager * manager, PyEngine * pyengine, PyObject * pyType, PushMode pushMode, PyObject * args )
{
    auto & cspType = pyTypeAsCspType( pyType );

    PyObject * pyProperties;
    PyObject * type;

    auto * redisManager = dynamic_cast<RedisAdapterManager*>( manager );
    if( !redisManager )
        CSP_THROW( TypeError, "Expected RedisAdapterManager" );

    if( !PyArg_ParseTuple( args, "O!O!",
                           &PyType_Type, &type,
                           &PyDict_Type, &pyProperties ) )
        CSP_THROW( PythonPassthrough, "" );

    return redisManager -> getInputAdapter( cspType, pushMode, fromPython<Dictionary>( pyProperties ) );
}

static OutputAdapter * create_redis_output_adapter( csp::AdapterManager * manager, PyEngine * pyengine, PyObject * args )
{
    PyObject * pyProperties;
    PyObject * pyType;

    auto * redisManager = dynamic_cast<RedisAdapterManager*>( manager );
    if( !redisManager )
        CSP_THROW( TypeError, "Expected RedisAdapterManager" );

    if( !PyArg_ParseTuple( args, "OO!",
                           &pyType,
                           &PyDict_Type, &pyProperties ) )
        CSP_THROW( PythonPassthrough, "" );

    auto & cspType = pyTypeAsCspType( pyType );

    return redisManager -> getOutputAdapter( cspType, fromPython<Dictionary>( pyProperties ) );
}

REGISTER_ADAPTER_MANAGER( _redis_adapter_manager, create_redis_adapter_manager );
REGISTER_INPUT_ADAPTER(   _redis_input_adapter,   create_redis_input_adapter );
REGISTER_OUTPUT_ADAPTER(  _redis_output_adapter,  create_redis_output_adapter );

static PyModuleDef _redisadapterimpl_module = {
        PyModuleDef_HEAD_INIT,
        "_redisadapterimpl",
        "_redisadapterimpl c++ module",
        -1,
        NULL, NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC PyInit__redisadapterimpl(void)
{
    PyObject* m;

    m = PyModule_Create( &_redisadapterimpl_module);
    if( m == NULL )
        return NULL;

    if( !InitHelper::instance().execute( m ) )
        return NULL;

    return m;
}

}
