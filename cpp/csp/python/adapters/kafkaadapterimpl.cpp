#include <csp/adapters/kafka/KafkaAdapterManager.h>
#include <csp/engine/PushInputAdapter.h>
#include <csp/python/Conversions.h>
#include <csp/python/Exception.h>
#include <csp/python/InitHelper.h>
#include <csp/python/PyAdapterManagerWrapper.h>
#include <csp/python/PyEngine.h>
#include <csp/python/PyInputAdapterWrapper.h>
#include <csp/python/PyOutputAdapterWrapper.h>

using namespace csp::adapters::kafka;

namespace csp::python
{

//AdapterManager
csp::AdapterManager * create_kafka_adapter_manager( PyEngine * engine, const Dictionary & properties )
{
    return engine -> engine() -> createOwnedObject<KafkaAdapterManager>( properties );
}

static InputAdapter * create_kafka_input_adapter( csp::AdapterManager * manager, PyEngine * pyengine, PyObject * pyType, PushMode pushMode, PyObject * args )
{
    auto & cspType = pyTypeAsCspType( pyType );

    PyObject * pyProperties;
    PyObject * type;

    auto * kafkaManager = dynamic_cast<KafkaAdapterManager*>( manager );
    if( !kafkaManager )
        CSP_THROW( TypeError, "Expected KafkaAdapterManager" );

    if( !PyArg_ParseTuple( args, "O!O!",
                           &PyType_Type, &type,
                           &PyDict_Type, &pyProperties ) )
        CSP_THROW( PythonPassthrough, "" );

    return kafkaManager -> getInputAdapter( cspType, pushMode, fromPython<Dictionary>( pyProperties ) );
}

static OutputAdapter * create_kafka_output_adapter( csp::AdapterManager * manager, PyEngine * pyengine, PyObject * args )
{
    PyObject * pyProperties;
    PyObject * pyType;

    auto * kafkaManager = dynamic_cast<KafkaAdapterManager*>( manager );
    if( !kafkaManager )
        CSP_THROW( TypeError, "Expected KafkaAdapterManager" );

    if( !PyArg_ParseTuple( args, "OO!",
                           &pyType,
                           &PyDict_Type, &pyProperties ) )
        CSP_THROW( PythonPassthrough, "" );

    auto & cspType = pyTypeAsCspType( pyType );

    return kafkaManager -> getOutputAdapter( cspType, fromPython<Dictionary>( pyProperties ) );
}

REGISTER_ADAPTER_MANAGER( _kafka_adapter_manager, create_kafka_adapter_manager );
REGISTER_INPUT_ADAPTER(   _kafka_input_adapter,   create_kafka_input_adapter );
REGISTER_OUTPUT_ADAPTER(  _kafka_output_adapter,  create_kafka_output_adapter );

static PyModuleDef _kafkaadapterimpl_module = {
        PyModuleDef_HEAD_INIT,
        "_kafkaadapterimpl",
        "_kafkaadapterimpl c++ module",
        -1,
        NULL, NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC PyInit__kafkaadapterimpl(void)
{
    PyObject* m;

    m = PyModule_Create( &_kafkaadapterimpl_module);
    if( m == NULL )
        return NULL;

    if( !InitHelper::instance().execute( m ) )
        return NULL;

    return m;
}

}
