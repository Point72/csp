#include <csp/python/adapters/ArrowInputAdapter.h>
#include <csp/python/Conversions.h>
#include <csp/python/Exception.h>
#include <csp/python/PyEngine.h>
#include <csp/python/PyInputAdapterWrapper.h>
#include <Python.h>

namespace csp::python
{

static InputAdapter * record_batch_input_adapter_creator( csp::AdapterManager * manager, PyEngine * pyengine, PyObject * pyType, PushMode pushMode, PyObject * args )
{
    PyObject * pyTsColName = nullptr;
    PyObject * pySource = nullptr;
    PyObject * pySchema = nullptr;
    int expectSmallBatches;

    if( !PyArg_ParseTuple( args, "OOOp", &pyTsColName, &pySource, &pySchema, &expectSmallBatches ) )
        CSP_THROW( PythonPassthrough, "" );

    if( !PyIter_Check( pySource ) )
        CSP_THROW( ValueError, "Source is not a valid iterator/generator of record batches" );

    auto tsColName = fromPython<csp::CspType::StringCType>( pyTsColName );
    auto cspType = pyTypeAsCspType( pyType );
    auto source = PyObjectPtr::incref( pySource );
    auto schema = PyObjectPtr::incref( pySchema );

    return pyengine -> engine() -> createOwnedObject<csp::python::arrow::RecordBatchInputAdapter>( cspType, std::move( schema ), std::move( tsColName ), std::move( source ), expectSmallBatches != 0 );
}

REGISTER_INPUT_ADAPTER( _record_batch_input_adapter_creator, record_batch_input_adapter_creator );

static PyModuleDef _arrowadapterimpl_module = {
        PyModuleDef_HEAD_INIT,
        "_arrowadapterimpl",
        "_arrowadapterimpl c++ module",
        -1,
        NULL, NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC PyInit__arrowadapterimpl( void )
{
    PyObject *m;

    m = PyModule_Create( &_arrowadapterimpl_module );
    if( m == NULL )
    {
        return NULL;
    }

    if( !InitHelper::instance().execute( m ) )
    {
        return NULL;
    }

    return m;
}

}
