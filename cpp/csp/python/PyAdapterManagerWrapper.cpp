#include <csp/engine/AdapterManager.h>
#include <csp/engine/StatusAdapter.h>
#include <csp/python/Conversions.h>
#include <csp/python/Exception.h>
#include <csp/python/PyEngine.h>
#include <csp/python/PyAdapterManagerWrapper.h>
#include <csp/python/PyInputAdapterWrapper.h>

namespace csp::python
{

PyObject * PyAdapterManagerWrapper::create( Creator creator, PyObject * args )
{
    CSP_BEGIN_METHOD;

    PyEngine * pyEngine     = nullptr;
    PyObject * pyProperties = nullptr;

    if( !PyArg_ParseTuple( args, "O!O!",
                           &PyEngine::PyType, &pyEngine, 
                           &PyDict_Type, &pyProperties ) )
        CSP_THROW( PythonPassthrough, "" );

    auto * adapterMgr = creator( pyEngine, fromPython<Dictionary>( pyProperties ) );

    return PyCapsule_New( adapterMgr, "adapterMgr", nullptr );
    CSP_RETURN_NULL;
}

csp::AdapterManager * PyAdapterManagerWrapper::extractAdapterManager( PyObject * wrapper )
{
    return ( csp::AdapterManager * ) PyCapsule_GetPointer( wrapper, "adapterMgr" );
}

static StatusAdapter * create_status_adapter( csp::AdapterManager * manager, PyEngine * pyengine, PyObject * pyType, PushMode pushMode, PyObject * args )
{
    auto & cspType = pyTypeAsCspType( pyType );
    return manager -> createStatusAdapter( cspType, pushMode );
}

REGISTER_INPUT_ADAPTER( _status_adapter, create_status_adapter );

}
