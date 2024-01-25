#include <csp/engine/ConstInputAdapter.h>
#include <csp/python/Conversions.h>
#include <csp/python/PyEngine.h>
#include <csp/python/Exception.h>
#include <csp/python/PyInputAdapterWrapper.h>

namespace csp::python
{

static InputAdapter * const_creator( csp::AdapterManager * manager, PyEngine * engine, 
                                     PyObject * pyType, PushMode pushMode, PyObject * args )
{
    PyObject * pyValue  = nullptr;
    PyObject * pyDelay  = nullptr;

    if( !PyArg_ParseTuple( args, "OO", 
                           &pyValue, 
                           &pyDelay ) )
        CSP_THROW( PythonPassthrough, "" );

    auto delay = fromPython<TimeDelta>( pyDelay );

    auto & cspType = pyTypeAsCspType( pyType );

    InputAdapter *adapter = switchCspType( cspType,
                                           [ engine, &cspType, pyValue, delay ]( auto tag ) -> InputAdapter *
                                           {
                                               using T = typename decltype(tag)::type;
                                               return engine -> engine() -> createOwnedObject<ConstInputAdapter<T>>( cspType,
                                                                                                                     fromPython<T>(
                                                                                                                             pyValue,
                                                                                                                             *cspType ),
                                                                                                                     delay );
                                           } );
    return adapter;
}

REGISTER_INPUT_ADAPTER( _const, const_creator );

}
