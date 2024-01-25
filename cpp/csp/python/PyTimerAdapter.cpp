#include <csp/engine/TimerInputAdapter.h>
#include <csp/python/Conversions.h>
#include <csp/python/Exception.h>
#include <csp/python/PyEngine.h>
#include <csp/python/PyInputAdapterWrapper.h>

namespace csp::python
{

static InputAdapter * timer_creator( csp::AdapterManager * manager, PyEngine * pyengine, PyObject * pyType, PushMode pushMode, PyObject * args )
{
    PyObject * pyInterval = nullptr;
    PyObject * pyValue    = nullptr;
    int        allowDeviation;

    if( !PyArg_ParseTuple( args, "OOp", &pyInterval, &pyValue, &allowDeviation ) )
        CSP_THROW( PythonPassthrough, "" );

    auto interval = fromPython<TimeDelta>( pyInterval );

    auto cspType = pyTypeAsCspType( pyType );

    return switchCspType( cspType,
                          [ engine = pyengine -> engine(), &cspType, interval, pyValue, allowDeviation ](
                                  auto tag ) -> InputAdapter *
                          {
                              using T = typename decltype(tag)::type;
                              return engine -> createOwnedObject<TimerInputAdapter<T>>(
                                      cspType, interval, fromPython<T>( pyValue, *cspType ), bool( allowDeviation ) );
                          } );
}

REGISTER_INPUT_ADAPTER( _timer, timer_creator );

}
