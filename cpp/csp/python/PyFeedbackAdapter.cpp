#include <csp/engine/Feedback.h>
#include <csp/python/Conversions.h>
#include <csp/python/PyEngine.h>
#include <csp/python/PyInputAdapterWrapper.h>
#include <csp/python/PyOutputAdapterWrapper.h>

namespace csp::python
{

static OutputAdapter * output_creator( csp::AdapterManager * manager, PyEngine * pyengine, PyObject * args )
{
    PyObject * pyType;
    PyInputAdapterWrapper * pyFeedbackInput = nullptr;
 
    if( !PyArg_ParseTuple( args, "OO!", 
                           &pyType,
                           &PyInputAdapterWrapper::PyType, &pyFeedbackInput ) )
        CSP_THROW( PythonPassthrough, "" );

    auto & cspType = pyTypeAsCspType( pyType );
    return switchCspType( cspType,
                          [ pyengine, pyFeedbackInput ]( auto tag ) -> OutputAdapter *
                          {
                              using T = typename decltype(tag)::type;
                              return pyengine -> engine()
                                              -> createOwnedObject<FeedbackOutputAdapter<T>>( pyFeedbackInput -> adapter() );
                          } );
}

static InputAdapter * input_creator( csp::AdapterManager * manager, PyEngine * engine, 
                                     PyObject * pyType, PushMode pushMode, PyObject * args )
{
    auto & cspType = pyTypeAsCspType( pyType );
    return switchCspType( cspType.get(),
                          [ engine, &cspType, pushMode ]( auto tag ) -> InputAdapter *
                          {
                              using T = typename decltype(tag)::type;
                              return engine -> engine() -> createOwnedObject<FeedbackInputAdapter<T>>( cspType, pushMode );
                          } );
}

REGISTER_INPUT_ADAPTER( _feedback_input_adapter, input_creator );
REGISTER_OUTPUT_ADAPTER( _feedback_output_adapter, output_creator );

}
