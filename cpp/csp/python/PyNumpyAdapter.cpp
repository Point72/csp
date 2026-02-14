//this is included first so that we do include without NO_IMPORT_ARRAY defined, which is done in NumpyInputAdapter.h
#include <numpy/ndarrayobject.h>
//
// BUT!  really we can't have this here and in NumpyConversions.cpp....
//

#include <csp/python/Conversions.h>
#include <csp/python/Exception.h>
#include <csp/python/NumpyInputAdapter.h>
#include <csp/python/PyEngine.h>
#include <csp/python/PyInputAdapterWrapper.h>

static void * init_nparray()
{
    csp::python::AcquireGIL gil;
    import_array();
    return nullptr;
}
static void * s_init_array;

namespace csp::python
{

static InputAdapter * numpy_adapter_creator( csp::AdapterManager * manager, PyEngine * pyengine, PyObject * pyType, PushMode pushMode, PyObject * args )
{
    s_init_array = init_nparray();

    PyObject * type;
    PyArrayObject * pyDatetimes = nullptr;
    PyArrayObject * pyValues    = nullptr;

    if( !PyArg_ParseTuple( args, "OO!O!",
                           &type,
                           &PyArray_Type, &pyDatetimes,
                           &PyArray_Type, &pyValues ) )
        CSP_THROW( PythonPassthrough, "" );

    auto cspType = pyTypeAsCspType( pyType );

    return switchCspType( cspType,
                          [ engine = pyengine -> engine(), &cspType, pyDatetimes, pyValues ](
                                  auto tag ) -> InputAdapter *
                          {
                              using T = typename decltype(tag)::type;
                              return engine -> createOwnedObject<NumpyInputAdapter<T>>(
                                      cspType, pyDatetimes, pyValues );
                          } );
}

REGISTER_INPUT_ADAPTER( _npcurve, numpy_adapter_creator );

}
