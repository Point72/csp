#include <csp/engine/PullInputAdapter.h>
#include <csp/python/Conversions.h>
#include <csp/python/Exception.h>
#include <csp/python/PyInputAdapterWrapper.h>
#include <csp/python/PyEngine.h>
#include <csp/python/PyObjectPtr.h>

namespace csp::python
{

template<typename T>
class PyPullInputAdapter : public PullInputAdapter<T>
{
public:
    PyPullInputAdapter( Engine * engine, PyObjectPtr pyadapter, 
                        PyObject * pyType, PushMode pushMode ) : PullInputAdapter<T>( engine, pyTypeAsCspType( pyType ), pushMode ),
                                                                 m_pyadapter( pyadapter ),
                                                                 m_pyType( PyObjectPtr::incref( ( PyObject * ) pyType ) )
    {
    }

    void start( DateTime start, DateTime end ) override
    {
        PyObjectPtr rv = PyObjectPtr::own( PyObject_CallMethod( m_pyadapter.ptr(), "start", "OO", 
                                                                PyObjectPtr::own( toPython( start ) ).ptr(),
                                                                PyObjectPtr::own( toPython( end ) ).ptr() ) );
        if( !rv.ptr() )
            CSP_THROW( PythonPassthrough, "" );

        PullInputAdapter<T>::start( start, end );
    }

    void stopAdapter() override
    {
        PyObjectPtr rv = PyObjectPtr::own( PyObject_CallMethod( m_pyadapter.ptr(), "stop", nullptr ) ); 

        if( !rv.ptr() )
            CSP_THROW( PythonPassthrough, "" );
    }

    bool next( DateTime & time, T & value ) override
    {
        PyObjectPtr rv = PyObjectPtr::own( PyObject_CallMethod( m_pyadapter.ptr(), "next", nullptr ) ); 

        if( !rv.ptr() )
        {
            if( PyErr_Occurred() == PyExc_KeyboardInterrupt )
            {
                this -> rootEngine() -> shutdown();
                return false;
            }
            CSP_THROW( PythonPassthrough, "" );
        }
        
        if( rv.ptr() == Py_None )
            return false;

        if( !PyTuple_Check( rv.ptr() ) || PyTuple_GET_SIZE( rv.ptr() ) != 2 )
            CSP_THROW( TypeError, "PyPullInputAdapter::next expects None or ( datetime, value ), got " << rv );

        time  = fromPython<DateTime>( PyTuple_GET_ITEM( rv.ptr(), 0 ) );
        PyObject * pyValue = PyTuple_GET_ITEM( rv.ptr(), 1 );
        try
        {
            if( !validatePyType( this -> dataType(), m_pyType.ptr(), pyValue ) )
                CSP_THROW( TypeError, "" );
            value = fromPython<T>( pyValue, *this -> dataType() );
        }
        catch( TypeError & e )
        {
            CSP_THROW( TypeError, "\"" << Py_TYPE( m_pyadapter.ptr() ) -> tp_name << "\" pull adapter expected output type to be of type \"" 
                       << pyTypeToString( m_pyType.ptr() ) << "\" got type \"" << Py_TYPE( pyValue ) -> tp_name << "\"" );

        }
        return true;

    }

private:
    PyObjectPtr m_pyadapter;
    PyObjectPtr m_pyType;
};

static InputAdapter * pypulladapter_creator( csp::AdapterManager * manager, PyEngine * pyengine, 
                                             PyObject * pyType, PushMode pushMode, PyObject * args )
{
    PyTypeObject * pyAdapterType = nullptr;
    PyObject * adapterArgs = nullptr;
    if( !PyArg_ParseTuple( args, "O!O!", &PyType_Type, &pyAdapterType,
                           &PyTuple_Type, &adapterArgs ) )
        CSP_THROW( PythonPassthrough, "" );

    PyObject * pyAdapter = PyObject_Call( ( PyObject * ) pyAdapterType, adapterArgs, nullptr );
    if( !pyAdapter )
        CSP_THROW( PythonPassthrough, "" );

    return switchPyType( pyType, 
                         [engine = pyengine -> engine(), pyAdapter,pyType,pushMode] ( auto tag ) -> InputAdapter *
                         { 
                             using T = typename decltype(tag)::type;
                             return engine -> createOwnedObject<PyPullInputAdapter<T>>( PyObjectPtr::own( pyAdapter ), pyType, pushMode ); 
                         } );
}


// NOTE: no python object is exported as its not needed
// at this time.

REGISTER_INPUT_ADAPTER( _pulladapter, pypulladapter_creator );

}
