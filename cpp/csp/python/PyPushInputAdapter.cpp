#include <csp/engine/PushInputAdapter.h>
#include <csp/python/Conversions.h>
#include <csp/python/Exception.h>
#include <csp/python/PyInputAdapterWrapper.h>
#include <csp/python/PyPushInputAdapter.h>
#include <csp/python/PyEngine.h>
#include <csp/python/PyObjectPtr.h>

namespace csp::python
{

static int PushBatch_init( PyPushBatch * self, PyObject * args, PyObject * kwargs )
{
    CSP_BEGIN_METHOD;
    if( PyTuple_GET_SIZE( args ) != 1 )
        CSP_THROW( TypeError, "PushBatch expected engine as single positional argument" );

    PyObject * pyengine = PyTuple_GET_ITEM( args, 0 );
    if( pyengine -> ob_type != &PyEngine::PyType )
        CSP_THROW( TypeError, "PushBatch expected engine as single positional argument" );

    csp::RootEngine * engine = ( ( PyEngine * ) pyengine ) -> rootEngine();
    new( &self -> batch ) PushBatch( engine );
    CSP_RETURN_INT;
}

static void PushBatch_dealloc( PyPushBatch * self )
{
    CSP_BEGIN_METHOD;
    self -> batch.~PushBatch();
    Py_TYPE( self ) -> tp_free( self ); 
    CSP_RETURN;
}

static PyObject * PushBatch_enter( PyObject * self )
{
    Py_INCREF( self );
    return self;
}

static PyObject * PushBatch_exit( PyPushBatch * self, PyObject * args )
{
    PyObject * exception = PyTuple_GET_ITEM( args, 0 );
    if( exception != Py_None )
        self -> batch.clear();

    //We flush here since we want to flush on exit from scope, not when object is destroyed ( which can outlast the scope! )
    self -> batch.flush();
    Py_RETURN_NONE;       
}

static PyMethodDef PushBatch_methods[] = {
    { "__enter__", (PyCFunction) PushBatch_enter, METH_NOARGS,  "enter context" },
    { "__exit__",  (PyCFunction) PushBatch_exit,  METH_VARARGS, "exit context" },
    {NULL}
};

PyTypeObject PyPushBatch::PyType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "_cspimpl.PushBatch",      /* tp_name */
    sizeof(PyPushBatch),       /* tp_basicsize */
    0,                         /* tp_itemsize */
    ( destructor ) PushBatch_dealloc, /* tp_dealloc */
    0,                         /* tp_print */
    0,                         /* tp_getattr */
    0,                         /* tp_setattr */
    0,                         /* tp_reserved */
    0,                         /* tp_repr */
    0,                         /* tp_as_number */
    0,                         /* tp_as_sequence */
    0,                         /* tp_as_mapping */
    0,                         /* tp_hash  */
    0,                         /* tp_call */
    0,                         /* tp_str */
    0,                         /* tp_getattro */
    0,                         /* tp_setattro */
    0,                         /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,        /* tp_flags */
    "csp push batch",          /* tp_doc */
    0,                         /* tp_traverse */
    0,                         /* tp_clear */
    0,                         /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    0,                         /* tp_iter */
    0,                         /* tp_iternext */
    PushBatch_methods,         /* tp_methods */
    0,                         /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    ( initproc ) PushBatch_init,  /* tp_init */
    0,
    PyType_GenericNew,
};

class PyPushInputAdapter : public PushInputAdapter
{
public:
    PyPushInputAdapter( Engine * engine, AdapterManager * manager,
                        PyObjectPtr pyadapter, PyObject * pyType,
                        PushMode pushMode, PushGroup * pushGroup ) : PushInputAdapter( engine, pyTypeAsCspType( pyType ), pushMode, pushGroup ),
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
    }

    void stop() override
    {
        PyObjectPtr rv = PyObjectPtr::own( PyObject_CallMethod( m_pyadapter.ptr(), "stop", nullptr ) ); 

        if( !rv.ptr() )
        {
            if( PyErr_Occurred() == PyExc_KeyboardInterrupt )
            {
                PyErr_Clear();
                rv = PyObjectPtr::own( PyObject_CallMethod( m_pyadapter.ptr(), "stop", nullptr ) );
            }

            if( !rv.ptr() )
                CSP_THROW( PythonPassthrough, "" );
        }
    }

    virtual void pushPyTick( PyObject * value, PushBatch * batch ) = 0;

protected:
    PyObjectPtr m_pyadapter;
    PyObjectPtr m_pyType;
};

template<typename T>
class TypedPyPushInputAdapter : public PyPushInputAdapter
{
public:
    TypedPyPushInputAdapter( Engine * engine, AdapterManager * manager, PyObjectPtr pyadapter, PyObject * pyType,
                             PushMode pushMode, PyObjectPtr pyPushGroup, PushGroup * pushGroup ):
        PyPushInputAdapter( engine, manager, pyadapter, pyType, pushMode, pushGroup ),
        m_pyPushGroup( pyPushGroup )
    {
    }

    void pushPyTick( PyObject * value, PushBatch * batch ) override
    {
        try
        {
            if( !validatePyType( this -> dataType(), m_pyType.ptr(), value ) )
                CSP_THROW( TypeError, "" );
            pushTick<T>( std::move(fromPython<T>( value, *this -> dataType() )), batch );
        }
        catch( const TypeError & )
        {
            CSP_THROW( TypeError, "\"" << Py_TYPE( m_pyadapter.ptr() ) -> tp_name << "\" push adapter expected output type to be of type \"" 
                       << pyTypeToString( m_pyType.ptr() ) << "\" got type \"" << Py_TYPE( value ) -> tp_name << "\"" );
        }
    }

private:
    PyObjectPtr m_pyPushGroup;
};

struct PyPushInputAdapter_PyObject
{
    PyObject_HEAD
    PyPushInputAdapter * adapter;

    static PyObject * pushTick( PyPushInputAdapter_PyObject * self, PyObject * args, PyObject **kwargs )
    {
        CSP_BEGIN_METHOD;
        PyObject *pyvalue = nullptr;
        PyObject *pybatch = nullptr;

        Py_ssize_t len = PyTuple_GET_SIZE( args );
        if( len < 1 || len > 2 )
            CSP_THROW( TypeError, "push_tick takes value and optional batch as positional arguments" );

        pyvalue = PyTuple_GET_ITEM( args, 0 );
        
        PushBatch * batch = nullptr;
        if( len == 2 )
        {
            pybatch = PyTuple_GET_ITEM( args, 1 );
            if( pybatch -> ob_type != &PyPushBatch::PyType )
                CSP_THROW( TypeError, "push_tick expected PushBatch type as second argument, got " << pybatch -> ob_type -> tp_name );

            batch = &( ( PyPushBatch * ) pybatch ) -> batch;
        }

        self -> adapter -> pushPyTick( pyvalue, batch );

        CSP_RETURN_NONE;
    }

    static PyObject * shutdown_engine( PyPushInputAdapter_PyObject * self, PyObject * pyException )
    {
        CSP_BEGIN_METHOD;
        
        self -> adapter -> rootEngine() -> shutdown( PyEngine_shutdown_make_exception( pyException ) );

        CSP_RETURN_NONE;
    }

    static PyTypeObject PyType;
};

static PyMethodDef PyPushInputAdapter_PyObject_methods[] = {
    { "push_tick",          (PyCFunction) PyPushInputAdapter_PyObject::pushTick, METH_VARARGS, "push new tick" },
    { "shutdown_engine",    (PyCFunction) PyPushInputAdapter_PyObject::shutdown_engine, METH_O, "shutdown_engine" },
    {NULL}
};

PyTypeObject PyPushInputAdapter_PyObject::PyType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "_cspimpl.PyPushInputAdapter", /* tp_name */
    sizeof(PyPushInputAdapter_PyObject),    /* tp_basicsize */
    0,                         /* tp_itemsize */
    0,                         /* tp_dealloc */
    0,                         /* tp_print */
    0,                         /* tp_getattr */
    0,                         /* tp_setattr */
    0,                         /* tp_reserved */
    0,                         /* tp_repr */
    0,                         /* tp_as_number */
    0,                         /* tp_as_sequence */
    0,                         /* tp_as_mapping */
    0,                         /* tp_hash  */
    0,                         /* tp_call */
    0,                         /* tp_str */
    0,                         /* tp_getattro */
    0,                         /* tp_setattro */
    0,                         /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
    "csp push input adapter",  /* tp_doc */
    0,                         /* tp_traverse */
    0,                         /* tp_clear */
    0,                         /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    0,                         /* tp_iter */
    0,                         /* tp_iternext */
    PyPushInputAdapter_PyObject_methods,    /* tp_methods */
    0,                         /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    0,                         /* tp_init */
    0,
    PyType_GenericNew,
};

static InputAdapter * pypushinputadapter_creator( csp::AdapterManager * manager, PyEngine * pyengine, 
                                                  PyObject * pyType, PushMode pushMode, PyObject * args )
{
    PyTypeObject * pyAdapterType = nullptr;
    PyObject * adapterArgs = nullptr;
    PyObject * pyPushGroup;

    if( !PyArg_ParseTuple( args, "O!OO!", &PyType_Type, &pyAdapterType, &pyPushGroup, &PyTuple_Type, &adapterArgs ) )
        CSP_THROW( PythonPassthrough, "" );

    if( !PyType_IsSubtype( pyAdapterType, &PyPushInputAdapter_PyObject::PyType ) )
        CSP_THROW( TypeError, "Expected PushInputAdapter derived type, got " << pyAdapterType -> tp_name );

    csp::PushGroup *pushGroup = nullptr;
    if( pyPushGroup != Py_None )
    {
        pushGroup = ( csp::PushGroup * ) PyCapsule_GetPointer( pyPushGroup, nullptr );
        if( !pushGroup )
        {
            PyErr_Clear();
            CSP_THROW( TypeError, "Expected PushGroup instance for push group, got: " << PyObjectPtr::incref( pyPushGroup ) );
        }
    }

    PyPushInputAdapter_PyObject * pyAdapter = ( PyPushInputAdapter_PyObject * ) PyObject_Call( ( PyObject * ) pyAdapterType, adapterArgs, nullptr );
    if( !pyAdapter )
        CSP_THROW( PythonPassthrough, "" );

    switchPyType( pyType,
                  [&]( auto tag )
                  {
                      pyAdapter -> adapter = pyengine -> engine() -> createOwnedObject<TypedPyPushInputAdapter<typename decltype(tag)::type>>( 
                          manager, PyObjectPtr::own( ( PyObject * ) pyAdapter ), pyType, pushMode, PyObjectPtr::incref( pyPushGroup ), pushGroup );
                  } );
    
    return pyAdapter -> adapter;
}

//PushGroup
static void destroy_push_group( PyObject * capsule )
{
    delete ( ( PushGroup * ) PyCapsule_GetPointer( capsule, nullptr ) );
}

static PyObject * create_push_group( PyObject * module, PyObject * args )
{
    return PyCapsule_New( new PushGroup(), nullptr, destroy_push_group );
}


REGISTER_TYPE_INIT( &PyPushInputAdapter_PyObject::PyType, "PyPushInputAdapter" );
REGISTER_TYPE_INIT( &PyPushBatch::PyType,                 "PushBatch" );

REGISTER_INPUT_ADAPTER( _pushadapter, pypushinputadapter_creator );
REGISTER_MODULE_METHOD( "PushGroup", create_push_group, METH_NOARGS, "PushGroup" );

}

