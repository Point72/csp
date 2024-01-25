#include <csp/engine/AdapterManager.h>
#include <csp/python/Conversions.h>
#include <csp/python/Exception.h>
#include <csp/python/InitHelper.h>
#include <csp/python/PyInputAdapterWrapper.h>
#include <csp/python/PyAdapterManager.h>
#include <csp/python/PyCspType.h>
#include <csp/python/PyEngine.h>
#include <csp/python/PyObjectPtr.h>

namespace csp::python
{

class PyManagedSimInputAdapter : public ManagedSimInputAdapter
{
public:
    PyManagedSimInputAdapter( Engine * engine, AdapterManager * manager,
                              PyObjectPtr pyadapter, PyObject * pyType,
                              PushMode pushMode ) : ManagedSimInputAdapter( engine, pyTypeAsCspType( pyType ), manager, pushMode ),
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
            CSP_THROW( PythonPassthrough, "" );
    }

    virtual void pushPyTick( PyObject * value ) = 0;

protected:
    PyObjectPtr m_pyadapter;
    PyObjectPtr m_pyType;
};

template<typename T>
class TypedPyManagedSimInputAdapter : public PyManagedSimInputAdapter
{
public:
    TypedPyManagedSimInputAdapter( Engine * engine, AdapterManager * manager, PyObjectPtr pyadapter, PyObject * pyType, PushMode pushMode ):
        PyManagedSimInputAdapter( engine, manager, pyadapter, pyType, pushMode )
    {
        
    }
    
    void pushPyTick( PyObject * value ) override
    {
        try
        {
            if( !validatePyType( this -> dataType(), m_pyType.ptr(), value ) )
                CSP_THROW( TypeError, "" );
        
            pushTick<T>( fromPython<T>( value, *dataType() ) );
        }
        catch( const TypeError & )
        {
            CSP_THROW( TypeError, "\"" << Py_TYPE( m_pyadapter.ptr() ) -> tp_name << "\" managed sim adapter expected output type to be of type \"" 
                       << pyTypeToString( m_pyType.ptr() ) << "\" got type \"" << Py_TYPE( value ) -> tp_name << "\"" );
        }

    }
};

struct PyManagedSimInputAdapter_PyObject
{
    PyObject_HEAD
    PyManagedSimInputAdapter * adapter;

    static PyObject * pushTick( PyManagedSimInputAdapter_PyObject * self, PyObject * pyValue )
    {
        CSP_BEGIN_METHOD;
        self -> adapter -> pushPyTick( pyValue );
        CSP_RETURN_NONE;
    }

    static PyTypeObject PyType;
};
    
static PyMethodDef PyManagedSimInputAdapter_PyObject_methods[] = {
    { "push_tick", (PyCFunction) PyManagedSimInputAdapter_PyObject::pushTick, METH_O, "push new tick" },
    {NULL}
};

PyTypeObject PyManagedSimInputAdapter_PyObject::PyType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "_cspimpl.Pymanagedsiminputadapter", /* tp_name */
    sizeof(PyManagedSimInputAdapter_PyObject),    /* tp_basicsize */
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
    "csp maanged sim adapter",     /* tp_doc */
    0,                         /* tp_traverse */
    0,                         /* tp_clear */
    0,                         /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    0,                         /* tp_iter */
    0,                         /* tp_iternext */
    PyManagedSimInputAdapter_PyObject_methods,    /* tp_methods */
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
    0,                         /* tp_free */ /* Low-level free-memory routine */
    0,                         /* tp_is_gc */ /* For PyObject_IS_GC */
    0,                         /* tp_bases */
    0,                         /* tp_mro */ /* method resolution order */
    0,                         /* tp_cache */
    0,                         /* tp_subclasses */
    0,                         /* tp_weaklist */
    0,                         /* tp_del */
    0,                         /* tp_version_tag */
    0                          /* tp_finalize */
};

REGISTER_TYPE_INIT( &PyManagedSimInputAdapter_PyObject::PyType, "PyManagedSimInputAdapter" );

static InputAdapter * pymanagedsimadapter_creator( csp::AdapterManager * manager, PyEngine * engine, 
                                                   PyObject * pyType, PushMode pushMode, PyObject * args )
{
    PyTypeObject * pyAdapterType = nullptr;
    PyObject * adapterArgs = nullptr;
    if( !PyArg_ParseTuple( args, "O!O!", &PyType_Type, &pyAdapterType,
                           &PyTuple_Type, &adapterArgs ) )
        CSP_THROW( PythonPassthrough, "" );

    if( !PyType_IsSubtype( pyAdapterType, &PyManagedSimInputAdapter_PyObject::PyType ) )
        CSP_THROW( TypeError, "Expected PyManagedSimInputAdapter derived type, got " << pyAdapterType -> tp_name );

    PyManagedSimInputAdapter_PyObject * pyAdapter = ( PyManagedSimInputAdapter_PyObject * ) PyObject_Call( ( PyObject * ) pyAdapterType, adapterArgs, nullptr );
    if( !pyAdapter )
        CSP_THROW( PythonPassthrough, "" );

    switchPyType( pyType,
                  [&]( auto tag )
                  {
                      pyAdapter -> adapter = engine -> engine() -> createOwnedObject<TypedPyManagedSimInputAdapter<typename decltype(tag)::type>>( 
                          manager, PyObjectPtr::own( ( PyObject * ) pyAdapter ), pyType, pushMode );
                  } );

    return pyAdapter -> adapter;
}

REGISTER_INPUT_ADAPTER( _managedsimadapter, pymanagedsimadapter_creator );

}
