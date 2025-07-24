#include <csp/engine/BasketInfo.h>
#include <csp/engine/Node.h>
#include <csp/python/Common.h>
#include <csp/python/Conversions.h>
#include <csp/python/Exception.h>
#include <csp/python/InitHelper.h>
#include <csp/python/PyBasketInputProxy.h>
#include <csp/python/PyInputProxy.h>
#include <csp/python/PyIterator.h>
#include <csp/python/PyNode.h>

namespace csp::python
{

template<typename ITER,typename GETTER>
struct TsIterator
{
    TsIterator( ITER i ) : it( i ) {}

    PyObject * iternext()
    {
        CSP_BEGIN_METHOD;

        if( !it )
        {
            PyErr_SetNone( PyExc_StopIteration );
            return NULL;
        }

        PyObject * rv = getter( it );
        ++it;
        return rv;

        CSP_RETURN_NULL;
    }

    ITER   it;
    GETTER getter;
};

struct ValueGetter 
{
    template<typename IterT> PyObject * operator()( const IterT & iter ) { return lastValueToPython( iter.get() ); } 
};

template<typename PyBasketInputProxyT>
struct KeyGetter 
{
    PyBasketInputProxyT * proxy;
    template<typename IterT> PyObject * operator()( const IterT & iter ) { return toPython( proxy -> key( iter.elemId() ) ); }
};

template<typename PyBasketInputProxyT>
struct ItemGetter 
{
    PyBasketInputProxyT * proxy;
    template<typename IterT> PyObject * operator()( const IterT & iter ) 
    { 
        PyObject * rv = PyTuple_New( 2 );
        if( !rv )
            CSP_THROW( PythonPassthrough, "" );
        PyTuple_SET_ITEM( rv, 0, toPython( proxy -> key( iter.elemId() ) ) );
        PyTuple_SET_ITEM( rv, 1, lastValueToPython( iter.get() ) );
        return rv;
    }
};

using ValidValuesIterator                                        = TsIterator<InputBasketInfo::valid_iterator,ValueGetter>;
template<typename PyBasketInputProxyT> using ValidKeysIterator   = TsIterator<InputBasketInfo::valid_iterator,KeyGetter<PyBasketInputProxyT>>;
template<typename PyBasketInputProxyT> using ValidItemsIterator  = TsIterator<InputBasketInfo::valid_iterator,ItemGetter<PyBasketInputProxyT>>;

using TickedValuesIterator                                        = TsIterator<InputBasketInfo::ticked_iterator,ValueGetter>;
template<typename PyBasketInputProxyT> using TickedKeysIterator   = TsIterator<InputBasketInfo::ticked_iterator,KeyGetter<PyBasketInputProxyT>>;
template<typename PyBasketInputProxyT> using TickedItemsIterator  = TsIterator<InputBasketInfo::ticked_iterator,ItemGetter<PyBasketInputProxyT>>;

PyBaseBasketInputProxy::PyBaseBasketInputProxy( PyNode * node, INOUT_ID_TYPE id ) : m_node( node ),
                                                                                    m_id( id )
{
}

bool PyBaseBasketInputProxy::ticked() const
{
    return basketInfo() -> ticked();
}

bool PyBaseBasketInputProxy::valid() const
{
    return basketInfo() -> allValid();
}

InputBasketInfo * PyBaseBasketInputProxy::basketInfo()
{ 
    return m_node -> inputBasket( m_id ); 
}

bool PyBaseBasketInputProxy::makeActive()
{
    return m_node -> makeBasketActive( m_id );
}

bool PyBaseBasketInputProxy::makePassive()
{
    return m_node -> makeBasketPassive( m_id );
}

void PyBaseBasketInputProxy::setBufferingPolicy( int32_t tickCount, TimeDelta tickHistory )
{
    auto * basket = basketInfo();
    for( size_t elemId = 0; elemId < basket -> size(); ++elemId )
    {
        auto * ts = const_cast<TimeSeriesProvider *>( m_node -> input( InputId( m_id, elemId ) ) );

        if( tickCount > 0 )
            ts -> setTickCountPolicy( tickCount );
 
        if( !tickHistory.isNone() && tickHistory > TimeDelta::ZERO() )
            ts -> setTickTimeWindowPolicy( tickHistory );
    }

    //if its dynamic, set the policy on the basket as well so that newly added keys get the same policy applied
    if( basket -> isDynamicBasket() )
    {
        auto * dynBasket = static_cast<DynamicInputBasketInfo*>( basket );
        if( tickCount > 0 )
            dynBasket -> setTickCountPolicy( tickCount );

        if( !tickHistory.isNone() && tickHistory > TimeDelta::ZERO() )
            dynBasket -> setTickTimeWindowPolicy( tickHistory );            
    }
}

//reactive methods
static PyObject * PyBaseBasketInputProxy_ticked( PyBaseBasketInputProxy * proxy )
{
    CSP_BEGIN_METHOD;
    return toPython( proxy -> ticked() );
    CSP_RETURN_NONE;
}

static PyObject * PyBaseBasketInputProxy_valid( PyBaseBasketInputProxy * proxy )
{
    CSP_BEGIN_METHOD;
    return toPython( proxy -> valid() );
    CSP_RETURN_NONE;
}

static PyObject * PyBaseBasketInputProxy_make_active( PyBaseBasketInputProxy * proxy, PyObject * args )
{
    CSP_BEGIN_METHOD;
    return toPython( proxy -> makeActive() );
    CSP_RETURN_NONE;
}

static PyObject * PyBaseBasketInputProxy_make_passive( PyBaseBasketInputProxy * proxy, PyObject * args )
{
    CSP_BEGIN_METHOD;
    return toPython( proxy -> makePassive() );
    CSP_RETURN_NONE;
}

static PyObject * PyBaseBasketInputProxy_validvalues( PyBaseBasketInputProxy * proxy )
{
    CSP_BEGIN_METHOD;
    ValidValuesIterator iter( proxy -> basketInfo() -> begin_valid() );
    return PyIterator<ValidValuesIterator>::create( iter );
    CSP_RETURN_NONE;
}

static PyObject * PyBaseBasketInputProxy_set_buffering_policy( PyBaseBasketInputProxy * proxy, PyObject * args, PyObject * kwargs )
{
    CSP_BEGIN_METHOD;
    PyObject * tickCount   = nullptr;
    PyObject * tickHistory = nullptr;

    static const char * kwlist[] = { "tick_count", "tick_history", nullptr };
    if( !PyArg_ParseTupleAndKeywords( args, kwargs, "|O!O", ( char ** ) kwlist, 
                                      &PyLong_Type, &tickCount, 
                                      &tickHistory ) )
        CSP_THROW( PythonPassthrough, "" );

    if( !tickCount && !tickHistory )
        CSP_THROW( TypeError, "csp.set_buffering_policy expected at least one of tick_count or tick_history" );

    proxy -> setBufferingPolicy( tickCount ? fromPython<int32_t>( tickCount ) : -1,
                                 tickHistory ? fromPython<TimeDelta>( tickHistory ) : TimeDelta::NONE() );
    CSP_RETURN_NONE;
}

template<typename PyBasketInputProxyT>
static PyObject * PyBaseBasketInputProxy_validitems( PyBasketInputProxyT * proxy )
{
    CSP_BEGIN_METHOD;
    ValidItemsIterator<PyBasketInputProxyT> iter( proxy -> basketInfo() -> begin_valid() );
    iter.getter.proxy = proxy;
    return PyIterator<ValidItemsIterator<PyBasketInputProxyT>>::create( iter );
    CSP_RETURN_NONE;
}

template<typename PyBasketInputProxyT>
static PyObject * PyBaseBasketInputProxy_validkeys( PyBasketInputProxyT * proxy )
{
    CSP_BEGIN_METHOD;
    ValidKeysIterator<PyBasketInputProxyT> iter( proxy -> basketInfo() -> begin_valid() );
    iter.getter.proxy = proxy;
    return PyIterator<ValidKeysIterator<PyBasketInputProxyT>>::create( iter );
    CSP_RETURN_NONE;
}

static PyObject * PyBaseBasketInputProxy_tickedvalues( PyBaseBasketInputProxy * proxy )
{
    CSP_BEGIN_METHOD;
    TickedValuesIterator iter( proxy -> basketInfo() -> begin_ticked() );
    return PyIterator<TickedValuesIterator>::create( iter );    
    CSP_RETURN_NONE;
}

template<typename PyBasketInputProxyT>
static PyObject * PyBaseBasketInputProxy_tickeditems( PyBasketInputProxyT * proxy )
{
    CSP_BEGIN_METHOD;
    TickedItemsIterator<PyBasketInputProxyT> iter( proxy -> basketInfo() -> begin_ticked() );
    iter.getter.proxy = proxy;
    return PyIterator<TickedItemsIterator<PyBasketInputProxyT>>::create( iter );
    CSP_RETURN_NONE;
}

template<typename PyBasketInputProxyT>
static PyObject * PyBaseBasketInputProxy_tickedkeys( PyBasketInputProxyT * proxy )
{
    CSP_BEGIN_METHOD;
    TickedKeysIterator<PyBasketInputProxyT> iter( proxy -> basketInfo() -> begin_ticked() );
    iter.getter.proxy = proxy;
    return PyIterator<TickedKeysIterator<PyBasketInputProxyT>>::create( iter );
    CSP_RETURN_NONE;
}

static Py_ssize_t PyBaseBasketInputProxy_len( PyBaseBasketInputProxy * proxy ) 
{
    CSP_BEGIN_METHOD;
    return proxy -> basketInfo() -> size(); 
    CSP_RETURN_INT;
}



//ListBasket specific methods
PyListBasketInputProxy::PyListBasketInputProxy( PyNode * node, INOUT_ID_TYPE id, 
                                                size_t shape ) : PyBaseBasketInputProxy( node, id )
{
    for( size_t elemId = 0; elemId < shape; ++elemId )
        m_proxies.emplace_back( PyInputProxyPtr::own( PyInputProxy::create( node, InputId( id, elemId ) ) ) );
}

PyListBasketInputProxy * PyListBasketInputProxy::create( PyNode * node, INOUT_ID_TYPE id, size_t shape )
{
    if( shape > InputId::maxBasketElements() )
        CSP_THROW( ValueError, "List basket size of " << shape << " exceeds basket size limit of " << InputId::maxBasketElements() << " in node " << node -> name() );

    PyListBasketInputProxy * proxy = ( PyListBasketInputProxy * ) PyType.tp_new( &PyType, nullptr, nullptr );
    new( proxy ) PyListBasketInputProxy( node, id, shape );
    return proxy;
}

static void PyListBasketInputProxy_dealloc( PyListBasketInputProxy * self )
{
    CSP_BEGIN_METHOD;
    self -> ~PyListBasketInputProxy();
    PyListBasketInputProxy::PyType.tp_free( self ); 
    CSP_RETURN;
}

//proxy by index
static PyObject * PyListBasketInputProxy_getproxy( PyListBasketInputProxy * proxy, PyObject * idx )
{
     CSP_BEGIN_METHOD;
     PyObject * rv = proxy -> proxy( fromPython<int64_t>( idx ) );
     Py_INCREF( rv );
     return rv;
     CSP_RETURN_NONE;
}

//value by index ( [n] )
static PyObject * PyListBasketInputProxy_getvalue( PyListBasketInputProxy * proxy, Py_ssize_t idx )
{
     CSP_BEGIN_METHOD;
     auto * in_proxy = proxy -> proxy( idx );
     if( !in_proxy -> valid() )
         CSP_THROW( RuntimeException, "list basket element " << idx << " is not valid" );

     return in_proxy -> lastValue();
     CSP_RETURN_NONE;
}

static PyMethodDef PyListBasketInputProxy_methods[] = {
    { "set_buffering_policy", (PyCFunction) PyBaseBasketInputProxy_set_buffering_policy,   METH_VARARGS | METH_KEYWORDS,  "set basket buffering policy" },
    { "make_active",     (PyCFunction) PyBaseBasketInputProxy_make_active,    METH_NOARGS,  "make input active" },
    { "make_passive",    (PyCFunction) PyBaseBasketInputProxy_make_passive,   METH_NOARGS,  "make input passive" },
    { "validvalues",     (PyCFunction) PyBaseBasketInputProxy_validvalues,    METH_NOARGS,  "iterator of all valid values" },
    { "tickedvalues",    (PyCFunction) PyBaseBasketInputProxy_tickedvalues,   METH_NOARGS,  "iterator of all ticked values" },
    { "validkeys",       (PyCFunction) PyBaseBasketInputProxy_validkeys<PyListBasketInputProxy>,    METH_NOARGS,  "iterator of all valid input keys" },
    { "validitems",      (PyCFunction) PyBaseBasketInputProxy_validitems<PyListBasketInputProxy>,   METH_NOARGS,  "iterator of key,value tuples of all valid inputs" },
    { "tickedkeys",      (PyCFunction) PyBaseBasketInputProxy_tickedkeys<PyListBasketInputProxy>,   METH_NOARGS,  "iterator of all ticked input keys" },
    { "tickeditems",     (PyCFunction) PyBaseBasketInputProxy_tickeditems<PyListBasketInputProxy>,  METH_NOARGS,  "iterator of key,value tuples of all ticked inputs" },
    { "_proxy",          (PyCFunction) PyListBasketInputProxy_getproxy,       METH_O,       "Access input proxy" },
    { NULL }
};


static PySequenceMethods PyListBasketInputProxy_SeqMethods = {
    (lenfunc) PyBaseBasketInputProxy_len,   /*sq_length */
    0,                                      /*sq_concat */
    0,                                      /*sq_repeat */
    (ssizeargfunc) PyListBasketInputProxy_getvalue, /*sq_item */
    0,                                      /*was_sq_slice */
    0,                                      /*sq_ass_item */
    0,                                      /*was_sq_ass_slice */
    0                                       /*sq_contains */
};

static PyNumberMethods PyListBasketInputProxy_NumberMethods = {
    0, /* binaryfunc nb_add */
    0, /* binaryfunc nb_subtract */
    0, /* binaryfunc nb_multiply */
    0, /* binaryfunc nb_remainder */
    0, /* binaryfunc nb_divmod */
    0, /* ternaryfunc nb_power */
    ( unaryfunc ) PyBaseBasketInputProxy_valid,  /* unaryfunc nb_negative */
    ( unaryfunc ) PyBaseBasketInputProxy_ticked, /* unaryfunc nb_positive */
    0, /* unaryfunc nb_absolute */
    0, /* inquiry nb_nonzero */ 
    0, /* unaryfunc nb_invert */
    0, /* binaryfunc nb_lshift */
    0, /* binaryfunc nb_rshift */
    0, /* binaryfunc nb_and */
    0, /* binaryfunc nb_xor */
    0, /* binaryfunc nb_or */
    0, /* unaryfunc nb_int */
    0, /* void * reserved */
    0, /* unaryfunc nb_float */
};


PyTypeObject PyListBasketInputProxy::PyType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "_cspimpl.PyListBasketInputProxy",   /* tp_name */
    sizeof(PyListBasketInputProxy),      /* tp_basicsize */
    0,                         /* tp_itemsize */
    ( destructor ) PyListBasketInputProxy_dealloc, /* tp_dealloc */
    0,                         /* tp_print */
    0,                         /* tp_getattr */
    0,                         /* tp_setattr */
    0,                         /* tp_reserved */
    0,                         /* tp_repr */
    &PyListBasketInputProxy_NumberMethods,/* tp_as_number */
    &PyListBasketInputProxy_SeqMethods,                         /* tp_as_sequence */
    0,                         /* tp_as_mapping */
    0,                         /* tp_hash  */
    0,                         /* tp_call */
    0,                         /* tp_str */
    0,                         /* tp_getattro */
    0,                         /* tp_setattro */
    0,                         /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,        /* tp_flags */
    "csp list input proxy",    /* tp_doc */
    0,                         /* tp_traverse */
    0,                         /* tp_clear */
    0,                         /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    0,                         /* tp_iter */
    0,                         /* tp_iternext */
    PyListBasketInputProxy_methods,      /* tp_methods */
    0,                         /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    0,                         /* tp_init */
    0,                         /* tp_alloc */
    PyType_GenericNew,         /* tp_new */
};


//DictBasket specific methods
PyDictBasketInputProxy::PyDictBasketInputProxy( PyNode * node, INOUT_ID_TYPE id, 
                                                PyObject * shape ) : PyBaseBasketInputProxy( node, id ),
                                                                     m_shape( PyObjectPtr::incref( shape ) )
{
    m_proxyMapping = PyObjectPtr::own( PyDict_New() );
    if( !m_proxyMapping.ptr() )
        CSP_THROW( PythonPassthrough, "" );

    Py_ssize_t numElements = PyList_GET_SIZE( shape );
    for( Py_ssize_t elemId = 0; elemId < numElements; ++elemId )
    {
        auto proxy = PyInputProxyPtr::own( PyInputProxy::create( node, InputId( id, elemId ) ) );

        if( PyDict_SetItem( m_proxyMapping.ptr(), PyList_GET_ITEM( shape, elemId ), proxy.ptr() ) < 0 )
            CSP_THROW( PythonPassthrough, "" );
    }
}

PyDictBasketInputProxy * PyDictBasketInputProxy::create( PyNode * node, INOUT_ID_TYPE id, PyObject * shape )
{
    if( !PyList_Check( shape ) )
        CSP_THROW( TypeError, "Invalid shape for dict basket, expect list got: " << Py_TYPE( shape ) -> tp_name );

    if( ( size_t ) PyList_GET_SIZE( shape ) > InputId::maxBasketElements() )
        CSP_THROW( ValueError, "Dict basket size of " << PyList_GET_SIZE( shape ) << " exceeds basket size limit of " << 
               InputId::maxBasketElements() << " in node " << node -> name() );

    PyDictBasketInputProxy * proxy = ( PyDictBasketInputProxy * ) PyType.tp_new( &PyType, nullptr, nullptr );
    new( proxy ) PyDictBasketInputProxy( node, id, shape );
    return proxy;
}

static void PyDictBasketInputProxy_dealloc( PyDictBasketInputProxy * self )
{
    CSP_BEGIN_METHOD;
    self -> ~PyDictBasketInputProxy();
    PyDictBasketInputProxy::PyType.tp_free( self ); 
    CSP_RETURN;
}

PyInputProxy * PyDictBasketInputProxy::proxyByKey( PyObject * key )
{
    PyInputProxy * in_proxy = static_cast<PyInputProxy *>( PyDict_GetItem( m_proxyMapping.ptr(), key ) );
    if( !in_proxy )
        CSP_THROW( KeyError, "key " << PyObjectPtr::incref( key ) << " %s is not a member of the dict basket" );
    return in_proxy;
}

//proxy by key
static PyObject * PyDictBasketInputProxy_getproxy( PyDictBasketInputProxy * proxy, PyObject * key )
{
     CSP_BEGIN_METHOD;
     PyObject * rv = proxy -> proxyByKey( key );
     Py_INCREF( rv );
     return rv;
     CSP_RETURN_NONE;
}

//value by key ( [k] )
static PyObject * PyDictBasketInputProxy_getvalue( PyDictBasketInputProxy * proxy, PyObject * key )
{
     CSP_BEGIN_METHOD;
     auto * in_proxy = proxy -> proxyByKey( key );
     if( !in_proxy -> valid() )
         CSP_THROW( RuntimeException, "dict basket element " << PyObjectPtr::incref( key ) << " is not valid" );

     return in_proxy -> lastValue();
     CSP_RETURN_NONE;
}

bool PyDictBasketInputProxy::contains( PyObject * key )
{
    return PyDict_Contains( m_proxyMapping.ptr(), key );
}

static bool PyDictBasketInputProxy_contains( PyDictBasketInputProxy * proxy, PyObject * key )
{
    return proxy -> contains( key );
}

static PyObject * PyDictBasketInputProxy_keys( PyDictBasketInputProxy * proxy )
{
    PyObject * keys = proxy -> shape();
    Py_INCREF( keys );
    return keys;
}

static PyMethodDef PyDictBasketInputProxy_methods[] = {
    { "set_buffering_policy", (PyCFunction) PyBaseBasketInputProxy_set_buffering_policy,   METH_VARARGS | METH_KEYWORDS,  "set basket buffering policy" },
    { "make_active",     (PyCFunction) PyBaseBasketInputProxy_make_active,    METH_NOARGS,  "make input active" },
    { "make_passive",    (PyCFunction) PyBaseBasketInputProxy_make_passive,   METH_NOARGS,  "make input passive" },
    { "validvalues",     (PyCFunction) PyBaseBasketInputProxy_validvalues,    METH_NOARGS,  "iterator of all valid values" },
    { "tickedvalues",    (PyCFunction) PyBaseBasketInputProxy_tickedvalues,   METH_NOARGS,  "iterator of all ticked values" },
    { "validkeys",       (PyCFunction) PyBaseBasketInputProxy_validkeys<PyDictBasketInputProxy>,    METH_NOARGS,  "iterator of all valid input keys" },
    { "validitems",      (PyCFunction) PyBaseBasketInputProxy_validitems<PyDictBasketInputProxy>,   METH_NOARGS,  "iterator of key,value tuples of all valid inputs" },
    { "tickedkeys",      (PyCFunction) PyBaseBasketInputProxy_tickedkeys<PyDictBasketInputProxy>,   METH_NOARGS,  "iterator of all ticked input keys" },
    { "tickeditems",     (PyCFunction) PyBaseBasketInputProxy_tickeditems<PyDictBasketInputProxy>,  METH_NOARGS,  "iterator of key,value tuples of all ticked inputs" },
    { "keys",            (PyCFunction) PyDictBasketInputProxy_keys,           METH_NOARGS,  "list of all keys on the basket" },
    { "_proxy",          (PyCFunction) PyDictBasketInputProxy_getproxy,       METH_O,       "Access input proxy" },
    { NULL }
};


static PyMappingMethods PyDictBasketInputProxy_MappingMethods = {
    (lenfunc) PyBaseBasketInputProxy_len,   /*mp_length */
    (binaryfunc) PyDictBasketInputProxy_getvalue, /*mp_subscript */
    0                                       /*mp_ass_subscript */
};

static PySequenceMethods PyDictBasketInputProxy_SeqMethods = {
        0,                                      /*sq_length */
        0,                                      /*sq_concat */
        0,                                      /*sq_repeat */
        0,                                      /*sq_item */
        0,                                      /*was_sq_slice */
        0,                                      /*sq_ass_item */
        0,                                      /*was_sq_ass_slice */
        (objobjproc) PyDictBasketInputProxy_contains     /*sq_contains */
};

static PyNumberMethods PyDictBasketInputProxy_NumberMethods = {
    0, /* binaryfunc nb_add */
    0, /* binaryfunc nb_subtract */
    0, /* binaryfunc nb_multiply */
    0, /* binaryfunc nb_remainder */
    0, /* binaryfunc nb_divmod */
    0, /* ternaryfunc nb_power */
    ( unaryfunc ) PyBaseBasketInputProxy_valid,  /* unaryfunc nb_negative */
    ( unaryfunc ) PyBaseBasketInputProxy_ticked, /* unaryfunc nb_positive */
    0, /* unaryfunc nb_absolute */
    0, /* inquiry nb_nonzero */ 
    0, /* unaryfunc nb_invert */
    0, /* binaryfunc nb_lshift */
    0, /* binaryfunc nb_rshift */
    0, /* binaryfunc nb_and */
    0, /* binaryfunc nb_xor */
    0, /* binaryfunc nb_or */
    0, /* unaryfunc nb_int */
    0, /* void * reserved */
    0, /* unaryfunc nb_float */
};


PyTypeObject PyDictBasketInputProxy::PyType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "_cspimpl.PyDictBasketInputProxy",   /* tp_name */
    sizeof(PyDictBasketInputProxy),      /* tp_basicsize */
    0,                         /* tp_itemsize */
    ( destructor ) PyDictBasketInputProxy_dealloc, /* tp_dealloc */
    0,                         /* tp_print */
    0,                         /* tp_getattr */
    0,                         /* tp_setattr */
    0,                         /* tp_reserved */
    0,                         /* tp_repr */
    &PyDictBasketInputProxy_NumberMethods,/* tp_as_number */
    &PyDictBasketInputProxy_SeqMethods,      /* tp_as_sequence */
    &PyDictBasketInputProxy_MappingMethods,  /* tp_as_mapping */
    0,                         /* tp_hash  */
    0,                         /* tp_call */
    0,                         /* tp_str */
    0,                         /* tp_getattro */
    0,                         /* tp_setattro */
    0,                         /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,        /* tp_flags */
    "csp dict input proxy",    /* tp_doc */
    0,                         /* tp_traverse */
    0,                         /* tp_clear */
    0,                         /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    0,                         /* tp_iter */
    0,                         /* tp_iternext */
    PyDictBasketInputProxy_methods,      /* tp_methods */
    0,                         /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    0,                         /* tp_init */
    0,                         /* tp_alloc */
    PyType_GenericNew,         /* tp_new */
};

PyDynamicBasketInputProxy::PyDynamicBasketInputProxy( PyNode * node, INOUT_ID_TYPE id, PyObject * shape ) : PyDictBasketInputProxy( node, id, shape )
{
    auto * dynBasket = const_cast<DynamicInputBasketInfo *>( static_cast<const DynamicInputBasketInfo *>( basketInfo() ) );
    dynBasket -> setChangeCallback( [this]( const DialectGenericType & key, bool added, int64_t elemId, int64_t replaceId )
                                    {
                                        handleShapeChange( key, added, elemId, replaceId );
                                    } );
    m_shapeProxy = PyInputProxyPtr::own( PyInputProxy::create( m_node, InputId( m_id, -1 ) ) );
}

PyDynamicBasketInputProxy * PyDynamicBasketInputProxy::create( PyNode * node, INOUT_ID_TYPE id )
{
    PyObjectPtr shape = PyObjectPtr::own( PyList_New( 0 ) );
    PyDynamicBasketInputProxy * proxy = ( PyDynamicBasketInputProxy * ) PyType.tp_new( &PyType, nullptr, nullptr );
    new( proxy ) PyDynamicBasketInputProxy( node, id, shape.get() );
    return proxy;
}

void PyDynamicBasketInputProxy::handleShapeChange( const DialectGenericType & key, bool added, int64_t elemId, int64_t replaceId )
{
    auto & pyKey = reinterpret_cast<const PyObjectPtr &>( key );

    if( added )
    {
        auto proxy = PyInputProxyPtr::own( PyInputProxy::create( m_node, InputId( m_id, elemId ) ) );

        if( PyDict_SetItem( m_proxyMapping.ptr(), pyKey.get(), proxy.get() ) < 0 )
            CSP_THROW( PythonPassthrough, "" );

        //current impl elements are always added at the end
        CSP_ASSERT( elemId == PyList_GET_SIZE( m_shape.get() ) );
        PyList_Append( m_shape.get(), pyKey.get() );
    }
    else
    {
        if( PyDict_DelItem( m_proxyMapping.ptr(), pyKey.get() ) < 0 )
            CSP_THROW( PythonPassthrough, "" );

        //update shape, assumption are made on algorithm used to update ids ( asserted below )
        if( replaceId != -1 )
        {
            CSP_ASSERT( replaceId == ( PyList_GET_SIZE( m_shape.get() ) -1 ) );
            //free up the key being deleted 
            Py_DECREF( PyList_GET_ITEM( m_shape.get(), elemId ) );

            //get borrowed ref to key that will take up the slot
            PyObject * replacedKey = PyList_GET_ITEM( m_shape.get(), replaceId );
            
            //set elemId slot = replaceId slot in our proxy dictionary
            PyInputProxy * replacedproxy = static_cast<PyInputProxy *>( PyDict_GetItem( m_proxyMapping.get(), replacedKey ) );
            CSP_ASSERT( replacedproxy != nullptr );
            replacedproxy -> setElemId( elemId );

            //set replaced key in new location in shape
            PyList_SET_ITEM( m_shape.get(), elemId, replacedKey );
        }
        else
        {
            CSP_ASSERT( elemId == ( PyList_GET_SIZE( m_shape.get() ) -1 ) );
            Py_DECREF( PyList_GET_ITEM( m_shape.get(), elemId ) );
        }

        //force resize down of last element.  There is no C-API to do this for some reason.. but this is what the python C code
        //does internally on pop ( essentially )
#if IS_PRE_PYTHON_3_11
        Py_SIZE( m_shape.get() ) -= 1;
#else
        Py_SET_SIZE( m_shape.get(), Py_SIZE( m_shape.get() ) - 1 );
#endif
    }
}

static void PyDynamicBasketInputProxy_dealloc( PyDynamicBasketInputProxy * self )
{
    CSP_BEGIN_METHOD;
    self -> ~PyDynamicBasketInputProxy();
    PyDynamicBasketInputProxy::PyType.tp_free( self ); 
    CSP_RETURN;
}

PyObject * PyDynamicBasketInputProxy::shape_getter( PyDynamicBasketInputProxy * self, void * )
{
    return self -> m_shapeProxy -> lastValue();
}

PyObject * PyDynamicBasketInputProxy::shapeproxy_getter( PyDynamicBasketInputProxy * self, void * )
{
    return self -> m_shapeProxy.incref().release();
}

//ugh, python 3.6 PyGetSet strings are incorrectly declre char * instead of const char *...
static char s_shapename[] = "shape";
static char s_shapedoc[] = "timeseries of the dynamic basket shape ts[csp.DynamicBasketEvents]";

static char s_shapeproxyname[] = "_shapeproxy";

static PyGetSetDef PyDynamicBasketInputProxy_getset[] = {
    { s_shapename, ( getter ) PyDynamicBasketInputProxy::shape_getter, NULL, s_shapedoc, NULL },
    { s_shapeproxyname, ( getter ) PyDynamicBasketInputProxy::shapeproxy_getter, NULL, NULL, NULL },
    { NULL }
};
    

PyTypeObject PyDynamicBasketInputProxy::PyType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "_cspimpl.PyDynamicBasketInputProxy",   /* tp_name */
    sizeof(PyDynamicBasketInputProxy),      /* tp_basicsize */
    0,                         /* tp_itemsize */
    ( destructor ) PyDynamicBasketInputProxy_dealloc, /* tp_dealloc */
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
    "csp dynamic basket input proxy",    /* tp_doc */
    0,                         /* tp_traverse */
    0,                         /* tp_clear */
    0,                         /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    0,                         /* tp_iter */
    0,                         /* tp_iternext */
    0,                         /* tp_methods */
    0,                         /* tp_members */
    PyDynamicBasketInputProxy_getset, /* tp_getset */
    &PyDictBasketInputProxy::PyType, /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    0,                         /* tp_init */
    0,                         /* tp_alloc */
    PyType_GenericNew,         /* tp_new */
};

REGISTER_TYPE_INIT( &PyListBasketInputProxy::PyType,    "PyListBasketInputProxy" );
REGISTER_TYPE_INIT( &PyDictBasketInputProxy::PyType,    "PyDictBasketInputProxy" );
REGISTER_TYPE_INIT( &PyDynamicBasketInputProxy::PyType, "PyDynamicBasketInputProxy" );

}
