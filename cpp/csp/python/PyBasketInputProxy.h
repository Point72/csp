#ifndef _IN_CSP_PYTHON_PYBASKETINPUTPROXY_H
#define _IN_CSP_PYTHON_PYBASKETINPUTPROXY_H

#include <csp/engine/InputId.h>
#include <csp/python/Conversions.h>
#include <Python.h>

namespace csp { class InputBasketInfo; }

namespace csp::python
{

class PyInputProxy;
using PyInputProxyPtr = PyPtr<PyInputProxy>; 
class PyNode;

class PyBaseBasketInputProxy : public PyObject
{
public:
    PyBaseBasketInputProxy( PyNode * node, INOUT_ID_TYPE id );

    bool makeActive();
    bool makePassive();

    bool valid() const;
    bool ticked() const;

    void setBufferingPolicy( int32_t tickCount, TimeDelta tickHistory );

    const InputBasketInfo * basketInfo() const { return const_cast<PyBaseBasketInputProxy *>( this ) -> basketInfo(); }
    InputBasketInfo * basketInfo();

protected:

    PyNode       * m_node;
    INOUT_ID_TYPE  m_id;
};


class PyListBasketInputProxy final: public PyBaseBasketInputProxy
{
public:
    PyListBasketInputProxy( PyNode * node, INOUT_ID_TYPE id, size_t shape );

    static PyListBasketInputProxy * create( PyNode * node, INOUT_ID_TYPE id, size_t shape );

    int64_t key( int64_t elemId ) const { return elemId; }

    static PyTypeObject PyType;

    //borrowed ref
    PyInputProxy * proxy( int64_t elemId ) 
    { 
        if( elemId < 0 || size_t(elemId) >= m_proxies.size() )
            CSP_THROW( RangeError, "basket index out of range" );

        return m_proxies[ elemId ].ptr(); 
    }

private:
    std::vector<PyInputProxyPtr> m_proxies;
};


class PyDictBasketInputProxy : public PyBaseBasketInputProxy
{
public:
    PyDictBasketInputProxy( PyNode * node, INOUT_ID_TYPE id, PyObject * shape );

    static PyDictBasketInputProxy * create( PyNode * node, INOUT_ID_TYPE id, PyObject * shape );

    PyObjectPtr key( int64_t elemId ) const { return PyObjectPtr::incref( PyList_GET_ITEM( m_shape.ptr(), elemId ) ); }
    
    PyInputProxy * proxyByKey( PyObject * key );

    bool contains( PyObject * key );

    //borrowed ref
    PyObject * shape() { return m_shape.get(); }

    static PyTypeObject PyType;

protected:
    PyObjectPtr m_shape;
    PyObjectPtr m_proxyMapping;
};

class PyDynamicBasketInputProxy final : public PyDictBasketInputProxy
{
public:
    PyDynamicBasketInputProxy( PyNode * node, INOUT_ID_TYPE id, PyObject * shape );

    static PyDynamicBasketInputProxy * create( PyNode * node, INOUT_ID_TYPE id );

    static PyObject * shape_getter( PyDynamicBasketInputProxy * self, void * );
    static PyObject * shapeproxy_getter( PyDynamicBasketInputProxy * self, void * );

    static PyTypeObject PyType;

private:
    PyInputProxyPtr m_shapeProxy;
    void handleShapeChange( const DialectGenericType & key, bool added, int64_t elemId, int64_t replaceId );
};

}

#endif
