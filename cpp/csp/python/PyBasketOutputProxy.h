#ifndef _IN_CSP_PYTHON_PYBASKETOUTPUTPROXY_H
#define _IN_CSP_PYTHON_PYBASKETOUTPUTPROXY_H

#include <csp/engine/InputId.h>
#include <csp/python/Conversions.h>
#include <Python.h>

namespace csp::python
{

class PyOutputProxy;
using PyOutputProxyPtr = PyPtr<PyOutputProxy>;
class PyNode;

class PyBaseBasketOutputProxy : public PyObject
{
public:
    PyBaseBasketOutputProxy( Node * node, INOUT_ID_TYPE id );

protected:
    Node *                        m_node;
    INOUT_ID_TYPE                 m_id;
};

class PyListBasketOutputProxy final : public PyBaseBasketOutputProxy
{
public:
    PyListBasketOutputProxy( PyObject *pyType, Node * node, INOUT_ID_TYPE id, size_t shape );

    static PyListBasketOutputProxy * create( PyObject *pyType, Node * node, INOUT_ID_TYPE id, size_t shape );

    static PyTypeObject PyType;

    PyOutputProxy * proxy( int64_t elemId )
    {
        if( elemId < 0 || size_t(elemId) >= m_proxies.size() )
            CSP_THROW( RangeError, "basket index out of range" );

        return m_proxies[ elemId ].ptr();
    }

private:
    std::vector<PyOutputProxyPtr> m_proxies;
};

class PyDictBasketOutputProxy : public PyBaseBasketOutputProxy
{
public:
    PyDictBasketOutputProxy( PyObject *pyType, Node * node, INOUT_ID_TYPE id, PyObject * shape );

    static PyDictBasketOutputProxy * create( PyObject *pyType, Node * node, INOUT_ID_TYPE id, PyObject * shape );

    PyOutputProxy * proxyByKey( PyObject * key );

    static PyTypeObject PyType;

protected:
    PyObjectPtr m_proxyMapping;
};

class PyDynamicBasketOutputProxy final : public PyDictBasketOutputProxy
{
public:
    PyDynamicBasketOutputProxy( PyObject *pyType, Node * node, INOUT_ID_TYPE id, PyObject * shape );

    static PyDynamicBasketOutputProxy * create( PyObject *pyType, Node * node, INOUT_ID_TYPE id );

    static PyTypeObject PyType;

    PyOutputProxy * getOrCreateProxy( PyObject * key );
    void removeProxy( PyObject * key );

private:
    PyObjectPtr m_elemType;
    std::vector<PyObjectPtr> m_keyMapping;
};

}

#endif
