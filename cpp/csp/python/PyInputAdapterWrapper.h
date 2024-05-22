#ifndef _IN_CSP_PYTHON_PYINPUTADAPTERWRAPPER_H
#define _IN_CSP_PYTHON_PYINPUTADAPTERWRAPPER_H

#include <Python.h>
#include <csp/engine/Enums.h>
#include <csp/python/Exception.h>
#include <csp/python/InitHelper.h>

namespace csp { class AdapterManager; class InputAdapter; }

namespace csp::python
{

class PyEngine;

class CSPIMPL_EXPORT PyInputAdapterWrapper : public PyObject
{
public:
    PyInputAdapterWrapper( InputAdapter * adapter ) : m_adapter( adapter )
    {}

    InputAdapter * adapter() { return m_adapter; }

    using Creator = std::function<csp::InputAdapter *( csp::AdapterManager * manager, PyEngine * pyengine, 
                                                       PyObject *, PushMode pushMode, PyObject * args )>;

    static PyObject * createAdapter( Creator creator, PyObject * args );
    static PyObject * create( InputAdapter * adapter );

    static PyTypeObject PyType;

private:
    InputAdapter * m_adapter;
};

#define REGISTER_INPUT_ADAPTER( METHOD_NAME, CREATOR_FUNC ) \
    static PyObject * create_##METHOD_NAME( PyObject *, PyObject * args ) { return PyInputAdapterWrapper::createAdapter( CREATOR_FUNC, args ); } \
    REGISTER_MODULE_METHOD( #METHOD_NAME, create_##METHOD_NAME, METH_VARARGS, #METHOD_NAME );

}

#endif
