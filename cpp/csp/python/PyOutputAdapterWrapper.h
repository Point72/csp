#ifndef _IN_CSP_PYTHON_PYOUTPUTADAPTERWRAPPER_H
#define _IN_CSP_PYTHON_PYOUTPUTADAPTERWRAPPER_H

#include <csp/python/Conversions.h>
#include <csp/python/Exception.h>
#include <csp/python/InitHelper.h>

namespace csp { class OutputAdapter; }

namespace csp::python
{

class PyEngine;

class CSPIMPL_EXPORT PyOutputAdapterWrapper final: public PyObject
{
public:
    PyOutputAdapterWrapper( OutputAdapter * adapter ) : m_adapter( adapter )
    {}

    OutputAdapter * adapter() { return m_adapter; }

    using Creator = std::function<csp::OutputAdapter *( csp::AdapterManager * manager, PyEngine * pyengine, PyObject * args )>;

    static PyObject * createAdapter( Creator creator, PyObject * args );
    static PyTypeObject PyType;

private:
    OutputAdapter * m_adapter;
};

#define REGISTER_OUTPUT_ADAPTER( METHOD_NAME, CREATOR_FUNC ) \
    static PyObject * create_##METHOD_NAME( PyObject *, PyObject * args ) { return PyOutputAdapterWrapper::createAdapter( CREATOR_FUNC, args ); } \
    REGISTER_MODULE_METHOD( #METHOD_NAME, create_##METHOD_NAME, METH_VARARGS, #METHOD_NAME );

}

#endif
