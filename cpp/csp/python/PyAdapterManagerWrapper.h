#ifndef _IN_CSP_PYTHON_PYADAPTERMANAGERWRAPPER_H
#define _IN_CSP_PYTHON_PYADAPTERMANAGERWRAPPER_H

#include <Python.h>
#include <csp/python/Exception.h>
#include <csp/python/InitHelper.h>

namespace csp { class AdapterManager; }

namespace csp::python
{

class PyEngine;

class CSPIMPL_EXPORT PyAdapterManagerWrapper
{
public:
    using Creator = std::function<csp::AdapterManager *( PyEngine * pyengine, const Dictionary & properties )>;

    static PyObject * create( Creator creator, PyObject * args );
    static csp::AdapterManager * extractAdapterManager( PyObject * wrapper );
};

#define REGISTER_ADAPTER_MANAGER( METHOD_NAME, CREATOR_FUNC ) \
    static PyObject * create_##METHOD_NAME( PyObject *, PyObject * args ) { return csp::python::PyAdapterManagerWrapper::create( CREATOR_FUNC, args ); } \
    REGISTER_MODULE_METHOD( #METHOD_NAME, create_##METHOD_NAME, METH_VARARGS, #METHOD_NAME );

#define REGISTER_ADAPTER_MANAGER_CUSTOM_CREATOR( METHOD_NAME, CREATOR_FUNC ) \
    static PyObject * create_##METHOD_NAME( PyObject *, PyObject * args ) { return CREATOR_FUNC( args ); } \
    REGISTER_MODULE_METHOD( #METHOD_NAME, create_##METHOD_NAME, METH_VARARGS, #METHOD_NAME );

}

#endif
