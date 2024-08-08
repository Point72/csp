#ifndef _IN_CSP_PYTHON_PYCPPNODE_H
#define _IN_CSP_PYTHON_PYCPPNODE_H

#include <Python.h>
#include <csp/engine/CppNode.h>
#include <csp/python/Exception.h>
#include <csp/python/InitHelper.h>

namespace csp::python
{

/* 
So here's whats happening here...
we have c++ nodes defined in C++ space under csp/cppnodes.  Those compile .cpp files that define the nodes, they arent in headers at all.
This allows us to build 1 library instead of 1 library per dialect ( not that it matters in practice now but who knows in the future ... )
We also need to expose the nodes to python with creator methods per node to return PyNodeWrappers wrapping the C++ node instances.  So we
need to register nodes in a python module as well ( see cspbaselibimpl.cpp for example ).  I norder to avoid the need for a registration factory
in c++, we have macros to fwd declare the creator methods that create the C++ node ( CPPNODE_CREATE_FWD_DECL ).  so the python code fwd declares the creator
and relies on linking in the c+ node library properly.
*/

CSPIMPL_EXPORT PyObject * pycppnode_create( PyObject * args, csp::CppNode::Creator creatorFn );

#define _REGISTER_CPPNODE( Name, Creator ) \
static PyObject * Name##_cppnode_create( PyObject * module, PyObject * args ) \
{\
    CSP_BEGIN_METHOD;                    \
    return csp::python::pycppnode_create( args, Creator ); \
    CSP_RETURN_NULL;                     \
}\
REGISTER_MODULE_METHOD( #Name, Name##_cppnode_create, METH_VARARGS, #Name );

#define REGISTER_CPPNODE( Namespace, NodeName ) CPPNODE_CREATE_FWD_DECL( Namespace, NodeName ) \
    _REGISTER_CPPNODE( NodeName, Namespace::CPPNODE_CREATE_METHOD( NodeName ) )

#endif

}
