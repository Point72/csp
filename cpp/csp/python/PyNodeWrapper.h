#ifndef _IN_CSP_PYTHON_PYNODEWRAPPER_H
#define _IN_CSP_PYTHON_PYNODEWRAPPER_H

#include <csp/core/Platform.h>
#include <csp/engine/Node.h>
#include <Python.h>

namespace csp::python
{

//simple wrapper for python level access when wiring
class CSP_PUBLIC PyNodeWrapper : public PyObject
{
public:
    csp::Node * node() { return m_node; }

    static PyNodeWrapper * create( csp::Node * node );
    static PyTypeObject PyType;

private:

    PyNodeWrapper( csp::Node * node ) : m_node( node ) {}
    csp::Node * m_node;
};

}

#endif