#include <csp/engine/Node.h>
#include <Python.h>

namespace csp::python
{

//simple wrapper for python level access when wiring
class PyNodeWrapper : public PyObject
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
