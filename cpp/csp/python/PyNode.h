#ifndef _IN_CSP_PYTHON_PYNODE_H
#define _IN_CSP_PYTHON_PYNODE_H

#include <csp/core/Time.h>
#include <csp/engine/Node.h>
#include <csp/python/PyObjectPtr.h>
#include <Python.h>

namespace csp::python
{

class PyEngine;

class PyNode final: public csp::Node
{
public:
    PyNode( csp::Engine * engine, PyObjectPtr gen, PyObjectPtr inputs, PyObjectPtr outputs,
            NodeDef def );
    ~PyNode();

    void executeImpl() override;
    void start() override;
    void stop() override;
    bool makeActive( InputId id ) override;
    bool makePassive( InputId id ) override;

    //see .cpp for reason why this is overloaded
    void createAlarm( CspTypePtr & type, size_t id ) override;

    const char * name() const override;

    static PyNode * create( PyEngine * engine, PyObject * inputs, PyObject * outputs, PyObject * gen );

private:
    void init( PyObjectPtr inputs, PyObjectPtr outputs );
    void call_gen();

    PyObjectPtr  m_gen;
    PyObject *** m_localVars; //array of PyObject ** objects

    //array that contains the count of each passive input when we last converted it to Python
    //the indexing corresponds to the input index as seen by the node
    uint32_t *   m_passiveCounts; 
};

};

#endif
