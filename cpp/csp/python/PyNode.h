#ifndef _IN_CSP_PYTHON_PYNODE_H
#define _IN_CSP_PYTHON_PYNODE_H

#include <csp/core/Time.h>
#include <csp/engine/Node.h>
#include <csp/python/Common.h>
#include <csp/python/PyObjectPtr.h>
#include <Python.h>

#if !IS_PRE_PYTHON_3_11
#if !IS_PRE_PYTHON_3_13
#    define Py_BUILD_CORE 1
#endif
#include <internal/pycore_code.h>
#include <internal/pycore_frame.h>
#if !IS_PRE_PYTHON_3_14
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4576)
#endif
#include <internal/pycore_genobject.h>
#include <internal/pycore_stackref.h>
#ifdef _MSC_VER
#pragma warning(pop)
#endif
#endif
#if !IS_PRE_PYTHON_3_13
#    undef Py_BUILD_CORE
#endif
#endif

namespace csp::python
{

#if !IS_PRE_PYTHON_3_14
using FrameLocalVar = _PyStackRef;
#else
using FrameLocalVar = PyObject *;
#endif

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

    PyObjectPtr      m_gen;
    FrameLocalVar ** m_localVars;

    //array that contains the count of each passive input when we last converted it to Python
    //the indexing corresponds to the input index as seen by the node
    uint32_t *   m_passiveCounts; 
};

};

#endif
