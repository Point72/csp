#ifndef _IN_PYTHON_PYNUMBANODE_H
#define _IN_PYTHON_PYNUMBANODE_H

#include <csp/core/Time.h>
#include <csp/engine/Node.h>
#include <csp/python/PyObjectPtr.h>
#include <Python.h>

namespace csp::python
{
class PyEngine;

typedef void (*CallbackType)(void *node, void *state);

class PyNumbaNode final : public csp::Node
{
public:
    // TODO: Add suppot for initialization callback as well
    PyNumbaNode(csp::Engine *engine,
                void *stateObject, CallbackType numbaInitCallback, CallbackType numbaImplCallback, PyObjectPtr inputs,
                PyObjectPtr outputs,
                NodeDef def, PyObject *dataReference);

    ~PyNumbaNode();

    void executeImpl() override;

    void start() override;

    void stop() override;

    const char *name() const override;

    static PyNumbaNode *create(PyEngine *engine, PyObject *inputs, PyObject *outputs,
                               PyObject *stateObject, PyObject *numbaInitCallback, PyObject *numbaImplCallback,
                               PyObject *dataReference);

private:
    void init(PyObjectPtr inputs, PyObjectPtr outputs);

    void call_callback();

    void *m_stateObject;
    CallbackType m_numbaInitCallback;
    CallbackType m_numbaImplCallback;
    PyObjectPtr m_dataReference;
};
};

#endif
