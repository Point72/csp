#ifndef _IN_CSP_PYTHON_PYNUMBANODE_H
#define _IN_CSP_PYTHON_PYNUMBANODE_H

#include <csp/core/Time.h>
#include <csp/engine/Node.h>
#include <csp/python/PyObjectPtr.h>
#include <Python.h>
#include <cstdint>

namespace csp::python
{

class PyEngine;

// Type used for ticked/valid flags - must match int8_t used in Numba generated code
using ValidType = int8_t;

// Lifecycle phase constants - must match values in numba_config.py
constexpr ValidType LIFECYCLE_EXECUTE = 0;  
constexpr ValidType LIFECYCLE_START = 1;    
constexpr ValidType LIFECYCLE_STOP = 2;     

/**
 * The compiled function signature matches the source categories registered
 * by CSP (Constant, Output, State, Lifecycle, Signal):
 *
 *   void(*)( void **              outputs,
 *            ValidType *          outputs_ticked,
 *            void **              state_variables,
 *            ValidType            lifecycle_phase,
 *            const void * const * inputs,
 *            const ValidType *    inputs_ticked,
 *            const ValidType *    inputs_valid );
 */
class PyNumbaNode final : public csp::Node
{
public:
    using CompiledFuncT = void(*)(
        void **              outputs,
        ValidType *          outputs_ticked,
        void **              state_variables,
        ValidType            lifecycle_phase,
        const void * const * inputs,
        const ValidType *    inputs_ticked,
        const ValidType *    inputs_valid
    );

    PyNumbaNode(
        csp::Engine * engine,
        CompiledFuncT compiledFunc,
        PyObjectPtr inputs,
        PyObjectPtr outputs,
        PyObjectPtr stateVariables,
        PyObjectPtr nrtStateIndices,
        PyObjectPtr structStateIndices,
        PyObjectPtr structStateSizes,
        csp::NodeDef def,
        PyObject * dataReference
    );

    ~PyNumbaNode();

    void executeImpl() override;
    void start() override;
    void stop() override;
    const char * name() const override;

    static PyNumbaNode * create(
        PyEngine * engine,
        PyObject * compiledFuncPtr,
        PyObject * inputs,
        PyObject * outputs,
        PyObject * stateVariables,
        PyObject * nrtStateIndices,
        PyObject * structStateIndices,
        PyObject * structStateSizes,
        PyObject * dataReference
    );

private:
    void initInputArrays( PyObjectPtr inputs );
    void initOutputArrays( PyObjectPtr outputs );
    void initStateArrays( PyObjectPtr stateVariables, PyObjectPtr nrtStateIndices, 
                         PyObjectPtr structStateIndices, PyObjectPtr structStateSizes );

    // The compiled Numba function
    CompiledFuncT m_compiledFunc = nullptr;

    // Active input arrays
    const void ** m_inputArgs = nullptr;
    ValidType *   m_inputValid = nullptr;
    ValidType *   m_inputTicked = nullptr;
    int64_t *     m_inputEnumStorage = nullptr;  // Storage for enum values (int64)
    size_t        m_inputCount = 0;

    // Output arrays
    void **     m_outputArgs = nullptr;
    ValidType * m_outputTicked = nullptr;
    void **     m_outputValueSlots = nullptr;  // One fixed-size by-value slot per output
    size_t      m_outputCount = 0;

    // State arrays
    void ** m_stateArgs = nullptr;
    size_t m_stateCount = 0;

    // Keep Python objects alive
    PyObjectPtr m_dataReference;
};

}  // namespace csp::python

#endif  // _IN_CSP_PYTHON_PYNUMBANODE_H
