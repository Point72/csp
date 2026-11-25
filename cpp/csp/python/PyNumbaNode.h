#ifndef _IN_CSP_PYTHON_PYNUMBANODE_H
#define _IN_CSP_PYTHON_PYNUMBANODE_H

#include <csp/core/Time.h>
#include <csp/engine/Node.h>
#include <csp/engine/Struct.h>
#include <csp/python/PyObjectPtr.h>
#include <Python.h>
#include <cstdint>
#include <memory>
#include <vector>

namespace csp::python
{

class PyEngine;

// Type used for ticked/valid flags - must match int8_t used in Numba generated code
using ValidType = int8_t;

// Values must match numba_cfunc_compiler.core.names.
enum class LifecyclePhase : ValidType
{
    Execute = 0,
    Start = 1,
    Stop = 2,
};

/**
 * The compiled function signature matches the source categories registered
 * by CSP (Constant, Output, State, Lifecycle, Signal):
 *
 *   int8_t(*)( void **            outputs,
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
    using CompiledFuncT = int8_t(*)(
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
        PyObjectPtr stateVariables,
        csp::NodeDef def,
        PyObject * dataReference
    );

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
        PyObject * dataReference
    );

private:
    void initStateArrays( PyObjectPtr stateVariables );
    void invokeCompiled( LifecyclePhase phase );

    // The compiled Numba function
    CompiledFuncT m_compiledFunc = nullptr;

    // Active input arrays
    std::vector<const void *> m_inputArgs;
    std::vector<ValidType> m_inputValid;
    std::vector<ValidType> m_inputTicked;
    std::vector<int64_t> m_inputEnumStorage;  // Stable int64 values passed to Numba

    // Output arrays
    std::vector<void *> m_outputArgs;
    std::vector<ValidType> m_outputTicked;
    std::vector<std::unique_ptr<char[]>> m_outputValueSlots;  // Primitive output storage
    std::vector<StructPtr> m_outputStructValues;  // Fresh native storage for each callback

    // State arrays
    std::vector<void *> m_stateArgs;
    std::vector<std::unique_ptr<char[]>> m_ownedStateBuffers;

    // Keep Python objects alive
    PyObjectPtr m_dataReference;
};

}  // namespace csp::python

#endif  // _IN_CSP_PYTHON_PYNUMBANODE_H
