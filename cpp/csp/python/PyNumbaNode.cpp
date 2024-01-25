#include <csp/engine/InputId.h>
#include <csp/engine/Node.h>
#include <csp/python/Conversions.h>
#include <csp/python/Exception.h>
#include <csp/python/InitHelper.h>
#include <csp/python/PyEngine.h>
#include <csp/python/PyInputAdapterWrapper.h>
#include <csp/python/PyBasketInputProxy.h>
#include <csp/python/PyInputProxy.h>
#include <csp/python/PyOutputProxy.h>
#include <csp/python/PyCspType.h>
#include <csp/python/PyNode.h>
#include <csp/python/PyNodeWrapper.h>
#include <csp/python/PyNumbaNode.h>

#include <frameobject.h>


#ifdef DEBUG_NUMBA_NODE
#include <iostream>
#define DEBUG_ONLY(v) do {v;} while(false)
#else
#define DEBUG_ONLY(v) do {} while(false)
#endif

#define DEBUG_PRINT(v) DEBUG_ONLY(std::cout << v << std::endl)
#define DEBUG_PRINT_WITH_LOC(v) DEBUG_PRINT(__FILE__ << ':' << __LINE__ << ": " << v)


namespace csp::python
{
PyNumbaNode::PyNumbaNode(csp::Engine *engine,
                         void *stateObject, CallbackType numbaInitCallback, CallbackType numbaImplCallback,
                         PyObjectPtr inputs,
                         PyObjectPtr outputs,
                         NodeDef def,
                         PyObject *dataReference) : csp::Node(def, engine),
                                                    m_stateObject(stateObject),
                                                    m_numbaInitCallback(numbaInitCallback),
                                                    m_numbaImplCallback(numbaImplCallback),
                                                    m_dataReference(PyObjectPtr::incref(dataReference))
{
    DEBUG_PRINT_WITH_LOC("Created numba node: " << (int64_t) (this) << " stateObject=" << (int64_t) (stateObject)
                                                << " numbaImplCallback="
                                                << (int64_t) numbaImplCallback);
    init(inputs, outputs);
}

PyNumbaNode::~PyNumbaNode()
{
}

void PyNumbaNode::init(PyObjectPtr inputs, PyObjectPtr outputs)
{

    for (int input_idx = 0; input_idx < numInputs(); ++input_idx)
    {
        PyObject * inputType = PyTuple_GET_ITEM(inputs.ptr(), input_idx);

        //if basket
        if (PyTuple_Check(inputType))
        {
            PyObject * shape = PyTuple_GET_ITEM(inputType, 0);
            std::uint64_t basketSize;
            if (PyLong_Check(shape))
                basketSize = fromPython<std::uint64_t>(shape);
            else
            {
                if (!PyList_Check(shape))
                    CSP_THROW(TypeError, "Expected basket type as int or list, got " << Py_TYPE(shape)->tp_name);
                basketSize = PyList_GET_SIZE(shape);
            }

            initInputBasket(input_idx, basketSize, false);
        }
    }
}

void PyNumbaNode::start()
{
    m_numbaInitCallback(this, m_stateObject);
}

void PyNumbaNode::stop()
{
//        PyObjectPtr rv = PyObjectPtr::own(PyObject_CallMethod(m_gen.ptr(), "close", nullptr));
//        if (!rv.ptr())
//            CSP_THROW(PythonPassthrough, "");
}

PyNumbaNode *PyNumbaNode::create(PyEngine *pyengine, PyObject *inputs, PyObject *outputs,
                                 PyObject *stateObject, PyObject *numbaInitCallback, PyObject *numbaImplCallback,
                                 PyObject *dataReference)
{
    //parse inputs/outputs, create outputs time series, etc
    Py_ssize_t numInputs = PyTuple_GET_SIZE(inputs);
    Py_ssize_t numOutputs = PyTuple_GET_SIZE(outputs);

    static_assert(sizeof(void *) == sizeof(int64_t));

    void *stateObjectPtr = reinterpret_cast<void *>(fromPython<int64_t>(stateObject));
    CallbackType numbaInitCallbackPtr = reinterpret_cast<CallbackType>(fromPython<int64_t>(numbaInitCallback));
    CallbackType numbaImplCallbackPtr = reinterpret_cast<CallbackType>(fromPython<int64_t>(numbaImplCallback));


    if( size_t( numInputs ) > InputId::maxBasketElements() )
        CSP_THROW( ValueError, "number of inputs exceeds limit of " << InputId::maxBasketElements() );

    if( size_t( numOutputs ) > OutputId::maxBasketElements() )
        CSP_THROW( ValueError, "number of outputs exceeds limit of " << OutputId::maxBasketElements() );


    return pyengine->engine()->createOwnedObject<PyNumbaNode>(stateObjectPtr, numbaInitCallbackPtr,
                                                              numbaImplCallbackPtr, PyObjectPtr::incref(inputs),
                                                              PyObjectPtr::incref(outputs),
                                                              NodeDef(numInputs, numOutputs), dataReference);
}

const char *PyNumbaNode::name() const
{
    // TODO: propagate name from python here?
    return "PyNumbaNode";
}

void PyNumbaNode::call_callback()
{
    m_numbaImplCallback(this, m_stateObject);
//        if (!m_gen->ob_type->tp_iternext(m_gen.ptr()))
//            CSP_THROW(PythonPassthrough, "");
}

void PyNumbaNode::executeImpl()
{
    call_callback();
}

static PyObject *PyNumbaNode_create(PyObject * module, PyObject * args)
{
    CSP_BEGIN_METHOD ;

        PyEngine *engine;
        PyObject * inputs;
        PyObject * outputs;
        PyObject * state;
        PyObject * initCallback;
        PyObject * implCallback;
        PyObject * dataReference;

        if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!O!",
                              &PyEngine::PyType, &engine,
                              &PyTuple_Type, &inputs,
                              &PyTuple_Type, &outputs,
                              &PyLong_Type, &state,
                              &PyLong_Type, &initCallback,
                              &PyLong_Type, &implCallback,
                              &PyTuple_Type, &dataReference))
            CSP_THROW(PythonPassthrough, "");

        auto node = PyNumbaNode::create(engine, inputs, outputs, state, initCallback, implCallback, dataReference);
        return PyNodeWrapper::create(node);
    CSP_RETURN_NULL;
}

REGISTER_MODULE_METHOD("PyNumbaNode", PyNumbaNode_create, METH_VARARGS, "PyNumbaNode");

}

#define IMPLEMENT_TYPE_GETTERS_AND_SETTERS(typ_name, c_typ_name)                                      \
c_typ_name __csp_get_node_ ## typ_name ## _value__(int64_t node_ptr, int32_t input_index)             \
{                                                                                                     \
auto node = reinterpret_cast<csp::python::PyNumbaNode *>(node_ptr);                                   \
auto res = node->tsinput(input_index)->lastValueTyped<c_typ_name>();                                  \
return res;                                                                                           \
}                                                                                                     \
void __csp_return_ ## typ_name ## _value__(int64_t node_ptr, int32_t output_index, c_typ_name value)  \
{                                                                                                     \
    auto node = reinterpret_cast<csp::python::PyNumbaNode *>(node_ptr);                               \
    node->tsoutput(output_index)->outputTickTyped(node->cycleCount(), node->now(), value);            \
}

namespace
{
inline csp::python::PyNumbaNode *getNodeFromInt(int64_t node_ptr)
{
    return reinterpret_cast<csp::python::PyNumbaNode *>(node_ptr);
}


inline csp::TimeSeriesProvider *getNonConstInput(csp::python::PyNumbaNode *node, int32_t input_index)
{
    return const_cast<csp::TimeSeriesProvider *>(node->tsinput(input_index));
}

}


extern "C" {

int64_t __csp_create_datetime_nanoseconds__(int32_t year, int32_t month, int32_t day,
                                            int32_t hour, int32_t minute, int32_t second,
                                            int32_t nanosecond)
{
    return csp::DateTime(year, month, day,
                         hour, minute, second,
                         nanosecond).asNanoseconds();
}

bool __csp_numba_node_ticked__(int64_t node_ptr, int32_t input_index)
{
    auto node = getNodeFromInt(node_ptr);
    return node->inputTicked(input_index);
}

bool __csp_numba_node_valid__(int64_t node_ptr, int32_t input_index)
{
    auto node = getNodeFromInt(node_ptr);
    return node->tsinput(input_index)->valid();
}

bool __csp_numba_node_make_passive__(int64_t node_ptr, int32_t input_index)
{
    auto node = getNodeFromInt(node_ptr);
    return getNonConstInput(node, input_index)->removeConsumer(node, csp::InputId(input_index));
}

bool __csp_numba_node_make_active__(int64_t node_ptr, int32_t input_index)
{
    auto node = getNodeFromInt(node_ptr);
    return getNonConstInput(node, input_index)->addConsumer(node, csp::InputId(input_index), true);
}


IMPLEMENT_TYPE_GETTERS_AND_SETTERS(int64, int64_t)
IMPLEMENT_TYPE_GETTERS_AND_SETTERS(double, double)
IMPLEMENT_TYPE_GETTERS_AND_SETTERS(bool, bool)

}
