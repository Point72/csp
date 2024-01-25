#include <csp/python/Conversions.h>
#include <csp/python/Exception.h>
#include <csp/python/PyInputAdapterWrapper.h>
#include <csp/python/PyCspType.h>
#include <csp/python/PyNodeWrapper.h>
#include <csp/python/PyObjectPtr.h>

namespace csp::python
{

PyNodeWrapper *PyNodeWrapper::create(csp::Node *node)
{
    PyNodeWrapper *wrapper = (PyNodeWrapper *) PyType.tp_new(&PyType, nullptr, nullptr);
    new(wrapper) PyNodeWrapper(node);
    return wrapper;
}


static PyObject *PyNodeWrapper_linkFrom(PyNodeWrapper *self, PyObject *args)
{
    CSP_BEGIN_METHOD ;

    //output_idx, output_basket_idx, node_to, input_idx, input_basket_idx
    int outputIdx, outputBasketIdx, inputIdx, inputBasketIdx;
    PyObject *source;
    if (!PyArg_ParseTuple(args, "Oiiii", &source,
                          &outputIdx, &outputBasketIdx,
                          &inputIdx, &inputBasketIdx))
        return nullptr;

    OutputId outputId(outputIdx, outputBasketIdx);
    InputId inputId(inputIdx, inputBasketIdx);

    if( PyType_IsSubtype(Py_TYPE(source), &PyNodeWrapper::PyType) )
    {
        auto * sourceNode = static_cast<PyNodeWrapper * >( source ) -> node();
        if( self -> node() -> isInputBasket( inputIdx ) && self -> node() -> inputBasket( inputIdx ) -> isDynamicBasket() )
        {
            CSP_ASSERT( sourceNode -> outputBasket( outputIdx ) -> isDynamicBasket() );

            auto * outputBasket = sourceNode -> outputBasket( outputIdx );
            static_cast<DynamicOutputBasketInfo *>( outputBasket ) -> linkInputBasket( self -> node(), inputIdx );
        }
        else
            self -> node() -> link( sourceNode -> output( outputId ), inputId );
    } 
    else if( PyType_IsSubtype(Py_TYPE(source), &PyInputAdapterWrapper::PyType) )
    {
        self -> node() -> link( static_cast<PyInputAdapterWrapper *>( source ) -> adapter(), inputId );
    } 
    else
        CSP_THROW(TypeError, "link_from expected PyNode or PyInputAdapter as source, got " << Py_TYPE( source ) -> tp_name);

    CSP_RETURN_NONE;
}

static PyObject *PyNodeWrapper_createAlarm(PyNodeWrapper *self, PyObject *args)
{
    CSP_BEGIN_METHOD ;
    int inputIdx;
    PyObject *type;
    if (!PyArg_ParseTuple(args, "iO", &inputIdx, &type))
        return nullptr;

    auto & cspType = pyTypeAsCspType( type );
    self -> node() -> createAlarm( cspType, inputIdx );
    CSP_RETURN_NONE;
}

static PyObject *PyNodeWrapper_createOutput(PyNodeWrapper *self, PyObject *args)
{
    CSP_BEGIN_METHOD ;
    int index;
    PyObject *type;
    if( !PyArg_ParseTuple( args, "iO", &index, &type ) )
        return nullptr;

    if( PyTuple_Check( type ) )
    {
        PyObject * shape = (PyObject * ) PyTuple_GET_ITEM( type, 0 );
        type = PyTuple_GET_ITEM( type, 1 );
        auto & cspType = pyTypeAsCspType( type );

        if( shape == Py_None )
        {
            self -> node() -> createDynamicBasketOutput( cspType, index );
        }
        else
        {
            //static Basket outputs
            std::uint64_t basketSize;
            if( PyLong_Check( shape ) )
                basketSize = fromPython<std::uint64_t>( shape );
            else
            {
                if( !PyList_Check( shape ) )
                    CSP_THROW( TypeError, "Expected basket shape as int or list, got " << Py_TYPE( shape ) -> tp_name );
                basketSize = PyList_GET_SIZE( shape );
            }
            self -> node() -> createBasketOutput( cspType, index, basketSize );
        }
    }
    else
    {
        auto & cspType = pyTypeAsCspType( type );
        self -> node() -> createOutput( cspType, index );
    }
        
    CSP_RETURN_NONE;
}

static PyObject *PyNodeWrapper_tp_str(PyNodeWrapper *self)
{
    CSP_BEGIN_METHOD
    return PyUnicode_FromString( self -> node() -> name() );

    CSP_RETURN_NULL
}

static PyMethodDef PyNodeWrapper_methods[] = {
        {"link_from",     (PyCFunction) PyNodeWrapper_linkFrom,     METH_VARARGS, "links node's output to target's input"},
        {"create_alarm",  (PyCFunction) PyNodeWrapper_createAlarm,  METH_VARARGS, "create a single alarm input"},
        {"create_output", (PyCFunction) PyNodeWrapper_createOutput, METH_VARARGS, "create an output"},
        {NULL}
};

PyTypeObject PyNodeWrapper::PyType = {
        PyVarObject_HEAD_INIT(NULL, 0)
        "_cspimpl.PyNodeWrapper",  /* tp_name */
        sizeof(PyNodeWrapper),     /* tp_basicsize */
        0,                         /* tp_itemsize */
        0,                         /* tp_dealloc */
        0,                         /* tp_print */
        0,                         /* tp_getattr */
        0,                         /* tp_setattr */
        0,                         /* tp_reserved */
        0,                         /* tp_repr */
        0,                         /* tp_as_number */
        0,                         /* tp_as_sequence */
        0,                         /* tp_as_mapping */
        0,                         /* tp_hash  */
        0,                         /* tp_call */
        (reprfunc) PyNodeWrapper_tp_str, /* tp_str */
        0,                         /* tp_getattro */
        0,                         /* tp_setattro */
        0,                         /* tp_as_buffer */
        Py_TPFLAGS_DEFAULT,        /* tp_flags */
        "csp node wrapper",        /* tp_doc */
        0,                         /* tp_traverse */
        0,                         /* tp_clear */
        0,                         /* tp_richcompare */
        0,                         /* tp_weaklistoffset */
        0,                         /* tp_iter */
        0,                         /* tp_iternext */
        PyNodeWrapper_methods,     /* tp_methods */
        0,                         /* tp_members */
        0,                         /* tp_getset */
        0,                         /* tp_base */
        0,                         /* tp_dict */
        0,                         /* tp_descr_get */
        0,                         /* tp_descr_set */
        0,                         /* tp_dictoffset */
        0,                         /* tp_init */
        0,                         /* tp_alloc */
        PyType_GenericNew,         /* tp_new */
        0,                         /* tp_free */ /* Low-level free-memory routine */
        0,                         /* tp_is_gc */ /* For PyObject_IS_GC */
        0,                         /* tp_bases */
        0,                         /* tp_mro */ /* method resolution order */
        0,                         /* tp_cache */
        0,                         /* tp_subclasses */
        0,                         /* tp_weaklist */
        0,                         /* tp_del */
        0,                         /* tp_version_tag */
        0                          /* tp_finalize */
};

REGISTER_TYPE_INIT(&PyNodeWrapper::PyType, "PyNodeWrapper");

}
