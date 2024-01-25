#include <csp/engine/Dictionary.h>
#include <csp/python/Conversions.h>
#include <csp/python/PyCppNode.h>
#include <csp/python/PyEngine.h>
#include <csp/python/PyNodeWrapper.h>

namespace csp::python
{

//not pytype is in/out value
CppNode::Shape extractShape( const char * nodeName, PyObject *& pytype )
{
    CppNode::Shape shape;
    if( PyTuple_Check( pytype ) )
    {
        PyObject * pyshape = (PyObject * ) PyTuple_GET_ITEM( pytype, 0 );
        pytype = PyTuple_GET_ITEM( pytype, 1 );

        if( PyLong_Check( pyshape ) )
        {
            auto longShape = fromPython<std::uint64_t>( pyshape );
            if(  longShape > InputId::maxBasketElements() )
                CSP_THROW( ValueError, "basket size exceeds limit of " << InputId::maxBasketElements() << " on node \"" << nodeName << "\"");

            shape = longShape;
        }
        else
        {
            if( !PyList_Check( pyshape ) )
                CSP_THROW( TypeError, "Expected basket shape as int or list, got " << Py_TYPE( pyshape ) -> tp_name );
            std::vector<std::string> keys;
            for( int i = 0; i < PyList_GET_SIZE( pyshape ); ++i )
            {
                if( !PyUnicode_Check( PyList_GET_ITEM( pyshape, i ) ) )
                    CSP_THROW( NotImplemented, "cppimpl nodes dont support non-string basket keys" );
                keys.emplace_back( fromPython<std::string>( PyList_GET_ITEM( pyshape, i ) ) );
            }
            shape = std::move( keys );
        }
    }

    return shape;
}

PyObject * pycppnode_create( PyObject * args, CppNode::Creator creatorFn )
{
    const char * nodeName;
    csp::python::PyEngine * pyengine;
    PyObject * pyinputs;
    PyObject * pyoutputs;
    PyObject * pyscalars;
    if( !PyArg_ParseTuple( args, "sO!O!O!O!",
                           &nodeName,
                           &csp::python::PyEngine::PyType, &pyengine,
                           &PyTuple_Type, &pyinputs,
                           &PyTuple_Type, &pyoutputs,
                           &PyDict_Type,  &pyscalars ) )                           
        CSP_THROW( PythonPassthrough, "" );

    csp::CppNode::NodeDef nodedef;
    const char * name;
    PyObject * pytype;
    int index;

    for( int i = 0; i < PyTuple_GET_SIZE( pyinputs ); ++i )
    {
        int isAlarm;
        if( !PyArg_ParseTuple( PyTuple_GET_ITEM( pyinputs, i ), "sOip", &name, &pytype, &index, &isAlarm ) )
            CSP_THROW( PythonPassthrough, "" );

        CppNode::Shape shape = extractShape( nodeName, pytype );

        nodedef.inputs[ name ] = { INOUT_ID_TYPE( index ), pyTypeAsCspType( pytype ), ( bool ) isAlarm, shape };
    }

    for( int i = 0; i < PyTuple_GET_SIZE( pyoutputs ); ++i )
    {
        if( !PyArg_ParseTuple( PyTuple_GET_ITEM( pyoutputs, i ), "sOi", &name, &pytype, &index ) )
            CSP_THROW( PythonPassthrough, "" );

        CppNode::Shape shape = extractShape( nodeName, pytype );
        nodedef.outputs[ name ] = { INOUT_ID_TYPE( index ), pyTypeAsCspType( pytype ), false, shape };
    }

    nodedef.scalars = fromPython<csp::Dictionary>( pyscalars );
    auto node = creatorFn( pyengine -> engine(), nodedef );
    return PyNodeWrapper::create( node );
}

}
