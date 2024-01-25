#include <csp/engine/BasketInfo.h>
#include <csp/engine/DynamicEngine.h>
#include <csp/engine/DynamicNode.h>
#include <csp/engine/GraphOutputAdapter.h>
#include <csp/python/Conversions.h>
#include <csp/python/Exception.h>
#include <csp/python/InitHelper.h>
#include <csp/python/PyEngine.h>
#include <csp/python/PyCspType.h>
#include <csp/python/PyNodeWrapper.h>

namespace csp::python
{

static PyObject * PyDynamicNode_create( PyObject * module, PyObject * args )
{
    CSP_BEGIN_METHOD;

    PyEngine * pyEngine;
    const char * name;
    PyObject * inputs;
    PyObject * outputs;
    PyObject * pyBuilder;
    PyObject * pyDynamicArgs;
    PyObject * pySubgraphArgs;

    if( !PyArg_ParseTuple( args, "O!sO!O!OO!O!",
                           &PyEngine::PyType, &pyEngine,
                           &name,
                           &PyTuple_Type, &inputs,
                           &PyTuple_Type, &outputs,
                           &pyBuilder,
                           &PyTuple_Type, &pyDynamicArgs,
                           &PyList_Type, &pySubgraphArgs) )
        CSP_THROW( PythonPassthrough, "" );

    PyObjectPtr pySubgraphArgsObj = PyObjectPtr::incref( pySubgraphArgs );
    PyObjectPtr pyBuilderObj = PyObjectPtr::incref( pyBuilder );
    PyObjectPtr pyOutputsObj = PyObjectPtr::incref( outputs );

    //vector of ts input index -> arg location to snap into
    std::vector<std::pair<size_t,int>> snapMap;
    std::vector<INOUT_ID_TYPE> snapIds; //used for makepassive call

    //args that requested snapkey
    std::vector<int> snapkeyArgs;

    //same for attach, even though we currently only allow ataching to trigger input 0
    std::vector<std::pair<size_t,int>> attachMap;

    int len = PyTuple_GET_SIZE( pyDynamicArgs );
    for( int i = 0; i < len; ++i )
    {
        PyObject * item = PyTuple_GET_ITEM( pyDynamicArgs, i );
        auto ts_idx  = fromPython<std::uint64_t>( PyTuple_GET_ITEM( item, 0 ) );
        auto arg_idx = fromPython<std::uint32_t>( PyTuple_GET_ITEM( item, 1 ) );
        const char * arg_type = PyUnicode_AsUTF8( PyTuple_GET_ITEM( item, 2 ) );
        if( strcmp( arg_type, "snap" ) == 0 )
        {
            snapMap.emplace_back( ts_idx, arg_idx );
            snapIds.emplace_back( ts_idx );
        }
        else if( strcmp( arg_type, "snapkey" ) == 0 )
            snapkeyArgs.emplace_back( arg_idx );
        else if( strcmp( arg_type, "attach" ) == 0 )
            attachMap.emplace_back( ts_idx, arg_idx );
        else
            CSP_THROW( ValueError, "Internal Error: unrecognized dynamic arg type: " << arg_type );
    }

    DynamicNode::EngineBuilder builder = [pyBuilderObj, 
                                          snapMap = std::move( snapMap ), 
                                          attachMap = std::move( attachMap ),
                                          snapkeyArgs = std::move( snapkeyArgs ),
                                          pySubgraphArgsObj, 
                                          pyOutputsObj ]( DynamicNode * pNode, DynamicEngine * engine, const DialectGenericType & newkey )
        {
            //this is the point where we adjust the "raw args" for any dynamic args

            //csp.snap
            for( auto & entry : snapMap )
            {
                auto ts_idx = entry.first;
                auto arg_idx = entry.second;

                //TODO support basket snap ?
                auto * ts = pNode -> tsinput( ts_idx );
                if( !ts -> valid() )
                    CSP_THROW( RuntimeException, "csp.snap input ( sub_graph arg " << arg_idx << " ) is not valid at time of dynamic creation on csp.dynamic node '" << pNode -> name() << "'" );
                auto * pyArg = lastValueToPython( ts );

                //decref previous arg
                Py_DECREF( PyList_GET_ITEM( pySubgraphArgsObj.get(), arg_idx ) );
                PyList_SET_ITEM( pySubgraphArgsObj.get(), arg_idx, pyArg );
            }

            //csp.snapkey
            for( auto & arg_idx : snapkeyArgs )
            {
                //we only allow snapkey on the trigger basket, so just pass along new key
                auto * pykey = toPython( newkey );
                Py_DECREF( PyList_GET_ITEM( pySubgraphArgsObj.get(), arg_idx ) );
                PyList_SET_ITEM( pySubgraphArgsObj.get(), arg_idx, pykey );
            }

            //csp.attach
            for( auto & entry : attachMap )
            {
                auto arg_idx = entry.second;
                auto edge_id = pNode -> elemId( newkey );

                PyObject * edge = PyList_GET_ITEM( pySubgraphArgsObj.get(), arg_idx );

                //python side wiring already setup the Edge for csp.attach, just need to update the elemId properly
                if( PyObject_SetAttrString( edge, "basket_idx", toPython( edge_id ) ) < 0 )
                    CSP_THROW( PythonPassthrough, "" );                
            }

            //temporary python wrapper for graph building
            PyPtr<PyEngine> pyEngineWrapper = PyPtr<PyEngine>::own( PyEngine::create( engine ) );
            PyObjectPtr rv = PyObjectPtr::check( PyObject_CallFunctionObjArgs( pyBuilderObj.get(), pyEngineWrapper.get(), pySubgraphArgsObj.get(), NULL ) );

            //collect timeseries that need to be fed back out over dynamic basket outputs
            DynamicNode::Outputs outputs;
            Py_ssize_t numOutputs = PyTuple_GET_SIZE( pyOutputsObj.get() );
            for( auto i = 0; i < numOutputs; ++i )
            {
                PyObject * key = PyTuple_GET_ITEM( pyOutputsObj.get(), i );
                outputs.emplace_back( engine -> outputTs( fromPython<std::string>( key ) ) );
            }

            return outputs;
        };

    Py_ssize_t numInputs = PyTuple_GET_SIZE( inputs );
    Py_ssize_t numOutputs = PyTuple_GET_SIZE( outputs );

    NodeDef def( numInputs, numOutputs );
    DynamicNode * node = pyEngine -> engine() -> createOwnedObject<DynamicNode>( name, snapIds, builder, def );

    return PyNodeWrapper::create( node );
    CSP_RETURN_NULL;
}

REGISTER_MODULE_METHOD( "PyDynamicNode", PyDynamicNode_create, METH_VARARGS, "PyDynamicNode" );

}
