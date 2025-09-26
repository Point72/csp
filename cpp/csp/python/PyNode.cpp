#include <csp/engine/Node.h>
#include <csp/python/Common.h>
#include <csp/python/Conversions.h>
#include <csp/python/Exception.h>
#include <csp/python/InitHelper.h>
#include <csp/python/PyEngine.h>
#include <csp/python/PyBasketInputProxy.h>
#include <csp/python/PyBasketOutputProxy.h>
#include <csp/python/PyInputProxy.h>
#include <csp/python/PyOutputProxy.h>
#include <csp/python/PyCspType.h>
#include <csp/python/PyNode.h>
#include <csp/python/PyNodeWrapper.h>

#include <limits>
#include <frameobject.h>

#if !IS_PRE_PYTHON_3_11
#if !IS_PRE_PYTHON_3_13
#    define Py_BUILD_CORE 1
#endif
#include <internal/pycore_code.h>
#include <internal/pycore_frame.h>
#if !IS_PRE_PYTHON_3_13
#    undef Py_BUILD_CORE
#endif
#endif

namespace csp::python
{

static const std::string NODEREF_VAR="node_p";
static const std::string INPUT_VAR_VAR="input_var";
static const std::string INPUT_PROXY_VAR="input_proxy";
static const std::string OUTPUT_PROXY_VAR="output_proxy";

static const uint32_t ACTIVE_PCOUNT=std::numeric_limits<uint32_t>::max();

PyNode::PyNode( csp::Engine * engine, PyObjectPtr gen, PyObjectPtr inputs, PyObjectPtr outputs,
                NodeDef def ) : csp::Node( def, engine ),
                                m_gen( gen ),
                                m_localVars( nullptr ),
                                m_passiveCounts( nullptr )
{
    init( inputs, outputs );
}

PyNode::~PyNode()
{
    free( m_localVars );
    free( m_passiveCounts );
}

void PyNode::init( PyObjectPtr inputs, PyObjectPtr outputs )
{
    PyGenObject * pygen = ( PyGenObject * ) m_gen.ptr();

    //call gen first yield to setup locals
    call_gen();

    //python stack locals are laid out as
    //node proxy
    //in proxies
    //out proxies
    //in locals ( non-basket only )

    m_localVars = ( PyObject *** ) calloc( numInputs(), sizeof( PyObject ** ) );

    //printf( "Starting %s slots: %ld rank: %d\n", name(), slots, rank() );
#if IS_PRE_PYTHON_3_11
    PyCodeObject * code = ( PyCodeObject * ) pygen -> gi_code;
    Py_ssize_t numCells = PyTuple_GET_SIZE( code -> co_cellvars );
    size_t cell2argIdx = 0;
    for( int stackloc = code -> co_argcount; stackloc < code -> co_nlocals + numCells; ++stackloc )
    {
        PyObject **var = &pygen -> gi_frame -> f_localsplus[stackloc];

        bool isCell = *var && PyCell_Check( *var );
        //printf( "RBA: stack: %d var: ", stackloc );
        //PyObject_Print( *var, stdout, 0 );
        //printf( " isCell: %d\n", isCell );

        if( isCell )
        {
            //might be a scalar argument cell
            if( code -> co_cell2arg &&
                code -> co_cell2arg[ cell2argIdx++ ] != CO_CELL_NOT_AN_ARG )
                continue;
            var = &( ( ( PyCellObject * ) *var ) -> ob_ref );
        }
//PY311+ changes
#else
    _PyInterpreterFrame * frame = ( _PyInterpreterFrame * ) pygen -> gi_iframe;
#if IS_PRE_PYTHON_3_12
    PyCodeObject * code = frame -> f_code;
#else
    PyPtr<PyCodeObject> code = PyPtr<PyCodeObject>::own( PyGen_GetCode( ( PyGenObject * ) pygen ) );
#endif
    int localPlusIndex = 0;
    for( int stackloc = code -> co_argcount; stackloc < code -> co_nlocalsplus; ++stackloc, ++localPlusIndex )
    {
        PyObject **var = &frame -> localsplus[stackloc];

        auto kind = _PyLocals_GetKind(code -> co_localspluskinds, localPlusIndex );
        bool isCell = *var && PyCell_Check(*var);

        //printf( "RBA: stack: %d idx: %d var: ", stackloc, localPlusIndex );
        //PyObject_Print( *var, stdout, 0 );
        //printf( " isCell: %d kind: %x\n", isCell, kind );

        if( isCell )
            var = &( ( ( PyCellObject * ) *var ) -> ob_ref );
        else if( kind == CO_FAST_CELL )
            continue;
#endif
        //null var indicates a stack slot for a local state variable ( state hasnt initialized yet )
        //we can skip those
        if( !*var )
            continue;

        if( !PyTuple_Check( *var ) )
            CSP_THROW( TypeError, "expected tuple types in stack, got " << (*var) -> ob_type -> tp_name << " in node " << name() );

        std::string vartype = PyUnicode_AsUTF8( PyTuple_GET_ITEM( *var, 0 ) );
        int index           = fromPython<int64_t>( PyTuple_GET_ITEM( *var, 1 ) );

        if( vartype == INPUT_VAR_VAR )
        {
            CSP_ASSERT( !isInputBasket( index ) );

            m_localVars[ index ] = var;
            //These vars will be "deleted" from the python stack after start
            continue;
        }

        //decref tuple at this point its no longer needed and will be replaced
        Py_DECREF( *var );

        PyObject * newvalue = nullptr;
        if( vartype == NODEREF_VAR )
            newvalue = toPython( reinterpret_cast<uint64_t>( static_cast<csp::Node*>(this) ) );
        else if( vartype == INPUT_PROXY_VAR )
        {
            PyObject * inputType = PyTuple_GET_ITEM( inputs.ptr(), index );

            //if basket
            if( PyTuple_Check( inputType ) )
            {
                PyObject * shape = PyTuple_GET_ITEM( inputType, 0 );
                if( shape == Py_None )
                {
                    //None shape is used to flag dynamic baskets
                    initInputBasket( index, 0, true );
                    newvalue = PyDynamicBasketInputProxy::create( this, index );
                }
                else if( PyLong_Check( shape ) )
                {
                    std::uint64_t basketSize = fromPython<std::uint64_t>( shape );
                    initInputBasket( index, basketSize, false );
                    newvalue = PyListBasketInputProxy::create( this, index, basketSize );
                }
                else
                {
                    if( !PyList_Check( shape ) )
                        CSP_THROW( TypeError, "Expected input basket shape as int or list, got " << Py_TYPE( shape ) -> tp_name << " on node \"" << name() << "\"" );
                    size_t basketSize = PyList_GET_SIZE( shape );
                    initInputBasket( index, basketSize, false );
                    newvalue = PyDictBasketInputProxy::create( this, index, shape );
                }

                inputType = PyTuple_GET_ITEM( inputType, 1 );
            }
            else
                newvalue = PyInputProxy::create( this, InputId( index ) );
        }
        else if( vartype == OUTPUT_PROXY_VAR )
        {
            PyObject * outputType = PyTuple_GET_ITEM( outputs.ptr(), index );

            //if basket type
            if( PyTuple_Check( outputType ) )
            {
                PyObject * shape = (PyObject * ) PyTuple_GET_ITEM( outputType, 0 );
                outputType = PyTuple_GET_ITEM( outputType, 1 );

                if( shape == Py_None )
                {
                    //None shape is used to flag dynamic baskets
                    newvalue = PyDynamicBasketOutputProxy::create( outputType, this, index );
                }
                else if( PyLong_Check( shape ) )
                {
                  std::uint64_t basketSize = fromPython<std::uint64_t>( shape );
                    if( basketSize > OutputId::maxBasketElements() )
                        CSP_THROW( ValueError, "output basket size exceeds limit of " << OutputId::maxBasketElements() << " on node \"" << name() << "\"");
                    newvalue = PyListBasketOutputProxy::create( outputType, this, index, basketSize );
                }
                else
                {
                    if( !PyList_Check( shape ) )
                        CSP_THROW( TypeError, "Expected output basket shape as int or list, got " << Py_TYPE( shape ) -> tp_name << " on node \"" << name() << "\"" );
                    newvalue = PyDictBasketOutputProxy::create( outputType, this, index, shape );
                }
            }
            else
            {
                newvalue = PyOutputProxy::create( outputType, this, OutputId( index ) );
            }
        }
        else
            CSP_THROW( ValueError, "Unexpected var type " << vartype );

        *var = newvalue;
    }
}

void PyNode::start()
{
    //yield to execute start block, if any
    call_gen();
}

void PyNode::stop()
{
    if( this -> rootEngine() -> interrupted() && PyErr_CheckSignals() == -1 )
    {
        // When an interrupt occurs a KeyboardInterrupt exception is raised in Python, which we need to clear
        // before calling "close" on the generator. Else, the close method will fail due to the unhandled
        // exception, and we lose the state of the generator before the "finally" block that calls stop() is executed.
        PyErr_Clear();
    }

    PyObjectPtr rv = PyObjectPtr::own( PyObject_CallMethod( m_gen.ptr(), "close", nullptr ) );
    if( !rv.ptr() )
        CSP_THROW( PythonPassthrough, "" );
}

PyNode * PyNode::create( PyEngine * pyengine, PyObject * inputs, PyObject * outputs, PyObject * gen )
{
    //parse inputs/outputs, create outputs time series, etc
    Py_ssize_t numInputs = PyTuple_GET_SIZE( inputs );
    Py_ssize_t numOutputs = PyTuple_GET_SIZE( outputs );

    if( size_t( numInputs ) >= InputId::maxId() )
        CSP_THROW( ValueError, "number of inputs exceeds limit of " << InputId::maxBasketElements() );

    if( size_t( numOutputs ) > OutputId::maxId() )
        CSP_THROW( ValueError, "number of outputs exceeds limit of " << OutputId::maxBasketElements() );

    return pyengine -> engine() -> createOwnedObject<PyNode>( PyObjectPtr::incref( gen ),
                                                              PyObjectPtr::incref( inputs ),
                                                              PyObjectPtr::incref( outputs ),
                                                              NodeDef( numInputs, numOutputs ) );
}

inline bool PyNode::makeActive( InputId id )
{
    if( likely( m_passiveCounts != nullptr ) )
        m_passiveCounts[ id.id ] = ACTIVE_PCOUNT;

    return Node::makeActive( id );
}

inline bool PyNode::makePassive( InputId id )
{
    if( unlikely( m_passiveCounts == nullptr ) )
    {
        // Allocate the passive conversion count array upon first csp.make_passive call
        m_passiveCounts = ( uint32_t* ) malloc( sizeof( uint32_t ) * numInputs() );
        for( size_t i = 0; i < numInputs(); i++ )
            m_passiveCounts[ i ] = ACTIVE_PCOUNT;
    }
    m_passiveCounts[ id.id ] = input( id ) -> count();

    return Node::makePassive( id );
}

void PyNode::createAlarm( CspTypePtr & type, size_t id )
{
    //For alarms in PyNode we intentionally create them as PyObjectPtr.  We know they arent exposed
    //to other nodes so no point in converting back and forth to/from python
    this -> Node::createAlarm<PyObjectPtr>( CspType::DIALECT_GENERIC(), id );
}

const char * PyNode::name() const
{
    return PyUnicode_AsUTF8( ( ( PyGenObject * ) m_gen.ptr() ) -> gi_name );
}

void PyNode::call_gen()
{
    if( !m_gen -> ob_type -> tp_iternext( m_gen.ptr() ) )
        CSP_THROW( PythonPassthrough, "" );
}

void PyNode::executeImpl()
{
    for( size_t idx = 0; idx < numInputs(); ++idx )
    {
        if( !isInputBasket( idx ) )
        {
            // we only need to update locals for inputs that have changed since the last node call
            // thus: we convert active inputs that ticked this cycle or passive inputs that ticked since we last converted them
            auto * ts = tsinput( idx );
            auto count = ts -> count();
            bool passiveConvert = ( m_passiveCounts != nullptr && m_passiveCounts[ idx ] < count );

            if( tsinputTicked( idx ) || passiveConvert )
            {
                Py_XDECREF( *m_localVars[ idx ] );
                *m_localVars[ idx ] = lastValueToPython( ts );
                if( passiveConvert )
                    m_passiveCounts[ idx ] = count;
            }
        }
    }

    call_gen();
}

static PyObject * PyNode_create( PyObject * module, PyObject * args )
{
    CSP_BEGIN_METHOD;

    PyEngine * engine;
    PyObject * inputs;
    PyObject * outputs;
    PyObject * gen;

    if( !PyArg_ParseTuple( args, "O!O!O!O!",
                           &PyEngine::PyType, &engine,
                           &PyTuple_Type, &inputs,
                           &PyTuple_Type, &outputs,
                           &PyGen_Type, &gen ) )
        CSP_THROW( PythonPassthrough, "" );

    auto node = PyNode::create( engine, inputs, outputs, gen );
    return PyNodeWrapper::create( node );
    CSP_RETURN_NULL;
}

REGISTER_MODULE_METHOD( "PyNode", PyNode_create, METH_VARARGS, "PyNode" );

}
