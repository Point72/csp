#include <csp/engine/InputId.h>
#include <csp/engine/Node.h>
#include <csp/engine/CspEnum.h>
#include <csp/engine/Struct.h>
#include <csp/python/Conversions.h>
#include <csp/python/Exception.h>
#include <csp/python/InitHelper.h>
#include <csp/python/PyEngine.h>
#include <csp/python/PyNodeWrapper.h>
#include <csp/python/PyNumbaNode.h>
#include <csp/python/PyCspType.h>
#include <csp/python/PyCspEnum.h>
#include <csp/python/PyStruct.h>

#include <algorithm>
#include <unordered_set>
#include <unordered_map>
#include <cstring>
#include <iostream>

namespace csp::python
{

namespace
{
constexpr size_t OUTPUT_VALUE_SLOT_BYTES = sizeof( int64_t );
}

PyNumbaNode::PyNumbaNode(
    csp::Engine * engine,
    CompiledFuncT compiledFunc,
    PyObjectPtr inputs,
    PyObjectPtr outputs,
    PyObjectPtr stateVariables,
    PyObjectPtr nrtStateIndices,
    PyObjectPtr structStateIndices,
    PyObjectPtr structStateSizes,
    NodeDef def,
    PyObject * dataReference
) : csp::Node( def, engine ),
    m_compiledFunc( compiledFunc ),
    m_dataReference( PyObjectPtr::incref( dataReference ) )
{
    initInputArrays( inputs );
    initOutputArrays( outputs );
    initStateArrays( stateVariables, nrtStateIndices, structStateIndices, structStateSizes );
}

PyNumbaNode::~PyNumbaNode()
{
    delete[] m_inputArgs;
    delete[] m_inputValid;
    delete[] m_inputTicked;
    delete[] m_inputEnumStorage;
    delete[] m_outputArgs;
    delete[] m_outputTicked;
    
    for( size_t i = 0; i < m_stateCount; ++i )
    {
        if( m_stateArgs[ i ] != nullptr )
            delete[] static_cast<char *>( m_stateArgs[ i ] );
    }
    delete[] m_stateArgs;
    
    // Clean up output value slots
    for( size_t i = 0; i < m_outputCount; ++i )
        delete[] static_cast<char *>( m_outputValueSlots[ i ] );
    delete[] m_outputValueSlots;
}

void PyNumbaNode::initInputArrays( PyObjectPtr inputs )
{
    m_inputCount = numInputs();
    if( m_inputCount == 0 )
    {
        m_inputArgs = nullptr;
        m_inputValid = nullptr;
        m_inputTicked = nullptr;
        m_inputEnumStorage = nullptr;
        return;
    }

    m_inputArgs = new const void *[ m_inputCount ];
    m_inputValid = new ValidType[ m_inputCount ];
    m_inputTicked = new ValidType[ m_inputCount ];
    m_inputEnumStorage = new int64_t[ m_inputCount ];

    for( size_t i = 0; i < m_inputCount; ++i )
    {
        m_inputArgs[ i ] = nullptr;
        m_inputValid[ i ] = 0;
        m_inputTicked[ i ] = 0;
        m_inputEnumStorage[ i ] = 0;
    }
}

void PyNumbaNode::initOutputArrays( PyObjectPtr outputs )
{
    m_outputCount = numOutputs();
    if( m_outputCount == 0 )
    {
        m_outputArgs = nullptr;
        m_outputTicked = nullptr;
        m_outputValueSlots = nullptr;
        return;
    }

    m_outputArgs = new void *[ m_outputCount ];
    m_outputTicked = new ValidType[ m_outputCount ];
    m_outputValueSlots = new void *[ m_outputCount ];

    for( size_t i = 0; i < m_outputCount; ++i )
    {
        m_outputTicked[ i ] = 0;
        // Numba writes primitive-compatible outputs by value into these fixed-size slots.
        m_outputValueSlots[ i ] = new char[ OUTPUT_VALUE_SLOT_BYTES ];
        std::memset( m_outputValueSlots[ i ], 0, OUTPUT_VALUE_SLOT_BYTES );
        m_outputArgs[ i ] = m_outputValueSlots[ i ];
    }
}

void PyNumbaNode::initStateArrays( PyObjectPtr stateVariables, PyObjectPtr nrtStateIndices,
                                   PyObjectPtr structStateIndices, PyObjectPtr structStateSizes )
{
    if( !stateVariables.ptr() || stateVariables.ptr() == Py_None || !PyTuple_Check( stateVariables.ptr() ) )
    {
        m_stateCount = 0;
        m_stateArgs = nullptr;
        return;
    }

    m_stateCount = PyTuple_Size( stateVariables.ptr() );
    if( m_stateCount == 0 )
    {
        m_stateArgs = nullptr;
        return;
    }

    m_stateArgs = new void *[ m_stateCount ];

    // Parse NRT state indices
    std::unordered_set<size_t> nrtSet;
    if( nrtStateIndices.ptr() && PyTuple_Check( nrtStateIndices.ptr() ) )
    {
        Py_ssize_t nrtCount = PyTuple_Size( nrtStateIndices.ptr() );
        for( Py_ssize_t i = 0; i < nrtCount; ++i )
        {
            PyObject * idx = PyTuple_GET_ITEM( nrtStateIndices.ptr(), i );
            if( PyLong_Check( idx ) )
            {
                size_t idxVal = static_cast<size_t>( PyLong_AsLongLong( idx ) );
                nrtSet.insert( idxVal );
            }
        }
    }

    // Parse struct state indices and sizes
    std::unordered_map<size_t, size_t> structSizeMap;
    if( structStateIndices.ptr() && PyTuple_Check( structStateIndices.ptr() ) &&
        structStateSizes.ptr() && PyTuple_Check( structStateSizes.ptr() ) )
    {
        Py_ssize_t structCount = PyTuple_Size( structStateIndices.ptr() );
        Py_ssize_t sizeCount = PyTuple_Size( structStateSizes.ptr() );
        if( structCount == sizeCount )
        {
            for( Py_ssize_t i = 0; i < structCount; ++i )
            {
                PyObject * idx = PyTuple_GET_ITEM( structStateIndices.ptr(), i );
                PyObject * size = PyTuple_GET_ITEM( structStateSizes.ptr(), i );
                if( PyLong_Check( idx ) && PyLong_Check( size ) )
                {
                    size_t idxVal = static_cast<size_t>( PyLong_AsLongLong( idx ) );
                    size_t sizeVal = static_cast<size_t>( PyLong_AsLongLong( size ) );
                    structSizeMap[ idxVal ] = sizeVal;
                }
            }
        }
    }

    // Initialize state slots
    for( size_t i = 0; i < m_stateCount; ++i )
    {
        PyObject * stateVal = PyTuple_GET_ITEM( stateVariables.ptr(), i );
        
        if( nrtSet.find( i ) != nrtSet.end() )
        {
            m_stateArgs[ i ] = nullptr;
        }
        else if( structSizeMap.find( i ) != structSizeMap.end() )
        {
            size_t structSize = structSizeMap[ i ];
            char * buf = new char[ structSize ];
            std::memset( buf, 0, structSize );
            
            if( PyObject_IsInstance( stateVal, ( PyObject * ) &PyStruct::PyType ) )
            {
                PyStruct * pyStruct = reinterpret_cast<PyStruct *>( stateVal );
                const void * srcData = pyStruct -> struct_.get();
                std::memcpy( buf, srcData, structSize );
            }
            
            m_stateArgs[ i ] = buf;
        }
        else if( PyObject_IsInstance( stateVal, ( PyObject * ) &PyCspEnum::PyType ) )
        {
            char * buf = new char[ 8 ];
            PyCspEnum * pyEnum = reinterpret_cast<PyCspEnum *>( stateVal );
            *reinterpret_cast<int64_t *>( buf ) = pyEnum -> enum_.value();
            m_stateArgs[ i ] = buf;
        }
        else if( PyLong_Check( stateVal ) )
        {
            char * buf = new char[ 8 ];
            *reinterpret_cast<int64_t *>( buf ) = PyLong_AsLongLong( stateVal );
            m_stateArgs[ i ] = buf;
        }
        else if( PyFloat_Check( stateVal ) )
        {
            char * buf = new char[ 8 ];
            *reinterpret_cast<double *>( buf ) = PyFloat_AsDouble( stateVal );
            m_stateArgs[ i ] = buf;
        }
        else if( PyBool_Check( stateVal ) )
        {
            char * buf = new char[ 8 ];
            *reinterpret_cast<int8_t *>( buf ) = ( stateVal == Py_True ) ? 1 : 0;
            m_stateArgs[ i ] = buf;
        }
        else
        {
            m_stateArgs[ i ] = nullptr;
        }
    }
}

void PyNumbaNode::start()
{
    m_compiledFunc(
        m_outputArgs,
        m_outputTicked,
        m_stateArgs,
        LIFECYCLE_START,
        m_inputArgs,
        m_inputTicked,
        m_inputValid
    );
}

void PyNumbaNode::stop()
{
    m_compiledFunc(
        m_outputArgs,
        m_outputTicked,
        m_stateArgs,
        LIFECYCLE_STOP,
        m_inputArgs,
        m_inputTicked,
        m_inputValid
    );
}

const char * PyNumbaNode::name() const
{
    return "PyNumbaNode";
}

void PyNumbaNode::executeImpl()
{
    const uint64_t currentCycleCount = rootEngine() -> cycleCount();

    // Update input pointers and flags
    for( size_t i = 0; i < m_inputCount; ++i )
    {
        const TimeSeriesProvider * ts = tsinput( i );
        if( ts )
        {
            bool valid = ts -> valid();
            bool ticked = ts -> lastCycleCount() == currentCycleCount;
            m_inputValid[ i ] = valid ? 1 : 0;
            m_inputTicked[ i ] = ticked ? 1 : 0;
            
            if( valid )
            {
                const CspType * cspType = ts -> type();
                CSP_ASSERT( cspType != nullptr );
                switch( cspType -> type() )
                {
                    case CspType::Type::INT64:
                        m_inputArgs[ i ] = &( ts -> lastValueTyped<int64_t>() );
                        break;
                    case CspType::Type::DOUBLE:
                        m_inputArgs[ i ] = &( ts -> lastValueTyped<double>() );
                        break;
                    case CspType::Type::BOOL:
                        m_inputArgs[ i ] = &( ts -> lastValueTyped<bool>() );
                        break;
                    case CspType::Type::DATETIME:
                        m_inputArgs[ i ] = &( ts -> lastValueTyped<DateTime>() );
                        break;
                    case CspType::Type::TIMEDELTA:
                        m_inputArgs[ i ] = &( ts -> lastValueTyped<TimeDelta>() );
                        break;
                    case CspType::Type::ENUM:
                        m_inputEnumStorage[ i ] = ts -> lastValueTyped<CspEnum>().value();
                        m_inputArgs[ i ] = &m_inputEnumStorage[ i ];
                        break;
                    case CspType::Type::STRUCT:
                        m_inputArgs[ i ] = ts -> lastValueTyped<StructPtr>().get();
                        break;
                    default:
                        break;
                }
            }
        }
    }

    // Clear output ticked flags before calling compiled function
    std::fill_n( m_outputTicked, m_outputCount, static_cast<ValidType>( 0 ) );

    // Call the compiled Numba function
    m_compiledFunc(
        m_outputArgs,
        m_outputTicked,
        m_stateArgs,
        LIFECYCLE_EXECUTE,
        m_inputArgs,
        m_inputTicked,
        m_inputValid
    );

    const DateTime currentTime = now();

    // Process output ticks
    for( size_t i = 0; i < m_outputCount; ++i )
    {
        if( m_outputTicked[ i ] )
        {
            TimeSeriesProvider * ts = tsoutput( i );
            if( ts )
            {
                const CspType * cspType = ts -> type();
                CSP_ASSERT( cspType != nullptr );
                switch( cspType -> type() )
                {
                    case CspType::Type::INT64:
                        ts -> outputTickTyped<int64_t>(
                            currentCycleCount, currentTime,
                            *static_cast<int64_t *>( m_outputValueSlots[ i ] )
                        );
                        break;
                    case CspType::Type::DOUBLE:
                        ts -> outputTickTyped<double>(
                            currentCycleCount, currentTime,
                            *static_cast<double *>( m_outputValueSlots[ i ] )
                        );
                        break;
                    case CspType::Type::BOOL:
                        ts -> outputTickTyped<bool>(
                            currentCycleCount, currentTime,
                            *static_cast<bool *>( m_outputValueSlots[ i ] )
                        );
                        break;
                    case CspType::Type::DATETIME:
                        ts -> outputTickTyped<DateTime>(
                            currentCycleCount, currentTime,
                            DateTime::fromNanoseconds( *static_cast<int64_t *>( m_outputValueSlots[ i ] ) )
                        );
                        break;
                    case CspType::Type::TIMEDELTA:
                        ts -> outputTickTyped<TimeDelta>(
                            currentCycleCount, currentTime,
                            TimeDelta::fromNanoseconds( *static_cast<int64_t *>( m_outputValueSlots[ i ] ) )
                        );
                        break;
                    case CspType::Type::ENUM:
                    {
                        auto enumType = static_cast<const CspEnumType *>( cspType );
                        int64_t enumValue = *static_cast<int64_t *>( m_outputValueSlots[ i ] );
                        CspEnum enumInstance = enumType -> meta() -> create( enumValue );
                        ts -> outputTickTyped<CspEnum>(
                            currentCycleCount, currentTime,
                            enumInstance
                        );
                        break;
                    }
                    default:
                        break;
                }
            }
        }
    }

    // Clear input ticked flags for next cycle
    std::fill_n( m_inputTicked, m_inputCount, static_cast<ValidType>( 0 ) );
}

PyNumbaNode * PyNumbaNode::create(
    PyEngine * pyengine,
    PyObject * compiledFuncPtr,
    PyObject * inputs,
    PyObject * outputs,
    PyObject * stateVariables,
    PyObject * nrtStateIndices,
    PyObject * structStateIndices,
    PyObject * structStateSizes,
    PyObject * dataReference
)
{
    // Parse compiled function pointer
    if( !PyLong_Check( compiledFuncPtr ) )
        CSP_THROW( TypeError, "compiledFuncPtr must be an integer (function pointer)" );
    
    CompiledFuncT funcPtr = reinterpret_cast<CompiledFuncT>( PyLong_AsVoidPtr( compiledFuncPtr ) );
    if( !funcPtr )
        CSP_THROW( ValueError, "compiledFuncPtr is null" );

    // Count inputs/outputs
    Py_ssize_t numInputs = 0;
    Py_ssize_t numOutputs = 0;

    if( inputs && PyTuple_Check( inputs ) )
        numInputs = PyTuple_Size( inputs );
    
    if( outputs && PyTuple_Check( outputs ) )
        numOutputs = PyTuple_Size( outputs );

    if( size_t( numInputs ) > InputId::maxBasketElements() )
        CSP_THROW( ValueError, "number of inputs exceeds limit of " << InputId::maxBasketElements() );

    if( size_t( numOutputs ) > OutputId::maxBasketElements() )
        CSP_THROW( ValueError, "number of outputs exceeds limit of " << OutputId::maxBasketElements() );

    return pyengine -> engine() -> createOwnedObject<PyNumbaNode>(
        funcPtr,
        PyObjectPtr::incref( inputs ),
        PyObjectPtr::incref( outputs ),
        PyObjectPtr::incref( stateVariables ),
        PyObjectPtr::incref( nrtStateIndices ),
        PyObjectPtr::incref( structStateIndices ),
        PyObjectPtr::incref( structStateSizes ),
        csp::NodeDef( numInputs, numOutputs ),
        dataReference
    );
}

// Python module method to create PyNumbaNode
static PyObject * PyNumbaNode_create( PyObject * module, PyObject * args )
{
    CSP_BEGIN_METHOD;

    PyEngine * engine;
    PyObject * compiledFuncPtr;
    PyObject * inputs;
    PyObject * outputs;
    PyObject * stateVariables;
    PyObject * nrtStateIndices;
    PyObject * structStateIndices;
    PyObject * structStateSizes;
    PyObject * dataReference;

    if( !PyArg_ParseTuple( args, "O!OOOOOOOO",
                          &PyEngine::PyType, &engine,
                          &compiledFuncPtr,
                          &inputs,
                          &outputs,
                          &stateVariables,
                          &nrtStateIndices,
                          &structStateIndices,
                          &structStateSizes,
                          &dataReference ) )
        CSP_THROW( PythonPassthrough, "" );
    

    auto * node = PyNumbaNode::create(
        engine,
        compiledFuncPtr,
        inputs,
        outputs,
        stateVariables,
        nrtStateIndices,
        structStateIndices,
        structStateSizes,
        dataReference
    );
    return PyNodeWrapper::create( node );

    CSP_RETURN_NULL;
}

REGISTER_MODULE_METHOD( "PyNumbaNode", PyNumbaNode_create, METH_VARARGS, "Create a Numba-compiled CSP node" );

}  // namespace csp::python
