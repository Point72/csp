#include <csp/engine/Node.h>
#include <csp/engine/CspEnum.h>
#include <csp/engine/PartialSwitchCspType.h>
#include <csp/engine/Struct.h>
#include <csp/python/Conversions.h>
#include <csp/python/Exception.h>
#include <csp/python/InitHelper.h>
#include <csp/python/PyEngine.h>
#include <csp/python/PyNodeWrapper.h>
#include <csp/python/PyNumbaNode.h>
#include <csp/python/PyCspType.h>
#include <csp/python/PyStruct.h>

#include <algorithm>
#include <cstring>
#include <type_traits>

namespace csp::python
{

namespace
{
constexpr size_t OUTPUT_VALUE_SLOT_BYTES = sizeof( int64_t );

using NumbaTypeSwitch = PartialSwitchCspType<
    CspType::Type::BOOL,
    CspType::Type::INT64,
    CspType::Type::DOUBLE,
    CspType::Type::DATETIME,
    CspType::Type::TIMEDELTA,
    CspType::Type::ENUM,
    CspType::Type::STRUCT
>;
}

PyNumbaNode::PyNumbaNode(
    csp::Engine * engine,
    CompiledFuncT compiledFunc,
    PyObjectPtr stateVariables,
    NodeDef def,
    PyObject * dataReference
) : csp::Node( def, engine ),
    m_compiledFunc( compiledFunc ),
    m_inputArgs( def.numInputs, nullptr ),
    m_inputValid( def.numInputs, 0 ),
    m_inputTicked( def.numInputs, 0 ),
    m_inputEnumStorage( def.numInputs ),
    m_outputArgs( def.numOutputs, nullptr ),
    m_outputTicked( def.numOutputs, 0 ),
    m_outputValueSlots( def.numOutputs ),
    m_outputStructValues( def.numOutputs ),
    m_dataReference( PyObjectPtr::incref( dataReference ) )
{
    for( size_t i = 0; i < m_outputArgs.size(); ++i )
    {
        // Numba writes primitive-compatible outputs by value into these fixed-size slots.
        m_outputValueSlots[ i ] = std::make_unique<char[]>( OUTPUT_VALUE_SLOT_BYTES );
        m_outputArgs[ i ] = m_outputValueSlots[ i ].get();
    }
    initStateArrays( stateVariables );
}

void PyNumbaNode::initStateArrays( PyObjectPtr stateVariables )
{
    if( !stateVariables.ptr() || stateVariables.ptr() == Py_None || !PyTuple_Check( stateVariables.ptr() ) )
        return;

    const size_t stateCount = PyTuple_Size( stateVariables.ptr() );

    // Keep only buffers allocated here; the compiled callback owns container pointers.
    auto allocateStateBuffer = [this]( size_t size )
    {
        m_ownedStateBuffers.emplace_back( new char[ size ]() );
        return m_ownedStateBuffers.back().get();
    };

    // Initialize state slots
    for( size_t i = 0; i < stateCount; ++i )
    {
        PyObject * stateVal = PyTuple_GET_ITEM( stateVariables.ptr(), i );
        void * stateArg = nullptr;
        
        if( PyObject_TypeCheck( stateVal, &PyStruct::PyType ) )
        {
            auto * pyStruct = reinterpret_cast<PyStruct *>( stateVal );
            size_t structSize = pyStruct -> structMeta() -> size();
            char * buf = allocateStateBuffer( structSize );
            std::memcpy( buf, pyStruct -> struct_.get(), structSize );
            stateArg = buf;
        }
        else if( PyBool_Check( stateVal ) )
        {
            char * buf = allocateStateBuffer( 1 );
            *buf = stateVal == Py_True;
            stateArg = buf;
        }
        else if( PyLong_Check( stateVal ) )
        {
            // CSP enums are IntEnum instances, so their Numba representation
            // is initialized through the same int64 path as Python integers.
            int64_t stateValue = fromPython<int64_t>( stateVal );
            char * buf = allocateStateBuffer( sizeof( int64_t ) );
            *reinterpret_cast<int64_t *>( buf ) = stateValue;
            stateArg = buf;
        }
        else if( PyFloat_Check( stateVal ) )
        {
            char * buf = allocateStateBuffer( sizeof( double ) );
            *reinterpret_cast<double *>( buf ) = PyFloat_AsDouble( stateVal );
            stateArg = buf;
        }
        m_stateArgs.push_back( stateArg );
    }
}

void PyNumbaNode::invokeCompiled( LifecyclePhase phase )
{
    const int8_t status = m_compiledFunc(
        m_outputArgs.data(),
        m_outputTicked.data(),
        m_stateArgs.data(),
        static_cast<ValidType>( phase ),
        m_inputArgs.data(),
        m_inputTicked.data(),
        m_inputValid.data()
    );
    if( status != 1 )
        CSP_THROW( RuntimeException, "Compiled Numba callback failed" );
}

void PyNumbaNode::start()
{
    invokeCompiled( LifecyclePhase::Start );
}

void PyNumbaNode::stop()
{
    invokeCompiled( LifecyclePhase::Stop );
}

const char * PyNumbaNode::name() const
{
    return "PyNumbaNode";
}

void PyNumbaNode::executeImpl()
{
    const uint64_t currentCycleCount = rootEngine() -> cycleCount();

    // Refresh pointers: buffered values can move, and struct targets can change on each tick.
    for( size_t i = 0; i < m_inputArgs.size(); ++i )
    {
        const TimeSeriesProvider * ts = tsinput( i );
        if( ts )
        {
            bool valid = ts -> valid();
            m_inputValid[ i ] = valid;
            m_inputTicked[ i ] = ts -> lastCycleCount() == currentCycleCount;
            
            if( valid )
            {
                const CspType * cspType = ts -> type();
                CSP_ASSERT( cspType != nullptr );
                NumbaTypeSwitch::invoke( cspType, [this, ts, i]( auto tag )
                {
                    using CType = typename decltype( tag )::type;
                    if constexpr( std::is_same_v<CType, CspEnum> )
                    {
                        // CSP stores an enum pointer; Numba needs an int64 value that stays alive through the callback.
                        m_inputEnumStorage[ i ] = ts -> lastValueTyped<CspEnum>().value();
                        m_inputArgs[ i ] = &m_inputEnumStorage[ i ];
                    }
                    else if constexpr( std::is_same_v<CType, StructPtr> )
                    {
                        m_inputArgs[ i ] = ts -> lastValueTyped<StructPtr>().get();
                    }
                    else
                        m_inputArgs[ i ] = &( ts -> lastValueTyped<CType>() );
                } );
            }
        }
    }

    // Clear output ticked flags before calling compiled function
    std::fill( m_outputTicked.begin(), m_outputTicked.end(), static_cast<ValidType>( 0 ) );

    for( size_t i = 0; i < m_outputArgs.size(); ++i )
    {
        const CspType * type = tsoutput( i ) -> type();
        if( type -> type() == CspType::Type::STRUCT )
        {
            m_outputStructValues[ i ] = static_cast<const CspStructType *>( type ) -> meta() -> create();
            m_outputArgs[ i ] = m_outputStructValues[ i ].get();
        }
    }

    // Call the compiled Numba function
    invokeCompiled( LifecyclePhase::Execute );

    const DateTime currentTime = now();

    // Process output ticks
    for( size_t i = 0; i < m_outputArgs.size(); ++i )
    {
        if( m_outputTicked[ i ] )
        {
            TimeSeriesProvider * ts = tsoutput( i );
            if( ts )
            {
                const CspType * cspType = ts -> type();
                CSP_ASSERT( cspType != nullptr );
                NumbaTypeSwitch::invoke( cspType, [this, ts, cspType, i, currentCycleCount, currentTime]( auto tag )
                {
                    using CType = typename decltype( tag )::type;
                    if constexpr( std::is_same_v<CType, DateTime> )
                    {
                        ts -> outputTickTyped<DateTime>(
                            currentCycleCount, currentTime,
                            DateTime::fromNanoseconds( *static_cast<int64_t *>( m_outputArgs[ i ] ) )
                        );
                    }
                    else if constexpr( std::is_same_v<CType, TimeDelta> )
                    {
                        ts -> outputTickTyped<TimeDelta>(
                            currentCycleCount, currentTime,
                            TimeDelta::fromNanoseconds( *static_cast<int64_t *>( m_outputArgs[ i ] ) )
                        );
                    }
                    else if constexpr( std::is_same_v<CType, CspEnum> )
                    {
                        auto enumType = static_cast<const CspEnumType *>( cspType );
                        int64_t enumValue = *static_cast<int64_t *>( m_outputArgs[ i ] );
                        ts -> outputTickTyped<CspEnum>(
                            currentCycleCount, currentTime,
                            enumType -> meta() -> create( enumValue )
                        );
                    }
                    else if constexpr( std::is_same_v<CType, StructPtr> )
                        ts -> outputTickTyped<StructPtr>( currentCycleCount, currentTime, m_outputStructValues[ i ] );
                    else
                        ts -> outputTickTyped<CType>(
                            currentCycleCount, currentTime,
                            *static_cast<CType *>( m_outputArgs[ i ] )
                        );
                } );
            }
        }
    }
}

PyNumbaNode * PyNumbaNode::create(
    PyEngine * pyengine,
    PyObject * compiledFuncPtr,
    PyObject * inputs,
    PyObject * outputs,
    PyObject * stateVariables,
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

    return pyengine -> engine() -> createOwnedObject<PyNumbaNode>(
        funcPtr,
        PyObjectPtr::incref( stateVariables ),
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
    PyObject * dataReference;

    if( !PyArg_ParseTuple( args, "O!OOOOO",
                          &PyEngine::PyType, &engine,
                          &compiledFuncPtr,
                          &inputs,
                          &outputs,
                          &stateVariables,
                          &dataReference ) )
        CSP_THROW( PythonPassthrough, "" );
    

    auto * node = PyNumbaNode::create(
        engine,
        compiledFuncPtr,
        inputs,
        outputs,
        stateVariables,
        dataReference
    );
    return PyNodeWrapper::create( node );

    CSP_RETURN_NULL;
}

REGISTER_MODULE_METHOD( "PyNumbaNode", PyNumbaNode_create, METH_VARARGS, "Create a Numba-compiled CSP node" );

}  // namespace csp::python
