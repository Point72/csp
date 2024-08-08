#ifndef _IN_CSP_ENGINE_ENGINE_H
#define _IN_CSP_ENGINE_ENGINE_H

#include <csp/core/Exception.h>
#include <csp/core/Time.h>
#include <csp/engine/CycleStepTable.h>
#include <csp/engine/DialectGenericType.h>
#include <memory>
#include <unordered_map>
#include <vector>

namespace csp
{

class AdapterManager;
class Consumer;
class GraphOutputAdapter;
class InputAdapter;
class Node;
class OutputAdapter;
class TimeSeriesProvider;

class Engine;
class RootEngine;

struct EngineOwned
{
    //force objects derived from EngineOwned to go through engine createOwnedObject methods
    //little trick here to force all derivations to fail
    void * operator new( size_t sz ) = delete;
    void * operator new( size_t sz, Engine * ) { return ::operator new( sz ); }

    void operator delete( void * ptr, Engine * ) { ::operator delete(ptr); }
    void operator delete( void* ptr )            { ::operator delete(ptr); }
};

class Engine
{
    using AdapterManagers   = std::vector<std::shared_ptr<AdapterManager>>;
    using InputAdapters     = std::vector<std::unique_ptr<InputAdapter>>;
    using OutputAdapters    = std::vector<std::unique_ptr<OutputAdapter>>;
    using Nodes             = std::vector<std::unique_ptr<Node>>;

public:
    Engine( CycleStepTable & stepTable,
            RootEngine * root = nullptr );
    virtual ~Engine();

    void start();
    void stop();

    const RootEngine * rootEngine() const { return m_rootEngine; }
    RootEngine * rootEngine()             { return m_rootEngine; }

    bool isRootEngine() const { return static_cast<void *>( m_rootEngine ) == this; }

    void scheduleConsumer( Consumer * consumer );

    template<typename T, typename ... ArgsT >
    T * createOwnedObject( ArgsT&&... args )
    {
        //TODO use block allocator
        T * obj = new( this ) T( this, std::forward<ArgsT>( args )... );
        registerOwnedObject( std::unique_ptr<T>( obj ) );
        return obj;
    }

    void registerOwnedObject( std::unique_ptr<Node> node );
    void registerOwnedObject( std::unique_ptr<InputAdapter> input );
    void registerOwnedObject( std::unique_ptr<OutputAdapter> output );
    void registerOwnedObject( std::shared_ptr<AdapterManager> mgr );

    const InputAdapters & inputAdapters() const   { return m_inputAdapters; }
    const OutputAdapters & outputAdapters() const { return m_outputAdapters; }
    const Nodes & nodes() const                   { return m_nodes; }

    //for results of csp.run calls.  See more detail in GraphOutputAdapter.h
    void registerGraphOutput( const DialectGenericType & key, std::shared_ptr<GraphOutputAdapter> adapter );
    csp::GraphOutputAdapter * graphOutput( const DialectGenericType & key );
    const std::vector<DialectGenericType> & graphOutputKeys() const { return m_graphOutputKeys; }

protected:

    //assign ranks, returns maxrank
    int32_t computeRanks();
    void    preRun();

    RootEngine     * m_rootEngine;
    CycleStepTable & m_cycleStepTable;

    InputAdapters    m_inputAdapters;
    OutputAdapters   m_outputAdapters;
    Nodes            m_nodes;
    AdapterManagers  m_adapterManagers;

    //graph outputs.  Consider writing an insert-ordered map so we only need one container
    std::unordered_map<DialectGenericType, std::shared_ptr<GraphOutputAdapter> > m_graphOutputs;
    std::vector<DialectGenericType>                                              m_graphOutputKeys;
};

inline void Engine::scheduleConsumer( Consumer * consumer )
{
    m_cycleStepTable.schedule( consumer );
}

};

#endif
