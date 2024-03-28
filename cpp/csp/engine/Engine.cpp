#include <csp/engine/AdapterManager.h>
#include <csp/engine/Engine.h>
#include <csp/engine/InputAdapter.h>
#include <csp/engine/Node.h>
#include <csp/engine/GraphOutputAdapter.h>
#include <csp/engine/OutputAdapter.h>
#include <csp/engine/PushInputAdapter.h>
#include <csp/engine/RootEngine.h>
#include <signal.h>
#include <mutex>

namespace csp
{

Engine::Engine( CycleStepTable & stepTable, RootEngine * root ) : 
    m_rootEngine( root ? root : static_cast<RootEngine *>( this ) ),
    m_cycleStepTable( stepTable )
{
}

Engine::~Engine()
{
}

void Engine::registerOwnedObject( std::shared_ptr<AdapterManager> manager )
{
    //AdapterManagers may get created in a dynamic graph the first time around, but we want them shared across dynamics
    //so they always get registered and effectively owned by root engine
    if( !isRootEngine() )
        rootEngine() -> registerOwnedObject( manager );

    m_adapterManagers.emplace_back( manager ); 
}

void Engine::registerOwnedObject( std::unique_ptr<InputAdapter> adapter )
{
    m_inputAdapters.emplace_back( std::move( adapter ) );
}

void Engine::registerOwnedObject( std::unique_ptr<OutputAdapter> adapter )
{
    m_outputAdapters.emplace_back( std::move( adapter ) );
}

void Engine::registerOwnedObject( std::unique_ptr<Node> node )
{
    m_nodes.emplace_back( std::move( node ) );
}

void Engine::registerGraphOutput( const DialectGenericType & key, std::shared_ptr<GraphOutputAdapter> adapter )
{
    auto rv = m_graphOutputs.emplace( key, adapter );
    if( !rv.second )
        CSP_THROW( ValueError, "graph output key \"" << key << "\" is already bound" );

    m_graphOutputKeys.emplace_back( key );

    //If were in a dynamic engine we register the key to catch duplicates up front.  We also register in root
    //so that it gets processed at the end of the run.
    if( !isRootEngine() )
        rootEngine() -> registerGraphOutput( key, adapter );
}

csp::GraphOutputAdapter * Engine::graphOutput( const DialectGenericType & key )
{
    auto it = m_graphOutputs.find( key );
    return it == m_graphOutputs.end() ? nullptr : it -> second.get();
}

static int32_t calcRank( Consumer * consumer, std::unordered_set<Consumer *> & visited, std::vector<Consumer *> & path  )
{
    if( consumer -> rank() >= 0 )
        return consumer -> rank();

    //we dont have to check visited vs done in our cycle check since done is checked above by seeing if rank was
    //already computed.  So we only insert into visited
    auto rv = visited.insert( consumer );
    if( !rv.second )
    {
        std::stringstream oss;
        oss << "Illegal cycle found in graph, path:\n\t** ";
        for( size_t idx = path.size() - 1; idx > 0; --idx )
        {
            //flag the end of the cycle clearly in the output, which is when we first see the current node again ( unless its the first iteration )
            bool end_of_cycle = ( path[idx] == consumer ) && ( idx != path.size() - 1 );
            oss << path[idx] -> name() << ( end_of_cycle ? " ** " : "" ) << " -> ";
        }

        oss << path[0] -> name();
        CSP_THROW( RuntimeException, oss.str() );
    }

    int rank = 0;
    for( auto input_iter = consumer -> inputs(); input_iter; ++input_iter )
    {
        int input_rank = 0;
        if( input_iter -> node() &&
            //its possible that we are a dynamic engine and the input is wired from the parent engine
            //in this case we treat it as if its a 0-ranked input
            input_iter -> node() -> engine() == consumer -> engine() )
        {
            path.push_back( input_iter -> node() );
            input_rank = calcRank( input_iter -> node(), visited, path );
            path.pop_back();
        }

        rank = std::max( rank, input_rank + 1 );
    }

    consumer -> setRank( rank );
    return rank;
}

int32_t Engine::computeRanks()
{
    //We will start with output nodes and work our way back to avoid recomputing ranks repeatedly if
    //we want input -> out
    std::vector<Consumer * > roots;
    for( auto & node : m_nodes )
    {
        if( node -> numOutputs() == 0 )
            roots.emplace_back( node.get() );
    }

    for( auto & adapter : m_outputAdapters )
        roots.emplace_back( adapter.get() );

    for( auto & entry : m_graphOutputs )
    {
        //might be a dynamic engine graph output registered in root
        if( entry.second -> engine() == this )
            roots.emplace_back( entry.second.get() );
    }
    
    //these are used for cycle detection, which can happen with delayed bindings
    std::vector<Consumer *> path;
    std::unordered_set<Consumer *> visited;

    int32_t maxRank = 0;
    for( auto * consumer : roots )
    {
        //we dont include outputs in the path since its confusing to see in the cycle ( and clearly not part of it )
        path.push_back( consumer );
        maxRank = std::max( maxRank, calcRank( consumer, visited, path ) );
        path.pop_back();
    }

    return maxRank;
}

void Engine::start()
{
    int32_t maxRank = computeRanks();
    m_cycleStepTable.resize( maxRank );

    auto start = std::max( m_rootEngine -> now(), m_rootEngine -> startTime() );
    auto end   = m_rootEngine -> endTime();

    //start up managers
    for( auto & manager : m_adapterManagers )
    {
        manager -> start( start, end );
        manager -> setStarted();
    }

    //start up output adapters
    for( auto & adapter : m_outputAdapters )
    {
        adapter -> start();
        adapter -> setStarted();
    }

    for( auto & entry : m_graphOutputs )
    {
        auto & graphOutputAdapter = entry.second;
        if( graphOutputAdapter -> engine() == this )
        {
            graphOutputAdapter -> start();
            graphOutputAdapter -> setStarted();
        }
    }

    //start up input adapters
    for( auto & adapter : m_inputAdapters )
    {
        adapter -> start( start, end );
        adapter -> setStarted();
    }

    //see registerOwnedObject( AdapterManager ) above, we register adapter managers with root.  At this point we dont
    //need the list of mgrs created in a dynamic engine anymore, so we clear out the mem ( and effetively take them out of the stop() list for dynamic shutdown )
    if( !isRootEngine() )
        m_adapterManagers.clear();

    //startup nodes
    for( auto & node : m_nodes )
    {
        node -> start();
        node -> setStarted();
    }
}

void Engine::stop()
{
    // Ensure we only stop nodes/adapters that have started in the case an exception occurs during startup
    for( auto & node : m_nodes )
    {
        if( node -> started() )
            node -> stop();
    }

    for( auto & adapter : m_inputAdapters )
    {
        if( adapter -> started() )
            adapter -> stop();
    }

    for( auto & entry : m_graphOutputs )
    {
        auto & graphOutputAdapter = entry.second;
        if( graphOutputAdapter -> started() && graphOutputAdapter -> engine() == this )
            graphOutputAdapter -> stop();
    }

    for( auto & adapter : m_outputAdapters )
    {
        if( adapter -> started() )
            adapter -> stop();
    }

    for( auto & manager : m_adapterManagers )
    {
        if( manager -> started() )
            manager -> stop();
    }
}

}
