#include <csp/engine/csp_autogen/autogen_types.h>
#include <csp/engine/OutputAdapter.h>
#include <csp/engine/BasketInfo.h>
#include <csp/engine/DynamicEngine.h>
#include <csp/engine/DynamicNode.h>

namespace csp
{

DynamicNode::DynamicNode( Engine * engine, const std::string & name,
                          std::vector<INOUT_ID_TYPE> snapIds, 
                          EngineBuilder builder,
                          NodeDef def ) :
    Node( def, engine ),
    m_engineBuilder( builder ),
    m_name( name ),
    m_snapIds( std::move( snapIds ) )
{
    //input 0 is always the dynamic basket trigger, we can create that here
    initInputBasket( 0, 0, true );
}

DynamicNode::~DynamicNode()
{
    m_instanceMap.clear();
}

void DynamicNode::start()
{
    //no need to keep csp.snap inputs active
    for( auto id : m_snapIds )
        makePassive( InputId( id ) );

    //dont need this anymore
    m_snapIds.clear();
}

void DynamicNode::stop()
{
    for( auto & instance : m_instanceMap )
        instance.second -> stop();
}

void DynamicNode::handleEvent( InputId id )
{
    //see comment in addDynamicInstance... these ids can be stub ids, we dont rely on them, skip Node::handleEvent logic
    //we do need to let dynamic basket input flow through the regular channels though
    if( id.id == ( INOUT_ID_TYPE ) -1 )
        Consumer::handleEvent( id );
    else
        Node::handleEvent( id );
}

void DynamicNode::executeImpl()
{
    DynamicInputBasketInfo * dynBasket = static_cast<DynamicInputBasketInfo *>( inputBasket( 0 ) );
    auto * shapeTs = dynBasket -> shapeTs();

    bool shapeTicked = shapeTs -> lastCycleCount() == rootEngine() -> cycleCount();
    if( shapeTicked )
    {
        auto & events = shapeTs -> lastValueTyped<autogen::DynamicBasketEvents::Ptr>() -> events();
        for( auto & event : events )
        {
            if( event -> added() )
                addDynamicInstance( event -> key() );
            //note that we handle removes AFTER the end of the dynamic cycle, the reason is two-fold
            //1) there can be cases where remove and some tick happen in the same cycle.  to be consistent with runing on endtime, we process that tick
            //2) its possible some nodes in the to-be-removed sub engine are already scheduled in the cycle step table.  it would be difficult to get them removed

            //HOWEVER there is the pitfall that a csp.attach() input will be accessed on the same cycle as it being shutdown.  It will no longer be a valid ts in this last cycle
            //since it was already removed.  In python code it still "works" because the last value is cached in a localframe var.  in c++ nodes access will raise since buffer will be reset to empty.  
            //In BOTH cases, a proper check of csp.valid() will safely return false on the removed edge
        }
    }

    //execute the cycle, all inputs should have ticked in by now
    m_cycleStepTable.executeCycle( rootEngine() -> profiler(), true );

    if( shapeTicked )
    {
        auto & events = shapeTs -> lastValueTyped<autogen::DynamicBasketEvents::Ptr>() -> events();
        for( auto & event : events )
        {
            if( !event -> added() )
                removeDynamicInstance( event -> key() );
        }
    }

    //We defer csp.stop_engine shutdowns until after the cycle because we cant shutdown / destroy 
    //the dynamic engine while the node within the dynamic which is calling csp.stop_engine is being executed!
    if( unlikely( !m_dynamicShutdowns.empty() ) )
    {
        for( auto & key : m_dynamicShutdowns )
            removeDynamicInstance( key );
        m_dynamicShutdowns.clear();
    }
}

int64_t DynamicNode::elemId( const DialectGenericType & key )
{
    auto it = m_dynamicKeyMap.find( key );
    return it == m_dynamicKeyMap.end() ? InputId::ELEM_ID_NONE : it -> second;
}

void DynamicNode::addDynamicInstance( const DialectGenericType & key )
{
    //TODO remove all the places keeping track if key mappings, move into dynamic basket impl
    int64_t keyElemId = m_dynamicElemIdMap.size();

    m_dynamicKeyMap[ key ] = keyElemId;
    m_dynamicElemIdMap.emplace_back( key );

    auto shutdownFn = [this,key]() { m_dynamicShutdowns.insert( key ); };
    auto instanceEngine = std::make_unique<DynamicEngine>( m_cycleStepTable, rootEngine(), shutdownFn );

    //build it up
    auto outputs = m_engineBuilder( this, instanceEngine.get(), key );

    //Wire up all outputs as new dynamic output keys
    assert( outputs.size() == numOutputs() );

    for( INOUT_ID_TYPE idx = 0; idx < ( INOUT_ID_TYPE ) outputs.size(); ++idx )
    {
        auto * dynBasket = static_cast<DynamicOutputBasketInfo *>( outputBasket( idx ) );
        dynBasket -> addDynamicTs( key, outputs[idx] );
    }

    {
        //monitor will keep track of all events scheduled for "now" which we want to execute immediately after start
        //otherwise they will defer to the next engine cycle.  The class destructore will then process those events
        Scheduler::DynamicEngineStartMonitor monitor( rootEngine() -> dynamicEngineStartMonitor() );
        instanceEngine -> start();
    }

    //Note that all external inputs are wired by graph building as we add them all as inputs to the dynamic node up front.
    //However, there may be external inputs that ticked the same cycle as creating this engine.  In such a case we need to 
    //force-propagate the input on startup to all nodes in this engine so that the new graph will see it ticked properly on the first cycle
    for( auto input = inputs(); input; ++input )
    {
        if( inputTicked( input.inputId() ) )
        {
            input -> propagator().apply(
                [e=instanceEngine.get()]( Consumer * consumer, const InputId & inputId )
                {
                    if( consumer -> engine() == e )
                        consumer -> handleEvent( inputId );
                } );
        }
    }


    //Need to attach all inputs created in the dynamic instance as inputs to this outter DynamicNode
    //as those inputs tick, we need to get scheduled in the root engine so that we execute the cycle step in here
    //we provide a stub id that is never intended to be used.  DynamicNode::handleEvent is overriden to ensure Node::handleEvent doesnt 
    //act on these ids
    constexpr INOUT_ID_TYPE stubId = -1;
    for( auto & input : instanceEngine -> inputAdapters() )
        const_cast<InputAdapter *>( input.get() ) -> addConsumer( this, InputId{ stubId } );

    m_instanceMap[ key ] = std::move( instanceEngine );
}

void DynamicNode::removeDynamicInstance( const DialectGenericType & key )
{
    auto it = m_instanceMap.find( key );
    //This can happen if a dynamic shuts itself down before the trigger basket removes the key
    if( it == m_instanceMap.end() )
        return;

    it -> second -> stop();

    //we need to remove consumers of anything coming in from the outter engine
    //to optimize expensive removeConsumer calls we search for all inputs that are sourced externally
    std::unordered_set<const TimeSeriesProvider *> externalTs;
    for( auto input = inputs(); input; ++input )
        externalTs.insert( input.get() );

    for( auto & node : it -> second -> nodes() )
    {
        for( auto input_iter = node -> inputs(); input_iter; ++input_iter )
        {
            if( externalTs.find( input_iter.get() ) != externalTs.end() )
                const_cast<TimeSeriesProvider *>( input_iter.get() ) -> removeConsumer( node.get(), input_iter.inputId() );
        }
    }

    for( auto & adapter : it -> second -> outputAdapters() )
    {
        for( auto input_iter = adapter -> inputs(); input_iter; ++input_iter )
        {
            if( externalTs.find( input_iter.get() ) != externalTs.end() )
                const_cast<TimeSeriesProvider *>( input_iter.get() ) -> removeConsumer( adapter.get(), input_iter.inputId() );
        }
    }

    auto removeIt = m_dynamicKeyMap.find( key );
    int64_t removeId = removeIt -> second;

    if( ( size_t) removeId != ( m_dynamicKeyMap.size() - 1 ) )
    {
        auto replaceId = m_dynamicKeyMap.size() - 1;
        auto replacedKey = m_dynamicElemIdMap[ replaceId ];
        m_dynamicKeyMap[ replacedKey ] = removeId;
        m_dynamicElemIdMap[ removeId ] = replacedKey;
    }

    for( INOUT_ID_TYPE idx = 0; idx < numOutputs(); ++idx )
    {
        auto * dynBasket = static_cast<DynamicOutputBasketInfo *>( outputBasket( idx ) );
        dynBasket -> removeDynamicKey( key, removeId );
    }

    m_dynamicKeyMap.erase( removeIt );
    m_dynamicElemIdMap.pop_back();

    //destroy Engine
    m_instanceMap.erase( it );
}

}
