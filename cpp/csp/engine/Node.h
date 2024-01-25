#ifndef _IN_CSP_ENGINE_NODE_H
#define _IN_CSP_ENGINE_NODE_H

#include <csp/core/Exception.h>
#include <csp/engine/AlarmInputAdapter.h>
#include <csp/engine/BasketInfo.h>
#include <csp/engine/Consumer.h>
#include <csp/engine/Engine.h>
#include <csp/engine/InputId.h>
#include <csp/engine/TimeSeriesProvider.h>
#include <stdint.h>

namespace csp
{

class InputAdapter;
    
struct NodeDef
{
    NodeDef( INOUT_ID_TYPE numIn, INOUT_ID_TYPE numOut ) : numInputs( numIn ), numOutputs( numOut )
    {}
    
    INOUT_ID_TYPE numInputs;
    INOUT_ID_TYPE numOutputs;
};
    
class Node : public Consumer, public EngineOwned
{
public:
    Node( NodeDef def, Engine * );
    ~Node();
    
    //data access
    INOUT_ID_TYPE numInputs() const  { return m_def.numInputs; }
    INOUT_ID_TYPE numOutputs() const { return m_def.numOutputs; }

    const TimeSeriesProvider * input( InputId id ) const;
    TimeSeriesProvider * output( OutputId id ) const;
    
    const TimeSeriesProvider * tsinput( INOUT_ID_TYPE id ) const;
    TimeSeriesProvider * tsoutput( INOUT_ID_TYPE id ) const;

    const InputBasketInfo * inputBasket( INOUT_ID_TYPE id ) const;
    InputBasketInfo * inputBasket( INOUT_ID_TYPE id );

    OutputBasketInfo * outputBasket( INOUT_ID_TYPE id ) const;

    bool isInputBasket( INOUT_ID_TYPE id ) const;

    virtual bool makeActive( InputId id );
    virtual bool makePassive( InputId id );

    bool makeBasketActive( INOUT_ID_TYPE id );
    bool makeBasketPassive( INOUT_ID_TYPE id );

    bool inputTicked( InputId id ) const;
    bool tsinputTicked( INOUT_ID_TYPE id ) const;

    virtual void createAlarm( CspTypePtr & type, size_t id );

    void createOutput( CspTypePtr & type, size_t id );
    void createBasketOutput( CspTypePtr & type, size_t id, size_t size );
    void createDynamicBasketOutput( CspTypePtr & type, size_t id );

    void initInputBasket( size_t id, size_t size, bool dynamicBasket );

    void link( TimeSeriesProvider * input, InputId inputId );

    input_iterator inputs() const override { return input_iterator( m_inputs, m_def.numInputs ); }

    void handleEvent( InputId id ) override;

    void addDynamicInputBasketKey( INOUT_ID_TYPE basketIdx, const DialectGenericType & key, const TimeSeriesProvider * ts );
    void removeDynamicInputBasketKey( INOUT_ID_TYPE basketIdx, const DialectGenericType & key, int64_t elemId, int64_t replaceId );

protected:
    template<typename T>
    void createAlarm( CspTypePtr & type, size_t id );

private:
    using TimeSeriesInputBasketInfo  = TaggedPointerUnion<TimeSeriesProvider,InputBasketInfo>;
    using TimeSeriesOutputBasketInfo = TaggedPointerUnion<TimeSeriesProvider,OutputBasketInfo>;

    using DirtyInputBaskets = std::vector<INOUT_ID_TYPE>;

    void validateInputIndex( size_t id );
    void validateOutputIndex( size_t id );
    void validateInputBasketSize( size_t id, size_t size );
    void validateOutputBasketSize( size_t id, size_t size );

    //TimeSeriesProviders not owned, baskets are owned
    TimeSeriesInputBasketInfo * m_inputs;

    //owned
    TimeSeriesOutputBasketInfo * m_outputs;
    NodeDef  m_def;
};

inline void Node::createOutput( CspTypePtr & type, size_t id )
{
    CSP_ASSERT( !m_outputs[ id ] );
    validateOutputIndex( id );
    auto * ts = new TimeSeriesProvider();
    ts -> init( type, this );
    m_outputs[ id ].set( ts );
}

inline void Node::createBasketOutput( CspTypePtr & type, size_t id, size_t size )
{
    validateOutputIndex( id );
    validateOutputBasketSize( id, size );

    CSP_ASSERT( !m_outputs[ id ] );
    auto * basket = new OutputBasketInfo( type, this, size );
    m_outputs[ id ].set( basket );
}

inline void Node::createDynamicBasketOutput( CspTypePtr & type, size_t id )
{
    validateOutputIndex( id );
    CSP_ASSERT( !m_outputs[ id ] );
    OutputBasketInfo * basket = new DynamicOutputBasketInfo( type, this );
    m_outputs[ id ].set( basket );
}

template<typename T>
inline void Node::createAlarm( CspTypePtr & type, size_t id )
{
    validateInputIndex( id );
    auto alarm = engine() -> createOwnedObject<AlarmInputAdapter<T>>( type );
    link( alarm, InputId( id ) );
}

inline const TimeSeriesProvider * Node::input( InputId id ) const
{
    return m_inputs[ id.id ].isSet<TimeSeriesProvider>() ? tsinput( id.id ) : inputBasket( id.id ) -> elem( id.elemId );
}

inline TimeSeriesProvider * Node::output( OutputId id ) const
{
    return m_outputs[ id.id ].isSet<TimeSeriesProvider>() ? tsoutput( id.id ) : outputBasket( id.id ) -> elem( id.elemId );
}   

inline const TimeSeriesProvider * Node::tsinput( INOUT_ID_TYPE id ) const
{
    CSP_ASSERT( m_inputs[ id ].isSet<TimeSeriesProvider>() );
    return m_inputs[ id ].get<TimeSeriesProvider>();
}

inline TimeSeriesProvider * Node::tsoutput( INOUT_ID_TYPE id ) const
{
    CSP_ASSERT( m_outputs[ id ].isSet<TimeSeriesProvider>() );
    return m_outputs[ id ].get<TimeSeriesProvider>();
}

inline bool Node::isInputBasket( INOUT_ID_TYPE id ) const
{
    return m_inputs[ id ].isSet<InputBasketInfo>();
}

inline const InputBasketInfo * Node::inputBasket( INOUT_ID_TYPE id ) const
{
    return const_cast<Node*>( this ) -> inputBasket( id );
}

inline InputBasketInfo * Node::inputBasket( INOUT_ID_TYPE id )
{
    CSP_ASSERT( m_inputs[ id ].isSet<InputBasketInfo>() );
    return m_inputs[ id ].get<InputBasketInfo>();
}

inline OutputBasketInfo * Node::outputBasket( INOUT_ID_TYPE id ) const
{
    CSP_ASSERT( m_outputs[ id ].isSet<OutputBasketInfo>() );
    return m_outputs[ id ].get<OutputBasketInfo>();
}

inline bool Node::makeActive( InputId id )
{
    auto * ts = input( id );
    return const_cast<TimeSeriesProvider *>( ts ) -> addConsumer( this, id, true );
}

inline bool Node::makePassive( InputId id )
{
    auto * ts = input( id );
    return const_cast<TimeSeriesProvider *>( ts ) -> removeConsumer( this, id );
}

inline bool Node::makeBasketActive( INOUT_ID_TYPE id )
{
    bool rv = false;
    auto * basketInfo = inputBasket( id );
    for( auto it = basketInfo -> begin_inputs(); it; ++it )
        rv |= const_cast<TimeSeriesProvider *>( it.get() ) -> addConsumer( this, InputId( id, it.elemId() ), true );
    return rv;
}

inline bool Node::makeBasketPassive( INOUT_ID_TYPE id )
{
    bool rv = false;
    auto * basketInfo = inputBasket( id );
    for( auto it = basketInfo -> begin_inputs(); it; ++it )
        rv |= const_cast<TimeSeriesProvider *>( it.get() ) -> removeConsumer( this, InputId( id, it.elemId() ) );
    return rv;
}

inline bool Node::inputTicked( InputId id ) const
{
    return input( id ) -> lastCycleCount() == rootEngine() -> cycleCount();
}

inline bool Node::tsinputTicked( INOUT_ID_TYPE id ) const
{
    return tsinput( id ) -> lastCycleCount() == rootEngine() -> cycleCount();
}

inline void Node::handleEvent( InputId id )
{
    if( isInputBasket( id.id ) )
        inputBasket( id.id ) -> handleEvent( id.elemId );

    Consumer::handleEvent( id );
}

inline void Node::addDynamicInputBasketKey( INOUT_ID_TYPE basketIdx, const DialectGenericType & key, const TimeSeriesProvider * ts )
{
    CSP_ASSERT( isInputBasket( basketIdx ) );
    CSP_ASSERT( inputBasket( basketIdx ) -> isDynamicBasket() );
    auto * dynBasket = static_cast<DynamicInputBasketInfo*>( inputBasket( basketIdx ) );

    size_t elemId = dynBasket -> addDynamicKey( key, ts );
    if( elemId > InputId::maxElemId() )
        CSP_THROW( RangeError, "Hit dynamic key limit of " << InputId::maxElemId() );
    const_cast<TimeSeriesProvider *>( ts ) -> addConsumer( this, InputId( basketIdx, elemId ) );
}

inline void Node::removeDynamicInputBasketKey( INOUT_ID_TYPE basketIdx, const DialectGenericType & key, int64_t elemId, int64_t replaceId )
{
    CSP_ASSERT( isInputBasket( basketIdx ) );
    CSP_ASSERT( inputBasket( basketIdx ) -> isDynamicBasket() );
    auto * dynBasket = static_cast<DynamicInputBasketInfo*>( inputBasket( basketIdx ) );

    dynBasket -> removeDynamicKey( rootEngine() -> cycleCount(), key, elemId, replaceId );
    //note we dont need to removeConsumer since the removed ts is no longer valid
}

};

#endif
