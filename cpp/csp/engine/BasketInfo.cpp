#include <csp/engine/BasketInfo.h>
#include <csp/engine/csp_autogen/autogen_types.h>
#include <csp/engine/Node.h>

namespace csp
{

InputBasketInfo::InputBasketInfo( RootEngine * rootEngine, 
                                  size_t size, bool isDynamic ) : m_inputs( nullptr ),
                                                                  m_size( size ),
                                                                  m_lastCycleCount( 0 ),
                                                                  m_rootEngine( rootEngine ),
                                                                  m_valid( false ),
                                                                  m_isDynamic( isDynamic )
{
    //For dynamic baskets, the shape time series is stored at index -1
    //so that size / iteration still works cleanly
    auto numElements = m_size + ( isDynamic ? 1 : 0 );
    if( numElements > 0 )
    {
        m_inputs = ( TimeSeriesProvider const ** ) malloc( sizeof( TimeSeriesProvider * ) * numElements );
        memset( m_inputs, 0, sizeof( TimeSeriesProvider * ) * numElements );
        if( isDynamic )
            ++m_inputs;
    }
}

InputBasketInfo::~InputBasketInfo()
{
    if( m_isDynamic )
        --m_inputs;

    free( m_inputs );
}

int64_t DynamicInputBasketInfo::addDynamicKey( const DialectGenericType & key, const TimeSeriesProvider * ts )
{
    int64_t elemId = m_size;
    ++m_size;

    //we always add to the end of the basket, removes swap out last elem into removed slot
    if( elemId >= m_capacity )
    {
        CSP_ASSERT( elemId == m_capacity );
        m_capacity = std::max( 1, m_capacity * 2 );
        m_inputs = ( TimeSeriesProvider const ** ) realloc( m_inputs - 1, sizeof( TimeSeriesProvider * ) * ( m_capacity + 1 ) );
        ++m_inputs;
        std::fill( m_inputs + elemId, m_inputs + m_capacity, nullptr );
    }
    else
        CSP_ASSERT( !m_inputs[ elemId ] );

    m_inputs[ elemId ] = ts;
    const_cast<TimeSeriesProvider *>( ts ) -> setTickCountPolicy( m_tickCountPolicy );
    const_cast<TimeSeriesProvider *>( ts ) -> setTickTimeWindowPolicy( m_timeWindowPolicy );

    if( likely( ( bool ) m_changeCallback ) )
        m_changeCallback( key, true, elemId, -1 );
    return elemId;
}

void DynamicInputBasketInfo::removeDynamicKey( uint64_t engineCycle, const DialectGenericType & key, int64_t removeId, int64_t replaceId )
{
    CSP_ASSERT( m_size > 0 );

    //we need to update ticked items if replaceId is set ( being moved ) AND it ticked.  Note that we dont have to reset removeId
    //since we dont allow removing a key same cycle as it is ticked.
    //We check if replaceId has a tick this cycle to avoid doing a potentially expensive tickeditems scan if we dont need to
    if( replaceId != -1 && elem( replaceId ) -> lastCycleCount() == engineCycle )
    {
        //need to scan unfortunately, we had a long debate back and forth on this, we may revisit in the future
        for( auto & id : m_tickedInputs )
        {
            if( id == replaceId )
            {
                id = removeId;
                break;
            }
        }
    }

    --m_size;

    m_inputs[ removeId ] = nullptr;
    if( replaceId >= 0 )
    {
        m_inputs[ removeId ] = m_inputs[ replaceId ];
        m_inputs[ replaceId ] = nullptr;
    }

    if( likely( ( bool ) m_changeCallback ) )
        m_changeCallback( key, false, removeId, replaceId );
}


OutputBasketInfo::OutputBasketInfo( CspTypePtr & type, Node * node,
                                    size_t size, bool isDynamic ) : m_outputs( nullptr ),
                                                                    m_size( size ),
                                                                    m_isDynamic( isDynamic )
{
    if( m_size > 0 )
    {
        //for static baskets we do a single alloc so all time series are in the same memory block
        size_t allocSize = sizeof( TimeSeriesProvider * ) * m_size + sizeof( TimeSeriesProvider ) * size;
        void * mem = malloc( allocSize );
        m_outputs = ( TimeSeriesProvider * * ) mem;
        //location of first contiguous TSP
        TimeSeriesProvider * p = reinterpret_cast<TimeSeriesProvider *>( reinterpret_cast<uint8_t *>( mem ) + sizeof( TimeSeriesProvider * ) * m_size );
        for( int64_t elemId = 0; elemId < m_size; ++elemId )
        {
            new ( p ) TimeSeriesProvider();
            m_outputs[ elemId ] = p;
            m_outputs[ elemId ] -> init( type, node );
            ++p;
        }
    }
}

OutputBasketInfo::~OutputBasketInfo()
{
    for( int64_t i = 0; i < m_size; ++i )
        m_outputs[i] -> ~TimeSeriesProvider();
    free( m_outputs );
}


DynamicOutputBasketInfo::DynamicOutputBasketInfo( CspTypePtr & type, Node * node ) : OutputBasketInfo( type, node, 0, true ),
                                                                                     m_elemType( type ),
                                                                                     m_parentNode( node ),
                                                                                     m_capacity( 0 ),
                                                                                     m_ownTs( true )
{
    static CspTypePtr s_shapeType = std::make_shared<CspStructType>( csp::autogen::DynamicBasketEvent::meta() );
    m_shapeTs.init( s_shapeType, node );
}

DynamicOutputBasketInfo::~DynamicOutputBasketInfo()
{
    //we need to free indiviual timeseries here since we allocate TS's independently in dynamic baskets
    //static baskets do a single allocation for all ts
    //we may be a dynamic graph output basket, in which case we dont own the time series ( see comment in header )
    if( m_ownTs )
    {
        for( int64_t i = 0; i < m_capacity; ++i )
            delete m_outputs[i];
    }

    //set size to 0 so that ~OutputBasketInfo doesnt try to destroy providers again
    m_size = m_capacity = 0;
}

void DynamicOutputBasketInfo::linkInputBasket( Node * node, INOUT_ID_TYPE inputIdx )
{
    node -> inputBasket( inputIdx ) -> setElem( -1, &m_shapeTs );
    m_shapeTs.addConsumer( node, InputId( inputIdx ) );
}

void DynamicOutputBasketInfo::addShapeChange( const DialectGenericType & key, bool added )
{
    if( m_parentNode -> rootEngine() -> cycleCount() != m_shapeTs.lastCycleCount() )
    {
        auto events = autogen::DynamicBasketEvents::create();
        events -> set_events( {} );
        m_shapeTs.outputTickTyped<StructPtr>( m_parentNode -> rootEngine() -> cycleCount(), 
                                              m_parentNode -> rootEngine() -> now(), 
                                              events, false );
    }

    auto & events = m_shapeTs.lastValueTyped<autogen::DynamicBasketEvents::Ptr>() -> events();

    auto event = autogen::DynamicBasketEvent::create();
    event -> set_key( key );
    event -> set_added( added );

    const_cast<std::vector<autogen::DynamicBasketEvent::Ptr> &>( events ).emplace_back( event );
}

bool DynamicOutputBasketInfo::addCapacity()
{
    if( m_size == m_capacity )
    {
        m_capacity = std::max( 1, m_capacity * 2 );
        //in dynamic baskets we cant use the "single alloc" optimization we do for static baskets since
        //input baskets already refer to the TSPs by pointer ( which can be changed on realloc )
        //so we do a regular alloc of array of ptrs, then allocate each item independently
        m_outputs = ( TimeSeriesProvider ** ) realloc( m_outputs, sizeof( TimeSeriesProvider * ) * ( m_capacity ) );
        memset( m_outputs + m_size, 0, sizeof( TimeSeriesProvider * ) * ( m_capacity - m_size ) );
        return true;
    }

    return false;
}

void DynamicOutputBasketInfo::propagateAddKey( const DialectGenericType & key, const TimeSeriesProvider * ts )
{
    m_shapeTs.propagator().apply( [ts, &key]( Consumer * node, const InputId & id )
                                  {
                                      static_cast<Node *>( node ) -> addDynamicInputBasketKey( id.id, key, ts );
                                      //invoke Consumer::handleEvent not Node::handleEvent, we dont want shape tick to get into
                                      //tickedInputs of basket
                                      static_cast<Node *>( node ) -> Consumer::handleEvent( id );
                                  } );
}

int64_t DynamicOutputBasketInfo::addDynamicTs( const DialectGenericType & key, const TimeSeriesProvider * ts )
{
    assert( m_size == 0 || !m_ownTs );

    m_ownTs = false;
    addCapacity();

    int64_t elemId = m_size++;
    m_outputs[ elemId ] = const_cast<TimeSeriesProvider *>( ts );

    addShapeChange( key, true );
    propagateAddKey( key, ts );
    return elemId;

}

int64_t DynamicOutputBasketInfo::addDynamicKey( const DialectGenericType & key )
{
    assert( m_size == 0 || m_ownTs );

    m_ownTs = true;
    addCapacity();

    int64_t elemId = m_size++;

    if( !m_outputs[ elemId ] )
    {
        m_outputs[ elemId ] = new TimeSeriesProvider();
        m_outputs[ elemId ] -> init( m_elemType, m_parentNode );
    }

    addShapeChange( key, true );
    propagateAddKey( key, m_outputs[ elemId ] );
    return elemId;
}

int64_t DynamicOutputBasketInfo::removeDynamicKey( const DialectGenericType & key, int64_t elemId )
{
    CSP_ASSERT( m_size > 0 );
    int64_t replaceId = -1;

    m_outputs[ elemId ] -> reset();

    if( elemId != m_size - 1 )
    {
        std::swap( m_outputs[ elemId ], m_outputs[ m_size - 1 ] );
        replaceId = m_size - 1;

        //We need to update the input ids associated with this TS now that it was moved
        m_outputs[ elemId ] -> propagator().apply( [elemId]( Consumer *, const InputId & id )
                                                   { const_cast<InputId &>( id ).elemId = elemId; } );
    }

    --m_size;

    addShapeChange( key, false );

    m_shapeTs.propagator().apply( [&key,elemId,replaceId]( Consumer * node, const InputId & id )
                                  {
                                      static_cast<Node *>( node ) -> removeDynamicInputBasketKey( id.id, key, elemId, replaceId );
                                      static_cast<Node *>( node ) -> Consumer::handleEvent( id );
                                  } );

    return replaceId;
}

}
