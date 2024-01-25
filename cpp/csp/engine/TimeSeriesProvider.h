#ifndef _IN_CSP_ENGINE_TIMESERIESPROVIDER_H
#define _IN_CSP_ENGINE_TIMESERIESPROVIDER_H

#include <csp/core/Exception.h>
#include <csp/engine/CspType.h>
#include <csp/engine/EventPropagator.h>
#include <csp/engine/TimeSeries.h>

namespace csp
{

class Consumer;
class Node;

//Holder of a single timeseries buffer and all associated links to consumers
class TimeSeriesProvider
{
public:
    using DuplicatePolicyEnum = TimeSeries::DuplicatePolicyEnum;

    TimeSeriesProvider() : m_node( nullptr ), m_lastCycleCount( 0 ), m_type( nullptr )
    {}

    //Note type is expected to live for the lifetime of the engine, we dont hold a ref count
    void init( const CspTypePtr & type, Node * node = nullptr );
    void reset();

    void setNode( Node * node ) 
    { 
        CSP_ASSERT( m_node == nullptr );
        m_node = node; 
    }

    //We want want to find a better way to get 0-rank, node is only kept here for ranking logic seems wasteful
    Node *    node() const       { return m_node; }
    const CspType * type() const { return m_type; }

    //time series iface
    bool     valid() const          { return m_lastCycleCount > 0; }
    int32_t  numTicks() const       { return m_timeseries -> numTicks(); } //number of ticks available in buffer
    uint32_t count() const          { return m_timeseries -> count(); }
    uint64_t lastCycleCount() const { return m_lastCycleCount; }
    DateTime lastTime() const       { return m_timeseries -> lastTime(); }

    template< typename T > void outputTickTyped( uint64_t cycleCount, DateTime timestamp, const T & value, bool propagate = true );
    template< typename T > T &  reserveTickTyped( uint64_t cycleCount, DateTime timestamp );
    template< typename T > const T & lastValueTyped() const;

    DateTime                         timeAtIndex( int32_t index ) const;
    template< typename T > const T & valueAtIndex( int32_t index ) const;
    template< typename T > const TickBuffer<T> * dataline() const { return static_cast<const TimeSeriesTyped<T> * >( m_timeseries.get() ) -> dataline(); }
    const TickBuffer<DateTime> * timeline() const { return m_timeseries -> timeline(); }

    int32_t getValueIndex(DateTime time, DuplicatePolicyEnum duplicatePolicy = DuplicatePolicyEnum::LAST_VALUE ) const
    {
        return m_timeseries -> getValueIndex( time, duplicatePolicy );
    }

    int32_t   tickCountPolicy() const        { return m_timeseries -> tickCountPolicy(); }
    TimeDelta tickTimeWindowPolicy() const   { return m_timeseries -> tickTimeWindowPolicy(); }

    void setTickCountPolicy( int32_t tickCount );
    void setTickTimeWindowPolicy( TimeDelta window );

    //EventPropagator iface
    bool addConsumer( Consumer * consumer, InputId id, bool checkExists = false );
    bool removeConsumer( Consumer * consumer, InputId id );
    const EventPropagator & propagator() const { return m_propagator; }

    void propagate();

protected:

    std::unique_ptr<TimeSeries> m_timeseries;
  
private:
    template< typename T >
    void initBuffer();

    const char * name() const;

    EventPropagator m_propagator;
    Node          * m_node;  //null for input adapters, producing Node otherwise
    uint64_t        m_lastCycleCount;
    const CspType * m_type;
};

template<typename T >
inline void TimeSeriesProvider::initBuffer()
{
    m_timeseries.reset( new TimeSeriesTyped<T>() );
}

template< typename T >
inline void TimeSeriesProvider::outputTickTyped( uint64_t cycleCount, DateTime timestamp, const T & value, bool propagate_ )
{
    //should this always be checked?
    CSP_ASSERT( CspType::Type::fromCType<T>::type == m_type -> type() );

    if( m_lastCycleCount == cycleCount )
        CSP_THROW( RuntimeException, "Attempted to output twice on the same engine cycle at time " << timestamp );

    m_lastCycleCount = cycleCount;
    m_timeseries -> addTickTyped<T>( timestamp, value );

    if( propagate_ )
        propagate();
}

template< typename T >
inline T & TimeSeriesProvider::reserveTickTyped( uint64_t cycleCount, DateTime timestamp )
{
    //should this always be checked?
    CSP_ASSERT( CspType::Type::fromCType<T>::type == m_type -> type() );

    if( m_lastCycleCount == cycleCount )
        CSP_THROW( RuntimeException, name() << " attempted to output twice on the same engine cycle at time " << timestamp );

    m_lastCycleCount = cycleCount;
    propagate();
    return m_timeseries -> reserveSpaceForTick<T>( timestamp );
}

inline void TimeSeriesProvider::setTickCountPolicy( int32_t tickCount )
{
    if( tickCount > m_timeseries -> tickCountPolicy() )
        m_timeseries -> setTickCountPolicy( tickCount );
}

inline void TimeSeriesProvider::setTickTimeWindowPolicy( TimeDelta window )
{
    if( window > m_timeseries -> tickTimeWindowPolicy() )
        m_timeseries -> setTickTimeWindowPolicy( window );
}

template< typename T >
inline const T & TimeSeriesProvider::lastValueTyped() const
{
    return m_timeseries -> lastValueTyped<T>();
}

inline DateTime TimeSeriesProvider::timeAtIndex( int32_t index ) const
{
    return m_timeseries -> timeAtIndex( index );
}

template< typename T >
inline const T & TimeSeriesProvider::valueAtIndex( int32_t index ) const
{
    return m_timeseries -> valueAtIndex<T>( index );
}
    
inline void TimeSeriesProvider::propagate()
{
    m_propagator.propagate();
}
    
    
};
#endif
