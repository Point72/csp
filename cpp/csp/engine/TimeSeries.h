#ifndef _IN_CSP_ENGINE_TIMESERIES_H
#define _IN_CSP_ENGINE_TIMESERIES_H

#include <csp/core/Enum.h>
#include <csp/core/Exception.h>
#include <csp/core/Time.h>
#include <csp/engine/TickBuffer.h>

namespace csp
{

template<typename T>
class TickBufferAccess
{
public:
    TickBufferAccess() : m_buffer( nullptr ), m_lastValue() { }

    ~TickBufferAccess()
    {
        delete m_buffer;
    }

    T & valueAtIndex( uint32_t index )
    {
        if( likely( m_buffer != nullptr ) )
            return ( *m_buffer )[index];
        if( unlikely( index != 0 ) )
            CSP_THROW( RangeError, "Accessing value past index 0 when no buffering policy is set" );
        return m_lastValue;
    }

    const T & valueAtIndex( uint32_t index ) const
    {
        return const_cast<TickBufferAccess<T> *>( this ) -> valueAtIndex( index );
    }

    void setBuffer( uint32_t capacity, bool swap )
    {
        m_buffer = new TickBuffer<T>( capacity );
        if( unlikely( swap ) )
            m_buffer -> push_back( m_lastValue );
    }

    T & value()                          { return m_lastValue; }
    const T & value() const              { return const_cast<TickBufferAccess<T> *>( this ) -> value(); }
    uint32_t numTicks() const            { return ( unlikely( m_buffer != nullptr ) ? m_buffer -> numTicks() : 1 ); }
    TickBuffer<T> * buffer()             { return m_buffer; }
    const TickBuffer<T> * buffer() const { return const_cast<TickBufferAccess<T> *>( this ) -> buffer(); }

    void setValue( const T & val )       { m_lastValue = val; }
    void reset()                         { if( unlikely( m_buffer != nullptr ) ) m_buffer -> clear(); }

private:
    TickBuffer<T> * m_buffer;
    T m_lastValue;
};

class TimeSeries
{
public:
    // Should match the DuplicatePolicy enum in python!!!
    struct DuplicatePolicyTraits {
        enum _enum : uint8_t {
            UNKNOWN = 0,
            LAST_VALUE = 1,
            FIRST_VALUE = 2,
            ALL_VALUES = 3,
            NUM_TYPES
        };

    protected:
        _enum m_value;
    };

    using DuplicatePolicyEnum = csp::Enum<DuplicatePolicyTraits>;

public:
    TimeSeries();
    virtual ~TimeSeries() 
    {
    }

    void reset();

    uint32_t numTicks() const                    { return ( m_count > 0 ? m_timeline.numTicks() : 0 ); }
    uint32_t count() const                       { return m_count; }
    DateTime lastTime() const                    { return unlikely( m_timeline.buffer() != nullptr ) ? timeAtIndex( 0 ) : m_timeline.value(); }
    DateTime timeAtIndex( uint32_t index ) const;

    //reserveSpaceForTick exposed for use by BURST
    template< typename T > T & reserveSpaceForTick( DateTime timestamp );

    template< typename T > void addTickTyped( DateTime timestamp, const T & value );
    template< typename T > const T & lastValueTyped() const;
    template< typename T > const T & valueAtIndex( int32_t index ) const;
    const TickBuffer<DateTime> * timeline() const { return m_timeline.buffer(); }

    template< typename T > T & lastValueTyped();

    /**
     * @param time - The time index to look for in the
     * @param duplicatePolicy - Policy for handling duplicate time stamps
     * @return The index that matches the given timestamp with the given policy. The returned index is non negative,
     * and less then count(). -1 is returned if no value is found.
     * Example:
     *  Assume the following values: (09:30, 0), (09:31, 1), (09:31, 2), (09:33, 3)
     *  The following calls will return (roughly the syntax is incorrect, it's just to demonstate symantics):
     *      getValueIndex(09:29) // Returns -1 since it's before the first timestamp
     *      getValueIndex(09:30) // Returns 0
     *      getValueIndex(09:31) // Returns 2 since it's the last index for the timestamp
     *      getValueIndex(09:31, DuplicatePolicyEnum::FIRST_VALUE) // Returns 1 since it's the first value for timestamp
     *      getValueIndex(09:32, DuplicatePolicyEnum::FIRST_VALUE) // Returns 2, NOTE this is slightly counter intuitive,
     *                                                             // we return the first value 09:32 that is last at
     *                                                             // preceding the given timestamp!
     *      getValueIndex(09:35) // Returns 3 as this is the last index before the given timestamp
     */
    int32_t getValueIndex(DateTime time,
                          DuplicatePolicyEnum duplicatePolicy = DuplicatePolicyEnum::LAST_VALUE) const;

    /**
     * Same as getValueIndexRange, the difference is that it supports ALL_VALUES as duplicate policy in which case
     * the whole range is returned (both start and end indices are inclusive)
     * @param time
     * @param duplicatePolicy
     * @return
     */
    std::pair<int32_t, int32_t> getValueIndexRange(
            DateTime time, DuplicatePolicyEnum duplicatePolicy = DuplicatePolicyEnum::LAST_VALUE) const;

    
    virtual void setTickCountPolicy( int32_t tickCount ) = 0;
    virtual void setTickTimeWindowPolicy( TimeDelta window ) = 0;

    int32_t   tickCountPolicy() const      { return m_bufferTickCountPolicy; }
    TimeDelta tickTimeWindowPolicy() const { return m_bufferTimeWindowPolicy; }
    
protected:
    int32_t    m_bufferTickCountPolicy;
    uint32_t   m_count;
    TimeDelta  m_bufferTimeWindowPolicy;

    TickBufferAccess<DateTime>  m_timeline;
};

template< typename T >
class TimeSeriesTyped : public TimeSeries
{
public:
    TimeSeriesTyped() : m_dataline()
    {
    }

    ~TimeSeriesTyped()
    {
    }

    void reset()
    {
        TimeSeries::reset();
        m_dataline.reset();
        // TimeSeries is responsible for tracking the count
    }

    T & reserveSpaceForTick( DateTime timestamp )
    {
        m_count++;
        auto * timeBuffer = m_timeline.buffer();
        if( likely( !timeBuffer ) )
        {
            m_timeline.setValue( timestamp );
            return m_dataline.value();
        }

        auto * dataBuffer = m_dataline.buffer();
        if( unlikely( !m_bufferTimeWindowPolicy.isNone() && timeBuffer -> full() ) )
        {
            auto diff = timestamp - timeBuffer -> valueAtIndex( timeBuffer -> capacity() - 1 );
            if( unlikely( diff <= m_bufferTimeWindowPolicy ) )
            {
                //consider new size... if we double we will end up storing more than requested
                //if we increase perfectly, it might get expensive quickly.  Will go for accuracy in first attempt
                auto cur_capacity = timeBuffer -> capacity();
                auto newCapacity = cur_capacity > 0 ? cur_capacity * 2 : 1;
                timeBuffer -> growBuffer( newCapacity );
                dataBuffer -> growBuffer( newCapacity );
            }
        }

        timeBuffer -> push_back( timestamp );
        return dataBuffer -> prepare_write();
    }
    
    void addTick( DateTime timestamp, const T & value )
    {
        reserveSpaceForTick( timestamp ) = value;
    }

    T & lastValue()
    {
        return unlikely( m_dataline.buffer() != nullptr ) ? valueAtIndex( 0 ) : m_dataline.value();
    }

    T & valueAtIndex( int32_t index )
    {
        return m_dataline.valueAtIndex( index );
    }

    const T & lastValue() const                   { return const_cast<TimeSeriesTyped<T>*>( this ) -> lastValue(); }
    const T & valueAtIndex( int32_t index ) const { return const_cast<TimeSeriesTyped<T>*>( this ) -> valueAtIndex( index ); }
    const TickBuffer<T> * dataline() const        { return m_dataline.buffer(); }
    
    void setTickCountPolicy( int32_t tickCount )
    {
        if( tickCount > 1 )
        {
            if( likely( !m_timeline.buffer() ) )
                initializeBuffers( tickCount );
            else
            {
                m_timeline.buffer() -> growBuffer( tickCount );
                m_dataline.buffer() -> growBuffer( tickCount );
            }
            m_bufferTickCountPolicy = tickCount;
        }
    }

    void setTickTimeWindowPolicy( TimeDelta window )
    {
        if( likely( !m_timeline.buffer() ) )
            initializeBuffers( 1 );
        m_bufferTimeWindowPolicy = window;
    }

    void initializeBuffers( uint32_t capacity )
    {
        // Allocate tick buffers only on an as-needed basis
        m_timeline.setBuffer( capacity, ( bool )m_count );
        m_dataline.setBuffer( capacity, ( bool )m_count );
    }
    
private:
    /* 
        We only need to use a tick buffer if the buffering policy is greater than 1 or time-based.
        Since most of the time we only use the last/current value, don't bother allocating the tick buffer until 
        we know we actually need it, and instead just store the value in TickBufferAccess.
    */
    TickBufferAccess<T>  m_dataline;
};

/*
TimeSeries
*/

inline TimeSeries::TimeSeries() : 
    m_bufferTickCountPolicy( 1 ),
    m_count( 0 )
{
}

inline void TimeSeries::reset()
{
    m_bufferTickCountPolicy = 1;
    m_bufferTimeWindowPolicy = TimeDelta();
    m_count = 0;
    m_timeline.reset();
}

template< typename T >
inline T & TimeSeries::reserveSpaceForTick( DateTime timestamp )
{
    return static_cast<TimeSeriesTyped<T> * >( this ) -> reserveSpaceForTick( timestamp );
}

template< typename T >
inline void TimeSeries::addTickTyped( DateTime timestamp, const T & value )
{
    static_cast<TimeSeriesTyped<T> * >( this ) -> addTick( timestamp, value );
}

template< typename T >
inline const T & TimeSeries::lastValueTyped() const
{
    return const_cast<TimeSeries *>( this ) -> lastValueTyped<T>();
}

template< typename T >
inline T & TimeSeries::lastValueTyped()
{
    return static_cast<TimeSeriesTyped<T> * >( this ) -> lastValue();
}

template< typename T >
inline const T & TimeSeries::valueAtIndex( int32_t index ) const
{
    return static_cast<const TimeSeriesTyped<T> * >( this ) -> valueAtIndex( index );
}

inline DateTime TimeSeries::timeAtIndex( uint32_t index ) const
{ 
    return m_timeline.valueAtIndex( index );
}

inline int32_t TimeSeries::getValueIndex(DateTime time, DuplicatePolicyEnum duplicatePolicy) const {
    // For all values the range version of the function must be called
    CSP_ENSURE_TRUE(duplicatePolicy!= DuplicatePolicyEnum::ALL_VALUES);
    auto range = getValueIndexRange(time, duplicatePolicy);
    switch(duplicatePolicy){
        case DuplicatePolicyTraits::FIRST_VALUE:
            return range.second;
        case DuplicatePolicyTraits::LAST_VALUE:
            return range.first;
        default:
            CSP_THROW(InvalidArgument, "Unexpected duplicate policy" << duplicatePolicy);
    }
}

inline std::pair<int32_t, int32_t>
TimeSeries::getValueIndexRange(DateTime time, DuplicatePolicyEnum duplicatePolicy) const {
    // Note that the data in the time series is inverted the last value is at index 0

    struct DateTimeWithIndex {
        int32_t index;
        DateTime time;
    };
    static const auto NO_ELEMENTS_FOUND = std::make_pair<int32_t, int32_t>(-1, -1);

    int32_t num_values = numTicks();
    if (num_values == 0) {
        return NO_ELEMENTS_FOUND;
    }

    auto dateTimeWithIndexGetter = [&](int32_t index) { return DateTimeWithIndex{index, timeAtIndex(index)}; };

    // We will do a binary search of the value
    DateTimeWithIndex startI{dateTimeWithIndexGetter(0)};
    DateTimeWithIndex endI{dateTimeWithIndexGetter(num_values - 1)};

    if (endI.time > time) {
        return NO_ELEMENTS_FOUND;
    }

    if (startI.time <= time) {
        endI = startI;
    }

    // Generally it's not possible to find the first index using binary search. Consider the following example:
    // timestamps of timeseries: 09:30, 09:31, 09:31, ..., 09:31, 09:33
    // If we are looking for 09:32 the binary search should first deduce that 09:31 is the matching timestamp, it
    // should then first deduce that the 09:31 is the matching timesamp to be returned.
    // Invariants during loop:
    //     startI.time > time
    //     endI.time <= time
    /*
     */
    while (startI.index < endI.index) {
        DateTimeWithIndex midI{dateTimeWithIndexGetter((startI.index + endI.index + 1) / 2)};
        if (midI.time <= time) {
            if (unlikely(midI.index == endI.index)) {
                // We know that there are only 2 elements in the range. We know that startI.time > time and
                // endI.time <= time. So we know that we need to select endI as the candidate time
                startI = endI;
            } else {
                endI = midI;
            }
        } else {
            startI = midI;
        }
    }
    // Both startI and endI are pointing to the same element now, the element that is the last element before timestamp
    // we now need to do a binary search to find the first element if necessary. Instead of doing binary search though,
    // we will just search linearly here since we expect the number of elements for each timestamp to be small.
    if(duplicatePolicy != DuplicatePolicyEnum::LAST_VALUE && endI.time == time) {
        while(endI.index < num_values - 1) {
            auto auxI = dateTimeWithIndexGetter(endI.index + 1);
            if(auxI.time == endI.time) {
                endI = auxI;
            }
            else {
                break;
            }
        }
    }

    return std::make_pair(startI.index, endI.index);
}



};
#endif
