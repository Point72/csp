#ifndef _IN_CSP_ENGINE_TICKBUFFER_H
#define _IN_CSP_ENGINE_TICKBUFFER_H

/************************************************************************
 ** Tick buffer is the storage for CSP time series ticks
 ** it uses a circular buffer for storage but only allows pushing back
 ** and resizing 
 ***********************************************************************/
#include <csp/core/System.h>
#include <csp/core/Time.h>
#include <memory>
#include <stdint.h>
#include <algorithm>

namespace csp
{

template< typename T >
class TickBuffer
{
public:
    TickBuffer( uint32_t capacity = 1 );
    ~TickBuffer();
    
    void clear()
    {
        m_full = false;
        m_writeIndex = 0;
    }

    void growBuffer( uint32_t capacity );

    //number of available ticks
    //note subtelty of returning m_writeIndex if we arent full, this is to account for cases where we were full
    //but buffer was grown ( m_count can be < m_capacity but m_count is not number of ticks available in that case )
    uint32_t numTicks() const { return unlikely( !m_full ) ? m_writeIndex : m_capacity; }

    uint32_t capacity() const { return m_capacity; }
    bool     full() const     { return m_full; }
    bool     empty()    const { return !m_full && m_writeIndex == 0; }

    void     push_back( const T & value );

    //used by BURST
    T &      prepare_write();

    //index works in reverse, index 0 = latest tick
    T & operator[]( uint32_t index );
    T & valueAtIndex( uint32_t index ) { return (*this)[index]; }

    const T & operator[]( uint32_t index ) const   { return const_cast<TickBuffer<T> *>( this ) -> operator[]( index ); }
    const T & valueAtIndex( uint32_t index ) const { return const_cast<TickBuffer<T> *>( this ) -> valueAtIndex( index ); }

    //copies the ticks from start to end into a newly allocated array
    //as with operator[], index works in reverse with index 0 = latest tick
    //this is needed for fast copies of the data, since it is internally stored in a circular buffer
    T * flatten( uint32_t startIndex, uint32_t endIndex, uint32_t tailPadding = 0 ) const;

private:
    void raiseRangeError( uint32_t index ) const;

    //TODO consider optimizing
    // - for size 1 case
    // can try a union of { T t; T * t_buffer; } and a capacity and T * that always points to either &t ot t_buffer for quick read access

    T *       m_buffer;
    uint32_t  m_capacity;

    uint32_t  m_writeIndex; //write index, wraps around
    bool      m_full;
};

template< typename T >
inline TickBuffer<T>::TickBuffer( uint32_t capacity ) : m_capacity( capacity ), m_writeIndex( 0 ), m_full( false )
{
    m_buffer = new T[ capacity ];
}

template< typename T >
inline TickBuffer<T>::~TickBuffer()
{
    delete [] m_buffer;
}

template< typename T >
inline void TickBuffer<T>::growBuffer( uint32_t new_capacity )
{
    if( unlikely( new_capacity <= m_capacity ) )
        return;

    T * oldbuf = m_buffer;
    m_buffer = new T[new_capacity];

    if( !m_full )
        std::move( oldbuf, oldbuf + m_writeIndex, m_buffer );
    else
    {
        //two move calls to handle potential wrap arounds
        std::move( oldbuf + m_writeIndex, oldbuf + m_capacity,   m_buffer );
        std::move( oldbuf,                oldbuf + m_writeIndex, m_buffer + m_capacity - m_writeIndex );
        m_writeIndex = m_capacity;
    }
    delete [] oldbuf;

    m_capacity = new_capacity;
    m_full = false;
}

template< typename T >
inline T & TickBuffer<T>::prepare_write()
{
    auto index = m_writeIndex;
    if( ++m_writeIndex >= m_capacity )
    {
        m_writeIndex = 0;
        m_full = true;
    }
    return m_buffer[ index ];
}

template< typename T >
inline void TickBuffer<T>::push_back( const T & value )
{
    prepare_write() = value;
}

template< typename T >
inline void TickBuffer<T>::raiseRangeError( uint32_t index ) const {
    CSP_THROW( RangeError, "Invalid buffer access: index " << index << " out of range for buffer with numTicks: " << numTicks() << " capacity: " << m_capacity );
}

template< typename T >
inline T & TickBuffer<T>::operator[]( uint32_t index )
{
    if (unlikely(index >= numTicks()))
        raiseRangeError(index);

    int64_t raw_index = int64_t( m_writeIndex ) - index - 1;
    if( unlikely( raw_index < 0 ) )
        raw_index += m_capacity;
    
    return m_buffer[ raw_index ];
}

template<typename T>
T * TickBuffer<T>::flatten( uint32_t startIndex, uint32_t endIndex, uint32_t tailPadding ) const
{
    static_assert( std::is_trivially_copyable_v<T> );
    if( startIndex < endIndex )
        CSP_THROW( RangeError, "Invalid buffer flatten: endIndex " << endIndex << " greater than startIndex " << startIndex );
    if( startIndex >= m_capacity )
        CSP_THROW( RangeError, "Invalid buffer flatten: startIndex " << startIndex << " greater than capacity " << m_capacity );

    size_t len = startIndex - endIndex + 1;

    T * values = ( T * ) malloc( sizeof( T ) * ( len + tailPadding ) );

    int64_t raw_index = int64_t( m_writeIndex ) - startIndex - 1;
    if( unlikely( raw_index < 0 ) )
        raw_index += m_capacity;

    if( unlikely( raw_index + len > m_capacity ) ) // buffer wraps around
    {
        uint32_t beforeWrap = m_capacity - raw_index;
        // copy the values after the wrap (starting from index 0 on the raw buffer, and index beforeWrap on values
        std::copy( m_buffer, m_buffer + ( len - beforeWrap ), values + beforeWrap );
        len = beforeWrap;
    }

    std::copy( m_buffer + raw_index, m_buffer + raw_index + len, values );
    return values;
}

};
#endif
