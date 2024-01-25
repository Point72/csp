#ifndef _IN_CSP_ENGINE_WINDOWBUFFER_H
#define _IN_CSP_ENGINE_WINDOWBUFFER_H

#include <csp/engine/CspType.h>

/************************************************************************
 ** Window buffer is the storage for rolling window data in an interval.
 ** It uses a circular buffer for storage which allows pushing, popping,
 ** resizing, and indexing.
 ***********************************************************************/

namespace csp 
{

// Class declarations

template<typename T>
class WindowBuffer
{
public:
    WindowBuffer();
    ~WindowBuffer() { delete[] m_values; }
                
    // WindowBuffer is not copyable
    WindowBuffer<T> & operator=( const WindowBuffer<T> & other) = delete;
    WindowBuffer( const WindowBuffer<T> & other) = delete;

    // WindowBuffer is movable
    WindowBuffer<T> & operator=( WindowBuffer<T> && other);
    WindowBuffer( WindowBuffer<T> && other);

    // Update functions
    void push( const T & value ); // different between time and tick
    T pop_left();
    T pop_right();
    void clear();
    void remove_left( int64_t n );

    // Utilities

    // if index is non-negative, treat as offset from the rightIndex (i.e. buf[1] is the second most recent value )
    // if index is negative, treat as offset from the leftIndex (i.e. buf[-1] is earliest value)
    T & operator[]( int64_t index );
    void copy_values( std::vector<T>* out );

    template<typename OutputIt>
    void copy_values( OutputIt out );

    bool empty() const { return m_count == 0; }
    bool full() const { return m_count == m_capacity; }
    int64_t count() const { return m_count; }
    int64_t capacity() const { return m_capacity; }

    struct const_iterator
    {
        const_iterator( T* data, int64_t start, int64_t cap )
        {
            m_index = start;
            m_data = data;
            m_capacity = cap;
        }

        // comparisons assume iterator is being used on the same buffer
        bool operator==( const const_iterator & rhs ) const { return m_index == rhs.m_index; }
        bool operator!=( const const_iterator & rhs ) const { return !( *this == rhs ); }

        const_iterator & operator++()
        {
            m_index++;
            if( m_index == m_capacity ) 
                m_index = 0;
            return *this;
        }

        inline T & value() const { return m_data[m_index]; }

    private:
        T* m_data;
        int64_t m_index;
        int64_t m_capacity;
    };


    /*
    NOTE: The concept of begin() and end() does not extend naturally to the window buffer because of its circularity. In the case the buffer is at capacity,
    the begin() and end() iterators point to the same location. Thus, the only way you can properly iterate is by starting at the left() and keeping track
    of when you've visited the correct number of elements. You could similarly iterate in reverse order by starting at right() and decrementing.
    */

    const_iterator left() const { return const_iterator( m_values, m_leftIndex, m_capacity ); }

    const_iterator right() const { return const_iterator(  m_values, m_rightIndex, m_capacity ); }
        
protected:
    // Used in both tick and time intervals
    T* m_values;
    int64_t m_capacity;
    int64_t m_count;
    int64_t m_rightIndex;
    int64_t m_leftIndex;
};

template<typename T>
class FixedSizeWindowBuffer : public WindowBuffer<T>
{
public:
    ~FixedSizeWindowBuffer() {};
    FixedSizeWindowBuffer() = default;
    FixedSizeWindowBuffer( FixedSizeWindowBuffer<T> && rhs );
    FixedSizeWindowBuffer<T> & operator=( FixedSizeWindowBuffer<T> && rhs );
    FixedSizeWindowBuffer( int64_t capacity );
    void push( const T & value );
};

template<typename T>
class VariableSizeWindowBuffer : public WindowBuffer<T>
{
public:
    ~VariableSizeWindowBuffer() {};
    VariableSizeWindowBuffer() = default;
    VariableSizeWindowBuffer( VariableSizeWindowBuffer<T> && rhs );
    VariableSizeWindowBuffer<T> & operator=( VariableSizeWindowBuffer<T> && rhs );
    void push( const T & value );
    void extend( const std::vector<T> & values );
};

// WindowBuffer functions

template<typename T>
inline WindowBuffer<T>::WindowBuffer() 
    : m_values( nullptr ), m_capacity( 0 ), m_count( 0 ) , m_rightIndex( 0 ), m_leftIndex( 0 )
{ }

template<typename T>
inline T WindowBuffer<T>::pop_left()
{
    if( unlikely( empty() ) ) 
        CSP_THROW( RangeError, "Cannot pop from empty window buffer" ); 
    T v = std::move( m_values[m_leftIndex] );
    m_count--;
    m_leftIndex++;
    if( m_leftIndex == m_capacity )
        m_leftIndex = 0;

    return v;
}

template<typename T>
inline T WindowBuffer<T>::pop_right()
{
    if( unlikely( empty() ) ) 
        CSP_THROW( RangeError, "Cannot pop from empty window buffer" ); 
    T v = std::move( m_values[m_rightIndex] );
    m_count--;
    m_rightIndex--;
    if( m_rightIndex == -1 ) 
        m_rightIndex = m_capacity - 1;

    return v;
}

template<typename T>
inline void WindowBuffer<T>::clear()
{
    m_leftIndex = 0;
    m_rightIndex = 0;
    m_count = 0;
}

template<typename T>
inline void WindowBuffer<T>::remove_left( int64_t n )
{
    if( n > m_count )
        CSP_THROW( RangeError, "Window buffer of size " << m_count << " does not contain " << n << " elements for removal." ); 
    m_leftIndex += n;
    m_leftIndex %= m_capacity;
    m_count -= n;
}

template<typename T>
inline T & WindowBuffer<T>::operator[]( int64_t index )
{
    if( index >= 0 )
    {
        index = m_rightIndex - 1 - index;
        if( index < 0 ) 
            index += m_capacity;
        return m_values[index];
    }
    
    index = m_leftIndex - 1 - index;
    if( index >= m_capacity ) 
        index -= m_capacity;
    return m_values[index];
}

template<typename T>
inline void WindowBuffer<T>::copy_values( std::vector<T>* out )
// Copies values to the vector pointer out
{
    out -> clear();
    out -> resize( m_count );
    copy_values( out -> begin() );
}

template<typename T>
template<typename OutputIt>
inline void WindowBuffer<T>::copy_values( OutputIt out )
// Copies values to the array out, which has been allocated
{
    if( m_rightIndex > m_leftIndex )
        std::copy( m_values + m_leftIndex, m_values + m_rightIndex, out );
    else if( m_count )
    {
        std::copy( m_values + m_leftIndex, m_values + m_capacity, out  );
        std::copy( m_values, m_values + m_rightIndex, out + ( m_capacity - m_leftIndex ) );
    }
}

template<typename T>
inline WindowBuffer<T>::WindowBuffer( WindowBuffer<T> && other ) // move constructor
{
    m_capacity = other.m_capacity;
    m_count = other.m_count;
    m_rightIndex = other.m_rightIndex;
    m_leftIndex = other.m_leftIndex;

    // Move the data
    m_values = other.m_values;
    other.m_values = nullptr;
}

template<typename T>
inline WindowBuffer<T> & WindowBuffer<T>::operator=( WindowBuffer<T> && other ) // move assignment
{
    // Guard self assignment
    if ( this == &other )
        return *this;

    m_capacity = other.m_capacity;
    m_count = other.m_count;
    m_rightIndex = other.m_rightIndex;
    m_leftIndex = other.m_leftIndex;

    // Move the data
    delete[] m_values;
    m_values = other.m_values;
    other.m_values = nullptr;

    return *this;
}

// FixedSizeWindowBuffer functions

template<typename T>
inline FixedSizeWindowBuffer<T>::FixedSizeWindowBuffer( int64_t capacity ) 
{
    // deterministic capacity
    // this needed as inherited variables from template class are non-dependent names
    try
    {
        this -> m_values = new T[capacity];
    }
    catch( std::bad_alloc const & )
    {
        CSP_THROW( ValueError, "Tick specified interval is too large To use an expanding window, set interval=None.");
    }
    this -> m_capacity = capacity;
}

template<typename T>
inline void FixedSizeWindowBuffer<T>::push( const T & value) 
{
    if( likely( this -> full() ) ) 
    {
        // Overwrite the leftIndex value
        this -> m_leftIndex++;
        if( unlikely( this -> m_leftIndex == this -> m_capacity ) ) 
            this -> m_leftIndex = 0;
    }
    else 
        this -> m_count++;

    this -> m_values[this -> m_rightIndex] = value;
    this -> m_rightIndex++;
    // loop back around "clockwise"
    if( unlikely( this -> m_rightIndex == this -> m_capacity ) ) 
        this -> m_rightIndex = 0;
}

template<typename T>
inline FixedSizeWindowBuffer<T>::FixedSizeWindowBuffer( FixedSizeWindowBuffer<T> && rhs )
{
    WindowBuffer<T>::operator=( std::move( rhs ) );
}

template<typename T>
inline FixedSizeWindowBuffer<T> & FixedSizeWindowBuffer<T>::operator=( FixedSizeWindowBuffer<T> && rhs )
{
    WindowBuffer<T>::operator=( std::move( rhs ) );
    return *this;
}


// VariableSizeWindowBuffer functions

template<typename T>
inline void VariableSizeWindowBuffer<T>::push( const T & value )
{
    if( unlikely( !( this -> m_capacity ) ) )
    {
        this -> m_capacity = 1;
        this -> m_values = new T[1];
    }
    else if( unlikely( this -> full() ) ) 
    {
        // need to create new arrray and flatten all elements to maintain circularity
        T* old_values = this -> m_values;
        this -> m_values = new T[2*this -> m_capacity]; // double size
        std::move( old_values + this -> m_leftIndex, old_values + this -> m_capacity, this -> m_values );
        if( this -> m_leftIndex != 0 )
            std::move( old_values, old_values + this -> m_rightIndex, this -> m_values+ ( this -> m_capacity - this -> m_leftIndex ) );
            // wrap around
        
        delete[] old_values;
        this -> m_rightIndex = this -> m_capacity;
        this -> m_capacity *= 2;
        this -> m_leftIndex = 0;
    }
    
    this -> m_values[this -> m_rightIndex] = value;
    this -> m_rightIndex++;
    this -> m_count++;
    if( this -> m_rightIndex == this -> m_capacity ) 
        this -> m_rightIndex = 0;
}

        
template<typename T>
inline void VariableSizeWindowBuffer<T>::extend( const std::vector<T> & values )
{
    for( auto & v: values ) 
       this -> push( v );
}

template<typename T>
VariableSizeWindowBuffer<T>::VariableSizeWindowBuffer( VariableSizeWindowBuffer<T> && rhs )
{
    WindowBuffer<T>::operator=( std::move( rhs ) );
}

template<typename T>
inline VariableSizeWindowBuffer<T> & VariableSizeWindowBuffer<T>::operator=( VariableSizeWindowBuffer<T> && rhs )
{
    WindowBuffer<T>::operator=( std::move( rhs ) );
    return *this;
}


}
#endif
