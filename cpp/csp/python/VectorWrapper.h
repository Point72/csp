// Wrapper class that extends std::vector interface with Python list API

#ifndef _IN_CSP_PYTHON_VECTORWRAPPER_H
#define _IN_CSP_PYTHON_VECTORWRAPPER_H

#include <Python.h>
#include <algorithm>
#include <cmath>
#include <functional>
#include <vector>

namespace csp
{

template<typename StorageT>
class VectorWrapper
{
public:
    using IndexType = Py_ssize_t;

    VectorWrapper( std::vector<StorageT> & m_v ) : m_vector( m_v ) {};

    bool operator==( const VectorWrapper & rhs ) const { return m_vector == rhs.m_vector; }
    bool operator!=( const VectorWrapper & rhs ) const { return m_vector != rhs.m_vector; }

    StorageT & operator[]( IndexType index )
    {
        index = verify_index( index );
        return m_vector[ index ];
    }
    const StorageT & operator[]( IndexType index ) const
    {
        index = verify_index( index );
        return m_vector[ index ];
    }

    // Make negative indices (from the array end) to be positive (from the array start)
    // If the index pointed outside of the array, it will continue to do so after the call to this function
    inline IndexType normalizeIndex( IndexType index ) const
    {
        if( unlikely( index < 0 ) )
            index += size();
        return index;
    }

    struct Slice
    {
        IndexType start_index;
        IndexType stop_index;
        IndexType step;
        IndexType length;

        Slice( IndexType start_i, IndexType stop_i, IndexType sp, IndexType len ): start_index( start_i ), stop_index( stop_i ), step( sp ), length( len ) {}

        bool belongs( IndexType index )
        {
            return ( ( index - start_index ) % step == 0 ) && ( ( index - start_index ) / step >= 0 ) && ( ( index - start_index ) / step < length );
        }
    };

    // Make slice canonical:
    // - normalize start and stop indices as normalizeIndex() does
    // - if the step is 0, raise ValueError
    // - if the step is positive, start and stop indices will be between 0 and one-past-the-last array element inclusive
    // - if the step is negative, start and stop indices will be between -1 and the last array element inclusive
    // - number of elements in the slice will be computed
    inline Slice normalizeSlice( IndexType start_index, IndexType stop_index, IndexType step ) const;

    const std::vector<StorageT> & getVector() const                                                                     { return m_vector; }
    IndexType size() const                                                                                              { return ( IndexType ) m_vector.size(); }
    void append( const StorageT & value )                                                                               { m_vector.emplace_back( value ); }
    void insert( const StorageT & value, IndexType index );
    StorageT pop( IndexType index = -1 );
    void reverse()                                                                                                      { std::reverse( m_vector.begin(), m_vector.end() ); }
    void extend( const std::vector<StorageT> & other )                                                                  { m_vector.insert( m_vector.end(), other.begin(), other.end() ); }
    IndexType count( const StorageT & value ) const                                                                     { return ( IndexType ) std::count( m_vector.begin(), m_vector.end(), value ); }
    // Return the 0-based index of the first element that equals value in the subsequence [start_index, stop_index)
    IndexType index( const StorageT & value, IndexType start_index, IndexType stop_index ) const;
    void remove( const StorageT & value );
    void clear()                                                                                                        { m_vector.clear(); }
    void eraseItem( IndexType index );
    std::vector<StorageT> getSlice( IndexType start_index, IndexType stop_index, IndexType step ) const;
    void setSlice( const std::vector<StorageT> & other, IndexType start_index, IndexType stop_index, IndexType step );
    void eraseSlice( IndexType start_index, IndexType stop_index, IndexType step );
    void repeat( int n );
    bool contains( const StorageT & value ) const                                                                       { return std::find( m_vector.begin(), m_vector.end(), value ) != m_vector.end(); }

protected:
    std::vector<StorageT> & m_vector;

private:
    // Check that the index value is in the array bounds
    inline bool check_index_in_bounds( IndexType index ) const
    {
        return ( index >= 0 && index < size() );
    }
    // Make index to be strictly inside provided bounds
    inline IndexType force_index_in_bounds( IndexType index, IndexType l_bound, IndexType u_bound ) const
    {
        return std::max( l_bound, std::min( u_bound, index ) );
    }
    // Run normalizeIndex() on index and throw if it's not in array bounds
    inline IndexType verify_index( IndexType index ) const;
};

template<typename StorageT>
inline typename VectorWrapper<StorageT>::IndexType VectorWrapper<StorageT>::verify_index( IndexType index ) const
{
    // We allow list indices that are negative or point outside of the array
    index = normalizeIndex( index );
    // We don't allow indices to be past the array boundaries
    if( unlikely( !check_index_in_bounds( index ) ) )
        CSP_THROW( RangeError, "Index " << index << " is out of range." );
    return index;
}

template<typename StorageT>
inline typename VectorWrapper<StorageT>::Slice VectorWrapper<StorageT>::normalizeSlice( IndexType start_index, IndexType stop_index, IndexType step ) const
{    
    if( unlikely( step == 0 ) )
        CSP_THROW( ValueError, "Slice step cannot be zero." );

    IndexType sz = size();
    // Call Python API method that computes the size of the slice and properly adjusts start_index and stop_index
    IndexType slice_len = PySlice_AdjustIndices( sz, &start_index, &stop_index, step );
            
    return VectorWrapper<StorageT>::Slice( start_index, stop_index, step, slice_len );
}

template<typename StorageT>
void VectorWrapper<StorageT>::insert( const StorageT & value, IndexType index )
{
    // We allow list indices that are negative or point outside of the array
    // We allow indices for insert to be past the array boundaries, in which case the element is added to the beginning or end of the array
    auto it = m_vector.begin() + force_index_in_bounds( normalizeIndex( index ), 0, size() );
    m_vector.insert( it, value );
}

template<typename StorageT>
StorageT VectorWrapper<StorageT>::pop( IndexType index )
{
    index = verify_index( index );
    auto it = m_vector.begin() + index;
    StorageT elem = *it;
    m_vector.erase( it );
    return elem;
}

template<typename StorageT>
typename VectorWrapper<StorageT>::IndexType VectorWrapper<StorageT>::index( const StorageT & value, IndexType start_index, IndexType stop_index ) const
{
    // We allow list indices that are negative or point outside of the array
    VectorWrapper<StorageT>::IndexType sz = m_vector.size();
    auto normalized_slice = normalizeSlice( start_index, stop_index, 1 );
    if( likely( normalized_slice.start_index < sz ) )
    {
        auto start_it = m_vector.begin() + normalized_slice.start_index;
        auto stop_it = ( normalized_slice.stop_index < sz ) ? m_vector.begin() + normalized_slice.stop_index : m_vector.end();
        auto it = std::find( start_it, stop_it, value );
        if( likely( it != stop_it ) )
            return it - m_vector.begin();
    }
    CSP_THROW( ValueError, "Value not found." ); 
}

template<typename StorageT>
void VectorWrapper<StorageT>::remove( const StorageT & value )
{
    auto it = std::find( m_vector.begin(), m_vector.end(), value );
    if( likely( it != m_vector.end() ) )
        m_vector.erase( it );
    else
        CSP_THROW( ValueError, "Value not found." ); 
}

template<typename StorageT>
void VectorWrapper<StorageT>::eraseItem( IndexType index )
{
    index = verify_index( index );
    auto it = m_vector.begin() + index;
    m_vector.erase( it );
}

template<typename StorageT>
std::vector<StorageT> VectorWrapper<StorageT>::getSlice( IndexType start_index, IndexType stop_index, IndexType step ) const
{
    // We allow list indices that are negative or point outside of the array
    auto normalized_slice = normalizeSlice( start_index, stop_index, step );

    std::vector<StorageT> result;
    result.reserve( normalized_slice.length );

    // Go through the slice and copy its elements
    for( VectorWrapper<StorageT>::IndexType index = normalized_slice.start_index; normalized_slice.belongs( index ); index += normalized_slice.step )
        result.emplace_back( m_vector[ index ] );

    return result;
}

template<typename StorageT>
void VectorWrapper<StorageT>::setSlice( const std::vector<StorageT> & other, IndexType start_index, IndexType stop_index, IndexType step )
{
    // We allow list indices that are negative or point outside of the array
    VectorWrapper<StorageT>::IndexType sz = m_vector.size();
    auto normalized_slice = normalizeSlice( start_index, stop_index, step );

    // In case of step size 1, mismatch between lengths of the slice and the iterable to be set on the slice is allowed
    // In that case, we erase the full slice and insert the iterable in place of it
    if( normalized_slice.step == 1 && normalized_slice.length != ( VectorWrapper<StorageT>::IndexType ) other.size() )
    {
        auto start_it = ( normalized_slice.start_index < sz ) ? m_vector.begin() + normalized_slice.start_index : m_vector.end();
        auto stop_it = ( normalized_slice.stop_index < sz ) ? m_vector.begin() + normalized_slice.stop_index : m_vector.end();
        if( normalized_slice.length > 0 && normalized_slice.start_index < sz )
            m_vector.erase( start_it, stop_it );
        auto it = m_vector.begin() + normalized_slice.start_index;
        m_vector.insert( it, other.begin(), other.end() );
    }
    // In case of step size other then 1, mismatch between lengths of the slice and the iterable to be set on the slice is not allowed
    else
    {
        if( unlikely( normalized_slice.length != ( VectorWrapper<StorageT>::IndexType ) other.size() ) )
            CSP_THROW( ValueError, "Attempt to assign a sequence of mismatched size to extended slice." );
        
        // Go through the slice and assign its elements
        VectorWrapper<StorageT>::IndexType i = 0;
        for( VectorWrapper<StorageT>::IndexType index = normalized_slice.start_index; normalized_slice.belongs( index ); index += normalized_slice.step )
        {
            m_vector[ index ] = other[ i ];
            i++;
        }
    }
}

template<typename StorageT>
void VectorWrapper<StorageT>::eraseSlice( IndexType start_index, IndexType stop_index, IndexType step )
{
    // We allow list indices that are negative or point outside of the array
    VectorWrapper<StorageT>::IndexType sz = m_vector.size();
    auto normalized_slice = normalizeSlice( start_index, stop_index, step );

    if( normalized_slice.length != 0 )
    {
        std::vector<StorageT> result;
        result.reserve( sz - normalized_slice.length );
        // Go through the vector and copy elements that don't belong to the slice
        for( VectorWrapper<StorageT>::IndexType index = 0; index < sz; ++index )
            if( !normalized_slice.belongs( index ) )
                result.emplace_back( m_vector[ index ] );
        m_vector = std::move( result );
    }
}

template<typename StorageT>
void VectorWrapper<StorageT>::repeat( int n )
{    
    if( n <= 0 )
        m_vector.clear();
    else
    {
        VectorWrapper<StorageT>::IndexType sz = m_vector.size();
        m_vector.resize( sz * n );
        for( int i = 1; i < n; ++i )
            std::copy( m_vector.begin(), m_vector.begin() + sz, m_vector.begin() + i * sz );
    }
}

}

#endif
