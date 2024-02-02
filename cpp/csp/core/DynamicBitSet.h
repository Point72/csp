#ifndef _IN_CSP_CORE_DYNAMICBITSET_H
#define _IN_CSP_CORE_DYNAMICBITSET_H

#include <csp/core/Likely.h>
#include <type_traits>

namespace csp
{

template<typename NodeT = uint64_t, typename IndexT = int32_t>
class DynamicBitSet
{

public:
    using node_type     = NodeT;
    using index_type    = IndexT;

    static constexpr index_type npos = -1;

    DynamicBitSet() : m_nodes( nullptr ), m_size( 0 ), m_numNodes( 0 ) {}

    DynamicBitSet( index_type size ) : m_size( size )
    {
        assert( m_size > 0 );

        m_numNodes = ( m_size - 1 ) / _bits + 1;
        m_nodes = new node_type[ m_numNodes ];
        memset( m_nodes, 0, m_numNodes * sizeof( node_type ) );
    }

    DynamicBitSet( DynamicBitSet && other ) : m_size( other.m_size ), m_numNodes( other.m_numNodes )
    {
        m_nodes = other.m_nodes;
        other.m_nodes = nullptr;
    }

    DynamicBitSet & operator=( DynamicBitSet && other )
    {
        m_size = other.m_size;
        m_numNodes = other.m_numNodes;
        m_nodes = other.m_nodes;
        other.m_nodes = nullptr;

        return *this;
    }

    // Do not copy
    DynamicBitSet( const DynamicBitSet & other ) = delete;
    DynamicBitSet & operator=( const DynamicBitSet & other ) = delete;

    ~DynamicBitSet()
    {
        delete[] m_nodes; 
    }

    void set( index_type value )
    {
        auto [ node, bit ] = locateValue( value );
        m_nodes[ node ] |= mask( bit );
    }

    void reset( index_type value )
    {
        auto [ node, bit ] = locateValue( value );
        m_nodes[ node ] &= ~mask( bit );
    }

    index_type find_first() const
    {
        for( index_type i = 0; i < m_numNodes; ++i )
        {
            if( m_nodes[ i ] )
                return ( i << _logBits ) + ffs( m_nodes[ i ] ) - 1;
        }
        return npos;
    }

    index_type find_last() const
    {
        for( index_type i = m_numNodes - 1; i >= 0; --i )
        {
            if( m_nodes[ i ] )
                return ( i << _logBits ) + ( _bits - clz( m_nodes[ i ] ) ) - 1;
        }
        return npos;
    }

    index_type find_next( index_type value ) const
    {
        auto [ node, bit ] = locateValue( value );
        if( likely( bit != _bits - 1 ) )
        {
            node_type shifted = m_nodes[ node ] >> ( bit + 1 );
            if( shifted )
                return value + ffs( shifted );
        }

        do
        {
            node++;
        }
        while( node < m_numNodes && !m_nodes[ node ] );
        
        if( unlikely( node == m_numNodes ) )
            return npos;
        else
            bit = ffs( m_nodes[ node ] ) - 1;

        return indexValue( node, bit );
    }

    index_type find_prev( index_type value ) const
    {
        auto [ node, bit ] = locateValue( value );
        if( likely( bit != 0 ) )
        {
            auto shifted = m_nodes[ node ] << ( _bits - bit );
            if( shifted )
                return value - ( clz( shifted ) + 1 );
        }

        do
        {
            node--;
        }
        while( node >= 0 && !( m_nodes[ node ] ) );
        
        if( unlikely( node < 0 ) )
            return npos;
        else
            bit = _bits - clz( m_nodes[ node ] ) - 1;

        return indexValue( node, bit );
    }

    void resize( index_type size )
    {
        index_type newNodes = ( size - 1 ) / _bits + 1;
        if( newNodes > m_numNodes )
        {
            node_type * old = m_nodes;
            m_nodes = new node_type[ newNodes ];
            memcpy( m_nodes, old, m_numNodes * sizeof( node_type ) );
            memset( m_nodes + m_numNodes, 0, ( newNodes - m_numNodes ) * sizeof( node_type ) );

            m_numNodes = newNodes;
            m_size = size;
            delete[] old;
        }
    }

    size_t size() const { return ( size_t )m_size; }

private:
    using nbit_type = uint8_t;

    template<typename value_type, 
        std::enable_if_t<std::is_unsigned<value_type>::value, bool> = true>
    static constexpr nbit_type nbits() { return sizeof( value_type ) * 8; }

    template<typename U, 
        std::enable_if_t<std::is_unsigned<U>::value, bool> = true>
    static constexpr nbit_type log2( U n )           { return nbits<uint32_t>() - __builtin_clz( n ) - 1; }
    static constexpr nbit_type log2( uint64_t n )    { return nbits<uint64_t>() - __builtin_clzl( n ) - 1; }

    // clz (count leading zeros) returns number of leading zeros before MSB (i.e. clz(00110..) = 2 )
    // __builtin_clz auto-promotes to 32-bits: need to subtract off extra leading zeros
    static constexpr nbit_type clz( uint8_t n )  { return __builtin_clz( n ) - 24; }
    static constexpr nbit_type clz( uint16_t n ) { return __builtin_clz( n ) - 16; }
    static constexpr nbit_type clz( uint32_t n ) { return __builtin_clz( n ); }
    static constexpr nbit_type clz( uint64_t n ) { return __builtin_clzl( n ); }

    // ffs (find first set) returns offset of first set bit (i.e. ffs(..0110) = 2 ), with ffs(0) = 0
    template<typename U, 
        std::enable_if_t<std::is_unsigned<U>::value, bool> = true>
    static constexpr nbit_type ffs( U n )           { return __builtin_ffs( n ); }
    static constexpr nbit_type ffs( uint64_t n )    { return __builtin_ffsl( n ); }

    static constexpr node_type mask( nbit_type bitIndex )  { return ( node_type )1 << bitIndex; }

    std::pair<index_type, nbit_type> locateValue( index_type value ) const
    {
        return std::pair<index_type, nbit_type>( value >> _logBits, value & ( _bits - 1 ) );
    }

    index_type indexValue( index_type node, nbit_type bit ) const
    {
        return ( index_type )bit + ( node << _logBits );
    }

    static constexpr nbit_type _bits    = nbits<NodeT>();
    static constexpr nbit_type _logBits = log2( _bits );

    node_type * m_nodes;
    index_type  m_size;
    index_type  m_numNodes;

};

}


#endif