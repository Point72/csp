#ifndef _IN_CSP_CORE_ENUMBITSET_H
#define _IN_CSP_CORE_ENUMBITSET_H

#include <stddef.h>
#include <stdint.h>
#include <initializer_list>
#include <limits>

namespace csp
{

//Utility class to hold enums as a bitmask ( where enum values are incremental from 0 )
//enum must have a NUM_TYPES entry for number of entries
template< typename EnumT >
class EnumBitSet
{
    using value_type = uint64_t;

public:
    constexpr EnumBitSet() : m_bitset( 0 ) {}
    constexpr EnumBitSet( EnumT v ) : m_bitset( enumBit( v ) ) {}
    constexpr EnumBitSet( const EnumBitSet & rhs ) : m_bitset( rhs.m_bitset ) {}
    constexpr EnumBitSet( std::initializer_list<EnumT> enums );

    void add( EnumT e )           { m_bitset |= enumBit( e ); }
    void clear( EnumT e )         { m_bitset &= ~enumBit( e ); }
    bool isSet( EnumT e ) const   { return m_bitset & enumBit( e ); }
    void reset()                  { m_bitset = 0; }

    bool empty()                   { return !m_bitset; }
    explicit operator bool() const { return m_bitset; }

    size_t size() const    { return __builtin_popcountl( m_bitset ); }

    constexpr EnumBitSet operator ~() const                          { return EnumBitSet( ~m_bitset & ( ( ( (value_type) 1u ) << int( EnumT::NUM_TYPES ) ) - 1 ) ); }
    constexpr EnumBitSet operator |( typename EnumT::EnumV v ) const { return EnumBitSet( m_bitset | enumBit( v ) ); }
    constexpr EnumBitSet operator |( EnumT e ) const                 { return EnumBitSet( m_bitset | enumBit( e ) ); }
    constexpr EnumBitSet operator |( const EnumBitSet & rhs ) const  { return EnumBitSet( m_bitset | rhs.m_bitset ); }
    constexpr EnumBitSet operator &( typename EnumT::EnumV v ) const { return EnumBitSet( m_bitset & enumBit( v ) ); }
    constexpr EnumBitSet operator &( EnumT e ) const                 { return EnumBitSet( m_bitset & enumBit( e ) ); }
    constexpr EnumBitSet operator &( const EnumBitSet & rhs ) const  { return EnumBitSet( m_bitset & rhs.m_bitset ); }

    EnumBitSet & operator |=( typename EnumT::EnumV v ) { m_bitset |= enumBit( v ); return *this; }
    EnumBitSet & operator |=( EnumT e )                 { m_bitset |= enumBit( e ); return *this; }
    EnumBitSet & operator |=( const EnumBitSet & rhs )  { m_bitset |= rhs.m_bitset; return *this; }
    EnumBitSet & operator &=( const EnumBitSet & rhs )  { m_bitset &= rhs.m_bitset; return *this; }

    bool operator==( const EnumBitSet & rhs ) const    { return m_bitset == rhs.m_bitset; }
    bool operator!=( const EnumBitSet & rhs ) const    { return m_bitset != rhs.m_bitset; }

    struct iterator
    {
        iterator( value_type bitset ) : m_bitset( bitset ) {}
        
        EnumT operator*() const { return ( EnumT ) __builtin_ctzl( m_bitset ); } //returns lowest order bit
        iterator operator++()      { m_bitset &= ( m_bitset - 1 ); return *this; } //clear out least significant bit
        iterator operator++( int ) { iterator it = *this; m_bitset &= ( m_bitset - 1 ); return it; }

        bool operator==( iterator rhs ) { return m_bitset == rhs.m_bitset; }
        bool operator!=( iterator rhs ) { return !(*this == rhs ); }

    private:
        value_type m_bitset;
    };

    iterator begin() const { return iterator( m_bitset ); }
    iterator end() const   { return iterator( 0 ); }

    value_type rawBitSet() const { return m_bitset; }

    void setRawBitSet( value_type bitset ) { m_bitset = bitset; }

private:
    constexpr EnumBitSet( value_type v ) : m_bitset( v ) {}
    static constexpr value_type enumBit( typename EnumT::EnumV e ) { return static_cast<value_type>( 1 ) << ( unsigned char ) e; }

    static_assert( ( 1ul << ( int( EnumT::NUM_TYPES ) - 1 ) ) < std::numeric_limits<value_type>::max(), "Too many enums for EnumBitSet to hold" );

    value_type m_bitset;
};

template< typename EnumT >
inline constexpr EnumBitSet<EnumT>::EnumBitSet( std::initializer_list<EnumT> enums ) : m_bitset( 0 )
{
    for( auto e : enums )
        m_bitset |= enumBit( e );
}

}

#endif
