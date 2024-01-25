#ifndef _IN_CSP_CORE_TAGGEDPOINTERUNION_H
#define _IN_CSP_CORE_TAGGEDPOINTERUNION_H

#include <csp/core/System.h>

namespace csp
{

template<typename... Ts> struct TypeList {};

template<typename, typename>
struct IndexOf { };

template <typename T, typename... Ts>
struct IndexOf<T, TypeList<T, Ts...>>
{
    static constexpr std::size_t value = 0;
};

template <typename T, typename TOther, typename... Ts>
struct IndexOf<T, TypeList<TOther, Ts...> >
{
    static constexpr std::size_t value = IndexOf<T, TypeList<Ts...>>::value + 1;
};


template<typename... Ts>
class TaggedPointerUnion
{
public:
    static inline constexpr size_t NUM_TAGS = sizeof...(Ts);
    //we can be more efficient if needed for more types..., we can store an integeger value rather
    //than a bit per type.  but current use cases we only need 2 types so not bothering
    static inline constexpr size_t TAG_BITS = NUM_TAGS;
    static inline constexpr size_t TAG_MASK = ( 1UL << TAG_BITS ) - 1;

    TaggedPointerUnion()
    {
        reset();
    }

    void * raw() const { return m_ptr; }
    
    void * unmasked() const { return ( void * ) ( ( ( uint64_t ) m_ptr ) & ~TAG_MASK ); }

    void reset() { m_ptr = nullptr; }

    operator bool() const { return m_ptr != nullptr; }

    template<typename T> 
    void set( T * p )
    {
        m_ptr = p;
        setMask<T>();
    }

    //note that this will NOT do error checking, on caller to check isSet as needed
    template<typename T> 
    T * get() const
    {
        return ( T * ) unmasked();
    }

    template<typename T>
    bool isSet() const
    {
        constexpr size_t bitmask = 1 << typeBit<T>();
        return ( ( ( uint64_t ) m_ptr ) & bitmask ) == bitmask;
    }

    template< typename T >
    static constexpr size_t typeBit() { return IndexOf<T,TypeList<Ts...>>::value; }

private:
    template< typename T >
    void setMask()
    { 
        constexpr size_t bitmask = 1 << typeBit<T>();
        m_ptr = ( void * ) ( ( ( uint64_t ) m_ptr ) | bitmask );
    }

    void * m_ptr;
};

}

#endif
