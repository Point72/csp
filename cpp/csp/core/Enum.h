#ifndef _IN_CSP_CORE_ENUM_H
#define _IN_CSP_CORE_ENUM_H

#include <csp/core/Hash.h>
#include <csp/core/Exception.h>
#include <csp/core/Platform.h>
#include <memory>
#include <unordered_map>
#include <vector>
#include <assert.h>
#include <string.h>

namespace csp {

/*
Example usage:
Define your enum traits struct which must define a basic enum type named _enum with UNKNOWN = 0 and NUM_TYPES members ( where NUM_TYPES is number of enum values )
Along with an instance of said enum as a protected member m_value.  You can also add any methods you like on your traits:

struct MyEnumTraits
{
   enum _enum : unsigned char
   {
       UNKNOWN = 0,
       FOO,
       BAR,

       NUM_TYPES
   };

   //Optional: provide UNKNOWN_ON_INVALID_VALUE = true to return UNKNOWN on bad strings or values.
   //By default constructor and fromString will throw on invalid values
   //static const bool UNKNOWN_ON_INVALID_VALUE = true;

   bool isFoo() const { return m_value == FOO; }
   //etc...

protected:
    _enum m_value;
};

//Your enum type
using MyEnum = Enum<MyEnumTraits>

//In a cpp file, define your string mappings like so
INIT_CSP_ENUM( MyEnum,
    "UNKNOWN", 
    "A", 
    "B" 
);

*/

template<typename T>
struct EnumFriendHelper {
    typedef T type;
};

template<typename EnumV, bool B>
struct EnumUTypeHelper {
    using type = typename std::underlying_type<EnumV>::type;
};
template<typename EnumV>
struct EnumUTypeHelper<EnumV, false> {
    using type = EnumV;
};

template<typename T>
auto UnknownOnInvalidValue(int) -> decltype(T::UNKNOWN_ON_INVALID_VALUE) { return T::UNKNOWN_ON_INVALID_VALUE; }

template<typename T>
bool UnknownOnInvalidValue(long) { return false; }

START_PACKED
template<typename EnumTraits>
struct Enum : public EnumTraits
{
    using EnumV = typename EnumTraits::_enum;
    using Mapping = std::vector<std::string>;

    using UType = typename EnumUTypeHelper<EnumV, std::is_enum<EnumV>::value>::type;

    template<typename = std::enable_if<std::is_enum<EnumV>::value> >
    constexpr Enum( EnumV v ) { this->m_value = v; }

    constexpr Enum() { this->m_value = EnumTraits::UNKNOWN; }

    constexpr Enum( const Enum &rhs ) { this->m_value = rhs.m_value; }

    Enum( const char *s ) { this->m_value = reverseMap().fromString( s ); }
    Enum( const std::string &s ) : Enum( s.c_str() ) {}
    Enum( UType v );

    static const std::string &asString(EnumV v) { return mapping()[ v ]; }

    const std::string &asString() const { return asString( this -> m_value ); }
    const char *asCString() const { return asString( this -> m_value ).c_str(); }

    //this pulls in all comparison operators for free since w will auto-convert to the raw enum
    constexpr operator EnumV() const { return this -> m_value; }

    constexpr UType value() const { return this -> m_value; }

    //common convenience methods
    bool isKnown() const { return this -> m_value != EnumTraits::UNKNOWN; }
    bool isUnknown() const { return this -> m_value == EnumTraits::UNKNOWN; }

    struct iterator 
    {
        iterator( int v ) : m_v( v ) {}

        Enum operator*() { return Enum( ( EnumV ) m_v ); }
        bool operator==(const iterator &rhs) const { return m_v == rhs.m_v; }
        bool operator!=(const iterator &rhs) const { return !(*this == rhs); }

        iterator &operator++() {
            ++m_v;
            return *this;
        }

    private:
        int m_v;
    };

    static constexpr size_t numTypes() { return ( size_t ) EnumTraits::NUM_TYPES; }

    static iterator begin() { return iterator(0); }
    static iterator end() { return iterator(numTypes()); }

protected:
    using Aliases = std::unordered_multimap<std::string,std::string>;

    struct ReverseMap : public std::unordered_map<const char *, typename EnumTraits::_enum, hash::CStrHash, hash::CStrEq> 
    {
        using BaseT   = std::unordered_map<const char *, EnumV, hash::CStrHash, hash::CStrEq>;

        ReverseMap( const Mapping &mapping )
        {
            int v = 0;
            for( auto &s : mapping )
            {
                assert( this -> find( s.c_str() ) == this -> end() );
                (*this)[strdup(s.c_str())] = (EnumV) v++;
            }
        }

        ~ReverseMap() 
        {
            clear();
        }

        void clear() 
        {
            for( auto &entry : *this )
                free( const_cast<char *>( entry.first ) );

            BaseT::clear();
        }

        EnumV fromString( const char *s ) const 
        {
            auto it = this -> find( s );
            if( it == this -> end() )
            {
                if( UnknownOnInvalidValue<EnumTraits>( 0 ) )
                    return EnumTraits::UNKNOWN;

                CSP_THROW( ValueError, "Unrecognized enum value: " << s << " for enum " << typeid( EnumTraits ).name() );
            }

            return it->second;
        }
    };

    //This is defined by INIT macro
    static const Mapping & mapping();

    static const ReverseMap & reverseMap() 
    { 
        static ReverseMap s_reverseMap( mapping() );
        return s_reverseMap; 
    }

} END_PACKED;

template<typename EnumTraits>
Enum<EnumTraits>::Enum( UType v ) 
{
    if( v < 0 || v >= numTypes() )
    {
        if( UnknownOnInvalidValue<EnumTraits>(0) )
            this -> m_value = EnumTraits::UNKNOWN;
        else
            CSP_THROW(ValueError, "enum value: " << v << " out of range for enum " << typeid(EnumTraits).name());
    } else
        this -> m_value = (EnumV) v;
}

template<typename EnumTraits>
std::ostream &operator<<(std::ostream &o, Enum<EnumTraits> e) {
    o << e.asString();
    return o;
}
};

//Make all Enum types hashable
namespace std {

template<typename EnumTraits>
struct hash<csp::Enum<EnumTraits>> {
    size_t operator()(csp::Enum<EnumTraits> e) const {
        return std::hash<typename csp::Enum<EnumTraits>::EnumV>()(e);
    }
};

}

#define INIT_CSP_ENUM(ENUM, ...)                                \
    template<> const ENUM::Mapping & ENUM::mapping() {      \
        static ENUM::Mapping s_mapping( { __VA_ARGS__ } );  \
        return s_mapping;                                   \
    }

#endif
