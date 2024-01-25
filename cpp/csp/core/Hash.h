#ifndef _IN_CSP_CORE_HASH_H
#define _IN_CSP_CORE_HASH_H

#include <cstring>
#include <string_view>
#include <type_traits>
#include <utility>

namespace csp::hash
{

//C-string hash helpers
struct CStrHash
{
    std::size_t operator()( const char * s ) const noexcept
    {
        //Unabashedly stolen from Python 2.7 string hash
        const unsigned char * p = (const unsigned char *) s;
        std::size_t x = *p << 7;
        while( *p )
            x = (1000003*x) ^ *p++;
            
        return x;
    }
};

struct CStrEq
{
    bool operator()( const char * lhs, const char * rhs ) const noexcept
    {
        return strcmp( lhs, rhs ) == 0;
    }
};

inline size_t hash_bytes( const void * data, size_t len )
{
    return std::hash<std::string_view>{}( std::string_view( ( const char * ) data, len ) );
}

//Convenient hash of pair<T1,T2> so it can be used as a key in unordered_map
struct hash_pair 
{ 
    template <typename T1, typename T2> 
    size_t operator()( const std::pair<T1, T2> & p ) const
    { 
        return std::hash<T1>{}(p.first) ^ std::hash<T2>{}(p.second); 
    } 
};

}

#endif
