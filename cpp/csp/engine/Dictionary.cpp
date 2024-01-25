#include <csp/engine/Dictionary.h>
#include <set>

template<> struct std::hash<std::vector<csp::Dictionary::Data>>
{
    size_t operator()( const vector<csp::Dictionary::Data> & v )
    {
        size_t h = 0;
        for( auto &&entry : v )
            h ^= hash<csp::Dictionary::Value>()( entry._data );
        return h;
    }
};

namespace csp
{

Dictionary::Dictionary()
{}

Dictionary::~Dictionary()
{}

Dictionary::Dictionary( const Dictionary & rhs )
{
    m_map  = rhs.m_map;
    m_data = rhs.m_data;
}

Dictionary::Dictionary( Dictionary && rhs )
{
    m_map  = std::move( rhs.m_map );
    m_data = std::move( rhs.m_data);
}

Dictionary & Dictionary::operator=( const Dictionary & rhs )
{
    m_map  = rhs.m_map;
    m_data = rhs.m_data;
    return *this;
}

Dictionary & Dictionary::operator=( Dictionary && rhs )
{
    m_map  = std::move( rhs.m_map );
    m_data = std::move( rhs.m_data);
    return *this;
}

bool Dictionary::exists( const std::string & key ) const
{
    return m_map.find( key ) != m_map.end();
}

bool Dictionary::operator==( const Dictionary & rhs ) const
{
    if( m_data.size() != rhs.m_data.size())
    {
        return false;
    }
    for(auto&& entry:m_data)
    {
        auto &&rhsIt = rhs.m_map.find( entry.first );
        if( rhsIt == rhs.m_map.end())
        {
            return false;
        }
        if( entry.second != rhs.m_data[rhsIt->second].second)
        {
            return false;
        }
    }
    return true;
}

size_t Dictionary::hash() const
{
    size_t hash = 0;
    for( auto &&entry: m_data )
    {
        hash = hash ^ std::hash<std::string>()( entry.first ) ^ std::hash<Value>()( entry.second._data );
    }
    return hash;
}

}
