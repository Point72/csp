#ifndef _IN_CSP_CORE_DICTIONARY_H
#define _IN_CSP_CORE_DICTIONARY_H

#include <csp/core/Exception.h>
#include <csp/core/System.h>
#include <csp/core/Time.h>
#include <csp/engine/DialectGenericType.h>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace csp
{

class Dictionary;
using DictionaryPtr = std::shared_ptr<Dictionary>;

//for passing struct type information as parameters
class StructMeta;
using StructMetaPtr = std::shared_ptr<StructMeta>;

class Dictionary
{
public:
    struct Data;
    using Vector=std::vector<Data>;
    using Value = std::variant<std::monostate,bool,int32_t,uint32_t,int64_t,uint64_t,double,std::string,DateTime,TimeDelta,StructMetaPtr,DialectGenericType,DictionaryPtr,Vector,std::shared_ptr<Data>>;
    enum DictDataType { MONOSTATE, BOOL, INT32, UINT32, INT64, UINT64, DOUBLE, STRING, DATETIME, TIMEDELTA, STRUCTMETAPTR, DIALECTGENERICTYPE, DICTIONARYPTR, VECTOR, DATA };

    struct Data
    {
        Data( Value data ) : _data( std::move( data ) ) {}
        bool operator==( const Data & other ) const { return _data == other._data; }
        bool operator!=( const Data & other ) const { return _data != other._data; }
        Value _data;
    };
private:
    using Map = std::unordered_map<std::string, size_t>;
    using DataVector = std::vector<std::pair<std::string, Data>>;

public:
    Dictionary();
    Dictionary( const Dictionary & rhs );
    Dictionary( Dictionary && rhs );

    ~Dictionary();

    Dictionary & operator=( const Dictionary & rhs );
    Dictionary & operator=( Dictionary && rhs );

    template<typename T>
    bool insert( const std::string &key, const T & value );

    bool insert( const std::string &key, const char * value )
    {
        return insert( key, std::string( value ) );
    }

    bool insert( const std::string &key, char * value )
    {
        return insert( key, std::string( value ) );
    }

    //aka insertOrReplace, returns true if it replaced a value
    template<typename T>
    bool update( const std::string &key, const T & value );

    bool update( const std::string &key, const char * value )
    {
        return update( key, std::string( value ) );
    }

    bool update( const std::string &key, char * value )
    {
        return update( key, std::string( value ) );
    }


    bool operator==( const Dictionary & rhs ) const;
    bool operator!=( const Dictionary & rhs ) const { return !( *this == rhs ); }

    size_t hash() const;

    template<typename T>  struct ReturnType { using Type = T; };

    //will throw if missing
    template<typename T>
    typename ReturnType<T>::Type get( const std::string & key ) const;

    template<typename T>
    bool tryGet( const std::string & key, T& target ) const;

    const Value& getUntypedValue( const std::string & key ) const;

    template<typename T>
    typename ReturnType<T>::Type get( const std::string & key, const T & default_ ) const;

    bool exists( const std::string & key ) const;

    struct const_iterator
    {
        const_iterator(  const DataVector::const_iterator &it )
        :  m_it( it )
        {}

        const_iterator & operator++() { ++m_it; return *this; }
        bool operator==( const const_iterator & rhs ) const { return m_it == rhs.m_it; }
        bool operator!=( const const_iterator & rhs ) const { return !( *this == rhs ); }

        const std::string & key() const { return m_it -> first; }

        template<typename T>
        typename ReturnType<T>::Type value() const
        {
            return Dictionary::extractValue<T>( m_it -> first, m_it -> second._data );
        }

        const Value & getUntypedValue() const { return m_it -> second._data; }

        //check if value exists as given type
        //note this checks exact types, not coercible types ( it if its int32 and you check int64 itr will be false )
        template<typename T>
        bool hasValue() { return std::holds_alternative<T>( m_it -> second._data ); }

    private:
        DataVector::const_iterator m_it;
    };

    const_iterator begin() const { return const_iterator( m_data.begin()); }

    const_iterator end() const { return const_iterator( m_data.end()); }

    size_t size() const          { return m_data.size(); }
    bool   empty() const         { return m_data.empty(); }

private:
    template<typename T>
    static typename ReturnType<T>::Type extractValue( const std::string & key, const Value & value );

    template<typename From,typename To>
    static typename ReturnType<To>::Type cast( const From & value )
    { throw std::bad_variant_access(); }

    Map        m_map;
    DataVector m_data;
};

template<typename T>
inline bool Dictionary::insert( const std::string &key, const T & value )
{
    auto rv = m_map.emplace( key, m_data.size() );
    if(rv.second)
    {
        m_data.push_back( { key, Value( value ) } );
    }
    return rv.second;
}

template<typename T>
inline bool Dictionary::update( const std::string &key, const T & value )
{
    auto rv = m_map.emplace( key, m_data.size() );
    if( !rv.second )
    {
        m_data[rv.first->second].second = Value( value );
    }
    else
    {
        m_data.push_back( { key, Value( value ) } );
    }

    return !rv.second;
}

template<typename T>
inline typename Dictionary::ReturnType<T>::Type Dictionary::get( const std::string & key ) const
{
    return extractValue<T>( key, getUntypedValue(key) );
}

template<typename T>
inline bool Dictionary::tryGet( const std::string & key, T& target  ) const
{
    auto it = m_map.find( key );
    if( it == m_map.end() )
        return false;
    target = extractValue<T>(key, m_data[ it -> second ].second._data );
    return true;
}


inline const Dictionary::Value& Dictionary::getUntypedValue( const std::string &key ) const
{
    auto it = m_map.find( key );
    if( it == m_map.end() )
        CSP_THROW( KeyError, "Dictionary missing key \"" << key << "\"" );

    return m_data[it->second].second._data;
}

template<typename T>
inline typename Dictionary::ReturnType<T>::Type Dictionary::get( const std::string & key, const T & default_ ) const
{
    auto it = m_map.find( key );
    if( it == m_map.end() )
        return default_;

    return extractValue<T>( key, m_data[it -> second].second._data );
}

//allowed casts
template<> inline double   Dictionary::cast<int64_t,double>( const int64_t & value )     { return value; }
template<> inline double   Dictionary::cast<uint64_t,double>( const uint64_t & value )   { return value; }
template<> inline double   Dictionary::cast<int32_t,double>( const int32_t & value )     { return value; }
template<> inline double   Dictionary::cast<uint32_t,double>( const uint32_t & value )   { return value; }
template<> inline int32_t  Dictionary::cast<uint32_t,int32_t>( const uint32_t & value )
{
    if( value > (uint32_t) std::numeric_limits<int32_t>::max() )
        CSP_THROW( RangeError, "Dictionary value for uint32_t ( " << value << " ) is out of range for int32_t cast" );
    return value;
}

template<> inline uint32_t  Dictionary::cast<int32_t,uint32_t>( const int32_t & value )
{
    if( value < 0 )
        CSP_THROW( RangeError, "Dictionary value for int32_t ( " << value << " ) is out of range for uint32_t cast" );
    return value;
}

template<> inline uint64_t Dictionary::cast<int32_t,uint64_t>( const int32_t & value )   { return value; }
template<> inline uint64_t Dictionary::cast<uint32_t,uint64_t>( const uint32_t & value ) { return value; }
template<> inline int64_t  Dictionary::cast<int32_t,int64_t>( const int32_t & value )    { return value; }
template<> inline int64_t  Dictionary::cast<uint32_t,int64_t>( const uint32_t & value )  { return value; }
template<> inline int64_t  Dictionary::cast<uint64_t,int64_t>( const uint64_t & value )
{
    if( value > ( uint64_t ) std::numeric_limits<int64_t>::max() )
        CSP_THROW( RangeError, "Dictionary value for uint64_t ( " << value << " ) is out of range for int64_t cast" );
    return value;
}

template<> inline uint64_t  Dictionary::cast<int64_t,uint64_t>( const int64_t & value )
{
    if( value < 0 )
        CSP_THROW( RangeError, "Dictionary value for int64_t ( " << value << " ) is out of range for uint64_t cast" );
    return value;
}

//ensure std::string is extracted by reference
template<> struct Dictionary::ReturnType<std::string> { using Type = const std::string &; };

template<typename T>
inline typename Dictionary::ReturnType<T>::Type Dictionary::extractValue( const std::string & key, const Value & value )
{
    try
    {
        return std::get<T>( value );
    }
    catch( const std::bad_variant_access & err )
    {
        try
        {
            return std::visit( []( auto && arg ) -> typename ReturnType<T>::Type {
                    using ActualT = std::decay_t<decltype(arg)>;
                    return cast<ActualT,T>( arg );
                }, value );
        }
        catch( const std::bad_variant_access & err )
        {
            std::string curtypename = std::visit( []( auto && arg ) -> std::string { return cpp_type_name<std::decay_t<decltype(arg)>>(); }, value );
            CSP_THROW( TypeError, "Dictionary type-mismatch on key \"" << key << "\".  Expected type \"" << cpp_type_name<T>() << "\" got type: \"" << curtypename << "\"" );
        }
    }
}

}

namespace std
{

template<>
struct hash<csp::Dictionary>
{
    size_t operator()( const csp::Dictionary & d ) const
    {
        return d.hash();
    }
};

}

#endif
