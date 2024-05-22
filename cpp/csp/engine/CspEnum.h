#ifndef _IN_CSP_ENGINE_CSPENUM_H
#define _IN_CSP_ENGINE_CSPENUM_H

#include <csp/core/Exception.h>
#include <csp/core/Hash.h>
#include <limits>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

namespace csp
{

class CspEnumMeta;

class CspEnumInstance
{
public:
    CspEnumInstance( std::string name, int64_t value, csp::CspEnumMeta * meta ) : m_name( name ), m_value( value ), m_meta( meta ) {}
    CspEnumInstance( CspEnumInstance && o ) : m_name( o.m_name ), m_value( o.m_value ), m_meta( o.m_meta ) {}
    CspEnumInstance( const CspEnumInstance & o ) = delete;
    CspEnumInstance & operator=( CspEnumInstance o ) = delete;

    int64_t value() const { return m_value; }
    const std::string & name() const { return m_name; }
    const CspEnumMeta * meta() const { return m_meta; }

private:
    std::string m_name;
    int64_t m_value;
    CspEnumMeta * m_meta;
};

//As an optimization we do NOT attach meta or value to every instance of an enum.  Instead, the enum
//holds only a pointer to a singleton CspEnumInstance, which holds the value, name, and meta pointer.
class CspEnum
{
public:
    CspEnum();
    CspEnum( const CspEnum & other ) { m_instance = other.m_instance; }

    const int64_t value() const { return m_instance -> value(); }
    const CspEnumMeta * meta() const { return m_instance -> meta(); }
    const std::string & name() const { return m_instance -> name(); }

    // check instance to ensure value and meta are the same
    bool operator==( const CspEnum & rhs ) const { return m_instance == rhs.m_instance; }
    bool operator!=( const CspEnum & rhs ) const { return m_instance != rhs.m_instance; }

protected:
    explicit CspEnum( const CspEnumInstance * instance ) : m_instance( instance ) {}

    const CspEnumInstance * m_instance;

    friend class CspEnumMeta;
};

std::ostream &operator<<( std::ostream &os, const CspEnum & rhs );

class CspEnumMeta
{
public:
    using ValueDef = std::unordered_map<std::string,int64_t>;
    using Ptr = std::shared_ptr<CspEnumMeta>;

    CspEnumMeta( const std::string & name, const ValueDef & def );
    virtual ~CspEnumMeta();

    const std::string & name() const { return m_name; }
    size_t size() const              { return m_mapping.size(); }

    //note this will throw on invalid values
    CspEnum fromString( const char * key ) const
    {
        auto it = m_mapping.find( key );
        if( it == m_mapping.end() )
            CSP_THROW( ValueError, "Unrecognized enum name " << key << " for enum " << m_name );
        return CspEnum( it -> second -> second.get() );
    }

    CspEnum create( int64_t value ) const
    {
        auto found = m_instanceMap.find( value );
        if( found == m_instanceMap.end() )
            CSP_THROW( RuntimeException, "Unrecognized value " << value << " for enum " << m_name );
        return CspEnum( found -> second.get() );
    }

private:
    using InstanceMapping = std::unordered_map<int64_t, std::shared_ptr<CspEnumInstance>>;
    using Mapping         = std::unordered_map<const char *,InstanceMapping::iterator, hash::CStrHash, hash::CStrEq >;

    std::string    m_name;
    Mapping        m_mapping;

    InstanceMapping m_instanceMap;
};

}

namespace std
{

template<>
struct hash<csp::CspEnum>
{
    size_t operator()( csp::CspEnum e ) const
    {
        return std::hash<int64_t>()( e.value() );
    }
};

}

#endif
