#ifndef _IN_CSP_CORE_DIALECTGENERICTYPE_H
#define _IN_CSP_CORE_DIALECTGENERICTYPE_H

#include <csp/core/Platform.h>
#include <cstddef>
#include <functional>

namespace csp
{

// Note that this is intentionally exported with CSPTYPESIMPL_EXPORT because the actual impl is compiled into the dialect specific impl library
struct CSPTYPESIMPL_EXPORT DialectGenericType
{
public:
    DialectGenericType();
    ~DialectGenericType();
    DialectGenericType( const DialectGenericType & rhs );
    DialectGenericType( DialectGenericType && rhs );

    DialectGenericType & operator=( const DialectGenericType & rhs );
    DialectGenericType & operator=( DialectGenericType && rhs );

    DialectGenericType deepcopy() const;

    bool operator==( const DialectGenericType & rhs ) const;
    bool operator!=( const DialectGenericType & rhs ) const { return !( (*this)==rhs); }

    size_t hash() const;

private:
    [[maybe_unused]] void* m_data;
};

CSPTYPESIMPL_EXPORT std::ostream & operator<<( std::ostream & o, const DialectGenericType & obj );

}

namespace std
{

template<>
struct hash<csp::DialectGenericType>
{
    size_t operator()( const csp::DialectGenericType & obj ) const
    {
        return obj.hash();
    }
};

}

#endif
