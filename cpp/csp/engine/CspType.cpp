#include <csp/engine/CspType.h>
#include <mutex>

namespace csp
{

INIT_CSP_ENUM( CspType::Type,
           "UNKNOWN",
           "BOOL",
           "INT8",
           "UINT8",
           "INT16",
           "UINT16",
           "INT32",
           "UINT32",
           "INT64",
           "UINT64",
           "DOUBLE",
           "DATETIME",
           "TIMEDELTA",
           "DATE",
           "TIME",
           "ENUM",
           "STRING",
           "STRUCT",
           "ARRAY",
           "DIALECT_GENERIC" 
    );

CspTypePtr & CspArrayType::create( const CspTypePtr & elemType )
{
    using Cache = std::unordered_map<const CspType*,CspTypePtr>;
    static std::mutex s_mutex;
    static Cache      s_cache;

    std::lock_guard<std::mutex> guard( s_mutex );
    auto rv = s_cache.emplace( elemType.get(), nullptr );
    if( rv.second )
        rv.first -> second = std::make_shared<CspArrayType>( elemType );
    return rv.first -> second;
}

}
