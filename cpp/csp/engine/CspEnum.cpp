#include <csp/engine/CspEnum.h>

namespace csp
{

static CspEnumInstance s_stubInstance( "", 0, new CspEnumMeta( "", CspEnumMeta::ValueDef{{ "", 0 }} ) );

CspEnum::CspEnum()
{
    m_instance = &s_stubInstance;
}

CspEnumMeta::CspEnumMeta( const std::string & name, const ValueDef & def ) : m_name( name )
{
    for( auto [ key,value ] : def )
    {
        auto [rit, inserted] = m_instanceMap.emplace( value, std::make_shared<CspEnumInstance>( key, value, this ) );
        if( !inserted )
            CSP_THROW( TypeError, "CspEnum type " << name << " defined with multiple entries for " << value );

        m_mapping[ rit -> second -> name().c_str() ] = rit;
    }
}

CspEnumMeta::~CspEnumMeta()
{
}

std::ostream &operator<<( std::ostream &os, const CspEnum & rhs )
{
    os << rhs.name();
    return os;
};

}
