#ifndef _IN_CSP_ADAPTERS_STATUSADAPTER_H
#define _IN_CSP_ADAPTERS_STATUSADAPTER_H

#include <csp/engine/PushInputAdapter.h>
#include <string>

namespace csp
{

struct StatusMessage
{
    std::shared_ptr<StructMeta> meta;
    std::shared_ptr<Int64StructField> level;
    std::shared_ptr<Int64StructField> statusCode;
    std::shared_ptr<StringStructField> msg;
};

struct StatusLevelTraits
{
    enum _enum : unsigned char
    {
        DEBUG    = 0,
        INFO     = 1,
        WARNING  = 2,
        ERROR    = 3,
        CRITICAL = 4
    };
};

using StatusLevel = csp::Enum<StatusLevelTraits>;

class StatusAdapter : public PushInputAdapter
{
public:
    StatusAdapter( Engine * engine, CspTypePtr & type, PushMode pushMode, PushGroup * group = nullptr ) : PushInputAdapter( engine, type, pushMode, group )
    {
        if( type -> type() != CspType::Type::STRUCT )
            CSP_THROW( ValueError, "Status Adapter can only be created with struct ts type" );

        const CspStructType & sType = static_cast<const CspStructType&>( *type.get() );

        auto meta = sType.meta();

        m_statusAccess.meta       = meta;
        m_statusAccess.level      = meta -> getMetaField<int64_t>( "level", "Status" );
        m_statusAccess.statusCode = meta -> getMetaField<int64_t>( "status_code", "Status" );
        m_statusAccess.msg        = meta -> getMetaField<std::string>( "msg", "Status" );
    }

    void pushStatus( int64_t level, int64_t statusCode, const std::string & msg, PushBatch *batch = nullptr )
    {
        StructPtr data = m_statusAccess.meta -> create();
        m_statusAccess.level      -> setValue( data.get(), level );
        m_statusAccess.statusCode -> setValue( data.get(), statusCode );
        m_statusAccess.msg        -> setValue( data.get(), msg );

        pushTick( std::move( data ), batch );
    }
private:
    StatusMessage m_statusAccess;
};

}

#endif