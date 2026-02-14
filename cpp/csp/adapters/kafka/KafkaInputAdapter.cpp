#include <csp/adapters/kafka/KafkaInputAdapter.h>

#include <iostream>

namespace csp::adapters::kafka
{

KafkaInputAdapter::KafkaInputAdapter( Engine *engine, CspTypePtr &type,
                                      PushMode pushMode, PushGroup *group,
                                      const Dictionary &properties)
    : PushPullInputAdapter( engine, type, pushMode, group,
                            properties.get<bool>( "adjust_out_of_order_time") ),
      m_includeMsgBeforeStartTime( properties.get<bool>( "include_msg_before_start_time", false ) )
{
    if( type -> type() != CspType::Type::STRUCT &&
        type -> type() != CspType::Type::STRING )
        CSP_THROW( RuntimeException, "Unsupported type: " << type -> type() );


    if( properties.exists( "meta_field_map" ) )
    {
        const CspStructType & structType = static_cast<const CspStructType &>( *type );
        const Dictionary & metaFieldMap = *properties.get<DictionaryPtr>( "meta_field_map" );

        if( !metaFieldMap.empty() && type -> type() != CspType::Type::STRUCT )
            CSP_THROW( ValueError, "meta_field_map is not supported on non-struct types" );

        if( metaFieldMap.exists( "partition" ) )
        {
            std::string partitionFieldName = metaFieldMap.get<std::string>( "partition" );
            m_partitionField = structType.meta() -> field( partitionFieldName );
            if( !m_partitionField )
                CSP_THROW( ValueError, "field " << partitionFieldName << " is not a valid field on struct type " << structType.meta() -> name() );
            if( m_partitionField -> type() -> type() != CspType::Type::INT64 )
                CSP_THROW( ValueError, "field " << partitionFieldName << " must be of type int on struct type " << structType.meta() -> name() );
        }
        if( metaFieldMap.exists( "offset" ) )
        {
            std::string offsetFieldName = metaFieldMap.get<std::string>( "offset" );
            m_offsetField = structType.meta() -> field( offsetFieldName );
            if( !m_offsetField )
                CSP_THROW( ValueError, "field " << offsetFieldName << " is not a valid field on struct type " << structType.meta() -> name() );
            if( m_offsetField -> type() -> type() != CspType::Type::INT64 )
                CSP_THROW( ValueError, "field " << offsetFieldName << " must be of type int on struct type " << structType.meta() -> name() );
        }
        if( metaFieldMap.exists( "live" ) )
        {
            std::string liveFieldName = metaFieldMap.get<std::string>( "live" );
            m_liveField = structType.meta() -> field( liveFieldName );
            if( !m_liveField )
                CSP_THROW( ValueError, "field " << liveFieldName << " is not a valid field on struct type " << structType.meta() -> name() );
            if( m_liveField -> type() -> type() != CspType::Type::BOOL )
                CSP_THROW( ValueError, "field " << liveFieldName << " must be of type bool on struct type " << structType.meta() -> name() );
        }
        if( metaFieldMap.exists( "timestamp" ) )
        {
            std::string timestampFieldName = metaFieldMap.get<std::string>( "timestamp" );
            m_timestampField = structType.meta() -> field( timestampFieldName );
            if( !m_timestampField )
                CSP_THROW( ValueError, "field " << timestampFieldName << " is not a valid field on struct type " << structType.meta() -> name() );
            if( m_timestampField -> type() -> type() != CspType::Type::DATETIME )
                CSP_THROW( ValueError, "field " << timestampFieldName << " must be of type datetime on struct type " << structType.meta() -> name() );
        }
        if( metaFieldMap.exists( "key" ) )
        {
            std::string keyFieldName = metaFieldMap.get<std::string>( "key" );
            m_keyField = structType.meta() -> field( keyFieldName );
            if( !m_keyField )
                CSP_THROW( ValueError, "field " << keyFieldName << " is not a valid field on struct type " << structType.meta() -> name() );
            if( m_keyField -> type() -> type() != CspType::Type::STRING )
                CSP_THROW( ValueError, "field " << keyFieldName << " must be of type string on struct type " << structType.meta() -> name() );
        }
        if( properties.exists( "tick_timestamp_from_field" ) )
        {
            std::string timestampFieldName = properties.get<std::string>("tick_timestamp_from_field");
            m_tickTimestampField = structType.meta() -> field( timestampFieldName );
            if( !m_tickTimestampField )
                CSP_THROW( ValueError, "field " << timestampFieldName << " is not a valid field on struct type " << structType.meta() -> name() );
            if( m_tickTimestampField -> type() -> type() != CspType::Type::DATETIME )
                CSP_THROW( ValueError, "field " << timestampFieldName << " must be of type datetime on struct type " << structType.meta() -> name() );
        }
    }

    m_converter = utils::MessageStructConverterCache::instance().create( type, properties );
}

void KafkaInputAdapter::processMessage( RdKafka::Message* message, bool live, csp::PushBatch* batch )
{
    DateTime msgTime;
    auto ts = message -> timestamp();
    if( ts.type != RdKafka::MessageTimestamp::MSG_TIMESTAMP_NOT_AVAILABLE )
        msgTime = DateTime::fromMilliseconds( ts.timestamp );

    if( dataType() -> type() == CspType::Type::STRUCT )
    {
        auto tick = m_converter -> asStruct( message -> payload(), message -> len() );

        if( m_partitionField )
            m_partitionField -> setValue( tick.get(), message -> partition() );

        if( m_offsetField )
            m_offsetField -> setValue( tick.get(), message -> offset() );

        if( m_liveField )
            m_liveField -> setValue( tick.get(), live );

        if( m_timestampField && !msgTime.isNone() )
            m_timestampField -> setValue( tick.get(), msgTime );

        if( m_keyField )
            m_keyField -> setValue( tick.get(), *message -> key() );
        
        if( m_tickTimestampField )
            msgTime = m_tickTimestampField->value<DateTime>(tick.get());

        bool pushLive = shouldPushLive(live, msgTime);
        if( shouldProcessMessage( pushLive, msgTime ) )
            pushTick(pushLive, msgTime, std::move(tick), batch);
    }
    else if( dataType() -> type() == CspType::Type::STRING )
    {
        bool pushLive = shouldPushLive(live, msgTime);
        if( shouldProcessMessage( pushLive, msgTime ) )
            pushTick( pushLive, msgTime, std::string( ( const char * ) message -> payload(), message -> len() ) );
    }
}

}
