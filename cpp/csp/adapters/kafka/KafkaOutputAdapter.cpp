#include <csp/adapters/kafka/KafkaOutputAdapter.h>
#include <csp/adapters/kafka/KafkaPublisher.h>

namespace csp::adapters::kafka
{

KafkaOutputAdapter::KafkaOutputAdapter( Engine * engine, KafkaPublisher & publisher,
                                        CspTypePtr & type, const Dictionary & properties ) : OutputAdapter( engine ),
                                                                                             m_publisher( publisher )
{
    if( !publisher.isRawBytes() )
        m_dataMapper = utils::OutputDataMapperCache::instance().create( type, *properties.get<DictionaryPtr>( "field_map" ) );
    else if( type -> type() != CspType::Type::STRING )
        CSP_THROW( TypeError, "RAW_BYTES output expected ts[str] got ts type " << type -> type() );
}

KafkaOutputAdapter::KafkaOutputAdapter( Engine * engine, KafkaPublisher & publisher,
                                        CspTypePtr & type, const Dictionary & properties,
                                        const std::string & key ) : KafkaOutputAdapter( engine, publisher, type, properties )
{
    m_publisher.setKey( key );
}

KafkaOutputAdapter::KafkaOutputAdapter( Engine * engine, KafkaPublisher & publisher,
                                        CspTypePtr &type, const Dictionary & properties,
                                        const std::vector<std::string> & keyFields ) : KafkaOutputAdapter( engine, publisher, type, properties )
{
    addFields( keyFields, type );
}

KafkaOutputAdapter::~KafkaOutputAdapter()
{
}

void KafkaOutputAdapter::addFields( const std::vector<std::string> & keyFields, CspTypePtr & type, size_t i )
{
    std::string fieldName = keyFields[ i ];
    auto structMeta = static_cast<const CspStructType &>( *type ).meta();
    auto structField = structMeta -> field( fieldName );
    if( !structField )
        CSP_THROW( InvalidArgument, "Struct type " << structMeta -> name() << " missing required field " << fieldName );

    if( i == keyFields.size() - 1 )
    {
        CSP_TRUE_OR_THROW_RUNTIME( structField -> type() -> type() == CspType::Type::STRING,
                                   "Key field must be of type string, got " << structField -> type() -> type().asString() );
        m_structFields.emplace_back( structField );
    }
    else
    {
        CSP_TRUE_OR_THROW_RUNTIME( structField -> type() -> type() == CspType::Type::STRUCT,
                                   "Non-key field must be of type struct, got " << structField -> type() -> type().asString() );
        m_structFields.emplace_back( structField );
        auto t = structField -> type();
        addFields( keyFields, t, i + 1 );
    }
}

const std::string & KafkaOutputAdapter::getKey( const Struct * struct_ )
{
    for( size_t i = 0; i < m_structFields.size() - 1; ++i )
        struct_ = m_structFields[ i ] -> value<StructPtr>( struct_ ).get();

    return m_structFields[ m_structFields.size() - 1 ] -> value<std::string>( struct_ );
}

void KafkaOutputAdapter::executeImpl()
{
    if( m_publisher.isRawBytes() )
    {
        const std::string & value = input() -> lastValueTyped<std::string>();
        m_publisher.send( value.c_str(), value.length() );
    }
    else
    {
        auto & msgWriter = m_publisher.msgWriter();
        msgWriter.processTick( *m_dataMapper, input() );
        
        if( !m_structFields.empty() )
            m_publisher.setKey( getKey( input() -> lastValueTyped<StructPtr>().get() ) );
        
        m_publisher.scheduleEndCycleEvent();
    }
}

}
