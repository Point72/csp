#ifndef _IN_CSP_ADAPTERS_KAFKA_KAFKAOUTPUTADAPTER_H
#define _IN_CSP_ADAPTERS_KAFKA_KAFKAOUTPUTADAPTER_H

#include <csp/adapters/utils/MessageWriter.h>
#include <csp/engine/Dictionary.h>
#include <csp/engine/OutputAdapter.h>
#include <string>

namespace csp::adapters::kafka
{

class KafkaPublisher;

class KafkaOutputAdapter final: public OutputAdapter
{
public:
    KafkaOutputAdapter( Engine * engine, KafkaPublisher & publisher, CspTypePtr & type, const Dictionary & properties, const std::string & key );
    KafkaOutputAdapter( Engine * engine, KafkaPublisher & publisher, CspTypePtr & type, const Dictionary & properties, const std::vector<std::string> & keyFields );
    ~KafkaOutputAdapter();

    void executeImpl() override;

    const char * name() const override { return "KafkaOutputAdapter"; }

private:
    KafkaOutputAdapter( Engine * engine, KafkaPublisher & publisher, CspTypePtr & type, const Dictionary & properties );
    void addFields( const std::vector<std::string> & keyFields, CspTypePtr & type, size_t i = 0 );
    const std::string & getKey( const Struct * struct_ );

    KafkaPublisher &            m_publisher;
    utils::OutputDataMapperPtr  m_dataMapper;
    std::vector<StructFieldPtr> m_structFields;
};

}

#endif
