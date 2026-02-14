#ifndef _IN_CSP_ADAPTERS_KAFKA_KAFKAINPUTADAPTER_H
#define _IN_CSP_ADAPTERS_KAFKA_KAFKAINPUTADAPTER_H

#include <csp/adapters/utils/MessageStructConverter.h>
#include <csp/engine/PushPullInputAdapter.h>
#include <csp/engine/Struct.h>
#include <librdkafka/rdkafkacpp.h>

namespace csp::adapters::kafka
{

class KafkaInputAdapter final: public PushPullInputAdapter
{
public:
    KafkaInputAdapter( Engine * engine, CspTypePtr & type,
                       PushMode pushMode, PushGroup * group,
                       const Dictionary & properties );

    void processMessage( RdKafka::Message* message, bool live, csp::PushBatch* batch );

private:
    inline bool shouldPushLive( bool pushLive, DateTime msgTime )
    {
        return pushLive || flaggedLive() || msgTime.isNone();
    }

    inline bool shouldProcessMessage( bool pushLive, DateTime msgTime )
    {
        // This function encapsulates the logic for determining if a message should be processed
        // live always goes through, otherwise, we filter out
        // messages before engine starttime when not m_includeMsgBeforeStartTime
        return pushLive || m_includeMsgBeforeStartTime || 
               msgTime >= rootEngine()->startTime();
    }

    utils::MessageStructConverterPtr m_converter;
    StructFieldPtr m_partitionField;
    StructFieldPtr m_offsetField;
    StructFieldPtr m_liveField;
    StructFieldPtr m_timestampField;
    StructFieldPtr m_keyField;
    StructFieldPtr m_tickTimestampField;

    bool m_includeMsgBeforeStartTime;
};

}

#endif
