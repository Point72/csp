set(KAFKA_HEADER_FILES
        KafkaAdapterManager.h
        KafkaConsumer.h
        KafkaInputAdapter.h
        KafkaOutputAdapter.h
        KafkaPublisher.h
        KafkaSubscriber.h
)

set(KAFKA_SOURCE_FILES
        KafkaAdapterManager.cpp
        KafkaConsumer.cpp
        KafkaInputAdapter.cpp
        KafkaOutputAdapter.cpp
        KafkaPublisher.cpp
        KafkaSubscriber.cpp
        ${KAFKA_HEADER_FILES}
)

add_library(csp_kafka_adapter STATIC ${KAFKA_SOURCE_FILES})
set_target_properties(csp_kafka_adapter PROPERTIES PUBLIC_HEADER "${KAFKA_HEADER_FILES}")

find_package(DepsKafkaAdapter REQUIRED)

target_link_libraries(csp_kafka_adapter PUBLIC csp_adapter_utils RdKafka::rdkafka RdKafka::rdkafka++)

install(TARGETS csp_kafka_adapter
        PUBLIC_HEADER DESTINATION include/csp/adapters/kafka
        RUNTIME DESTINATION ${CSP_RUNTIME_INSTALL_SUBDIR}
        LIBRARY DESTINATION lib/
       )
