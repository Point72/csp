set(ADAPTER_UTILS_PUBLIC_HEADERS
        AvroMessageStructConverter.h
        AvroMessageWriter.h
        JSONMessageStructConverter.h
        JSONMessageWriter.h
        MessageEnums.h
        MessageWriter.h
        MessageStructConverter.h
        RawBytesMessageStructConverter.h
        StructAdapterInfo.h
        ValueDispatcher.h
        )

set(ADAPTER_UTILS_FILES
        JSONMessageStructConverter.cpp
        MessageWriter.cpp
        MessageEnums.cpp
        MessageStructConverter.cpp
        RawBytesMessageStructConverter.cpp
        )

find_package(DepsKafkaAdapter REQUIRED)

if(CSP_KAFKA_ENABLE_AVRO)
    list(APPEND ADAPTER_UTILS_FILES AvroMessageStructConverter.cpp)
endif()

add_library(csp_adapter_utils STATIC ${ADAPTER_UTILS_FILES})
set_target_properties(csp_adapter_utils PROPERTIES PUBLIC_HEADER "${ADAPTER_UTILS_PUBLIC_HEADERS}" PREFIX lib)

if(CSP_KAFKA_ENABLE_AVRO)
    target_compile_definitions(csp_adapter_utils PUBLIC CSP_USE_AVRO)
    target_link_libraries(csp_adapter_utils PUBLIC ${AVRO_LIBRARIES})
endif()

install(TARGETS csp_adapter_utils
        PUBLIC_HEADER DESTINATION include/csp/adapters/utils
        RUNTIME DESTINATION ${CSP_RUNTIME_INSTALL_SUBDIR}
        LIBRARY DESTINATION lib/
)
