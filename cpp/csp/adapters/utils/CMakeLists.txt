set(ADAPTER_UTILS_PUBLIC_HEADERS
        JSONMessageStructConverter.h
        JSONMessageWriter.h
        MessageEnums.h
        MessageWriter.h
        MessageStructConverter.h
        ProtobufHelper.h
        ProtobufMessageStructConverter.h
        RawBytesMessageStructConverter.h
        StructAdapterInfo.h
        ValueDispatcher.h
        )

set(ADAPTER_UTILS_FILES
        JSONMessageStructConverter.cpp
        MessageWriter.cpp
        MessageEnums.cpp
        MessageStructConverter.cpp
        ProtobufHelper.cpp
        ProtobufMessageStructConverter.cpp
        RawBytesMessageStructConverter.cpp
        )

add_library(csp_adapter_utils STATIC ${ADAPTER_UTILS_FILES})
set_target_properties(csp_adapter_utils PROPERTIES PUBLIC_HEADER "${ADAPTER_UTILS_PUBLIC_HEADERS}")

find_package(Protobuf REQUIRED)
target_link_libraries(csp_adapter_utils PRIVATE protobuf::libprotoc protobuf::libprotobuf protobuf::libprotobuf-lite)

install(TARGETS csp_adapter_utils
        PUBLIC_HEADER DESTINATION include/csp/adapters/utils
        RUNTIME DESTINATION bin/
        LIBRARY DESTINATION lib/
       )
