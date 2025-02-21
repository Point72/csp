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
# See https://github.com/Point72/csp/issues/454
if(WIN32)
    set(CSP_LIB_TYPE STATIC)
else()
    set(CSP_LIB_TYPE SHARED)
endif()

add_library(csp_adapter_utils ${CSP_LIB_TYPE} ${ADAPTER_UTILS_FILES})
set_target_properties(csp_adapter_utils PROPERTIES PUBLIC_HEADER "${ADAPTER_UTILS_PUBLIC_HEADERS}" PREFIX lib)

find_package(Protobuf REQUIRED)
target_link_libraries(csp_adapter_utils PRIVATE protobuf::libprotoc protobuf::libprotobuf protobuf::libprotobuf-lite)

install(TARGETS csp_adapter_utils
        PUBLIC_HEADER DESTINATION include/csp/adapters/utils
        RUNTIME DESTINATION ${CSP_RUNTIME_INSTALL_SUBDIR}
        LIBRARY DESTINATION lib/
)
