
if(CSP_BUILD_KAFKA_ADAPTER)
    add_library(kafkaadapterimpl SHARED kafkaadapterimpl.cpp)
    target_link_libraries(kafkaadapterimpl csp_core csp_engine cspimpl csp_kafka_adapter)
    install(TARGETS kafkaadapterimpl RUNTIME DESTINATION ${CSP_RUNTIME_INSTALL_SUBDIR})
endif()

if(CSP_BUILD_PARQUET_ADAPTER)
    add_library(parquetadapterimpl SHARED parquetadapterimpl.cpp)
    target_link_libraries(parquetadapterimpl csp_core csp_engine cspimpl csp_parquet_adapter)
    target_include_directories(parquetadapterimpl PUBLIC ${ARROW_INCLUDE_DIR} ${PARQUET_INCLUDE_DIR})
    install(TARGETS parquetadapterimpl RUNTIME DESTINATION ${CSP_RUNTIME_INSTALL_SUBDIR} )
endif()

if(CSP_BUILD_WS_CLIENT_ADAPTER)
    add_library(websocketadapterimpl SHARED websocketadapterimpl.cpp)
    target_link_libraries(websocketadapterimpl csp_core csp_engine cspimpl csp_websocket_client_adapter)
    install(TARGETS websocketadapterimpl RUNTIME DESTINATION ${CSP_RUNTIME_INSTALL_SUBDIR})
endif()
