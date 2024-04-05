
if(CSP_BUILD_KAFKA_ADAPTER)
    add_subdirectory(kafka)
endif()

if(CSP_BUILD_PARQUET_ADAPTER)
    add_subdirectory(parquet)
endif()

if(CSP_BUILD_WS_CLIENT_ADAPTER)
    add_subdirectory(websocket)
endif()

add_subdirectory(utils)
