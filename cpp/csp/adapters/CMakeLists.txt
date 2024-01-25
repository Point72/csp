
if(CSP_BUILD_KAFKA_ADAPTER)
    add_subdirectory(kafka)
endif()

if(CSP_BUILD_PARQUET_ADAPTER)
    add_subdirectory(parquet)
endif()

add_subdirectory(utils)
