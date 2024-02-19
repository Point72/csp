
if(CSP_BUILD_KAFKA_ADAPTER)
    add_library(kafkaadapterimpl SHARED kafkaadapterimpl.cpp)
    target_link_libraries(kafkaadapterimpl csp_core csp_engine cspimpl csp_kafka_adapter)
    install(TARGETS kafkaadapterimpl RUNTIME DESTINATION bin/ LIBRARY DESTINATION lib/)
endif()

if(CSP_BUILD_WS_CLIENT_ADAPTER)
    add_library(wsclientadapterimpl SHARED wsclientadapterimpl.cpp)
    target_link_libraries(wsclientadapterimpl csp_core csp_engine cspimpl csp_websocket_client_adapter)
    install(TARGETS wsclientadapterimpl RUNTIME DESTINATION bin/ LIBRARY DESTINATION lib/)
endif()

if(CSP_BUILD_PARQUET_ADAPTER)
    set(VENDORED_PYARROW_ROOW "${CMAKE_SOURCE_DIR}/cpp/csp/python/adapters/vendored/pyarrow-7.0.0/")
    set(ARROW_PYTHON_SRCS
        ${VENDORED_PYARROW_ROOW}/arrow/python/arrow_to_pandas.cc
        ${VENDORED_PYARROW_ROOW}/arrow/python/benchmark.cc
        ${VENDORED_PYARROW_ROOW}/arrow/python/common.cc
        ${VENDORED_PYARROW_ROOW}/arrow/python/datetime.cc
        ${VENDORED_PYARROW_ROOW}/arrow/python/decimal.cc
        ${VENDORED_PYARROW_ROOW}/arrow/python/deserialize.cc
        ${VENDORED_PYARROW_ROOW}/arrow/python/extension_type.cc
        ${VENDORED_PYARROW_ROOW}/arrow/python/gdb.cc
        ${VENDORED_PYARROW_ROOW}/arrow/python/helpers.cc
        ${VENDORED_PYARROW_ROOW}/arrow/python/inference.cc
        ${VENDORED_PYARROW_ROOW}/arrow/python/init.cc
        ${VENDORED_PYARROW_ROOW}/arrow/python/io.cc
        ${VENDORED_PYARROW_ROOW}/arrow/python/ipc.cc
        ${VENDORED_PYARROW_ROOW}/arrow/python/numpy_convert.cc
        ${VENDORED_PYARROW_ROOW}/arrow/python/numpy_to_arrow.cc
        ${VENDORED_PYARROW_ROOW}/arrow/python/python_to_arrow.cc
        ${VENDORED_PYARROW_ROOW}/arrow/python/pyarrow.cc
        ${VENDORED_PYARROW_ROOW}/arrow/python/serialize.cc
        ${VENDORED_PYARROW_ROOW}/arrow/python/csv.cc
        ${VENDORED_PYARROW_ROOW}/arrow/python/filesystem.cc)
    add_library(parquetadapterimpl SHARED parquetadapterimpl.cpp ${ARROW_PYTHON_SRCS})
    target_link_libraries(parquetadapterimpl csp_core csp_engine cspimpl csp_parquet_adapter)
    target_include_directories(parquetadapterimpl PUBLIC ${ARROW_INCLUDE_DIR} ${PARQUET_INCLUDE_DIR} "${VENDORED_PYARROW_ROOW}")
    install(TARGETS parquetadapterimpl RUNTIME DESTINATION bin/ LIBRARY DESTINATION lib/)
endif()
