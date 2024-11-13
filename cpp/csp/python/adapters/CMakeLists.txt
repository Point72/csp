
if(CSP_BUILD_KAFKA_ADAPTER)
    add_library(kafkaadapterimpl SHARED kafkaadapterimpl.cpp)
    target_link_libraries(kafkaadapterimpl csp_core csp_engine cspimpl csp_kafka_adapter)
    install(TARGETS kafkaadapterimpl RUNTIME DESTINATION ${CSP_RUNTIME_INSTALL_SUBDIR})
endif()

if(CSP_BUILD_PARQUET_ADAPTER)
    set(VENDORED_PYARROW_ROOT "${CMAKE_SOURCE_DIR}/cpp/csp/python/adapters/vendored/pyarrow-16.0.0/")
    set(ARROW_PYTHON_SRCS
        ${VENDORED_PYARROW_ROOT}/arrow/python/arrow_to_pandas.cc
        ${VENDORED_PYARROW_ROOT}/arrow/python/benchmark.cc
        ${VENDORED_PYARROW_ROOT}/arrow/python/common.cc
        ${VENDORED_PYARROW_ROOT}/arrow/python/datetime.cc
        ${VENDORED_PYARROW_ROOT}/arrow/python/decimal.cc
        ${VENDORED_PYARROW_ROOT}/arrow/python/deserialize.cc
        ${VENDORED_PYARROW_ROOT}/arrow/python/extension_type.cc
        ${VENDORED_PYARROW_ROOT}/arrow/python/gdb.cc
        ${VENDORED_PYARROW_ROOT}/arrow/python/helpers.cc
        ${VENDORED_PYARROW_ROOT}/arrow/python/inference.cc
        ${VENDORED_PYARROW_ROOT}/arrow/python/init.cc
        ${VENDORED_PYARROW_ROOT}/arrow/python/io.cc
        ${VENDORED_PYARROW_ROOT}/arrow/python/ipc.cc
        ${VENDORED_PYARROW_ROOT}/arrow/python/numpy_convert.cc
        ${VENDORED_PYARROW_ROOT}/arrow/python/numpy_to_arrow.cc
        ${VENDORED_PYARROW_ROOT}/arrow/python/python_to_arrow.cc
        ${VENDORED_PYARROW_ROOT}/arrow/python/pyarrow.cc
        ${VENDORED_PYARROW_ROOT}/arrow/python/serialize.cc
        ${VENDORED_PYARROW_ROOT}/arrow/python/csv.cc
        ${VENDORED_PYARROW_ROOT}/arrow/python/filesystem.cc)
    add_library(parquetadapterimpl SHARED parquetadapterimpl.cpp ${ARROW_PYTHON_SRCS})
    # Ignore warning regarding static datetime API initialization coming from vendored Arrow code
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set_target_properties(parquetadapterimpl PROPERTIES COMPILE_OPTIONS "-Wno-unused-variable")
    endif()
    target_link_libraries(parquetadapterimpl csp_core csp_engine cspimpl csp_parquet_adapter)
    target_include_directories(parquetadapterimpl PUBLIC ${ARROW_INCLUDE_DIR} ${PARQUET_INCLUDE_DIR} "${VENDORED_PYARROW_ROOT}")
    target_compile_definitions(parquetadapterimpl PUBLIC ARROW_PYTHON_STATIC)
    install(TARGETS parquetadapterimpl RUNTIME DESTINATION ${CSP_RUNTIME_INSTALL_SUBDIR} )
endif()

if(CSP_BUILD_WS_CLIENT_ADAPTER)
    add_library(websocketadapterimpl SHARED websocketadapterimpl.cpp)
    target_link_libraries(websocketadapterimpl csp_core csp_engine cspimpl csp_websocket_client_adapter)
    install(TARGETS websocketadapterimpl RUNTIME DESTINATION ${CSP_RUNTIME_INSTALL_SUBDIR})
endif()
