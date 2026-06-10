set(PARQUET_HEADER_FILES
        ArrowBackedArrayBuilder.h
        ArrowSingleColumnArrayBuilder.h
        FileNameGeneratorReplicator.h
        ParquetDictBasketOutputWriter.h
        ParquetInputAdapterManager.h
        ParquetOutputAdapter.h
        ParquetOutputAdapterManager.h
        ParquetOutputFilenameAdapter.h
        ParquetStatusUtils.h
        ParquetWriter.h
        RecordBatchSink.h
        RecordBatchStreamSource.h
        RecordBatchWithFlag.h
)

set(PARQUET_SOURCE_FILES
        ArrowBackedArrayBuilder.cpp
        ParquetDictBasketOutputWriter.cpp
        ParquetInputAdapterManager.cpp
        ParquetOutputAdapter.cpp
        ParquetOutputAdapterManager.cpp
        ParquetOutputFilenameAdapter.cpp
        ParquetWriter.cpp
        ${PARQUET_HEADER_FILES}
        )

add_library(csp_parquet_adapter STATIC ${PARQUET_SOURCE_FILES})
set_target_properties(csp_parquet_adapter PROPERTIES PUBLIC_HEADER "${PARQUET_HEADER_FILES}")
target_include_directories(csp_parquet_adapter PRIVATE ${ARROW_INCLUDE_DIR} ${PARQUET_INCLUDE_DIR})

target_link_libraries(csp_parquet_adapter PRIVATE csp_arrow_adapter csp_adapter_utils thrift::thrift lz4::lz4 utf8proc::utf8proc ${BROTLI_STATIC_LIB} ${CSP_PARQUET_LINK_LIBS} ${CSP_ARROW_LINK_LIBS})

install(TARGETS csp_parquet_adapter
        PUBLIC_HEADER DESTINATION include/csp/adapters/parquet
        RUNTIME DESTINATION ${CSP_RUNTIME_INSTALL_SUBDIR}
        LIBRARY DESTINATION lib/
       )
