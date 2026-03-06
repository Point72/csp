set(PARQUET_HEADER_FILES
        ArrowIPCFileReaderWrapper.h
        ArrowIPCFileWriterWrapper.h
        ArrowSingleColumnArrayBuilder.h
        DialectGenericListReaderInterface.h
        DialectGenericListWriterInterface.h
        FileReaderWrapper.h
        FileWriterWrapper.h
        ParquetFileWriterWrapper.h
        ParquetFileReaderWrapper.h
        FileWriterWrapperContainer.h
        ParquetDictBasketOutputWriter.h
        ParquetInputAdapterManager.h
        ParquetOutputAdapter.h
        ParquetOutputAdapterManager.h
        ParquetOutputFilenameAdapter.h
        ParquetReader.h
        ParquetReaderColumnAdapter.h
        ParquetStatusUtils.h
        ParquetWriter.h
)

set(PARQUET_SOURCE_FILES
        ArrowIPCFileReaderWrapper.cpp
        ArrowIPCFileWriterWrapper.cpp
        FileReaderWrapper.cpp
        FileWriterWrapper.cpp
        ParquetFileWriterWrapper.cpp
        ParquetFileReaderWrapper.cpp
        FileWriterWrapperContainer.cpp
        ParquetDictBasketOutputWriter.cpp
        ParquetInputAdapterManager.cpp
        ParquetOutputAdapter.cpp
        ParquetOutputAdapterManager.cpp
        ParquetOutputFilenameAdapter.cpp
        ParquetReader.cpp
        ParquetReaderColumnAdapter.cpp
        ParquetWriter.cpp
        ${PARQUET_HEADER_FILES}
        )

add_library(csp_parquet_adapter STATIC ${PARQUET_SOURCE_FILES})
set_target_properties(csp_parquet_adapter PROPERTIES PUBLIC_HEADER "${PARQUET_HEADER_FILES}")
target_include_directories(csp_parquet_adapter PRIVATE ${ARROW_INCLUDE_DIR} ${PARQUET_INCLUDE_DIR})

target_link_libraries(csp_parquet_adapter PRIVATE csp_adapter_utils thrift::thrift lz4::lz4 utf8proc::utf8proc ${BROTLI_STATIC_LIB} ${CSP_PARQUET_LINK_LIBS} ${CSP_ARROW_LINK_LIBS})

install(TARGETS csp_parquet_adapter
        PUBLIC_HEADER DESTINATION include/csp/adapters/parquet
        RUNTIME DESTINATION ${CSP_RUNTIME_INSTALL_SUBDIR}
        LIBRARY DESTINATION lib/
       )
