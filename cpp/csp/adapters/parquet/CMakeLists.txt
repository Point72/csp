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

find_package(Arrow REQUIRED)
find_package(Parquet REQUIRED)
find_package(Thrift REQUIRED)
find_package(lz4 REQUIRED)
find_package(utf8proc REQUIRED)
find_package(Brotli REQUIRED)

if(WIN32)
   if(CSP_USE_VCPKG)
     set(ARROW_PACKAGES_TO_LINK Arrow::arrow_static Parquet::parquet_static )
     target_compile_definitions(csp_parquet_adapter PUBLIC ARROW_STATIC)
     target_compile_definitions(csp_parquet_adapter PUBLIC PARQUET_STATIC)
   else()
     # use dynamic variants
     # Until we manage to get the fix for ws3_32.dll in arrow-16 into conda, manually fix the error here
     get_target_property(LINK_LIBS Arrow::arrow_shared INTERFACE_LINK_LIBRARIES)
     string(REPLACE "ws2_32.dll" "ws2_32" FIXED_LINK_LIBS "${LINK_LIBS}")
     set_target_properties(Arrow::arrow_shared PROPERTIES INTERFACE_LINK_LIBRARIES "${FIXED_LINK_LIBS}")
     set(ARROW_PACKAGES_TO_LINK parquet_shared arrow_shared)
   endif()
else()
  if(CSP_USE_VCPKG)
    # use static variants
    set(ARROW_PACKAGES_TO_LINK parquet_static arrow_static)
  else()
    # use dynamic variants
    set(ARROW_PACKAGES_TO_LINK parquet arrow)
  endif()
endif()

target_link_libraries(csp_parquet_adapter PRIVATE csp_adapter_utils thrift::thrift lz4::lz4 utf8proc::utf8proc ${BROTLI_STATIC_LIB} ${ARROW_PACKAGES_TO_LINK})

install(TARGETS csp_parquet_adapter
        PUBLIC_HEADER DESTINATION include/csp/adapters/parquet
        RUNTIME DESTINATION ${CSP_RUNTIME_INSTALL_SUBDIR}
        LIBRARY DESTINATION lib/
       )
