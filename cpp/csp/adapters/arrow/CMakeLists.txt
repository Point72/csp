set(ARROW_ADAPTER_HEADER_FILES
        ArrowFieldReader.h
        ArrowFieldWriter.h
        RecordBatchToStruct.h
        StructToRecordBatch.h
)

set(ARROW_ADAPTER_SOURCE_FILES
        ArrowFieldReader.cpp
        ArrowFieldWriter.cpp
        RecordBatchToStruct.cpp
        StructToRecordBatch.cpp
        ${ARROW_ADAPTER_HEADER_FILES}
)

add_library(csp_arrow_adapter STATIC ${ARROW_ADAPTER_SOURCE_FILES})
set_target_properties(csp_arrow_adapter PROPERTIES PUBLIC_HEADER "${ARROW_ADAPTER_HEADER_FILES}")

find_package(Arrow REQUIRED)

if(WIN32)
   if(CSP_USE_VCPKG)
     set(ARROW_PACKAGES_TO_LINK_ARROW Arrow::arrow_static)
     target_compile_definitions(csp_arrow_adapter PUBLIC ARROW_STATIC)
   else()
     set(ARROW_PACKAGES_TO_LINK_ARROW arrow_shared)
   endif()
else()
  if(CSP_USE_VCPKG)
    set(ARROW_PACKAGES_TO_LINK_ARROW arrow_static)
  else()
    set(ARROW_PACKAGES_TO_LINK_ARROW arrow)
  endif()
endif()

target_link_libraries(csp_arrow_adapter PRIVATE csp_core csp_types csp_engine ${ARROW_PACKAGES_TO_LINK_ARROW})

install(TARGETS csp_arrow_adapter
        PUBLIC_HEADER DESTINATION include/csp/adapters/arrow
        RUNTIME DESTINATION ${CSP_RUNTIME_INSTALL_SUBDIR}
        LIBRARY DESTINATION lib/
       )
